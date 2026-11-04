/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decoding_task_streaming_node.h"
#include "decoding_task_graph_utils.h"
#include "decoding_task_node_kind_registry.h"

#include <algorithm>
#include <stdexcept>

namespace cudaq::qec {

namespace {

// Normalize either decoder result representation into logical observable bits.
logical_outcome project_decoder_result(decoder_result result,
                                       const sparse_binary_matrix &o,
                                       bool already_observables) {
  logical_outcome outcome;
  outcome.converged = result.converged;
  outcome.bits.assign(o.num_rows(), 0);
  if (already_observables) {
    if (result.result.size() != outcome.bits.size())
      throw std::out_of_range("streaming node: decoder returned " +
                              std::to_string(result.result.size()) +
                              " observables, expected " +
                              std::to_string(outcome.bits.size()));
    for (std::size_t i = 0; i < outcome.bits.size(); ++i)
      outcome.bits[i] =
          static_cast<uint8_t>(convert_soft_to_hard(result.result[i]));
    return outcome;
  }
  if (o.num_cols() > result.result.size())
    throw std::out_of_range("streaming node: decoder returned " +
                            std::to_string(result.result.size()) +
                            " values, but O references " +
                            std::to_string(o.num_cols()) + " fault columns");
  project_into(csr_rows{o}, result.result.data(), outcome.bits.data());
  return outcome;
}

// Describe and validate the configuration accepted by the streaming node kind.
decoding::config::decoder_schema streaming_node_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"num_syndromes_per_round", param_kind::uint64});
  schema.validate = [](const cudaqx::heterogeneous_map &params) {
    if (params.contains("num_syndromes_per_round") &&
        params.get<std::size_t>("num_syndromes_per_round") == 0)
      throw std::runtime_error(
          "streaming: num_syndromes_per_round must be positive");
  };
  return schema;
}

const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "streaming";
  spec.aliases = {"Streaming"};
  // Native correction access always produces logical observables.
  spec.inner_decoder_observable_policy = [](const cudaqx::heterogeneous_map &) {
    return true;
  };
  register_node_kind(
      std::move(spec), streaming_node_params(), [](node_build_context &ctx) {
        const auto &params = *ctx.node_params;
        const auto spr = static_cast<uint32_t>(
            params.get<std::size_t>("num_syndromes_per_round", ctx.spr));
        if (spr == 0)
          throw std::runtime_error(
              "task graph streaming node requires a nonzero syndrome round");
        return std::make_unique<streaming_decoder_node>(
            std::move(ctx.d), std::move(ctx.o), std::move(ctx.dec), spr,
            ctx.num_measurements == 0, ctx.dem_rounds);
      });
  return true;
}();

} // namespace

// Validate the decoder-facing matrices and allocate persistent streaming state.
// When rounds of different widths divide D into no fixed windows, a decoder
// that ends its own shots has the node report exact intervals instead. Such a
// decoder's node knows an untagged shot's rounds: its template's, or else
// its schedule's.
streaming_decoder_node::streaming_decoder_node(
    std::shared_ptr<const sparse_binary_matrix> d,
    std::shared_ptr<const sparse_binary_matrix> o, std::unique_ptr<decoder> dec,
    uint32_t spr, bool equal_rounds, uint32_t num_rounds)
    : d_(std::move(d)), o_(std::move(o)), dec_(std::move(dec)), spr_(spr),
      num_rounds_(num_rounds) {
  if (!d_ || !o_ || !dec_)
    throw std::invalid_argument("streaming node requires D, O, and a decoder");
  correction_provider_ =
      dynamic_cast<streaming_correction_provider *>(dec_.get());
  if (spr_ == 0 || (equal_rounds && d_->num_cols() % spr_ != 0))
    throw std::invalid_argument(
        "streaming node syndrome count must be divisible by its round size");
  exact_ = !equal_rounds && correction_provider_ && !streaming_schedule();
  if (const auto schedule = streaming_schedule();
      correction_provider_ && num_rounds_ == 0 && schedule)
    num_rounds_ = schedule->size;
  if (exact_ && num_rounds_ == 0)
    throw std::invalid_argument("streaming node over rounds of different "
                                "widths needs its shot's round count");
  if (dec_->get_num_observables() != o_->num_rows())
    throw std::invalid_argument(
        "streaming node decoder and O must declare the same observable count");
  if (dec_->get_num_msyn_per_decode() != d_->num_cols())
    throw std::invalid_argument(
        "streaming node decoder syndrome size must match D column count");
  dets_.assign(d_->num_rows(), 0.0f);
  correction_frame_.assign(o_->num_rows(), 0);
  last_output_ = logical_outcome{};
  last_output_.get<logical_outcome>().bits.assign(o_->num_rows(), 0);
} // end - streaming_decoder_node()

// Forward one caller-preserved raw chunk, with its round type, and publish any
// completed correction.  An empty chunk is not forwarded: a decoder that
// overrides only the untagged form gets it with the tag dropped, counts it as
// a round, and may be handed a null buffer.
bool streaming_decoder_node::enqueue(const uint8_t *msyn, std::size_t len,
                                     round_type type) {
  if (len == 0)
    return false;
  // A tagged shot ends where the decoder says, not where it is metered.
  untagged_fill_ = 0;
  if (!dec_->enqueue_syndrome(msyn, len, type) ||
      (type != round_type::FINAL && !commits_more())) {
    pending_inner_input_ = true;
    return false;
  }

  capture_inner_corrections(type == round_type::FINAL);
  // A provisional publication is not the end of the shot. Keep a final flush
  // pending, for a session that closes without FINAL.
  pending_inner_input_ = correction_provider_ && type != round_type::FINAL;
  return true;
}

// Forward an untagged chunk as one, keeping the decoder's legacy contract: it
// alone ends its fixed-length shot, and the legacy read seals a completed one,
// so the shot publishes whole and nothing of it is left for a flush. The graph
// meters the chunk to the shot's end (untagged_chunk_limit), which the decoder
// would otherwise overrun and drop. A publication before the shot's end is
// provisional, as a tagged one is.
bool streaming_decoder_node::enqueue_untagged(const uint8_t *msyn,
                                              std::size_t len) {
  bool shot_end = false;
  if (len == 0)
    return false;
  untagged_fill_ += len;
  shot_end = untagged_fill_ >= num_msyn_per_decode();
  if (shot_end)
    untagged_fill_ = 0;
  if (!dec_->enqueue_syndrome(msyn, len) || (!shot_end && !commits_more())) {
    // A shot that ends unpublished leaves the next shot's commits its own.
    if (shot_end)
      shot_committed_ = 0;
    pending_inner_input_ = true;
    return false;
  }
  if (!shot_end && correction_provider_) {
    publish_inner_frame(correction_provider_->peek_obs_corrections());
    record_coverage(/*ends_shot=*/false, std::nullopt);
    // As for a tagged one: keep a flush pending, for a session that closes
    // before the shot ends.
    pending_inner_input_ = true;
    return true;
  }
  publish_inner_frame(dec_->get_obs_corrections());
  if (correction_provider_) {
    // The shot is D's width, so the node knows its rounds even when the
    // decoder does not report them.
    const auto rounds =
        correction_provider_->finalized_round_count().value_or(num_rounds_);
    record_coverage(/*ends_shot=*/true, rounds);
    completed_rounds_ += rounds;
  }
  pending_inner_input_ = false;
  return true;
} // end - enqueue_untagged()

// A provider's provisional frame publishes only once it commits more rounds
// of the shot than its node has covered; without commit progress, never
// before the shot ends. Any other decoder's publishes whenever it fires.
bool streaming_decoder_node::commits_more() const {
  return !correction_provider_ ||
         correction_provider_->committed_round_count().value_or(0) >
             shot_committed_;
}

// Snapshot the inner decoder and expose its correction delta for edge routing.
// A FINAL shot's rounds follow those of the shots before it in its stream, and
// end the stream.
void streaming_decoder_node::capture_inner_corrections(bool final) {
  publish_inner_frame(
      correction_provider_
          ? (final ? correction_provider_->finalize_obs_corrections()
                   : correction_provider_->peek_obs_corrections())
          : dec_->get_obs_corrections());
  if (!correction_provider_)
    return;
  record_coverage(final, final ? correction_provider_->finalized_round_count()
                               : std::nullopt);
  if (final)
    completed_rounds_ = 0;
} // end - capture_inner_corrections()

// A provider's publication covers the shot's rounds committed so far: those
// the decoder reports, or at the shot's end its @p rounds, when known. An
// exact node's covers those committed since its last, or, at an end whose
// rounds are unknown, joins the output whole at FINAL; a scheduled node's
// covers the shot from its first round, the graph starting the output where
// it was last consumed.
void streaming_decoder_node::record_coverage(
    bool ends_shot, std::optional<std::uint64_t> rounds) {
  const auto before = shot_committed_;
  const auto begin = completed_rounds_;
  if (ends_shot)
    shot_committed_ = std::max(before, rounds.value_or(before));
  else
    shot_committed_ = std::max(
        before, correction_provider_->committed_round_count().value_or(0));
  if (!exact_)
    coverage_ = committed_coverage{begin, begin + shot_committed_};
  else if (ends_shot && !rounds)
    coverage_ = committed_coverage{begin + before, whole_shot};
  else
    coverage_ = committed_coverage{begin + before, begin + shot_committed_};
  if (ends_shot)
    shot_committed_ = 0;
} // end - record_coverage()

// Publish the change in the inner frame since the last publication.
void streaming_decoder_node::publish_inner_frame(const uint8_t *current) {
  auto &outcome = last_output_.get<logical_outcome>();
  outcome.converged = true;
  for (std::size_t i = 0; i < outcome.bits.size(); ++i) {
    outcome.bits[i] = current[i] ^ correction_frame_[i];
    correction_frame_[i] = current[i];
  }
  output_valid_ = true;
}

// Publish a decoder correction withheld after its final incomplete enqueue.
bool streaming_decoder_node::flush() {
  if (!pending_inner_input_)
    return false;
  capture_inner_corrections(/*final=*/true);
  pending_inner_input_ = false;
  return true;
}

// Return the current graph-owned aggregate correction frame.
const uint8_t *streaming_decoder_node::corrections() const {
  return correction_frame_.data();
}

// Consume published corrections while preserving inner streaming context.
void streaming_decoder_node::clear_corrections() {
  dec_->clear_corrections();
  std::fill(correction_frame_.begin(), correction_frame_.end(), uint8_t{0});
  output_valid_ = false;
}

// Drop node-local streaming state without invoking the inner decoder.
void streaming_decoder_node::reset_streaming_state() {
  std::fill(correction_frame_.begin(), correction_frame_.end(), uint8_t{0});
  pending_inner_input_ = false;
  output_valid_ = false;
  untagged_fill_ = 0;
  completed_rounds_ = 0;
  shot_committed_ = 0;
  coverage_.reset();
}

// Report the validated raw measurement width shared by the decoder and D.
std::size_t streaming_decoder_node::num_msyn_per_decode() const {
  return d_->num_cols();
}

// Report the configured common graph round unit.
std::size_t streaming_decoder_node::num_msyn_per_round() const { return spr_; }

// Report the logical width defined by the node's observable projection.
std::size_t streaming_decoder_node::num_observables() const {
  return o_->num_rows();
}

// Expose the latest correction delta only after this node publishes.
const edge_value *
streaming_decoder_node::streaming_output(std::size_t slot) const {
  return slot == 0 && output_valid_ ? &last_output_ : nullptr;
}

// A decoder that ends its own shots has every publication report the rounds
// it commits (see record_coverage()), rather than the configured template
// fire schedule: after the shots before it in its stream, the shot after a
// FINAL one beginning a new stream. Any other node keeps its schedule.
std::optional<streaming_task_graph_node::committed_coverage>
streaming_decoder_node::streaming_coverage() const {
  return coverage_;
}

// Project a one-shot decoder result through O when it is fault-valued.
logical_outcome
streaming_decoder_node::project_result(decoder_result result) const {
  return project_decoder_result(std::move(result), *o_,
                                dec_->get_result_type() ==
                                    decode_result_type::observables);
}

// Apply D and run the inner decoder for the non-streaming graph path.
logical_outcome streaming_decoder_node::decode_one_shot(
    const measurement_results &measurements) const {
  if (measurements.bits.size() < d_->num_cols())
    throw std::out_of_range(
        "streaming node shot has fewer measurements than D requires");
  detect_into(csr_rows{*d_}, measurements.bits.data(), dets_.data());
  return project_result(dec_->decode(dets_));
}

// Validate the typed graph input and produce the node's one-shot output.
void streaming_decoder_node::execute_indexed(
    const std::vector<edge_value> &inputs,
    std::vector<edge_value> &outputs) const {
  const auto *measurements = inputs[0].get_if<measurement_results>();
  if (!measurements)
    throw std::invalid_argument(
        "streaming node input requires measurement_results");
  outputs[0] = decode_one_shot(*measurements);
}

// Declare the node's sole logical-outcome output port.
std::optional<edge_value_type>
streaming_decoder_node::output_type(std::size_t slot) const {
  if (slot == 0)
    return logical_outcome_type;
  return std::nullopt;
}

// Accept measurement results only on the node's sole input port.
bool streaming_decoder_node::accepts_input_type(std::size_t slot,
                                                edge_value_type type) const {
  return slot == 0 && type == measurement_results_type;
}

// Publish the fixed logical width used for graph shape validation.
std::optional<std::size_t>
streaming_decoder_node::output_value_size(std::size_t slot) const {
  return slot == 0 ? std::optional<std::size_t>(o_->num_rows()) : std::nullopt;
}

// Reset both graph-local and inner-decoder shot state.
void streaming_decoder_node::reset_decoder() {
  reset_streaming_state();
  dec_->reset_decoder();
}

// Launch native asynchronous decoding and defer projection until collection.
std::future<logical_outcome> streaming_decoder_node::run_async(
    const measurement_results &measurements) const {
  if (measurements.bits.size() < d_->num_cols())
    throw std::out_of_range(
        "streaming node shot has fewer measurements than D requires");
  std::vector<float_t> dets(d_->num_rows(), 0.0f);
  detect_into(csr_rows{*d_}, measurements.bits.data(), dets.data());
  auto pending = dec_->decode_async(dets);
  const auto o = o_;
  const bool observables =
      dec_->get_result_type() == decode_result_type::observables;
  return std::async(std::launch::deferred,
                    [o, observables, pending = std::move(pending)]() mutable {
                      return project_decoder_result(pending.get(), *o,
                                                    observables);
                    });
}

// Forward a usable inner-decoder CUDA device assignment to graph scheduling.
std::optional<int> streaming_decoder_node::cuda_device_id() const {
  const int id = dec_->get_cuda_device_id();
  return id >= 0 ? std::optional<int>(id) : std::nullopt;
}

} // namespace cudaq::qec

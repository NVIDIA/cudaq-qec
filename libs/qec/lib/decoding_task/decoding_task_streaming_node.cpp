/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
            ctx.num_measurements == 0);
      });
  return true;
}();

} // namespace

// Validate the decoder-facing matrices and allocate persistent streaming state.
streaming_decoder_node::streaming_decoder_node(
    std::shared_ptr<const sparse_binary_matrix> d,
    std::shared_ptr<const sparse_binary_matrix> o, std::unique_ptr<decoder> dec,
    uint32_t spr, bool equal_rounds)
    : d_(std::move(d)), o_(std::move(o)), dec_(std::move(dec)), spr_(spr) {
  if (!d_ || !o_ || !dec_)
    throw std::invalid_argument("streaming node requires D, O, and a decoder");
  correction_provider_ =
      dynamic_cast<streaming_correction_provider *>(dec_.get());
  if (spr_ == 0 || (equal_rounds && d_->num_cols() % spr_ != 0))
    throw std::invalid_argument(
        "streaming node syndrome count must be divisible by its round size");
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
}

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
  if (!dec_->enqueue_syndrome(msyn, len, type)) {
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
// would otherwise overrun and drop.
bool streaming_decoder_node::enqueue_untagged(const uint8_t *msyn,
                                              std::size_t len) {
  if (len == 0)
    return false;
  untagged_fill_ += len;
  if (untagged_fill_ >= num_msyn_per_decode())
    untagged_fill_ = 0;
  if (!dec_->enqueue_syndrome(msyn, len)) {
    pending_inner_input_ = true;
    return false;
  }
  publish_inner_frame(dec_->get_obs_corrections(), /*final=*/false);
  if (correction_provider_)
    if (const auto rounds = correction_provider_->finalized_round_count())
      completed_rounds_ += *rounds;
  pending_inner_input_ = false;
  return true;
}

// Snapshot the inner decoder and expose its correction delta for edge routing.
// A FINAL shot's rounds follow those of the shots before it in its stream, and
// end the stream.
void streaming_decoder_node::capture_inner_corrections(bool final) {
  publish_inner_frame(
      correction_provider_
          ? (final ? correction_provider_->finalize_obs_corrections()
                   : correction_provider_->peek_obs_corrections())
          : dec_->get_obs_corrections(),
      final);
  if (!final || !correction_provider_)
    return;
  final_coverage_.reset();
  if (const auto rounds = correction_provider_->finalized_round_count())
    final_coverage_ =
        committed_coverage{completed_rounds_, completed_rounds_ + *rounds};
  completed_rounds_ = 0;
}

// Publish the change in the inner frame since the last publication.
void streaming_decoder_node::publish_inner_frame(const uint8_t *current,
                                                 bool final) {
  auto &outcome = last_output_.get<logical_outcome>();
  outcome.converged = true;
  for (std::size_t i = 0; i < outcome.bits.size(); ++i) {
    outcome.bits[i] = current[i] ^ correction_frame_[i];
    correction_frame_[i] = current[i];
  }
  output_valid_ = true;
  published_final_ = final;
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
  published_final_ = false;
  untagged_fill_ = 0;
  completed_rounds_ = 0;
  final_coverage_.reset();
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

// Native frames accumulate from the start of the stream. Once FINAL finalizes
// the shot, use its physical length, after the shots before it, rather than
// the configured template fire schedule. A shot sealed at its configured
// length keeps the schedule, which also numbers back-to-back shots that never
// see FINAL; the shot after a FINAL one begins a new stream.
std::optional<streaming_task_graph_node::committed_coverage>
streaming_decoder_node::streaming_coverage() const {
  return published_final_ ? final_coverage_ : std::nullopt;
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

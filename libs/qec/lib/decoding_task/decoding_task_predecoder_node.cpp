/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "decoding_task_predecoder_node.h"
#include "decoding_task_graph_utils.h"
#include "decoding_task_window_node.h"
#include "cudaq/qec/logger.h"

#include <algorithm>
#include <any>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// Out-of-line destructor anchors the vtable to this translation unit.
predecoder_node::~predecoder_node() = default;

predecoder_node::predecoder_node(residual_source src,
                                 std::shared_ptr<const sparse_binary_matrix> d,
                                 std::shared_ptr<const sparse_binary_matrix> o,
                                 std::shared_ptr<const sparse_binary_matrix> h,
                                 std::unique_ptr<decoder> dec, uint32_t spr,
                                 uint32_t rounds, std::size_t num_measurements)
    : src_(src), d_(std::move(d)), o_(std::move(o)), h_(std::move(h)),
      dec_(std::move(dec)), spr_(spr) {
  const bool from_parity = src_ == residual_source::parity_check;
  const bool has_model_outputs =
      dec_ && dec_->get_result_type() == decode_result_type::observables &&
      dec_->get_output_request().requests(
          decoder_auxiliary_result_type::residual_detectors);
  if (!d_ || !o_ || !dec_)
    throw std::invalid_argument(
        "predecoder_node: D, O and the inner decoder are all required");
  if (from_parity && !h_)
    throw std::invalid_argument("predecoder_node: residual_from parity_check "
                                "needs H to map a prediction back to the "
                                "detectors it explains");
  if (spr_ == 0 || rounds == 0)
    throw std::invalid_argument("predecoder_node: rounds and syndrome bytes "
                                "per round must both be positive");
  // All are walked as rows; see as_csr().
  d_ = as_csr(d_);
  o_ = as_csr(o_);
  if (h_)
    h_ = as_csr(h_);
  // The parity-check derivation needs an error pattern to push through H.
  // Model derivation instead consumes pre_L plus residual detector metadata
  // and validates both widths in decode_into().
  if (from_parity && dec_->get_result_type() != decode_result_type::errors)
    throw std::invalid_argument(
        "predecoder_node: the inner decoder projects to observables itself "
        "(observables), so it cannot say which detectors it explained and "
        "no residual can be formed from it");
  if (!from_parity && !has_model_outputs)
    throw std::invalid_argument(
        "predecoder_node: residual_from model needs an inner decoder that "
        "returns observables plus residual_detectors metadata");
  // The inner reads this node's detectors under either derivation, so this one
  // is not H's to check.
  if (dec_->get_syndrome_size() != d_->num_rows())
    throw std::invalid_argument(
        "predecoder_node: the inner decoder reads " +
        std::to_string(dec_->get_syndrome_size()) +
        " detectors but D "
        "produces " +
        std::to_string(d_->num_rows()) +
        "; it was built on a different DEM than this node's");
  if (from_parity) {
    // The residual replaces detector events with the ones H says the
    // prediction explains, so the two have to be the same detectors in the
    // same order.
    if (h_->num_rows() != d_->num_rows())
      throw std::invalid_argument(
          "predecoder_node: H has " + std::to_string(h_->num_rows()) +
          " detector rows but D produces " + std::to_string(d_->num_rows()));
    // H comes from this node's DEM but `e` comes from the inner decoder, which
    // a caller may have supplied ready-built on some other matrix. Then H
    // would be indexed by columns that mean something else: with mismatched
    // dimensions that reads out of range, and with matching dimensions it is a
    // silently wrong residual. Shapes are all we can check, but they catch the
    // first case outright and make the second much harder to reach by
    // accident.
    if (dec_->get_block_size() != h_->num_cols())
      throw std::invalid_argument(
          "predecoder_node: the inner decoder predicts " +
          std::to_string(dec_->get_block_size()) + " mechanisms but H has " +
          std::to_string(h_->num_cols()) +
          " columns; it was built on a different DEM than this node's");
    if (o_->num_cols() > h_->num_cols())
      throw std::invalid_argument("predecoder_node: O references column " +
                                  std::to_string(o_->num_cols() - 1) +
                                  " but H has only " +
                                  std::to_string(h_->num_cols()) + " columns");
  }
  // `model` returns its two outputs through decoder_result, so decode_into()
  // checks both result widths against O and D.
  buf_.assign(num_measurements != 0 ? num_measurements
                                    : static_cast<std::size_t>(rounds) * spr_,
              0);
  dets_.assign(d_->num_rows(), 0.0f);
  corr_.assign(o_->num_rows(), 0);
  lo_val_ = edge_value{logical_outcome{}};
  lo_val_.get<logical_outcome>().bits.assign(o_->num_rows(), 0);
  residual_val_ = edge_value{detection_events{}};
  residual_val_.get<detection_events>().bits.assign(d_->num_rows(), 0.0f);
}

void predecoder_node::decode_into(const std::vector<float_t> &dets,
                                  logical_outcome &lo,
                                  detection_events &residual) const {
  const auto dr = dec_->decode(dets);

  // A pre-decoder cannot fail: it either explains part of the syndrome or
  // defers all of it, and deferring is the case the downstream decoder exists
  // for. Reporting dr.converged here would leave every combined outcome in
  // such a topology permanently unconverged, since xor_logical ANDs the flags.
  lo.converged = true;
  std::fill(lo.bits.begin(), lo.bits.end(), uint8_t{0});

  if (!dr.converged) {
    // Abstain: commit nothing and hand the syndrome on untouched, so the
    // consumer sees exactly what it would have with no pre-decoder in front
    // of it. A decoder that did not converge still returns a result vector,
    // and committing that would leave a correction nothing undoes.
    std::copy(dets.begin(), dets.end(), residual.bits.begin());
    return;
  }

  if (src_ == residual_source::model) {
    // Keep the decoder result contract intact: pre_L is the observable result,
    // while the model's second output is named auxiliary data.
    const auto n_obs = o_->num_rows(), n_dets = d_->num_rows();
    const std::vector<float_t> *model_residual = nullptr;
    if (dr.result.size() != n_obs)
      throw std::out_of_range("predecoder_node: residual_from model expects " +
                              std::to_string(n_obs) +
                              " observable values, but the inner decoder " +
                              "returned " + std::to_string(dr.result.size()));
    if (!dr.opt_results.has_value())
      throw std::out_of_range(
          "predecoder_node: residual_from model requires the inner decoder to "
          "return residual_detectors in opt_results");
    for (const auto &[key, value] : *dr.opt_results)
      if (key == "residual_detectors") {
        model_residual = std::any_cast<std::vector<float_t>>(&value);
        break;
      }
    if (!model_residual)
      throw std::out_of_range(
          "predecoder_node: residual_from model requires the inner decoder to "
          "return residual_detectors in opt_results");
    if (model_residual->size() != n_dets)
      throw std::out_of_range(
          "predecoder_node: residual_from model expects " +
          std::to_string(n_dets) +
          " residual detectors, but the inner decoder returned " +
          std::to_string(model_residual->size()));
    for (std::size_t i = 0; i < n_obs; ++i)
      lo.bits[i] = static_cast<uint8_t>(convert_soft_to_hard(dr.result[i]));
    for (std::size_t i = 0; i < n_dets; ++i)
      residual.bits[i] =
          static_cast<float_t>(convert_soft_to_hard((*model_residual)[i]));
    return;
  }

  const auto width = std::max(o_->num_cols(), h_->num_cols());
  if (width > dr.result.size())
    throw std::out_of_range("predecoder_node: decoder returned " +
                            std::to_string(dr.result.size()) +
                            " values, but O and H reference column " +
                            std::to_string(width - 1));

  project_into(csr_rows{*o_}, dr.result.data(), lo.bits.data());
  // The residual is the syndrome minus what this node just committed: the same
  // prediction, mapped back through H into detector space. Whatever the
  // prediction got wrong stays in the residual, which is the point -- the two
  // nodes' commits sum in GF(2) to the whole correction exactly because their
  // syndromes do.
  residual_into(csr_rows{*h_}, dets.data(), dr.result.data(),
                residual.bits.data());
}

bool predecoder_node::flush() {
  // Reported, not discarded: a partial window stays buffered at end of stream,
  // so the next shot's first rounds would complete this shot's window. Call
  // reset_decoder() between shots, not just clear_corrections().
  if (bytes_in_ > 0)
    CUDA_QEC_WARN("predecoder_node: {} of {} syndrome bytes never completed a "
                  "window and are not committed; they remain buffered, so "
                  "reset_decoder() is required before the next shot",
                  bytes_in_, buf_.size());
  return false;
}

bool predecoder_node::enqueue(const uint8_t *msyn, std::size_t len,
                              round_type /*type*/) {
  if (bytes_in_ + len > buf_.size()) {
    // Same contract as decoder::enqueue_syndrome and window_node::enqueue:
    // drop the chunk that does not fit, keep what has accumulated.
    CUDA_QEC_WARN("predecoder_node: ignoring a {}-byte syndrome; only {} of {} "
                  "bytes remain in the window",
                  len, buf_.size() - bytes_in_, buf_.size());
    return false;
  }
  std::copy(msyn, msyn + len, buf_.begin() + bytes_in_);
  bytes_in_ += len;
  if (bytes_in_ < buf_.size())
    return false;
  bytes_in_ = 0; // the window does not slide: a fire empties the buffer

  detect_into(csr_rows{*d_}, buf_.data(), dets_.data());
  auto &lo = lo_val_.get<logical_outcome>();
  decode_into(dets_, lo, residual_val_.get<detection_events>());
  for (std::size_t i = 0; i < corr_.size(); ++i)
    corr_[i] ^= lo.bits[i];
  outputs_valid_ = true;
  return true;
}

const uint8_t *predecoder_node::corrections() const { return corr_.data(); }

void predecoder_node::clear_corrections() {
  std::fill(corr_.begin(), corr_.end(), uint8_t{0});
}

void predecoder_node::reset_streaming_state() {
  bytes_in_ = 0;
  std::fill(buf_.begin(), buf_.end(), uint8_t{0});
  outputs_valid_ = false;
  clear_corrections();
}

void predecoder_node::reset_decoder() {
  reset_streaming_state();
  dec_->reset_decoder();
}

std::optional<edge_value_type>
predecoder_node::output_type(std::size_t slot) const {
  switch (slot) {
  case 0:
    return logical_outcome_type;
  case 1:
    return detection_events_type;
  default:
    return std::nullopt;
  }
}

bool predecoder_node::accepts_input_type(std::size_t slot,
                                         edge_value_type type) const {
  return slot == 0 && type == measurement_results_type;
}

const edge_value *predecoder_node::streaming_output(std::size_t slot) const {
  if (!outputs_valid_)
    return nullptr;
  if (slot == 0)
    return &lo_val_;
  return slot == 1 ? &residual_val_ : nullptr;
}

void predecoder_node::execute_indexed(const std::vector<edge_value> &inputs,
                                      std::vector<edge_value> &outputs) const {
  const auto &m = inputs[0].get<measurement_results>();
  if (m.bits.size() < d_->num_cols())
    throw std::out_of_range(
        "predecoder_node: shot has " + std::to_string(m.bits.size()) +
        " measurements, D needs " + std::to_string(d_->num_cols()));
  detect_into(csr_rows{*d_}, m.bits.data(), dets_.data());

  // Fill every declared slot: a wire may read either one.
  logical_outcome lo;
  lo.bits.assign(o_->num_rows(), 0);
  detection_events residual;
  residual.bits.assign(d_->num_rows(), 0.0f);
  decode_into(dets_, lo, residual);
  outputs[0] = std::move(lo);
  outputs[1] = std::move(residual);
}

std::optional<int> predecoder_node::cuda_device_id() const {
  const int id = dec_->get_cuda_device_id();
  return id >= 0 ? std::optional<int>(id) : std::nullopt;
}

decoder *predecoder_node::inner_decoder() const { return dec_.get(); }

// ---------------------------------------------------------------------------
// The `predecoder` node kind
// ---------------------------------------------------------------------------

namespace {

/// The one place the two spellings are known, shared by the schema's validate
/// hook and the factory: `from_topology` does not schema-check node params, so
/// the factory has to reject a bad value on its own.
predecoder_node::residual_source parse_residual_from(const std::string &value,
                                                     const std::string &where) {
  if (value == "parity_check")
    return predecoder_node::residual_source::parity_check;
  if (value == "model")
    return predecoder_node::residual_source::model;
  throw std::invalid_argument(
      where +
      ": residual_from must be 'parity_check' (the inner decoder "
      "returns an error pattern) or 'model' (it returns pre_L plus named "
      "residual detector metadata), not '" +
      value + "'");
}

/// A pre-decoder covers exactly the rounds its DEM expands to and empties on
/// every fire, so it declares neither `step_size` nor a guard: both need a
/// window that overlaps its successor, and overlapping residuals would hand
/// the consumer the same rounds twice. Not declaring them is what makes naming
/// one an unknown-key error rather than a special case in a validator -- the
/// same choice `single_window` makes.
decoding::config::decoder_schema predecoder_params() {
  decoding::config::decoder_schema schema;
  schema.params = window_kind_common_params();
  schema.params.push_back({"residual_from",
                           decoding::config::param_kind::string,
                           /*required=*/true});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    if (p.contains("window_size") && p.get<std::size_t>("window_size") == 0)
      throw std::runtime_error("predecoder: window_size must be at least 1");
    if (p.contains("residual_from"))
      (void)parse_residual_from(p.get<std::string>("residual_from"),
                                "predecoder");
  };
  return schema;
}

std::unique_ptr<task_graph_node> make_predecoder_node(node_build_context &ctx) {
  const auto &p = *ctx.node_params;
  const std::string where = "task graph node '" + ctx.id + "'";

  const auto size =
      static_cast<uint32_t>(p.get<std::size_t>("window_size", ctx.dem_rounds));
  if (size != ctx.dem_rounds)
    throw std::invalid_argument(
        where + ": window_size " + std::to_string(size) +
        " does not match the " + std::to_string(ctx.dem_rounds) +
        " rounds its DEM spans (window_size is what the DEM expands to)");
  if (p.contains("num_syndromes_per_round")) {
    const auto declared = p.get<std::size_t>("num_syndromes_per_round");
    if (declared != ctx.spr)
      throw std::invalid_argument(
          where + ": num_syndromes_per_round is " + std::to_string(declared) +
          " but the DEM has " + std::to_string(ctx.spr) +
          " syndrome bytes per round");
  }

  // Required, and with no default: a model whose export format is mis-declared
  // should fail the build, not the first decode.
  if (!p.contains("residual_from"))
    throw std::invalid_argument(
        where + ": a pre-decoder must declare residual_from, either "
                "'parity_check' or 'model'; the inner decoder's result type "
                "cannot say which it returns");
  const auto src =
      parse_residual_from(p.get<std::string>("residual_from"), where);

  // DEM-dependent rules live here, where the DEM itself is.
  if (src == predecoder_node::residual_source::parity_check && !ctx.h)
    throw std::invalid_argument(
        where + ": residual_from parity_check needs H to say which detectors "
                "a predicted error explains, and D and O cannot supply it; "
                "describe this node with dem_chunks, or give the flat form an "
                "h_sparse");
  // The parity-check residual is `dets XOR H*e` over fault columns. A decoder
  // that projects to observables returns a frame in a different space, so only
  // the model derivation may consume such a result: there the primary result
  // is pre_L and the residual detectors arrive as named auxiliary data.
  if (src == predecoder_node::residual_source::parity_check &&
      ctx.dec->get_result_type() != decode_result_type::errors)
    throw std::invalid_argument(
        where + ": a pre-decoder needs an inner decoder that returns an error "
                "pattern; this one projects to observables itself "
                "(observables), which cannot say which detectors it "
                "explained -- use it as the downstream decoder instead");
  if (src == predecoder_node::residual_source::model &&
      (ctx.dec->get_result_type() != decode_result_type::observables ||
       !ctx.dec->get_output_request().requests(
           decoder_auxiliary_result_type::residual_detectors)))
    throw std::invalid_argument(
        where + ": residual_from model needs an inner decoder that returns "
                "observables plus residual_detectors metadata");

  return std::make_unique<predecoder_node>(src, ctx.d, ctx.o, ctx.h,
                                           std::move(ctx.dec), ctx.spr, size,
                                           ctx.num_measurements);
}

const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "predecoder";
  spec.aliases = {"PreDecoder"};
  spec.requires_decoder = true;
  spec.requires_dem = true;
  // The ordinary parity-check form needs an error pattern and applies O at the
  // node. A model-residual predecoder gives O to its inner decoder so it can
  // emit pre_L itself.
  spec.inner_decoder_output_policy =
      [](const cudaqx::heterogeneous_map &params) {
        if (params.get<std::string>("residual_from", "") == "model")
          return decoder_output_request{
              decode_result_type::observables,
              decoder_auxiliary_result_type::residual_detectors};
        return decoder_output_request{decode_result_type::errors};
      };
  spec.inner_decoder_observable_policy =
      [](const cudaqx::heterogeneous_map &params) {
        return params.get<std::string>("residual_from", "") == "model";
      };
  register_node_kind(
      std::move(spec), predecoder_params(),
      [](node_build_context &ctx) { return make_predecoder_node(ctx); });
  return true;
}();

} // namespace

} // namespace cudaq::qec

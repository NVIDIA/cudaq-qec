/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decoding_task_window_node.h"
#include "decoding_task_graph_impl.h"
#include "decoding_task_graph_utils.h"
#include "decoding_task_memory_window_node.h"
#include "../hardware_guards.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/extended_dem.h"
#include "cudaq/qec/logger.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

namespace {

/// Zero everything outside the committed column range.  project_into() reads
/// the result at O's global column ids, so a zeroed entry contributes nothing
/// to any observable — the same outcome as dropping that column from O, with
/// no second matrix to build or store.
void mask_guard_range(std::vector<float_t> &result, std::size_t lo_col,
                      std::size_t hi_col) {
  const auto n = result.size();
  if (lo_col > 0)
    std::fill(result.begin(),
              result.begin() + static_cast<std::ptrdiff_t>(std::min(lo_col, n)),
              float_t{0});
  if (hi_col < n)
    std::fill(result.begin() + static_cast<std::ptrdiff_t>(hi_col),
              result.end(), float_t{0});
}

/// Reduce a decoder result to observable bits, the one way all three entry
/// points (streaming fire, one-shot execute_indexed, run_async) do it.
///
/// A decoder that projects to observables itself hands back a frame we take as
/// it stands; anything else returns fault columns, which are masked to the
/// committed range and then projected through O.  `bits` must already be sized
/// to O's row count.
void commit_into(decoder_result &dr, const sparse_binary_matrix &o,
                 bool projects_to_observables, std::size_t lo_col,
                 std::size_t hi_col, std::vector<uint8_t> &bits) {
  std::fill(bits.begin(), bits.end(), uint8_t{0});
  if (projects_to_observables) {
    if (dr.result.size() != bits.size())
      throw std::out_of_range(
          "window_node: decoder projects to observables and returned " +
          std::to_string(dr.result.size()) + " values, O has " +
          std::to_string(bits.size()) + " rows");
    for (std::size_t i = 0; i < bits.size(); ++i)
      bits[i] = static_cast<uint8_t>(convert_soft_to_hard(dr.result[i]));
    return;
  }
  if (o.num_cols() > dr.result.size())
    throw std::out_of_range(
        "window_node: decoder returned " + std::to_string(dr.result.size()) +
        " values, O references column " + std::to_string(o.num_cols() - 1));
  mask_guard_range(dr.result, lo_col, hi_col);
  project_into(csr_rows{o}, dr.result.data(), bits.data());
}

} // namespace

// ---------------------------------------------------------------------------
// window_node
// ---------------------------------------------------------------------------

window_node::window_node(std::shared_ptr<const sparse_binary_matrix> d,
                         std::shared_ptr<const sparse_binary_matrix> o,
                         std::unique_ptr<decoder> dec, uint32_t spr,
                         window_spec ws, std::vector<std::size_t> fault_offsets,
                         std::size_t num_measurements, bool push_candidates,
                         bool cancellable, std::string id)
    : d_(std::move(d)), o_(std::move(o)), dec_(std::move(dec)), spr_(spr),
      ws_(ws), pushes_(push_candidates), cancellable_(cancellable),
      id_(std::move(id)) {
  if (ws_.stride == 0 || ws_.stride > ws_.size)
    throw std::invalid_argument(
        "window_node: stride (step_size) must be between 1 and size "
        "(window_size), got " +
        std::to_string(ws_.stride) + " for size " + std::to_string(ws_.size));
  // Both are walked as rows by detect_into/project_into; see as_csr().
  d_ = as_csr(d_);
  o_ = as_csr(o_);
  if (ws_.start_guard > 0 || ws_.end_guard > 0) {
    // Guards are a fault-column mask, so there has to be a fault-column vector
    // to mask. A decoder that projects to observables itself hands back an
    // observable frame instead, and masking it would zero observables.
    if (dec_ && dec_->get_result_type() == decode_result_type::observables)
      throw std::invalid_argument(
          "window_node: guards need a decoder that returns an error pattern; "
          "this one projects to observables itself, leaving "
          "no fault columns to withhold");
    // A guard only means something for a window that overlaps its successor:
    // the rounds it withholds have to be committed by some other fire.
    if (ws_.stride >= ws_.size)
      throw std::invalid_argument(
          "window_node: guards need a sliding window (stride " +
          std::to_string(ws_.stride) + " must be less than size " +
          std::to_string(ws_.size) + ")");
    if (ws_.start_guard + ws_.end_guard >= ws_.size)
      throw std::invalid_argument(
          "window_node: guards leave no rounds to commit (start_guard " +
          std::to_string(ws_.start_guard) + " + end_guard " +
          std::to_string(ws_.end_guard) + " must be less than size " +
          std::to_string(ws_.size) + ")");
    if (ws_.stride != ws_.core())
      throw std::invalid_argument(
          "window_node: guarded windows must tile the stream exactly once "
          "(stride " +
          std::to_string(ws_.stride) +
          " must equal size - start_guard - end_guard = " +
          std::to_string(ws_.core()) + ")");
    // Guards are given in rounds but committed in columns, and per-round fault
    // counts differ, so the boundaries have to come from the DEM.
    if (fault_offsets.size() != static_cast<std::size_t>(ws_.size) + 1)
      throw std::invalid_argument(
          "window_node: guards need " + std::to_string(ws_.size + 1) +
          " fault-column offsets for a " + std::to_string(ws_.size) +
          "-round window, got " + std::to_string(fault_offsets.size()));
    lo_col_ = fault_offsets[ws_.start_guard];
    hi_col_ = fault_offsets[ws_.size - ws_.end_guard];
  }
  // Both paths hand the decoder exactly D's detector rows, and a caller may
  // have supplied a decoder built on some other DEM. A decoder expecting more
  // detectors than that reads past the end of the buffer we give it, which the
  // inner decoder is entitled not to check (the LUT's assert compiles out).
  if (dec_ && dec_->get_syndrome_size() != d_->num_rows())
    throw std::invalid_argument(
        "window_node: the inner decoder reads " +
        std::to_string(dec_->get_syndrome_size()) +
        " detectors but D produces " + std::to_string(d_->num_rows()) +
        "; it was built on a different DEM than this node's");
  // An error pattern is indexed by O's fault columns, so O must not reach past
  // what the decoder predicts. There is deliberately no companion check for
  // For observable output, `block_size` does not mean the observable count.
  // Decoders in that mode differ on what they leave it as -- some rebuild it as
  // the observable count, others keep H's fault-column count and switch only
  // the result vector -- and get_num_observables() is no stand-in either, since
  // such a decoder need not carry an O_sparse at all. The result width is the
  // only reliable signal, and it is not known until the first decode, where
  // commit_into() checks it.
  if (dec_ && dec_->get_result_type() != decode_result_type::observables &&
      o_->num_cols() > dec_->get_block_size())
    throw std::invalid_argument("window_node: O references column " +
                                std::to_string(o_->num_cols() - 1) +
                                " but the inner decoder predicts only " +
                                std::to_string(dec_->get_block_size()) +
                                " mechanisms");
  if (num_measurements != 0 && ws_.stride != ws_.size)
    throw std::invalid_argument(
        "window_node: rounds of different widths cannot slide (step_size must "
        "equal window_size)");
  buf_.assign(num_measurements != 0 ? num_measurements
                                    : static_cast<std::size_t>(ws_.size) * spr_,
              0);
  dets_.assign(d_->num_rows(), 0.0f);
  corr_.assign(o_->num_rows(), 0);
  // One edge value is one whole window (see enqueue_edge_value), and D's rows
  // already span the whole window, so this is the decoder's syndrome_size.
  det_buf_.assign(d_->num_rows(), 0.0f);
  // Pre-allocate streaming output buffer; reused in place on every fire.
  last_lo_val_ = edge_value{logical_outcome{}};
  last_lo_val_.get<logical_outcome>().bits.assign(o_->num_rows(), 0);
}

bool window_node::enqueue(const uint8_t *msyn, std::size_t len,
                          round_type /*type*/) {
  reject_after_flush("enqueue");
  if (bytes_in_ + len > buf_.size()) {
    // Same contract as decoder::enqueue_syndrome: drop the chunk that does not
    // fit and keep what has already accumulated, so an oversized call cannot
    // silently discard a partially filled window.  A slid window has only
    // stride * spr bytes of room, so this is the path a caller hits when it
    // sends whole windows instead of rounds.
    CUDA_QEC_WARN("window_node: ignoring a {}-byte syndrome; only {} of {} "
                  "bytes remain in the window",
                  len, buf_.size() - bytes_in_, buf_.size());
    return false;
  }
  std::copy(msyn, msyn + len, buf_.begin() + bytes_in_);
  bytes_in_ += len;
  if (bytes_in_ < buf_.size())
    return false;

  detect_into(csr_rows{*d_}, buf_.data(), dets_.data());
  fire(dets_);

  const std::size_t keep = (ws_.size - ws_.stride) * spr_;
  if (keep)
    std::memmove(buf_.data(), buf_.data() + ws_.stride * spr_, keep);
  bytes_in_ = keep;
  return true;
}

bool window_node::fire(const std::vector<float_t> &dets) {
  auto decoded = decode(dets, token_);
  if (!decoded)
    return false; // stopped before a result: nothing to commit
  auto &dr = *decoded;
  // Snapshot before commit_into() masks dr.result in place, and only when
  // columns above hi_col_ actually exist to withhold.  hi_col_ is set for any
  // guard, but a start-guard-only window has hi_col_ == the result width and
  // commits through the end, so there is nothing for flush() to do.  Sized on
  // the first fire, then copied into.
  last_result_valid_ = hi_col_ < dr.result.size();
  if (last_result_valid_) {
    if (last_result_.size() != dr.result.size()) {
      last_result_.assign(dr.result.begin(), dr.result.end());
      flush_scratch_.resize(dr.result.size());
    } else {
      std::copy(dr.result.begin(), dr.result.end(), last_result_.begin());
    }
  }
  // Write into pre-allocated last_lo_val_ — no per-fire allocation.
  auto &lo = last_lo_val_.get<logical_outcome>();
  lo.converged = dr.converged;
  // Keep the leading guard on the first fire of a shot: no earlier window
  // covered those rounds, so nothing else commits them.
  commit_into(dr, *o_, projects_to_observables(),
              fires_since_reset_ == 0 ? 0 : lo_col_, hi_col_, lo.bits);
  for (std::size_t i = 0; i < corr_.size(); ++i)
    corr_[i] ^= lo.bits[i];
  last_lo_valid_ = true;
  ++fires_since_reset_;
  if (pushes_)
    push_candidate(sink_, lo, dr);
  return true;
}

std::optional<decoder_result>
window_node::decode(const std::vector<float_t> &dets,
                    const cancellation_token &tok) const {
  if (!cancellable_)
    return dec_->decode(dets);
  return dec_->decode(dets, tok);
}

void window_node::push_candidate(const candidate_sink &sink,
                                 const logical_outcome &lo,
                                 const decoder_result &dr) const {
  std::optional<double> cost;
  if (dr.opt_results && dr.opt_results->contains("cost"))
    cost = dr.opt_results->get<double>("cost");
  sink.push(lo, id_, cost);
}

bool window_node::projects_to_observables() const {
  return dec_->get_result_type() == decode_result_type::observables;
}

const uint8_t *window_node::corrections() const { return corr_.data(); }

void window_node::clear_corrections() {
  std::fill(corr_.begin(), corr_.end(), uint8_t{0});
  // Re-opens a flushed node.  The double-commit hazard is a flushed tail and a
  // later overlapping fire coexisting in one accumulator; zeroing it dissolves
  // that, so streaming may resume.  This is the between-shots call on the wire
  // path -- get_corrections(reset=true) clears rather than resetting -- so
  // requiring reset_decoder() here would strand every session-backed caller.
  flushed_ = false;
}

void window_node::reset_streaming_state() {
  // The shot is being discarded with rounds still withheld.  Silent for any
  // caller that ends its stream with a FINAL round, which is what commits
  // them.
  if (fires_since_reset_ > 0 && last_result_valid_ && !flushed_)
    CUDA_QEC_WARN("window_node: reset with the last window's {} guard rounds "
                  "still uncommitted; tag the shot's last round FINAL to keep "
                  "them",
                  ws_.end_guard);
  flushed_ = false;
  last_result_valid_ = false;
  bytes_in_ = 0;
  std::fill(buf_.begin(), buf_.end(), uint8_t{0});
  fires_since_reset_ = 0;
  std::fill(det_buf_.begin(), det_buf_.end(), 0.0f);
  last_lo_valid_ = false;
  clear_corrections();
}

void window_node::reset_decoder() {
  reset_streaming_state();
  dec_->reset_decoder();
}

void window_node::reject_after_flush(const char *call) const {
  if (flushed_)
    throw std::logic_error(
        std::string("window_node: ") + call +
        " after flush() with its tail still in the corrections buffer. Those "
        "rounds are exactly the ones the next window's core commits, so "
        "resuming would XOR them back out and silently zero them. Read the "
        "corrections, then clear_corrections() for the next shot, or "
        "reset_decoder() to also drop the accumulated rounds.");
}

bool window_node::flush() {
  if (flushed_)
    return false;

  const std::size_t keep = (ws_.size - ws_.stride) * spr_;
  // Also what stops a snapshot cached before reset_decoder() from landing in
  // the new shot's corrections.
  if (fires_since_reset_ == 0) {
    if (bytes_in_ > 0)
      CUDA_QEC_WARN("window_node: {} of {} syndrome bytes never filled a "
                    "window and are not committed; they remain buffered, so "
                    "reset_decoder() is required before the next shot",
                    bytes_in_, buf_.size());
    return false;
  }
  if (bytes_in_ > keep)
    CUDA_QEC_WARN(
        "window_node: {} syndrome bytes never reached a full window and are "
        "not committed; they remain buffered, so reset_decoder() is required "
        "before the next shot (send whole rounds so the round count satisfies "
        "(rounds - window_size) % step_size == 0)",
        bytes_in_ - keep);

  // Nothing withheld.  Set under the same hi_col_ condition that withholds a
  // tail, so it covers the unguarded window too.
  if (!last_result_valid_)
    return false;

  // Latch only now.  The latch exists to stop a *committed* tail from being
  // XOR-cancelled by a later overlapping fire; with nothing committed there is
  // no hazard, and latching would brick the stream -- every unguarded window
  // returns false above, so a caller that flushes defensively would find its
  // next enqueue rejected for a tail that was never committed.
  flushed_ = true;

  // [hi_col_, end) is the complement of what that fire committed, so the two
  // together cover the window exactly once -- what XOR into corr_ requires.
  auto &lo = last_lo_val_.get<logical_outcome>();
  std::copy(last_result_.begin(), last_result_.end(), flush_scratch_.begin());
  mask_guard_range(flush_scratch_, hi_col_,
                   std::numeric_limits<std::size_t>::max());
  std::fill(lo.bits.begin(), lo.bits.end(), uint8_t{0});
  project_into(csr_rows{*o_}, flush_scratch_.data(), lo.bits.data());
  for (std::size_t i = 0; i < corr_.size(); ++i)
    corr_[i] ^= lo.bits[i];
  // Publishes the tail, not the last fire: routing reads streaming_output()
  // after a flush, so leaving the fire's value here would re-deliver it.
  //
  // Dormant consequence, since no kind can consume a window's output today: a
  // FINAL round that also fires routes twice in one call -- once for the fire,
  // once for the tail.  That is correct as two separate commits, but a
  // consumer that decodes per delivery would decode twice for one round.
  // Settle it when a kind can actually take this edge.
  last_lo_valid_ = true;
  return true;
}

std::size_t window_node::num_msyn_per_decode() const { return buf_.size(); }

std::optional<std::size_t> window_node::raw_chunk_limit() const {
  return buf_.size() - bytes_in_;
}
std::size_t window_node::num_msyn_per_round() const { return spr_; }
std::size_t window_node::num_observables() const { return corr_.size(); }

void window_node::execute_indexed(const std::vector<edge_value> &inputs,
                                  std::vector<edge_value> &outputs) const {
  // Either form: raw bits get the D-apply, detection events are already in
  // detector space (an upstream node's residual) and skip it, exactly as the
  // two streaming entry points do.
  const std::vector<float_t> *dets = nullptr;
  if (const auto *m = inputs[0].get_if<measurement_results>()) {
    if (m->bits.size() < d_->num_cols())
      throw std::out_of_range(
          "window_node: shot has " + std::to_string(m->bits.size()) +
          " measurements, D needs " + std::to_string(d_->num_cols()));
    detect_into(csr_rows{*d_}, m->bits.data(), dets_.data());
    dets = &dets_;
  } else if (const auto *de = inputs[0].get_if<detection_events>()) {
    if (de->bits.size() != d_->num_rows())
      throw std::out_of_range(
          "window_node: got " + std::to_string(de->bits.size()) +
          " detection events, D produces " + std::to_string(d_->num_rows()));
    dets = &de->bits;
  } else {
    throw std::invalid_argument("window_node: input slot 0 takes "
                                "measurement_results or detection_events");
  }
  auto decoded = decode(
      *dets, cancellable_ ? inputs[*token_slot()].get<cancellation_token>()
                          : cancellation_token{});
  logical_outcome lo;
  lo.bits.assign(o_->num_rows(), 0);
  // A stopped decode answers nothing: an unconverged zero outcome, no
  // candidate.
  if (decoded) {
    auto &dr = *decoded;
    lo.converged = dr.converged;
    // A one-shot execution has only one window, so neither guard applies:
    // no predecessor committed the leading rounds, and no successor -- nor a
    // flush, which the one-shot path lacks -- will commit the trailing ones.
    // Masking would discard them rather than defer them.
    commit_into(dr, *o_, projects_to_observables(), /*lo_col=*/0,
                /*hi_col=*/std::numeric_limits<std::size_t>::max(), lo.bits);
    if (pushes_)
      push_candidate(inputs[*sink_slot()].get<candidate_sink>(), lo, dr);
  }
  outputs[0] = std::move(lo);
}

std::vector<std::string> window_node::inputs() const {
  std::vector<std::string> names{"input"};
  if (pushes_)
    names.push_back("sink");
  if (cancellable_)
    names.push_back("token");
  return names;
}
std::vector<std::string> window_node::outputs() const { return {"output"}; }

std::future<logical_outcome>
window_node::run_async(const measurement_results &m) const {
  if (m.bits.size() < d_->num_cols())
    throw std::out_of_range(
        "decoding_task_graph: shot has " + std::to_string(m.bits.size()) +
        " measurements, D needs " + std::to_string(d_->num_cols()));
  std::vector<float_t> dets(d_->num_rows(), 0.0f);
  detect_into(csr_rows{*d_}, m.bits.data(), dets.data());
  auto pending = dec_->decode_async(dets);
  const sparse_binary_matrix *o_ptr = o_.get();
  // Read the decoder's result type here, on the calling thread: the worker
  // must not reach back into the node.
  return std::async(
      std::launch::deferred, [o_ptr, obs = projects_to_observables(),
                              pending = std::move(pending)]() mutable {
        auto dr = pending.get();
        logical_outcome lo;
        lo.converged = dr.converged;
        lo.bits.assign(o_ptr->num_rows(), 0);
        // Like execute_indexed(), this is one standalone shot:
        // the only window, so neither guard applies.
        commit_into(dr, *o_ptr, obs, /*lo_col=*/0,
                    std::numeric_limits<std::size_t>::max(), lo.bits);
        return lo;
      });
}

std::optional<edge_value_type>
window_node::output_type(std::size_t slot) const {
  if (slot == 0)
    return logical_outcome_type;
  return std::nullopt;
}

bool window_node::accepts_input_type(std::size_t slot,
                                     edge_value_type type) const {
  if (slot == 0)
    return type == measurement_results_type || type == detection_events_type;
  if (slot == sink_slot())
    return type == edge_value_type::of<candidate_sink>();
  return slot == token_slot() &&
         type == edge_value_type::of<cancellation_token>();
}

std::optional<std::size_t>
window_node::output_value_size(std::size_t slot) const {
  if (slot != 0)
    return std::nullopt;
  return o_->num_rows();
}

void window_node::validate_input_shape(
    std::size_t slot, edge_value_type type,
    std::optional<std::size_t> producer_size) const {
  if (slot != 0 || type != detection_events_type || !producer_size)
    return;
  if (*producer_size != d_->num_rows())
    throw std::invalid_argument("window_node: detection-events input has " +
                                std::to_string(*producer_size) +
                                " values, but D produces " +
                                std::to_string(d_->num_rows()));
}

bool window_node::supports_streaming_input(std::size_t slot) const {
  return slot == 0 || slot == sink_slot() || slot == token_slot();
}

std::optional<window_spec> window_node::derive_streaming_schedule(
    std::size_t output_slot, const std::vector<input_schedule> &inputs) const {
  if (output_slot != 0)
    return std::nullopt;
  // The handle inputs carry no window; the data input, if any, sets the pace.
  const input_schedule *data = nullptr;
  for (const auto &in : inputs) {
    if (in.slot != 0)
      continue;
    if (data)
      return std::nullopt;
    data = &in;
  }
  if (!data)
    return ws_;
  if (data->schedule.size != ws_.size)
    throw std::invalid_argument("window_node: edge-fed input covers " +
                                std::to_string(data->schedule.size) +
                                " rounds, but this window covers " +
                                std::to_string(ws_.size));
  // Delivery controls when this node fires; this node's own window controls
  // what each fire commits. In particular, do not inherit producer guards.
  return window_spec{ws_.size, data->schedule.stride, ws_.start_guard,
                     ws_.end_guard};
}

const edge_value *window_node::streaming_output(std::size_t slot) const {
  return (slot == 0 && last_lo_valid_) ? &last_lo_val_ : nullptr;
}

bool window_node::enqueue_edge_value(const edge_value &v, std::size_t to_slot) {
  if (to_slot == sink_slot()) {
    sink_ = v.get<candidate_sink>();
    return false;
  }
  if (to_slot == token_slot()) {
    token_ = v.get<cancellation_token>();
    return false;
  }
  reject_after_flush("enqueue_edge_value");
  if (to_slot != 0)
    throw std::invalid_argument("window_node::enqueue_edge_value: slot " +
                                std::to_string(to_slot) + " is not supported");
  const auto *dets = v.get_if<detection_events>();
  if (!dets)
    throw std::invalid_argument(
        "window_node::enqueue_edge_value: slot 0 requires detection_events "
        "(compile_wires should have caught this type mismatch)");
  validate_edge_input(to_slot);
  if (dets->bits.size() != d_->num_rows())
    throw std::out_of_range(
        "window_node: enqueue_edge_value: got " +
        std::to_string(dets->bits.size()) + " detectors, expected " +
        std::to_string(d_->num_rows()) +
        " (the upstream node must share this node's DEM and window_size)");

  std::copy(dets->bits.begin(), dets->bits.end(), det_buf_.begin());
  return fire(det_buf_);
}

void window_node::validate_edge_input(std::size_t to_slot) const {
  if (to_slot == sink_slot() || to_slot == token_slot())
    return;
  if (to_slot != 0)
    throw std::invalid_argument("window_node: slot " + std::to_string(to_slot) +
                                " takes no streaming input");
  // An edge carries one whole window of detection events: the upstream node
  // applied a D over the same rounds, and D's leading rows are that window's
  // first round measured against the initial state, which no later window's
  // rows reproduce.  That is also why a window fed this way cannot slide — it
  // would need a leading band it has no raw bytes left to rebuild.
  if (ws_.stride != ws_.size)
    throw std::invalid_argument(
        "window_node: a window fed detection events decodes one whole window "
        "per value and cannot slide (step_size " +
        std::to_string(ws_.stride) + " must equal window_size " +
        std::to_string(ws_.size) + ")");
}

const sparse_binary_matrix &window_node::d() const { return *d_; }
const sparse_binary_matrix &window_node::o() const { return *o_; }
std::shared_ptr<const sparse_binary_matrix> window_node::d_ptr() const {
  return d_;
}
std::shared_ptr<const sparse_binary_matrix> window_node::o_ptr() const {
  return o_;
}

std::optional<int> window_node::cuda_device_id() const {
  const int id = dec_->get_cuda_device_id();
  return id >= 0 ? std::optional<int>(id) : std::nullopt;
}

bool window_node::supports_graph_dispatch() const {
  // capture_graph() delegates the whole window (D-apply -> decode -> O-apply)
  // to the inner decoder's own captured graph, which fires once per full
  // window and resets to empty afterward -- it has no notion of retaining
  // (size - stride) rounds for the next fire (sliding_window) or of masking
  // guard columns out of the commit (guarded windows). Capturing either would
  // silently change results: a sliding window would fire every full window
  // instead of every stride, and a guarded window would commit rounds meant
  // to be withheld. Restrict DEVICE dispatch to fixed-T, unguarded windows
  // until sliding/guarded state is captured too.
  if (ws_.stride != ws_.size || ws_.start_guard != 0 || ws_.end_guard != 0)
    return false;
  return dec_->supports_graph_dispatch();
}

void *window_node::capture_graph(int reserved_sms) {
  // Decoder model inputs are immutable. Configuration-built nodes construct
  // the inner decoder with this window's D and O; a caller-provided decoder
  // instance is likewise responsible for carrying the model it captures.
  // The sanctioned caller of capture_decode_graph(): pins to dec_'s device
  // first so capture lands on the GPU every launch will use.
  return cudaq::qec::detail_affinity::capture_graph_pinned(*dec_, reserved_sms);
}

void window_node::release_graph(void *graph_resources) {
  dec_->release_decode_graph(graph_resources);
}

decoder *window_node::inner_decoder() const { return dec_.get(); }

void window_node::mask_guard_columns(std::vector<float_t> &result,
                                     bool first_fire) const {
  mask_guard_range(result, first_fire ? 0 : lo_col_, hi_col_);
}

void window_node::require_width(std::size_t result_size) const {
  if (o_->num_cols() > result_size)
    throw std::out_of_range(
        "decoding_task_graph: decoder returned " + std::to_string(result_size) +
        " values, O references column " + std::to_string(o_->num_cols() - 1));
}

// ---------------------------------------------------------------------------
// Node-kind construction shared by the window kinds
// ---------------------------------------------------------------------------

std::vector<decoding::config::param_spec> window_kind_common_params() {
  using decoding::config::param_kind;
  return {
      {"window_size", param_kind::uint64},
      // Redundant with the DEM, and therefore worth stating: a mismatch means
      // the author's model of the round layout differs from the DEM's.
      {"num_syndromes_per_round", param_kind::uint64},
  };
}

namespace {

/// Cross-field rules for `sliding_window`'s parameters, run by the schema
/// registry on both the YAML and programmatic paths. Bounds that need the DEM
/// (does `window_size` match the rounds it expands to?) belong to
/// make_window_node(), which has it.
void validate_sliding_window_params(const cudaqx::heterogeneous_map &p) {
  const auto guard = [&](const char *key) {
    return p.contains(key) ? p.get<std::size_t>(key) : 0u;
  };
  if (p.contains("guard_rounds") &&
      (p.contains("start_guard") || p.contains("end_guard")))
    throw std::runtime_error(
        "sliding_window: guard_rounds sets both guards, so it cannot be "
        "combined with start_guard or end_guard");

  const auto size =
      p.contains("window_size") ? p.get<std::size_t>("window_size") : 0u;
  if (p.contains("window_size") && size == 0)
    throw std::runtime_error("sliding_window: window_size must be at least 1");
  const auto stride =
      p.contains("step_size") ? p.get<std::size_t>("step_size") : 1u;
  if (stride == 0)
    throw std::runtime_error("sliding_window: step_size must be at least 1");
  if (p.contains("window_size") && stride > size)
    throw std::runtime_error(
        "sliding_window: step_size " + std::to_string(stride) +
        " must not exceed window_size " + std::to_string(size));

  const auto guards = p.contains("guard_rounds")
                          ? 2 * guard("guard_rounds")
                          : guard("start_guard") + guard("end_guard");
  if (guards == 0)
    return;
  // A guard withholds rounds from this window's commit on the understanding
  // that a neighboring window commits them, which only happens if the windows
  // overlap.
  if (stride >= size)
    throw std::runtime_error(
        "sliding_window: guards need overlapping windows (step_size " +
        std::to_string(stride) + " must be less than window_size " +
        std::to_string(size) + ")");
  if (guards >= size)
    throw std::runtime_error(
        "sliding_window: guards leave no rounds to commit (" +
        std::to_string(guards) + " guard rounds of a " + std::to_string(size) +
        "-round window)");
  if (stride != size - guards)
    throw std::runtime_error(
        "sliding_window: guarded windows must tile the stream exactly once "
        "(step_size " +
        std::to_string(stride) + " must equal window_size - guard rounds = " +
        std::to_string(size - guards) + ")");
}

/// The optional ports. `push_candidates` adds a `sink` input (a
/// `candidate_sink`) every outcome is pushed into, costed when the decoder
/// reports a `cost` among its optional results; `cancellable` adds a `token`
/// input (a `cancellation_token`) the decode honors through
/// `decoder::decode(syndrome, cancellation_token)`. Window kinds only; the
/// predecoder shares the common parameters but not these.
void add_port_params(std::vector<decoding::config::param_spec> &params) {
  using decoding::config::param_kind;
  params.push_back({"push_candidates", param_kind::boolean});
  params.push_back({"cancellable", param_kind::boolean});
}

decoding::config::decoder_schema sliding_window_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params = window_kind_common_params();
  add_port_params(schema.params);
  schema.params.push_back({"step_size", param_kind::uint64});
  schema.params.push_back({"start_guard", param_kind::uint64});
  schema.params.push_back({"end_guard", param_kind::uint64});
  schema.params.push_back({"guard_rounds", param_kind::uint64});
  schema.params.push_back({"arbitrary_length", param_kind::boolean});
  schema.validate = validate_sliding_window_params;
  return schema;
}

/// One window covering a fixed round count, decoded in one shot. It declares
/// neither a step size nor guards, so naming either is an unknown-key error
/// rather than a special case in a validator.
decoding::config::decoder_schema single_window_params() {
  decoding::config::decoder_schema schema;
  schema.params = window_kind_common_params();
  add_port_params(schema.params);
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    if (p.contains("window_size") && p.get<std::size_t>("window_size") == 0)
      throw std::runtime_error("single_window: window_size must be at least 1");
  };
  return schema;
}

/// Whether a window can delegate observable projection to its inner decoder.
bool window_inner_takes_observables(const cudaqx::heterogeneous_map &params) {
  return !params.get<bool>("arbitrary_length", false) &&
         params.get<std::size_t>("guard_rounds", 0) == 0 &&
         params.get<std::size_t>("start_guard", 0) == 0 &&
         params.get<std::size_t>("end_guard", 0) == 0;
}

/// The two kinds this node implements, registered from the node's own
/// translation unit: a kind costs a spec, a parameter schema, and a factory,
/// and nothing in the config layer, the JSON-schema export, or the Python
/// bindings mentions either by name.
const bool registered = [] {
  task_graph_node_kind_spec sliding;
  sliding.name = "sliding_window";
  sliding.aliases = {"SlidingWindow"};
  sliding.inner_decoder_output_policy =
      [](const cudaqx::heterogeneous_map &params) {
        return decoder_output_request{window_inner_takes_observables(params)
                                          ? decode_result_type::observables
                                          : decode_result_type::errors};
      };
  sliding.inner_decoder_observable_policy = window_inner_takes_observables;
  register_node_kind(std::move(sliding), sliding_window_params(),
                     [](node_build_context &ctx) {
                       return make_window_node(ctx, /*sliding=*/true);
                     });

  task_graph_node_kind_spec single;
  single.name = "single_window";
  single.aliases = {"SingleWindow"};
  register_node_kind(std::move(single), single_window_params(),
                     [](node_build_context &ctx) {
                       return make_window_node(ctx, /*sliding=*/false);
                     });
  return true;
}();

} // namespace

std::unique_ptr<task_graph_node> make_window_node(node_build_context &ctx,
                                                  bool sliding) {
  const auto &p = *ctx.node_params;
  const std::string where = "task graph node '" + ctx.id + "'";

  const auto size =
      static_cast<uint32_t>(p.get<std::size_t>("window_size", ctx.dem_rounds));
  if (size != ctx.dem_rounds)
    throw std::invalid_argument(
        where + ": window_size " + std::to_string(size) +
        " does not match the " + std::to_string(ctx.dem_rounds) +
        " rounds its DEM spans (window_size is what the DEM expands to)");

  uint32_t start_guard = 0, end_guard = 0, stride = size;
  if (sliding) {
    stride = static_cast<uint32_t>(p.get<std::size_t>("step_size", 1));
    if (p.contains("guard_rounds")) {
      start_guard = end_guard =
          static_cast<uint32_t>(p.get<std::size_t>("guard_rounds"));
    } else {
      start_guard = static_cast<uint32_t>(p.get<std::size_t>("start_guard", 0));
      end_guard = static_cast<uint32_t>(p.get<std::size_t>("end_guard", 0));
    }
  }
  if (stride == 0)
    throw std::invalid_argument(where + ": step_size must be at least 1");
  if (stride > size)
    throw std::invalid_argument(
        where + ": step_size " + std::to_string(stride) +
        " must not exceed window_size " + std::to_string(size));

  if (p.contains("num_syndromes_per_round")) {
    const auto declared = p.get<std::size_t>("num_syndromes_per_round");
    if (declared != ctx.spr)
      throw std::invalid_argument(
          where + ": num_syndromes_per_round is " + std::to_string(declared) +
          " but the DEM has " + std::to_string(ctx.spr) +
          " syndrome bytes per round");
  }

  // Flat matrices describe one whole experiment: there is no per-round column
  // layout to slide along, which is also why guards are unavailable there.
  if (ctx.fault_offsets.empty() && stride != size)
    throw std::invalid_argument(
        where +
        ": a window over flat D/O matrices cannot slide (step_size must equal "
        "window_size); describe the DEM with dem_chunks to stream rounds");

  if (sliding && p.get<bool>("arbitrary_length", false)) {
    if (!ctx.memory_spec || ctx.decoder_type.empty())
      throw std::invalid_argument(
          where + ": arbitrary_length requires a DEM recipe and decoder name");
    if (start_guard == 0 && end_guard == 0 && stride < size)
      end_guard = size - stride;
    return std::make_unique<memory_window_node>(
        ctx, window_spec{size, stride, start_guard, end_guard});
  }
  return std::make_unique<window_node>(
      ctx.d, ctx.o, std::move(ctx.dec), ctx.spr,
      window_spec{size, stride, start_guard, end_guard},
      std::move(ctx.fault_offsets), ctx.num_measurements,
      p.get<bool>("push_candidates", false), p.get<bool>("cancellable", false),
      ctx.id);
}

// ---------------------------------------------------------------------------
// decoding_task_graph public constructor (single window_node topology)
// ---------------------------------------------------------------------------

decoding_task_graph::decoding_task_graph(
    std::unique_ptr<decoder> dec, std::vector<std::vector<uint32_t>> d_sparse,
    std::vector<std::vector<uint32_t>> o_sparse, task_graph_options options) {
  if (!dec)
    throw std::invalid_argument(
        "decoding_task_graph: decoder must not be null");
  task_graph_node_config node;
  node.id = "node0";
  // Flat matrices are a fixed-T experiment by construction, and one
  // measurement column is one accumulation step.
  node.kind = "single_window";
  node.d_sparse = std::move(d_sparse);
  node.o_sparse = std::move(o_sparse);
  node.decoder_instance = std::move(dec);

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  impl_ =
      std::move(from_topology(std::move(nodes), {}, std::move(options))->impl_);
}

// ---------------------------------------------------------------------------
// decoding_task_graph::from_spec (single window_node topology)
// ---------------------------------------------------------------------------

std::unique_ptr<decoding_task_graph> decoding_task_graph::from_spec(
    const dem_chunks_spec &spec, const std::string &decoder_type,
    const cudaqx::heterogeneous_map &params, task_graph_options options) {
  // An unresolved round count is what makes this a stream: the spec repeats a
  // phase and does not say how often, so the window is the shortest chain the
  // connections admit and it advances one round at a time.
  const bool streaming =
      !spec.num_rounds.has_value() && spec.has_repeating_phase();

  task_graph_node_config node;
  node.id = "node0";
  node.kind = streaming ? "sliding_window" : "single_window";
  node.dem = spec;
  node.decoder_type = decoder_type;
  node.params = params;
  if (streaming)
    node.node_params.insert("step_size", std::size_t{1});

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  return from_topology(std::move(nodes), {}, std::move(options));
}

} // namespace cudaq::qec

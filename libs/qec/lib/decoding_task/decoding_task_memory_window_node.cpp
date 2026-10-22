/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "decoding_task_memory_window_node.h"
#include "decoding_task_graph_utils.h"
#include "decoding_task_node_kind_registry.h"
#include "cudaq/qec/logger.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cudaq::qec {

memory_window_node::memory_window_node(node_build_context &ctx,
                                       window_spec schedule)
    : window_node(ctx.d, ctx.o, std::move(ctx.dec), ctx.spr,
                  {schedule.size, schedule.size, 0, 0}, {},
                  ctx.num_measurements),
      spec_(ctx.memory_spec), schedule_(schedule) {
  const auto span = spec_->repeating_rounds();
  if (span.prefix == 0 || span.suffix == 0 ||
      span.prefix + span.suffix >= schedule_.size)
    throw std::invalid_argument(
        "memory window requires an initial prefix and terminal phases "
        "fitting in the window");
  if (ctx.decoder_type.empty())
    throw std::invalid_argument("memory window requires a decoder construction "
                                "recipe, not a decoder instance");
  if (ctx.decoder_params.contains("O"))
    throw std::invalid_argument(
        "memory window projects observables at graph level; inner decoder "
        "parameters must not supply O");
  if (schedule_.stride >= schedule_.size || schedule_.size < 3 ||
      schedule_.start_guard + schedule_.end_guard >= schedule_.size ||
      schedule_.end_guard == 0 || schedule_.stride != schedule_.core())
    throw std::invalid_argument("memory window requires a trailing guard and "
                                "stride equal to the committed core");
  const auto sequence = spec_->phase_sequence();
  prefix_phases_.assign(sequence.begin(), sequence.begin() + span.prefix);
  bulk_phase_ = spec_->repeating_phase();
  final_phases_.assign(sequence.end() - span.suffix, sequence.end());
  // Until FINAL arrives, these records are indistinguishable from bulk.
  // Hold every possible terminal record beyond the regular inference window.
  // Preserve the existing single-terminal-phase delivery contract.
  tail_lookahead_ = final_phases_.size() > 1 ? final_phases_.size() : 0;
  if (schedule_.start_guard < tail_lookahead_)
    throw std::invalid_argument(
        "memory window: leading guard must retain terminal phase lookahead");
  if (tail_lookahead_ > std::numeric_limits<uint32_t>::max() - schedule_.size)
    throw std::overflow_error("memory window: delivery window exceeds range");
  delivery_schedule_ = schedule_;
  delivery_schedule_.size += tail_lookahead_;
  delivery_schedule_.end_guard += tail_lookahead_;
  bulk_width_ = ctx.spr;
  for (const auto phase : prefix_phases_)
    prefix_widths_.push_back(measurement_width(spec_->get_phase(phase)));
  const auto declared_bulk_width =
      measurement_width(spec_->get_phase(bulk_phase_));
  final_width_ = measurement_width(spec_->get_phase(final_phases_.back()));
  if (!bulk_width_ || !final_width_ ||
      std::find(prefix_widths_.begin(), prefix_widths_.end(), 0) !=
          prefix_widths_.end())
    throw std::invalid_argument("memory window requires nonzero initial, bulk "
                                "and final measurement records");

  if (declared_bulk_width != ctx.spr || final_width_ != ctx.spr ||
      std::any_of(final_phases_.begin(), final_phases_.end(),
                  [&](auto id) {
                    return measurement_width(spec_->get_phase(id)) != ctx.spr;
                  }) ||
      std::any_of(prefix_widths_.begin(), prefix_widths_.end(),
                  [&](auto width) { return width != ctx.spr; }))
    throw std::invalid_argument(
        "arbitrary_length requires equal measurement widths in every phase");

  // Prepare every reachable physical-boundary layout before accepting input.
  // There is no model construction or allocation in the streaming hot path.
  const auto add = [&](std::size_t start_phase, bool terminal, std::size_t n) {
    models_.emplace(model_key{start_phase, terminal, n},
                    compile_model(start_phase, terminal, n, ctx));
  };
  // prefix.size()+1 distinguishes the first bulk window (preceded by the
  // last prefix phase) from subsequent bulk windows.
  for (std::size_t start = 0; start <= prefix_phases_.size() + 1; ++start) {
    if (start % schedule_.stride == 0 || start == prefix_phases_.size() + 1)
      add(start, false, schedule_.size);
    add(start, true, schedule_.size);
  }
  for (std::size_t n = spec_->minimum_rounds();
       n <= static_cast<std::size_t>(schedule_.size) + tail_lookahead_; ++n)
    if (n != schedule_.size)
      add(0, true, n);

  const auto widest =
      std::max({*std::max_element(prefix_widths_.begin(), prefix_widths_.end()),
                bulk_width_, final_width_});
  std::size_t detector_width = 0;
  for (const auto &[key, m] : models_) {
    for (std::size_t r = 1; r < m.detector_offsets.size(); ++r)
      detector_width = std::max(detector_width, m.detector_offsets[r] -
                                                    m.detector_offsets[r - 1]);
  }
  ring_.resize(static_cast<std::size_t>(schedule_.size) + tail_lookahead_ + 1);
  for (auto &r : ring_) {
    r.measurements.resize(widest);
    r.carry.resize(detector_width);
  }
  corrections_.resize(ctx.o->num_rows());
  output_ = logical_outcome{};
  output_.get<logical_outcome>().bits.resize(corrections_.size());
  next_fire_ = delivery_schedule_.size;
}

memory_window_node::model
memory_window_node::compile_model(std::size_t start_phase, bool terminal,
                                  std::size_t n,
                                  const node_build_context &ctx) {
  model m;
  std::vector<extended_dem> chunks;
  std::vector<std::vector<uint32_t>> rows;
  const auto previous_width = start_phase == 0 ? 0
                              : start_phase <= prefix_widths_.size()
                                  ? prefix_widths_[start_phase - 1]
                                  : bulk_width_;
  constexpr auto max_measurements = std::numeric_limits<uint32_t>::max();
  if (previous_width > max_measurements)
    throw std::overflow_error("memory window: context width exceeds CSR range");
  std::size_t previous_start = 0, previous_count = previous_width;
  std::size_t start = previous_width;
  m.measurement_offsets.push_back(start);
  m.detector_offsets.push_back(0);
  const std::vector<seam_id> seams{spec_->seam.to_seam, spec_->seam.from_seam};
  for (std::size_t r = 0; r < n; ++r) {
    const auto id = terminal && r >= n - final_phases_.size()
                        ? final_phases_[r - (n - final_phases_.size())]
                    : start_phase + r < prefix_phases_.size()
                        ? prefix_phases_[start_phase + r]
                        : bulk_phase_;
    const auto &phase = spec_->get_phase(id);
    const auto width = measurement_width(phase);
    if (width > max_measurements - start)
      throw std::overflow_error(
          "memory window: window width exceeds CSR range");
    chunks.push_back(dem_chunk_from_spec(phase, seams));
    const auto &chunk = chunks.back();
    const auto expected_rows =
        chunk.num_interior_rows() +
        (chunk.has_seam(spec_->seam.to_seam)
             ? chunk.get_seam(spec_->seam.to_seam).num_rows()
             : 0);
    std::vector<uint32_t> row;
    const auto before = rows.size();
    auto phase_d = phase.D_sparse;
    if (!spec_->has_D_sparse()) {
      if (expected_rows != ctx.spr)
        throw std::invalid_argument(
            "memory window: default D requires one detector per measurement");
      for (std::size_t k = 0; k < expected_rows; ++k) {
        if (previous_count)
          phase_d.push_back(k);
        phase_d.push_back(previous_count + k);
        phase_d.push_back(-1);
      }
    }
    for (const auto value : phase_d) {
      if (value == -1) {
        rows.push_back(std::move(row));
        row.clear();
      } else {
        const auto index = static_cast<std::uint64_t>(value);
        if (value < 0 || index >= previous_count + width)
          throw std::invalid_argument("memory window: D index is outside "
                                      "previous/current measurements");
        const auto column = index < previous_count
                                ? previous_start + index
                                : start + index - previous_count;
        if (column > std::numeric_limits<uint32_t>::max())
          throw std::overflow_error(
              "memory window: measurement index exceeds CSR range");
        row.push_back(static_cast<uint32_t>(column));
      }
    }
    if (!row.empty() || rows.size() - before != expected_rows)
      throw std::invalid_argument(
          "memory window: D rows do not match the phase's detector rows");
    previous_start = start;
    previous_count = width;
    start += width;
    m.measurement_offsets.push_back(start);
    m.detector_offsets.push_back(rows.size());
  }
  if (start > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error(
        "memory window: measurement count exceeds CSR range");
  m.d = sparse_binary_matrix::from_nested_csr(rows.size(), start, rows)
            .canonicalize();
  const auto closed =
      dem_close_all(chunks, spec_->seam.from_seam, spec_->seam.to_seam);
  m.h = sparse_binary_matrix(closed.detector_error_matrix,
                             sparse_binary_matrix_layout::csr)
            .canonicalize();
  m.o = sparse_binary_matrix(closed.observables_flips_matrix,
                             sparse_binary_matrix_layout::csr)
            .canonicalize();
  if (m.h.num_rows() != rows.size() || m.o.num_rows() != ctx.o->num_rows())
    throw std::invalid_argument(
        "memory window: model and measurement/output layouts disagree");
  m.fault_offsets = dem_chunks_to_fault_offsets(chunks);
  m.dec = get_decoder(ctx.decoder_type,
                      decoder_init(m.h, std::nullopt, closed.error_rates, m.d,
                                   closed.error_ids),
                      decoder_output_request{decode_result_type::errors},
                      ctx.decoder_params);
  if (m.dec->get_result_type() == decode_result_type::observables ||
      m.dec->get_syndrome_size() != m.h.num_rows() ||
      m.dec->get_block_size() != m.fault_offsets.back() ||
      m.dec->get_cuda_device_id() != inner_decoder()->get_cuda_device_id())
    throw std::invalid_argument("memory window: decoder must return matching "
                                "fault columns on the configured device");
  m.measurements.resize(start);
  m.residual.resize(rows.size());
  if (!terminal) {
    // decoder_init copied H into the decoder's immutable model. Replace our
    // row layout rather than retaining a second matrix for the carry path.
    m.h = m.h.to_csc();
    m.detector_rounds.resize(rows.size());
    for (std::size_t r = 0; r + 1 < m.detector_offsets.size(); ++r)
      std::fill(m.detector_rounds.begin() + m.detector_offsets[r],
                m.detector_rounds.begin() + m.detector_offsets[r + 1], r);
  }
  return m;
}

std::size_t
memory_window_node::measurement_width(const dem_chunk_spec &phase) const {
  if (phase.num_measurements)
    return *phase.num_measurements;
  if (spec_->measurements_per_round)
    return *spec_->measurements_per_round;
  return bulk_width_; // The existing inferred memory-experiment convention.
}

std::size_t memory_window_node::next_width() const {
  return rounds_ < prefix_widths_.size() ? prefix_widths_[rounds_]
                                         : bulk_width_;
}

std::optional<std::size_t> memory_window_node::raw_chunk_limit() const {
  return terminal_complete_ ? 0 : next_width() - partial_bytes_;
}

bool memory_window_node::enqueue(const uint8_t *data, std::size_t len,
                                 round_type type) {
  if (failed_)
    throw std::logic_error("memory window: reset required after failure");
  if (flushed_ || terminal_complete_)
    throw std::logic_error("memory window: input after terminal phase");
  // FINAL carries the final phase; an empty one would start the next ring
  // slot, which may still hold context the terminal model needs.
  if (len == 0 && type == round_type::FINAL)
    throw std::invalid_argument("memory window: empty FINAL round");
  if (len > *raw_chunk_limit() || (!data && len))
    throw std::invalid_argument(
        "memory window: input exceeds the next phase record");
  auto &r = round(rounds_);
  if (partial_bytes_ == 0) {
    r.width = next_width();
    std::fill(r.carry.begin(), r.carry.end(), uint8_t{0});
  }
  if (len)
    std::copy_n(data, len, r.measurements.begin() + partial_bytes_);
  partial_bytes_ += len;
  if (partial_bytes_ != r.width) {
    if (type == round_type::FINAL) {
      failed_ = true;
      throw std::invalid_argument("memory window: truncated final phase");
    }
    return false;
  }
  const bool terminal = type == round_type::FINAL;
  ++rounds_;
  partial_bytes_ = 0;
  if (terminal) {
    terminal_complete_ = true;
    return false; // flush publishes one complete, physically closed tail.
  }
  if (rounds_ == next_fire_) {
    bool fired;
    try {
      fired = decode_window(false);
    } catch (...) {
      failed_ = true;
      throw;
    }
    next_fire_ += schedule_.stride;
    return fired;
  }
  return false;
}

bool memory_window_node::decode_window(bool terminal) {
  const auto window_end = terminal ? rounds_ : rounds_ - tail_lookahead_;
  // A tagged FINAL can preempt the very first delayed regular fire. No
  // prefix has been committed yet, so its prepared initial model must cover
  // the whole shot (at most W + tail_lookahead rounds), not just its last W.
  const auto n = static_cast<std::size_t>(
      terminal && committed_ == 0
          ? window_end
          : std::min<std::uint64_t>(window_end, schedule_.size));
  const auto start = window_end - n;
  auto &m = models_.at({static_cast<std::size_t>(std::min<std::uint64_t>(
                            start, prefix_phases_.size() + 1)),
                        terminal, n});
  const auto context = m.measurement_offsets.front();
  if (context) {
    const auto &previous = round(start - 1);
    if (previous.width != context)
      throw std::logic_error("memory window: context width mismatch");
    std::copy_n(previous.measurements.begin(), context, m.measurements.begin());
  }
  const auto first_slot = start % ring_.size();
  auto slot = first_slot;
  for (std::size_t r = 0; r < n; ++r) {
    const auto &input = ring_[slot];
    if (input.width != m.measurement_offsets[r + 1] - m.measurement_offsets[r])
      throw std::logic_error(
          "memory window: buffered phase does not match the physical model");
    std::copy_n(input.measurements.begin(), input.width,
                m.measurements.begin() + m.measurement_offsets[r]);
    if (++slot == ring_.size())
      slot = 0;
  }
  detect_into(csr_rows{m.d}, m.measurements.data(), m.residual.data());
  slot = first_slot;
  for (std::size_t r = 0; r < n; ++r) {
    const auto &carry = ring_[slot].carry;
    for (auto d = m.detector_offsets[r]; d < m.detector_offsets[r + 1]; ++d)
      m.residual[d] = static_cast<float_t>(static_cast<uint8_t>(m.residual[d]) ^
                                           carry[d - m.detector_offsets[r]]);
    if (++slot == ring_.size())
      slot = 0;
  }
  auto result = m.dec->decode(m.residual);
  if (result.result.size() != m.fault_offsets.back())
    throw std::out_of_range(
        "memory window: decoder returned the wrong fault width");
  // Preserve the decoder's convergence contract: approximate decoders may
  // publish a nonconverged estimate, just as they do in the equal-width path.
  const auto end = terminal ? rounds_ : window_end - schedule_.end_guard;
  if (committed_ < start || committed_ >= end)
    throw std::logic_error("memory window: noncontiguous commit ownership");
  const auto lo = m.fault_offsets[committed_ - start];
  const auto hi = m.fault_offsets[end - start];
  std::fill(result.result.begin(), result.result.begin() + lo, float_t{0});
  std::fill(result.result.begin() + hi, result.result.end(), float_t{0});
  auto &out = output_.get<logical_outcome>();
  std::fill(out.bits.begin(), out.bits.end(), uint8_t{0});
  project_into(csr_rows{m.o}, result.result.data(), out.bits.data());
  out.converged = result.converged;
  // Carry committed effects through every retained detector, including leading
  // context. Removing their leading effects would make matching predict them
  // again. A successful terminal decode closes the epoch; the next decode
  // requires reset, which clears carry, so no terminal H walk is needed.
  if (!terminal) {
    scatter_fault_syndrome(m.h, result.result.data(), lo, hi, [&](auto d) {
      const auto r = m.detector_rounds[d];
      auto target_slot = first_slot + r;
      if (target_slot >= ring_.size())
        target_slot -= ring_.size();
      ring_[target_slot].carry[d - m.detector_offsets[r]] ^= uint8_t{1};
    });
  }
  for (std::size_t i = 0; i < corrections_.size(); ++i)
    corrections_[i] ^= out.bits[i];
  coverage_ = committed_coverage{committed_, end};
  committed_ = end;
  return true;
}

bool memory_window_node::flush() {
  if (failed_)
    throw std::logic_error("memory window: reset required after failure");
  if (flushed_)
    return false;
  if (partial_bytes_ || rounds_ < spec_->minimum_rounds()) {
    failed_ = true;
    throw std::invalid_argument(
        "memory window: flush requires a complete init/bulk/final shot");
  }
  bool fired;
  try {
    fired = decode_window(true);
  } catch (...) {
    failed_ = true;
    throw;
  }
  flushed_ = true;
  return fired;
}

void memory_window_node::clear_corrections() {
  if (flushed_) {
    reset_streaming_state();
    return;
  }
  std::fill(corrections_.begin(), corrections_.end(), uint8_t{0});
}

void memory_window_node::reset_streaming_state() {
  if (!flushed_ && (rounds_ || partial_bytes_))
    CUDA_QEC_WARN("memory_window_node: reset discards an unfinished shot");
  // The base reset calls virtual clear_corrections(); avoid recursive reset.
  flushed_ = false;
  window_node::reset_streaming_state();
  rounds_ = committed_ = partial_bytes_ = 0;
  next_fire_ = delivery_schedule_.size;
  terminal_complete_ = flushed_ = false;
  failed_ = false;
  coverage_.reset();
  for (auto &r : ring_) {
    std::fill(r.carry.begin(), r.carry.end(), uint8_t{0});
    r.width = 0;
  }
  std::fill(corrections_.begin(), corrections_.end(), uint8_t{0});
  std::fill(output_.get<logical_outcome>().bits.begin(),
            output_.get<logical_outcome>().bits.end(), uint8_t{0});
}

void memory_window_node::reset_decoder() {
  reset_streaming_state();
  inner_decoder()->reset_decoder();
  for (auto &[key, m] : models_)
    m.dec->reset_decoder();
}

void memory_window_node::validate_edge_input(std::size_t) const {
  throw std::invalid_argument(
      "memory window requires raw measurement records, not whole-window edges");
}
} // namespace cudaq::qec

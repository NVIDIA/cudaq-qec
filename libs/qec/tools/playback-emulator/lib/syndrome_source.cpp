/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "syndrome_source.h"

#include "stim.h"

#include "cuda-qx/core/tensor.h"
#include "cudaq.h"
#include "device/memory_circuit.h"
#include "cudaq/qec/code.h"

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>

namespace cudaq::qec::playback {

// ─── static_source ──────────────────────────────────────────────────────────

static_source::static_source(std::vector<std::vector<std::uint8_t>> rounds)
    : rounds_(std::move(rounds)) {}

std::vector<std::uint8_t> static_source::next_round() {
  if (next_ >= rounds_.size())
    return {};
  return rounds_[next_++];
}

void static_source::reset() { next_ = 0; }

// ─── stim_memory_source ─────────────────────────────────────────────────────
//
// Drives a live stim::FrameSimulator, advanced by exactly one round per
// next_round() call, or by the circuit's terminal segment per read_data()
// call. Both happen synchronously on the calling thread.

namespace {

constexpr std::size_t kSimdWidth = stim::MAX_BITWORD_WIDTH;

// Builds one of Stim's built-in memory-circuit families from `params`. Keys:
// "code" ("surface_code"/"repetition_code"/"color_code"), "task", "distance"
// (all required) and the four stim::CircuitGenParameters noise probabilities
// (optional, default 0).
stim::Circuit generate_circuit(const cudaqx::heterogeneous_map &params) {
  auto code = params.get<std::string>("code");
  auto task = params.get<std::string>("task");
  auto distance = params.get<std::size_t>("distance");
  auto rounds = params.get<std::size_t>("rounds", std::size_t{3});

  stim::CircuitGenParameters gen_params(static_cast<std::uint64_t>(rounds),
                                        static_cast<std::uint32_t>(distance),
                                        task);
  gen_params.after_clifford_depolarization =
      params.get<double>("after_clifford_depolarization", 0.0);
  gen_params.before_round_data_depolarization =
      params.get<double>("before_round_data_depolarization", 0.0);
  gen_params.before_measure_flip_probability =
      params.get<double>("before_measure_flip_probability", 0.0);
  gen_params.after_reset_flip_probability =
      params.get<double>("after_reset_flip_probability", 0.0);

  if (code == "surface_code")
    return stim::generate_surface_code_circuit(gen_params).circuit;
  if (code == "repetition_code")
    return stim::generate_rep_code_circuit(gen_params).circuit;
  if (code == "color_code")
    return stim::generate_color_code_circuit(gen_params).circuit;
  throw std::invalid_argument("stim_memory_source: unknown code family '" +
                              code + "'");
}

} // namespace

struct stim_memory_source::impl {
  stim::Circuit prefix, round_body, terminal_body;
  std::uint32_t prefix_width, round_width, terminal_width, num_qubits;
  std::uint64_t base_seed;

  std::uint64_t generation = 0; // bumped on every shot boundary
  bool prefix_played = false;   // cleared by rebuild_sim(); see next_round()
  std::unique_ptr<stim::FrameSimulator<kSimdWidth>> sim;

  // Generates the circuit from `params`, then splits it at its REPEAT block
  // into a prefix, the repeating stabilizer-round body, and whatever
  // terminal (data-qubit readout) segment follows, if any.
  impl(const cudaqx::heterogeneous_map &params, std::uint64_t seed)
      : base_seed(seed) {
    const stim::Circuit full = generate_circuit(params);
    std::size_t i = 0;
    for (; i < full.operations.size(); ++i)
      if (full.operations[i].gate_type == stim::GateType::REPEAT)
        break;
    if (i == full.operations.size())
      throw std::runtime_error(
          "stim_memory_source: circuit has no REPEAT block; a "
          "syndrome-extraction round can only be derived from a repeating "
          "block.");
    // Only the first REPEAT can be the round. A later one is copied into
    // terminal_body by safe_append(), which does not carry the block body it
    // refers to, so its measurements would silently go missing.
    for (std::size_t j = i + 1; j < full.operations.size(); ++j)
      if (full.operations[j].gate_type == stim::GateType::REPEAT)
        throw std::runtime_error(
            "stim_memory_source: circuit has more than one REPEAT block; "
            "exactly one is required, and it is the syndrome-extraction "
            "round.");
    for (std::size_t j = 0; j < i; ++j)
      prefix.safe_append(full.operations[j]);
    round_body = full.operations[i].repeat_block_body(full);
    for (std::size_t j = i + 1; j < full.operations.size(); ++j)
      terminal_body.safe_append(full.operations[j]);

    prefix_width = prefix.count_measurements();
    round_width = round_body.count_measurements();
    terminal_width = terminal_body.count_measurements();
    num_qubits = full.count_qubits();
    rebuild_sim();
  }

  void rebuild_sim() {
    stim::CircuitStats stats;
    stats.num_qubits = num_qubits;
    stats.max_lookback = std::max({prefix_width, round_width, terminal_width});
    sim = std::make_unique<stim::FrameSimulator<kSimdWidth>>(
        stats, stim::FrameSimulatorMode::STREAM_MEASUREMENTS_TO_DISK,
        /*batch_size=*/1, std::mt19937_64(base_seed + generation));
    sim->reset_all();
    prefix_played = false;
  }

  std::vector<std::uint8_t> read_lookback(std::uint32_t width) {
    std::vector<std::uint8_t> bits(width);
    for (std::uint32_t i = 0; i < width; ++i)
      bits[i] = sim->m_record.lookback(width - i)[0] ? 1 : 0;
    sim->m_record.mark_all_as_written();
    return bits;
  }

  // The first call after a shot boundary also plays the prefix, folding it
  // into the round rather than exposing it as an extra-wide result: keeps
  // running round_body until a full round_width of fresh bits is available.
  std::vector<std::uint8_t> next_round() {
    std::vector<std::uint8_t> bits;
    if (!prefix_played) {
      prefix_played = true;
      sim->safe_do_circuit(prefix);
      bits = read_lookback(prefix_width);
    }
    while (bits.size() < round_width) {
      sim->safe_do_circuit(round_body);
      auto rest = read_lookback(round_width);
      bits.insert(bits.end(), rest.begin(), rest.end());
    }
    if (bits.size() != round_width)
      throw std::runtime_error(
          "stim_memory_source: prefix width is not a multiple of "
          "round_width; a uniform round_width() can't be formed.");
    return bits;
  }

  // Runs the terminal segment against the simulator's current state then starts
  // a fresh shot.
  std::vector<std::uint8_t> read_data() {
    sim->safe_do_circuit(terminal_body);
    auto bits = read_lookback(terminal_width);
    ++generation;
    rebuild_sim();
    return bits;
  }

  void reset() {
    ++generation;
    rebuild_sim();
  }
};

stim_memory_source::stim_memory_source(const cudaqx::heterogeneous_map &params,
                                       std::uint64_t seed)
    : impl_(std::make_unique<impl>(params, seed)) {}

stim_memory_source::~stim_memory_source() = default;

std::vector<std::uint8_t> stim_memory_source::next_round() {
  return impl_->next_round();
}

std::vector<std::uint8_t> stim_memory_source::read_data() {
  return impl_->read_data();
}

void stim_memory_source::reset() { impl_->reset(); }

std::uint32_t stim_memory_source::round_width() const {
  return impl_->round_width;
}

std::uint32_t stim_memory_source::data_width() const {
  return impl_->terminal_width;
}

// ─── cudaq_memory_source ────────────────────────────────────────────────────

struct cudaq_memory_source::impl {
  const cudaq::qec::code &qec_code;
  cudaq::qec::operation state_prep;
  std::size_t max_rounds;
  cudaq::noise_model noise;
  std::uint64_t base_seed;

  std::uint64_t generation = 0;  // bumped on every shot boundary
  std::size_t current_round = 0; // next_round() calls made so far this shot

  std::size_t num_cols = 0; // numAncx + numAncz: width of a raw round
  std::size_t num_data = 0; // width of the raw data-qubit readout

  // round_bits[r - 1]: raw ancilla measurements for round r.
  std::vector<std::vector<std::uint8_t>> round_bits;
  // data_bits[r - 1]: raw data-qubit measurements for a shot that reads out
  // right after round r.
  std::vector<std::vector<std::uint8_t>> data_bits;

  impl(const cudaq::qec::code &c, cudaq::qec::operation prep,
       std::size_t maxRounds, cudaq::noise_model n, std::uint64_t seed)
      : qec_code(c), state_prep(prep), max_rounds(maxRounds),
        noise(std::move(n)), base_seed(seed) {
    if (max_rounds == 0)
      throw std::runtime_error("cudaq_memory_source: max_rounds must be >= 1.");
    rebuild_cache();
  }

  // Reruns the memory_circuit kernel for every 1 <= r <= max_rounds under
  // the same seed.
  void rebuild_cache() {
    round_bits.assign(max_rounds, {});
    data_bits.assign(max_rounds, {});

    if (!qec_code.contains_operation(cudaq::qec::operation::stabilizer_round))
      throw std::runtime_error(
          "cudaq_memory_source: code has no stabilizer_round operation.");
    if (!qec_code.contains_operation(state_prep))
      throw std::runtime_error(
          "cudaq_memory_source: code does not support the requested state "
          "prep.");

    auto &prep = qec_code.get_operation<cudaq::qec::code::one_qubit_encoding>(
        state_prep);
    auto &stabRound =
        qec_code.get_operation<cudaq::qec::code::stabilizer_round>(
            cudaq::qec::operation::stabilizer_round);

    const bool is_z_prep = state_prep == cudaq::qec::operation::prep0 ||
                           state_prep == cudaq::qec::operation::prep1;

    auto sched_x = qec_code.get_stabilizer_schedule_x();
    auto sched_z = qec_code.get_stabilizer_schedule_z();
    std::vector<std::size_t> xVec(sched_x.data(),
                                  sched_x.data() + sched_x.size());
    std::vector<std::size_t> zVec(sched_z.data(),
                                  sched_z.data() + sched_z.size());
    auto logical_obs =
        is_z_prep ? qec_code.get_observables_z() : qec_code.get_observables_x();
    const std::size_t num_obs = logical_obs.shape()[0];
    std::vector<std::size_t> obs_flat(logical_obs.data(),
                                      logical_obs.data() + logical_obs.size());

    num_data = qec_code.get_num_data_qubits();
    const std::size_t numAncx = qec_code.get_num_ancilla_x_qubits();
    const std::size_t numAncz = qec_code.get_num_ancilla_z_qubits();
    num_cols = numAncx + numAncz;

    for (std::size_t r = 1; r <= max_rounds; ++r) {
      cudaq::set_random_seed(static_cast<std::size_t>(base_seed + generation));

      cudaq::sample_options opts{
          .shots = 1, .noise = noise, .explicit_measurements = true};
      auto result = cudaq::sample(opts, cudaq::qec::memory_circuit, stabRound,
                                  prep, num_data, numAncx, numAncz, r, xVec,
                                  zVec, obs_flat, num_obs, !is_z_prep);

      // mzTable[0, meas_idx]: raw measurement layout is r*num_cols ancilla
      // bits (num_cols per round), then num_data data-qubit bits.
      cudaqx::tensor<std::uint8_t> mzTable(result.sequential_data());

      auto &round = round_bits[r - 1];
      round.resize(num_cols);
      const std::size_t round_start = (r - 1) * num_cols;
      for (std::size_t i = 0; i < num_cols; ++i)
        round[i] = mzTable.at({0, round_start + i});

      auto &shot_data = data_bits[r - 1];
      shot_data.resize(num_data);
      const std::size_t data_start = num_cols * r;
      for (std::size_t i = 0; i < num_data; ++i)
        shot_data[i] = mzTable.at({0, data_start + i});
    }
  }

  std::vector<std::uint8_t> next_round() {
    if (current_round >= max_rounds)
      return {};
    return round_bits[current_round++];
  }

  std::vector<std::uint8_t> read_data() {
    if (current_round == 0)
      throw std::runtime_error(
          "cudaq_memory_source::read_data() called before any "
          "next_round(); memory_circuit always performs at least one "
          "stabilizer round before a data readout.");
    auto bits = data_bits[current_round - 1];
    reset();
    return bits;
  }

  void reset() {
    ++generation;
    current_round = 0;
    rebuild_cache();
  }
};

cudaq_memory_source::cudaq_memory_source(const code &code, operation statePrep,
                                         std::size_t max_rounds,
                                         cudaq::noise_model noise,
                                         std::uint64_t seed)
    : impl_(std::make_unique<impl>(code, statePrep, max_rounds,
                                   std::move(noise), seed)) {}

cudaq_memory_source::~cudaq_memory_source() = default;

std::vector<std::uint8_t> cudaq_memory_source::next_round() {
  return impl_->next_round();
}

std::vector<std::uint8_t> cudaq_memory_source::read_data() {
  return impl_->read_data();
}

void cudaq_memory_source::reset() { impl_->reset(); }

std::size_t cudaq_memory_source::max_rounds() const {
  return impl_->max_rounds;
}

std::uint32_t cudaq_memory_source::round_width() const {
  return static_cast<std::uint32_t>(impl_->num_cols);
}

std::uint32_t cudaq_memory_source::data_width() const {
  return static_cast<std::uint32_t>(impl_->num_data);
}

} // namespace cudaq::qec::playback

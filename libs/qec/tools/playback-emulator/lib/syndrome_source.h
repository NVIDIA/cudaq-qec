/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file syndrome_source.h
/// @brief `syndrome_source` and its implementations: `static_source` (replay
/// pre-supplied rounds), `stim_memory_source` (JIT rounds from a generated
/// Stim memory circuit), and `cudaq_memory_source` (streams `memory_circuit`).

#include "cuda-qx/core/heterogeneous_map.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudaq {
class noise_model;
}

namespace cudaq::qec {
class code;
enum class operation;
} // namespace cudaq::qec

namespace cudaq::qec::playback {

/// Yields one round of bits per `next_round()` call, and an empty round to
/// signal exhaustion.
class syndrome_source {
public:
  virtual ~syndrome_source() = default;

  /// Empty vector => exhausted.
  virtual std::vector<std::uint8_t> next_round() = 0;

  /// The terminal data-qubit readout that ends a shot (`enqueue_data`'s
  /// source call). Default just pulls the next produced item, same as
  /// `next_round()`. A two-phase source (e.g. `stim_memory_source`) overrides
  /// this to run its distinct terminal segment instead.
  virtual std::vector<std::uint8_t> read_data() { return next_round(); }

  /// Rewind for a new shot.
  virtual void reset() {}

  /// True for a source that generates data on demand rather than replaying
  /// pre-supplied rounds; required for a `stream ... until=NAME`. A
  /// non-streamed source runs dry mid-runaway and reports SOURCE_EXHAUSTED
  /// where the experiment needs continued growth.
  virtual bool is_streamed() const { return false; }
};

/// Replay pre-supplied rounds. The reference source: any test that needs an
/// exactly reproducible input, an oracle comparison, or a clean timing
/// measurement should use it. Round widths may differ between elements.
class static_source : public syndrome_source {
public:
  explicit static_source(std::vector<std::vector<std::uint8_t>> rounds);

  std::vector<std::uint8_t> next_round() override;
  void reset() override;

private:
  std::vector<std::vector<std::uint8_t>> rounds_;
  std::size_t next_ = 0;
};

/// Just-in-time round generation from one of Stim's built-in memory-circuit
/// families. Each `next_round()` advances a persistent Pauli-frame
/// simulator by one round; `read_data()` runs the terminal segment.
class stim_memory_source : public syndrome_source {
public:
  /// `params` selects and configures one of Stim's built-in generated
  /// memory-circuit families: required keys "code", "task", "distance",
  /// plus stim::CircuitGenParameters's four noise probabilities (optional,
  /// default 0).
  stim_memory_source(const cudaqx::heterogeneous_map &params,
                     std::uint64_t seed);
  ~stim_memory_source() override;

  stim_memory_source(const stim_memory_source &) = delete;
  stim_memory_source &operator=(const stim_memory_source &) = delete;

  /// Always round_width() bits, including the first call after construction
  /// or a shot boundary (see the class doc comment for how a non-trivial
  /// prefix is folded in).
  std::vector<std::uint8_t> next_round() override;
  std::vector<std::uint8_t> read_data() override;
  void reset() override;
  bool is_streamed() const override { return true; }

  /// Width, in bits, of one syndrome-extraction round (not the terminal
  /// data readout, which may differ).
  std::uint32_t round_width() const;

  /// Width, in bits, of the terminal data-qubit readout. Zero if the
  /// circuit has no segment after its `REPEAT` block.
  std::uint32_t data_width() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

/// Streams raw `memory_circuit` measurements (ancilla per round, data at
/// readout) by re-launching the kernel once per `1 <= r <= max_rounds` under
/// the same seed, caching each launch's output for round-by-round replay.
/// O(max_rounds^2) cost, since Stim has no mid-circuit state to resume from.
class cudaq_memory_source : public syndrome_source {
public:
  /// `code` must outlive this object; `statePrep` and `noise` are copied.
  /// Throws if `max_rounds == 0` or `code` doesn't support `statePrep`
  /// (matching `sample_memory_circuit`'s own validation).
  cudaq_memory_source(const code &code, operation statePrep,
                      std::size_t max_rounds, cudaq::noise_model noise,
                      std::uint64_t seed);
  ~cudaq_memory_source() override;

  cudaq_memory_source(const cudaq_memory_source &) = delete;
  cudaq_memory_source &operator=(const cudaq_memory_source &) = delete;

  /// Always round_width() raw ancilla measurement bits, one call per
  /// stabilizer-extraction round, up to `max_rounds` calls per shot; an
  /// empty vector once that's exhausted.
  std::vector<std::uint8_t> next_round() override;

  /// Raw data-qubit measurement bits (data_width() of them) for a shot
  /// ending after however many `next_round()` calls have actually been made,
  /// then starts a fresh shot. Throws if called before any `next_round()`
  /// (`memory_circuit` always performs at least one round before readout).
  std::vector<std::uint8_t> read_data() override;

  /// Starts a fresh shot (new seed generation), discarding any unconsumed
  /// rounds from the current one.
  void reset() override;

  /// False: data is pregenerated, and cannot be used in contexts where the
  /// number of rounds is not known ahead of time
  bool is_streamed() const override { return false; }

  /// The `max_rounds` given at construction: how many `next_round()` calls
  /// a shot can serve before exhausting.
  std::size_t max_rounds() const;

  /// Width, in bits, of one round's raw ancilla measurements
  /// (numAncx + numAncz for the underlying code).
  std::uint32_t round_width() const;

  /// Width, in bits, of the raw data-qubit readout (numData for the
  /// underlying code).
  std::uint32_t data_width() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

} // namespace cudaq::qec::playback

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Tests syndrome sources. static_source (exact replay) pins down replay,
/// exhaustion, and rewind. stim_memory_source is cross-checked against a
/// whole-circuit run of Stim's built-ins; cudaq_memory_source is validated
/// against direct runs of the `memory_circuit` kernel.

#include "syndrome_source.h"

#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq.h"
#include "device/memory_circuit.h"
#include "stim.h"
#include "stim/simulators/frame_simulator_util.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/noise_model.h"

#include <chrono>
#include <gtest/gtest.h>
#include <span>

using namespace cudaq::qec::playback;
using cudaqx::heterogeneous_map;

using rounds_t = std::vector<std::vector<uint8_t>>;

// ─── static_source ──────────────────────────────────────────────────────────

TEST(StaticSource, ReplaysEveryRoundInOrderThenStaysExhausted) {
  const std::vector<uint8_t> wide(1000, 1);
  const rounds_t cases[] = {
      {},                       // nothing to replay: exhausted at once
      {{1, 0, 1}},              // exactly one round
      {{0, 1}, {1, 1, 0}, {0}}, // several, of differing widths
      {{1}, wide, {}},          // one bit, then 1000, then none
  };
  for (const auto &rounds : cases) {
    SCOPED_TRACE("rounds=" + std::to_string(rounds.size()));
    static_source src(rounds);
    EXPECT_FALSE(src.is_streamed());
    for (std::size_t i = 0; i < rounds.size(); ++i)
      EXPECT_EQ(src.next_round(), rounds[i]) << "round " << i;
    // Past the end it stays empty forever: no wraparound, no throw, no
    // corruption however many times it is asked.
    for (int i = 0; i < 10; ++i)
      EXPECT_TRUE(src.next_round().empty()) << "call " << i << " past the end";
  }
}

TEST(StaticSource, AZeroWidthRoundIsNotMistakenForExhaustion) {
  // An empty return is ambiguous by content alone -- it is both "this round
  // has no bits" and "there are no more rounds". Only a later call can tell
  // them apart, so a zero-width round sandwiched between two real ones must
  // not stop the replay.
  static_source src({{1}, {}, {1}});
  EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{1}));
  EXPECT_TRUE(src.next_round().empty()); // the zero-width round
  EXPECT_EQ(src.next_round(),
            (std::vector<uint8_t>{1}));  // proves it kept going
  EXPECT_TRUE(src.next_round().empty()); // now genuinely done
}

TEST(StaticSource, ResetRewindsToRoundZeroFromAnywhereAndIsIdempotent) {
  static_source src({{1}, {2}, {3}, {4}});
  const auto expect_full_replay = [&] {
    for (uint8_t v : {1, 2, 3, 4})
      EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{v}));
  };

  src.reset(); // before anything was consumed
  src.reset(); // and again: repeated resets are idempotent
  expect_full_replay();

  src.next_round(); // partially consumed, nowhere near the end
  src.next_round();
  src.reset(); // must rewind all the way, not to "wherever we were"
  expect_full_replay();

  ASSERT_TRUE(src.next_round().empty()); // fully exhausted
  src.reset();
  expect_full_replay();
}

TEST(StaticSource, PassesThroughNonBinaryValuesUnchanged) {
  // static_source is a pure replay buffer: next_round() returns a copy of
  // whatever it was handed, with no check that values are really 0/1 --
  // consistent with what replay means, though a caller cannot lean on it to
  // catch a malformed fixture.
  static_source src({{0, 1, 2, 255}});
  EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{0, 1, 2, 255}));
}

// ─── stim_memory_source ─────────────────────────────────────────────────────
//
// stim_memory_source only accepts parameters for one of Stim's six built-in
// memory-circuit families, so every test below drives it that way and
// cross-checks against a circuit built directly through stim's own API.

namespace {

constexpr std::size_t kSimdWidth = stim::MAX_BITWORD_WIDTH;

/// Every one of Stim's built-in generated-circuit families.
constexpr std::pair<const char *, const char *> kGeneratedTasks[] = {
    {"repetition_code", "memory"},
    {"surface_code", "rotated_memory_x"},
    {"surface_code", "rotated_memory_z"},
    {"surface_code", "unrotated_memory_x"},
    {"surface_code", "unrotated_memory_z"},
    {"color_code", "memory_xyz"},
};

struct shape {
  std::uint32_t distance;
  std::uint32_t rounds;
};

/// Distance changes round width; rounds changes how many times the round
/// repeats. Rounds start at 3: stim inlines the round body instead of
/// emitting a REPEAT block for 1 or 2 rounds.
constexpr shape kSweep[] = {
    {2, 3}, {3, 3}, {3, 5}, {3, 12}, {5, 3}, {5, 8}, {7, 4},
};

/// color_code:memory_xyz takes odd distances only, and emits a *second*
/// REPEAT block from 4 rounds up (see MoreThanOneRepeatBlockIsRejected).
constexpr shape kColorSweep[] = {{3, 2}, {3, 3}, {5, 2}, {5, 3}, {7, 3}};

/// One of Stim's built-in generated circuits, via stim's own public API --
/// entirely independent of stim_memory_source's internals.
stim::Circuit generate(const std::string &code, const std::string &task,
                       shape s, double noise = 0.0) {
  stim::CircuitGenParameters params(s.rounds, s.distance, task);
  params.after_clifford_depolarization = noise;
  params.before_measure_flip_probability = noise;
  if (code == "surface_code")
    return stim::generate_surface_code_circuit(params).circuit;
  if (code == "repetition_code")
    return stim::generate_rep_code_circuit(params).circuit;
  return stim::generate_color_code_circuit(params).circuit;
}

/// The heterogeneous_map stim_memory_source's constructor expects, for the
/// same (code, task, shape) generate() builds a reference circuit from.
heterogeneous_map params_for(const std::string &code, const std::string &task,
                             shape s, double noise = 0.0) {
  return heterogeneous_map{{"code", code},
                           {"task", task},
                           {"distance", static_cast<std::size_t>(s.distance)},
                           {"rounds", static_cast<std::size_t>(s.rounds)},
                           {"after_clifford_depolarization", noise},
                           {"before_measure_flip_probability", noise}};
}

} // namespace

TEST(StimMemorySource,
     RoundWidthMatchesTheGeneratedCircuitsPerRoundMeasurementCount) {
  // One extra round adds exactly round_width() measurements to the whole
  // circuit's count, since only the REPEAT block's replay count changed.
  for (const auto &[code, task] : kGeneratedTasks) {
    SCOPED_TRACE(std::string(code) + ":" + task);
    const shape s =
        std::string(code) == "color_code" ? shape{5, 3} : shape{3, 5};
    const std::size_t width =
        generate(code, task, {s.distance, s.rounds + 1}).count_measurements() -
        generate(code, task, s).count_measurements();

    stim_memory_source src(params_for(code, task, s), /*seed=*/1);
    EXPECT_EQ(src.round_width(), width);
    // Drawn well past the generated circuit's own REPEAT count: every round
    // is exactly round_width() wide and holds only real bits.
    for (int i = 0; i < 300; ++i) {
      auto round = src.next_round();
      ASSERT_EQ(round.size(), width) << "round " << i;
      for (auto b : round)
        ASSERT_LE(b, 1u) << "round " << i;
    }
  }
}

TEST(StimMemorySource, RoundsDefaultsTo3AndWorksForEveryFamily) {
  // next_round() replays the REPEAT block's body forever regardless of the
  // round count it was generated with, so omitting "rounds" (default 3)
  // must still produce a working, unbounded source for every family.
  for (const auto &[code, task] : kGeneratedTasks) {
    SCOPED_TRACE(std::string(code) + ":" + task);
    heterogeneous_map params{{"code", std::string(code)},
                             {"task", std::string(task)},
                             {"distance", std::size_t{3}}};
    stim_memory_source src(params, /*seed=*/1);
    EXPECT_GT(src.round_width(), 0u);
    EXPECT_EQ(src.next_round().size(), src.round_width());
  }
}

TEST(StimMemorySource, IsUnboundedAndOneSeedAlwaysGivesOneStream) {
  // Needs noise > 0: a noiseless memory circuit's outcome is deterministic,
  // so two different seeds would spuriously agree forever.
  auto params = params_for("repetition_code", "memory", {3, 5}, /*noise=*/0.2);
  stim_memory_source a(params, /*seed=*/42);
  EXPECT_TRUE(a.is_streamed());

  stim_memory_source same_seed(params, /*seed=*/42);
  for (int i = 0; i < 100; ++i)
    ASSERT_EQ(a.next_round(), same_seed.next_round()) << "round " << i;

  stim_memory_source b(params, /*seed=*/1), c(params, /*seed=*/2);
  bool any_different = false;
  for (int i = 0; i < 200 && !any_different; ++i)
    any_different = b.next_round() != c.next_round();
  EXPECT_TRUE(any_different);
}

TEST(StimMemorySource, ResetReseedsWithoutWedgingOrStallingTheSource) {
  // Covers the shapes that could wedge reset(): before any draw, back to
  // back with nothing in between, and alternating tightly with draws.
  const auto start = std::chrono::steady_clock::now();
  auto params = params_for("surface_code", "rotated_memory_z", {3, 5});
  stim_memory_source src(params, /*seed=*/11);

  src.reset();
  ASSERT_EQ(src.next_round().size(), src.round_width());

  for (int i = 0; i < 100; ++i)
    src.reset();
  ASSERT_EQ(src.next_round().size(), src.round_width());

  for (int iter = 0; iter < 50; ++iter) {
    for (int i = 0; i < 3; ++i)
      ASSERT_EQ(src.next_round().size(), src.round_width())
          << "iter=" << iter << " i=" << i;
    src.reset();
  }
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}

TEST(StimMemorySource, DestructionAfterUseDoesNotHangOrCrash) {
  auto params = params_for("repetition_code", "memory", {3, 5});
  for (int i = 0; i < 20; ++i) {
    stim_memory_source src(params, /*seed=*/static_cast<uint64_t>(i));
    src.next_round();
    src.next_round();
  }
}

TEST(StimMemorySource, RejectsAnUnknownCodeFamily) {
  EXPECT_THROW(
      stim_memory_source(params_for("not_a_real_code_family", "memory", {3, 5}),
                         /*seed=*/1),
      std::invalid_argument);
}

TEST(StimMemorySource, RejectsGeneratorParamsWithTooFewRoundsForARepeatBlock) {
  // Stim inlines the round body instead of emitting a REPEAT block for 1 or
  // 2 rounds, so there is no round for the constructor to derive.
  for (std::uint32_t rounds : {1u, 2u}) {
    SCOPED_TRACE(rounds);
    EXPECT_THROW(
        stim_memory_source(params_for("repetition_code", "memory", {3, rounds}),
                           /*seed=*/1),
        std::runtime_error);
  }
}

TEST(StimMemorySource, MoreThanOneRepeatBlockIsRejected) {
  // Stim emits a second REPEAT block for color_code:memory_xyz from four
  // rounds up; only a single REPEAT block is a valid round to stream.
  EXPECT_THROW(
      stim_memory_source(params_for("color_code", "memory_xyz", {3, 8}),
                         /*seed=*/1),
      std::runtime_error);
}

// ─── stim_memory_source vs. a single whole-circuit run ─────────────────────
//
// Cross-checks stim_memory_source's round-by-round output against a single
// whole-circuit `do_circuit()` call, for the same number of rounds, across
// all six of Stim's built-in generated-circuit families.

namespace {

std::string to_bit_string(const stim::simd_bits<kSimdWidth> &bits,
                          std::size_t count) {
  std::string out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    out += char('0' + (bits[i] ? 1 : 0));
  return out;
}

// One shot through stim's own batch sampler -- the entry point Python's
// `compile_sampler().sample()` reaches, rather than a hand-built
// FrameSimulator that could drift. `reference_sample` picks what comes
// back: all-zero yields the raw noise frame, a real sample yields absolutes.
std::string
sample_whole_circuit_once(const stim::Circuit &circuit, std::uint64_t seed,
                          const stim::simd_bits<kSimdWidth> &reference_sample) {
  // stim_memory_source seeds its own simulator with mt19937_64(seed), so the
  // reference has to be seeded identically or the two consume different
  // random streams and nothing below means anything.
  std::mt19937_64 rng(seed);
  const std::size_t n = circuit.count_measurements();
  auto table = stim::sample_batch_measurements<kSimdWidth>(
      circuit, reference_sample, /*num_samples=*/1, rng, /*transposed=*/false);

  std::string bits;
  bits.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    bits += char('0' + (table[i][0] ? 1 : 0));
  return bits;
}

stim::simd_bits<kSimdWidth> zero_reference(const stim::Circuit &circuit) {
  return stim::simd_bits<kSimdWidth>(circuit.count_measurements());
}

// Drives stim_memory_source round-by-round until every stabilizer-round bit
// the reference circuit has is covered, then one read_data() call for the
// terminal segment. How many next_round() calls that takes varies by family
// (prefix folds differently), so this counts by total bits, not round count.
std::string sample_via_stim_memory_source(const heterogeneous_map &params,
                                          std::uint64_t seed,
                                          std::size_t total_measurements) {
  stim_memory_source source(params, seed);
  const std::size_t stabilizer_bits_needed =
      total_measurements - source.data_width();
  std::string bits;
  while (bits.size() < stabilizer_bits_needed)
    for (auto b : source.next_round())
      bits += char('0' + b);
  for (auto b : source.read_data())
    bits += char('0' + b);
  return bits;
}

void check_task(const std::string &code, const std::string &task, shape s) {
  constexpr std::uint64_t kSeed = 424242;
  SCOPED_TRACE("distance=" + std::to_string(s.distance) +
               " rounds=" + std::to_string(s.rounds));

  const stim::Circuit gen = generate(code, task, s, /*noise=*/0.001);
  const std::string reference =
      sample_whole_circuit_once(gen, kSeed, zero_reference(gen));
  const std::string via_source =
      sample_via_stim_memory_source(params_for(code, task, s, /*noise=*/0.001),
                                    kSeed, gen.count_measurements());

  EXPECT_EQ(via_source, reference) << "mismatch for " << code << ":" << task;
  EXPECT_EQ(via_source.size(), gen.count_measurements())
      << "round-by-round generation didn't cover every measurement for " << code
      << ":" << task;
}

} // namespace

TEST(StimMemorySourceVsFullCircuit,
     EveryGeneratedCodeFamilyMatchesOneWholeCircuitRun) {
  // All six of Stim's built-in generated-circuit families, across a range of
  // distances and round counts, at the same round count on both sides.
  int cases = 0;
  for (const auto &[code, task] : kGeneratedTasks) {
    SCOPED_TRACE(std::string(code) + ":" + task);
    const bool is_color = std::string(code) == "color_code";
    for (const auto &s : is_color ? std::span<const shape>(kColorSweep)
                                  : std::span<const shape>(kSweep)) {
      check_task(code, task, s);
      ++cases;
    }
  }
  // The two sweeps are deliberately different sizes; assert the total so a
  // future edit cannot quietly shrink the matrix to nothing.
  EXPECT_EQ(cases, 5 * std::size(kSweep) + std::size(kColorSweep));
}

TEST(StimMemorySourceVsFullCircuit,
     AGeneratedMemoryCircuitsReferenceSampleIsAllZero) {
  // Why the check above can compare noise frames as measurement outcomes: a
  // memory circuit's stabilizers have noiseless result 0, so its reference
  // sample is all zero and `frame == outcome` for every bit.
  constexpr shape s{3, 3};
  for (const auto &[code, task] : kGeneratedTasks) {
    SCOPED_TRACE(std::string(code) + ":" + task);
    const stim::Circuit gen = generate(code, task, s);
    const auto reference =
        stim::TableauSimulator<kSimdWidth>::reference_sample_circuit(gen);
    EXPECT_EQ(to_bit_string(reference, gen.count_measurements()).find('1'),
              std::string::npos);
  }
}

// ─── cudaq_memory_source ────────────────────────────────────────────────────
//
// Validated against direct runs of `memory_circuit` itself (raw ancilla/
// data-qubit bits, not sample_memory_circuit's XOR-combined detectors).

namespace {

cudaq::noise_model make_noise() {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.05),
                              /*num_controls=*/1);
  return noise;
}

// Runs memory_circuit directly (bypassing sample_memory_circuit's
// detector-XOR step) and returns the raw measurement row: numRounds chunks
// of (numAncx + numAncz) ancilla bits, then numData data-qubit bits -- the
// same layout `cudaq_memory_source` reads internally.
std::vector<std::uint8_t> raw_measurements(const cudaq::qec::code &code,
                                           cudaq::qec::operation prep_op,
                                           std::size_t numRounds,
                                           cudaq::noise_model &noise,
                                           std::uint64_t seed) {
  auto &prep =
      code.get_operation<cudaq::qec::code::one_qubit_encoding>(prep_op);
  auto &stabRound = code.get_operation<cudaq::qec::code::stabilizer_round>(
      cudaq::qec::operation::stabilizer_round);
  const bool is_z_prep = prep_op == cudaq::qec::operation::prep0 ||
                         prep_op == cudaq::qec::operation::prep1;

  auto sched_x = code.get_stabilizer_schedule_x();
  auto sched_z = code.get_stabilizer_schedule_z();
  std::vector<std::size_t> xVec(sched_x.data(),
                                sched_x.data() + sched_x.size());
  std::vector<std::size_t> zVec(sched_z.data(),
                                sched_z.data() + sched_z.size());
  auto logical_obs =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                    logical_obs.data() + logical_obs.size());

  const std::size_t numData = code.get_num_data_qubits();
  const std::size_t numAncx = code.get_num_ancilla_x_qubits();
  const std::size_t numAncz = code.get_num_ancilla_z_qubits();

  cudaq::set_random_seed(static_cast<std::size_t>(seed));
  cudaq::sample_options opts{
      .shots = 1, .noise = noise, .explicit_measurements = true};
  auto result = cudaq::sample(opts, cudaq::qec::memory_circuit, stabRound, prep,
                              numData, numAncx, numAncz, numRounds, xVec, zVec,
                              obs_flat, num_obs, !is_z_prep);
  cudaqx::tensor<std::uint8_t> mzTable(result.sequential_data());

  const std::size_t width = mzTable.shape()[1];
  std::vector<std::uint8_t> out(width);
  for (std::size_t i = 0; i < width; ++i)
    out[i] = mzTable.at({0, i});
  return out;
}

} // namespace

// For every 1 <= r <= max_rounds, the source's next_round() output for
// round r must equal round r's raw ancilla bits from an independent, direct
// kernel run with numRounds=r at the same seed, and its cached data bits
// for round r must equal that same run's raw data-qubit measurements.
TEST(CudaqMemorySource, MatchesDirectKernelRunPerRound) {
  constexpr std::uint64_t kSeed = 99;
  constexpr std::size_t kMaxRounds = 5;

  auto code = cudaq::qec::get_code("repetition",
                                   cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();

  for (auto prep_op :
       {cudaq::qec::operation::prep0, cudaq::qec::operation::prep1}) {
    SCOPED_TRACE(prep_op == cudaq::qec::operation::prep0 ? "prep0" : "prep1");
    for (std::size_t r = 1; r <= kMaxRounds; ++r) {
      auto reference = raw_measurements(*code, prep_op, r, noise, kSeed);

      cudaq_memory_source local(*code, prep_op, r, noise, kSeed);
      ASSERT_EQ(local.round_width() * r + local.data_width(), reference.size());

      std::vector<std::uint8_t> streamed;
      for (std::size_t k = 0; k < r; ++k) {
        auto bits = local.next_round();
        ASSERT_EQ(bits.size(), local.round_width())
            << "round " << k + 1 << " of " << r;
        streamed.insert(streamed.end(), bits.begin(), bits.end());
      }
      const std::size_t ancilla_width = r * local.round_width();
      for (std::size_t i = 0; i < ancilla_width; ++i)
        EXPECT_EQ(streamed[i], reference[i])
            << "ancilla bit mismatch at r=" << r << ", index " << i;

      auto data_bits = local.read_data();
      ASSERT_EQ(data_bits.size(), local.data_width());
      for (std::size_t i = 0; i < data_bits.size(); ++i)
        EXPECT_EQ(data_bits[i], reference[ancilla_width + i])
            << "data bit mismatch at r=" << r << ", index " << i;
    }
  }
}

// Exercises the streaming interface end-to-end: a variable number of
// next_round() calls then one read_data() call, matching stim_memory_source's
// shape. Confirms calling read_data() early (a shorter shot) reproduces the
// same-seed direct kernel run for that shorter round count.
TEST(CudaqMemorySource, ReadDataAfterVariableRoundCountMatchesDirectCall) {
  constexpr std::uint64_t kSeed = 7;
  constexpr std::size_t kMaxRounds = 6;

  auto code = cudaq::qec::get_code("repetition",
                                   cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();

  for (std::size_t k = 1; k <= kMaxRounds; ++k) {
    cudaq_memory_source source(*code, cudaq::qec::operation::prep0, kMaxRounds,
                               noise, kSeed);
    for (std::size_t i = 0; i < k; ++i)
      ASSERT_FALSE(source.next_round().empty());
    auto data = source.read_data();

    auto reference =
        raw_measurements(*code, cudaq::qec::operation::prep0, k, noise, kSeed);
    const std::size_t ancilla_width = k * source.round_width();
    ASSERT_EQ(data.size(), reference.size() - ancilla_width);
    for (std::size_t i = 0; i < data.size(); ++i)
      EXPECT_EQ(data[i], reference[ancilla_width + i]) << "k=" << k;
  }
}

TEST(CudaqMemorySource, ARoundBudgetIsEnforcedAndDataNeedsARoundFirst) {
  auto code = cudaq::qec::get_code("repetition",
                                   cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();
  cudaq_memory_source source(*code, cudaq::qec::operation::prep0, 3, noise, 1);

  // The terminal readout only means anything after at least one round.
  EXPECT_THROW(source.read_data(), std::runtime_error);

  // Bounded, unlike stim_memory_source: it cannot serve a `stream ...
  // until=` that outlasts max_rounds, so is_streamed() must say so.
  EXPECT_FALSE(source.is_streamed());

  EXPECT_EQ(source.max_rounds(), 3u);
  for (int i = 0; i < 3; ++i)
    EXPECT_FALSE(source.next_round().empty()) << "round " << i;
  // Bounded, unlike stim_memory_source: past the budget it stays empty.
  EXPECT_TRUE(source.next_round().empty());
  EXPECT_TRUE(source.next_round().empty());
}

// reset() starts a fresh shot (new seed generation): re-running the same
// number of rounds afterward should reproduce a direct kernel run at the
// *bumped* seed, not the original one.
TEST(CudaqMemorySource, ResetAdvancesToANewSeedGeneration) {
  constexpr std::uint64_t kSeed = 55;
  auto code = cudaq::qec::get_code("repetition",
                                   cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();
  cudaq_memory_source source(*code, cudaq::qec::operation::prep0, 3, noise,
                             kSeed);
  source.reset();

  std::vector<std::uint8_t> streamed;
  for (int i = 0; i < 3; ++i) {
    auto bits = source.next_round();
    streamed.insert(streamed.end(), bits.begin(), bits.end());
  }

  auto reference = raw_measurements(*code, cudaq::qec::operation::prep0, 3,
                                    noise, kSeed + 1);
  ASSERT_LE(streamed.size(), reference.size());
  for (std::size_t i = 0; i < streamed.size(); ++i)
    EXPECT_EQ(streamed[i], reference[i]);
}

/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "stim.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_init.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
// Private header from libs/qec/lib; see this target's include directories.
#include "decoder_stats.h"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <random>
#include <thread>

#include "decoders/sliding_window.h"

namespace {
class decoder_init_probe final : public cudaq::qec::decoder {
public:
  explicit decoder_init_probe(cudaq::qec::decoder_init inputs)
      : decoder(std::move(inputs), cudaq::qec::decode_result_type::errors) {}

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {true, std::vector<cudaq::qec::float_t>(block_size, 0.0)};
  }

  using cudaq::qec::decoder::get_num_msyn_per_decode;
  using cudaq::qec::decoder::get_num_observables;
};

class observable_output_probe final : public cudaq::qec::decoder {
public:
  observable_output_probe(cudaq::qec::decoder_init inputs,
                          cudaq::qec::decode_result_type requested_output,
                          const cudaqx::heterogeneous_map &)
      : decoder(std::move(inputs), requested_output) {}

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    return {true, syndrome, std::nullopt};
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      observable_output_probe,
      static std::unique_ptr<cudaq::qec::decoder> create(
          cudaq::qec::decoder_init inputs,
          std::optional<cudaq::qec::decode_result_type> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<observable_output_probe>(
            std::move(inputs),
            output.value_or(cudaq::qec::decode_result_type::observables),
            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(observable_output_probe)

class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name(name) {
    if (const char *old = std::getenv(name))
      oldValue = old;
    setenv(name, value, 1);
  }

  ~ScopedEnv() {
    if (oldValue.has_value())
      setenv(name.c_str(), oldValue->c_str(), 1);
    else
      unsetenv(name.c_str());
  }

private:
  std::string name;
  std::optional<std::string> oldValue;
};

// Restore the log level on scope exit so a failed expectation cannot leave the
// rest of the suite logging.
class ScopedLogLevel {
public:
  explicit ScopedLogLevel(cudaq::qec::detail::log_level level)
      : previous(cudaq::qec::detail::get_log_level()) {
    cudaq::qec::detail::set_log_level(level);
  }
  ~ScopedLogLevel() { cudaq::qec::detail::set_log_level(previous); }

private:
  cudaq::qec::detail::log_level previous;
};
} // namespace

TEST(DecoderInputs, PreservesMatrixShapesAndMeasurementMap) {
  using matrix = cudaq::qec::sparse_binary_matrix;
  auto H = matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});
  auto O = matrix::from_nested_csr(3, 3, {{0}, {}, {2}});
  auto D = matrix::from_nested_csr(2, 5, {{0, 1}, {2, 3}});

  cudaq::qec::decoder_init inputs(std::move(H), std::move(O), {0.1, 0.2, 0.3},
                                  std::move(D));

  EXPECT_EQ(inputs.source(), cudaq::qec::decoder_model_source::matrices);
  EXPECT_EQ(inputs.num_detectors(), 2);
  EXPECT_EQ(inputs.num_error_mechanisms(), 3);
  EXPECT_EQ(inputs.num_observables(), 3);
  EXPECT_EQ(inputs.observable_flips_matrix().to_nested_csr(),
            (std::vector<std::vector<std::uint32_t>>{{0}, {}, {2}}));
  ASSERT_NE(inputs.measurement_to_detectors(), nullptr);
  EXPECT_EQ(inputs.measurement_to_detectors()->num_rows(), 2);
  EXPECT_EQ(inputs.measurement_to_detectors()->num_cols(), 5);
  EXPECT_EQ(inputs.error_rates(), (std::vector<double>{0.1, 0.2, 0.3}));

  // A supplied O with an all-zero row is still an observable model: the row
  // count is retained rather than collapsed.
  EXPECT_EQ(inputs.observable_flips_matrix().num_rows(), 3);
  EXPECT_EQ(inputs.observable_flips_matrix().num_cols(), 3);
  EXPECT_TRUE(inputs.observable_flips_matrix().to_nested_csr()[1].empty());

  auto decoder = cudaq::qec::get_decoder("sample_decoder", inputs);
  // Both come from the model handed to the factory; there is no second way to
  // supply either, so they cannot disagree with it.
  EXPECT_EQ(decoder->get_num_observables(), 3);
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 5);
}

TEST(DecoderInputs, BaseConstructionSizesRealtimeStateFromTheModel) {
  using matrix = cudaq::qec::sparse_binary_matrix;
  auto H = matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});
  auto O = matrix::from_nested_csr(3, 3, {{0}, {}, {2}});
  auto D = matrix::from_nested_csr(2, 5, {{0, 1}, {2, 3}});

  decoder_init_probe decoder(
      cudaq::qec::decoder_init(std::move(H), std::move(O), {}, std::move(D)));

  // Construction derives every realtime size from the model. Nothing arrives
  // later, so a decoder is usable the moment it exists.
  EXPECT_EQ(decoder.get_num_observables(), 3);
  EXPECT_EQ(decoder.get_num_msyn_per_decode(), 5);
  const auto *corrections = decoder.get_obs_corrections();
  ASSERT_NE(corrections, nullptr);
  EXPECT_EQ(corrections[0], 0);
  EXPECT_EQ(corrections[1], 0);
  EXPECT_EQ(corrections[2], 0);

  // A full measurement volume decodes with no further setup: the model gave
  // the base everything the realtime path needs.
  EXPECT_TRUE(decoder.enqueue_syndrome(std::vector<std::uint8_t>(5, 0)));
}

TEST(DecoderInputs, RawStimRemainsAuthoritative) {
  const std::string dem_text = "error(0.1) D0 L0\n"
                               "error(0.2) D1\n";

  auto inputs = cudaq::qec::decoder_init::from_stim_dem(dem_text);

  EXPECT_EQ(inputs.source(), cudaq::qec::decoder_model_source::stim_dem);
  ASSERT_TRUE(inputs.has_stim_dem());
  EXPECT_EQ(inputs.stim_dem(), dem_text);
  EXPECT_EQ(inputs.num_detectors(), 2);
  EXPECT_EQ(inputs.num_error_mechanisms(), 2);
  EXPECT_EQ(inputs.num_observables(), 1);
  EXPECT_EQ(inputs.error_rates(), (std::vector<double>{0.1, 0.2}));
  EXPECT_EQ(inputs.observable_flips_matrix().to_nested_csr(),
            (std::vector<std::vector<std::uint32_t>>{{0}}));
}

TEST(DecoderInputs, RejectsInconsistentDimensions) {
  using matrix = cudaq::qec::sparse_binary_matrix;
  auto H = matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});

  EXPECT_THROW(cudaq::qec::decoder_init(H, matrix::from_nested_csr(1, 2, {{0}}),
                                        {0.1, 0.2, 0.3}),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::decoder_init(H, matrix::from_nested_csr(1, 3, {{0}}),
                                        {0.1, 0.2}),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::decoder_init(H, matrix::from_nested_csr(1, 3, {{0}}),
                                        {0.1, 0.2, 0.3},
                                        matrix::from_nested_csr(1, 4, {{0}})),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::decoder_init(H, matrix::from_nested_csr(1, 3, {{0}}),
                                        {0.1, 0.2, 0.3}, std::nullopt,
                                        std::vector<std::size_t>{0, 1}),
               std::invalid_argument);
}

TEST(DecoderInputs, DerivationsKeepRawSourceAndFreshMatricesDoNot) {
  auto inputs = cudaq::qec::decoder_init::from_stim_dem(
      "error(0.1) D0 L0\n",
      cudaq::qec::sparse_binary_matrix::from_nested_csr(1, 2, {{0, 1}}));

  auto without_d = inputs.decoder_init_without_d();
  EXPECT_TRUE(without_d.has_stim_dem());
  EXPECT_EQ(without_d.stim_dem(), inputs.stim_dem());
  EXPECT_EQ(without_d.measurement_to_detectors(), nullptr);

  // Canonicalization preserves column identity and ordering, so it keeps it.
  auto canonical = inputs.canonicalize_H();
  EXPECT_TRUE(canonical.has_stim_dem());
  EXPECT_EQ(canonical.stim_dem(), inputs.stim_dem());

  // A caller that re-indexes detectors or errors builds fresh inputs, which
  // carry no raw source: the DEM text names the original detectors.
  cudaq::qec::decoder_init reindexed(
      cudaq::qec::sparse_binary_matrix::from_nested_csc(1, 1, {{0}}),
      cudaq::qec::sparse_binary_matrix::from_nested_csr(0, 1, {}), {0.1});
  EXPECT_FALSE(reindexed.has_stim_dem());
  EXPECT_THROW((void)reindexed.stim_dem(), std::logic_error);
}

TEST(DecoderOutputContract, OutputFormIsImmutablePerInstance) {
  auto H = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      2, 2, std::vector<std::vector<std::uint32_t>>{{0}, {1}});
  auto O = cudaq::qec::sparse_binary_matrix::from_nested_csr(
      1, 2, std::vector<std::vector<std::uint32_t>>{{0}});
  auto decoder = cudaq::qec::get_decoder(
      "single_error_lut", cudaq::qec::decoder_init(std::move(H), std::move(O)),
      cudaq::qec::decode_result_type::observables);

  EXPECT_EQ(decoder->get_result_type(),
            cudaq::qec::decode_result_type::observables);

  const std::vector<cudaq::qec::float_t> syndrome{1.0, 0.0};
  auto observables = decoder->decode(syndrome);
  EXPECT_EQ(observables.result, std::vector<cudaq::qec::float_t>({1.0}));

  auto error_decoder = cudaq::qec::get_decoder(
      "single_error_lut",
      cudaq::qec::decoder_init(
          cudaq::qec::sparse_binary_matrix::from_nested_csc(
              2, 2, std::vector<std::vector<std::uint32_t>>{{0}, {1}}),
          cudaq::qec::sparse_binary_matrix::from_nested_csr(
              1, 2, std::vector<std::vector<std::uint32_t>>{{0}})),
      cudaq::qec::decode_result_type::errors);
  auto errors = error_decoder->decode(syndrome);
  EXPECT_EQ(errors.result, std::vector<cudaq::qec::float_t>({1.0, 0.0}));
}

TEST(DecoderOutputContract, ModelDataIsRejectedInCustomParameters) {
  auto H = cudaq::qec::sparse_binary_matrix::from_nested_csc(1, 1, {{0}});
  cudaqx::heterogeneous_map params;
  params.insert("O", cudaqx::tensor<std::uint8_t>({1, 1}));
  EXPECT_THROW(cudaq::qec::get_decoder("single_error_lut", H, params),
               std::runtime_error);
}

TEST(DecoderUtils, CovertHardToSoft) {
  std::vector<int> in = {1, 0, 1, 1};
  std::vector<float> out;
  std::vector<float> expected_out = {1.0, 0.0, 1.0, 1.0};

  cudaq::qec::convert_vec_hard_to_soft(in, out);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  expected_out = {0.9, 0.1, 0.9, 0.9};
  cudaq::qec::convert_vec_hard_to_soft(in, out, 0.9f, 0.1f);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  std::vector<std::vector<int>> in2 = {{1, 0}, {0, 1}};
  std::vector<std::vector<double>> out2;
  std::vector<std::vector<double>> expected_out2 = {{0.9, 0.1}, {0.1, 0.9}};
  cudaq::qec::convert_vec_hard_to_soft(in2, out2, 0.9, 0.1);
  for (int r = 0; r < out2.size(); r++) {
    ASSERT_EQ(out2.size(), expected_out2.size());
    for (int c = 0; c < out2.size(); c++)
      ASSERT_EQ(out2[r][c], expected_out2[r][c]);
  }
}

TEST(DecoderUtils, CovertSoftToHard) {
  std::vector<float> in = {0.6, 0.4, 0.7, 0.8};
  std::vector<bool> out;
  std::vector<bool> expected_out = {true, false, true, true};

  cudaq::qec::convert_vec_soft_to_hard(in, out);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  expected_out = {true, true, true, true};
  cudaq::qec::convert_vec_soft_to_hard(in, out, 0.4f);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  std::vector<float> boundary_in = {0.499f, 0.5f, 0.501f};
  std::vector<uint8_t> boundary_out;
  std::vector<uint8_t> expected_boundary_out = {0, 1, 1};
  cudaq::qec::convert_vec_soft_to_hard(boundary_in, boundary_out);
  ASSERT_EQ(boundary_out.size(), expected_boundary_out.size());
  for (int i = 0; i < boundary_out.size(); i++)
    ASSERT_EQ(boundary_out[i], expected_boundary_out[i]);

  std::vector<std::vector<double>> in2 = {{0.6, 0.4}, {0.7, 0.8}};
  std::vector<std::vector<int>> out2;
  std::vector<std::vector<int>> expected_out2 = {{1, 0}, {1, 1}};
  cudaq::qec::convert_vec_soft_to_hard(in2, out2);
  for (int r = 0; r < out2.size(); r++) {
    ASSERT_EQ(out2.size(), expected_out2.size());
    for (int c = 0; c < out2.size(); c++)
      ASSERT_EQ(out2[r][c], expected_out2[r][c]);
  }
}

TEST(DecoderUtils, ConvertSoftToHardScalar) {
  // Default threshold (0.5): the canonical >= contract.
  EXPECT_TRUE(cudaq::qec::convert_soft_to_hard(0.6f));
  EXPECT_FALSE(cudaq::qec::convert_soft_to_hard(0.4f));
  EXPECT_TRUE(cudaq::qec::convert_soft_to_hard(0.5f));
  EXPECT_FALSE(cudaq::qec::convert_soft_to_hard(0.499f));
  EXPECT_TRUE(cudaq::qec::convert_soft_to_hard(0.501f));

  // Double-precision input.
  EXPECT_TRUE(cudaq::qec::convert_soft_to_hard(0.5));
  EXPECT_FALSE(cudaq::qec::convert_soft_to_hard(0.499));

  // Custom threshold.
  EXPECT_TRUE(cudaq::qec::convert_soft_to_hard(0.4f, 0.4f));
  EXPECT_FALSE(cudaq::qec::convert_soft_to_hard(0.3f, 0.4f));

  // Usable in a constant-expression context.
  static_assert(cudaq::qec::convert_soft_to_hard(0.5f));
  static_assert(!cudaq::qec::convert_soft_to_hard(0.49f));
}

TEST(DecoderUtils, ConvertVecSoftToTensorHard) {
  // Generate a million random floats between 0 and 1 using mt19937
  std::mt19937_64 gen(13);
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::vector<double> in(1000000);
  for (int i = 0; i < in.size(); i++)
    in[i] = dis(gen);

  // Test the conversion to a tensor
  cudaqx::tensor<uint8_t> out_tensor;
  auto t0 = std::chrono::high_resolution_clock::now();
  cudaq::qec::convert_vec_soft_to_tensor_hard(in, out_tensor);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = t1 - t0;
  std::cout << "Time taken for cudaqx::tensor: " << diff.count() * 1000.0
            << "ms" << std::endl;

  // Use the conversion to a vector as a baseline
  std::vector<uint8_t> out_vec(in.size());
  t0 = std::chrono::high_resolution_clock::now();
  cudaq::qec::convert_vec_soft_to_hard(in, out_vec);
  t1 = std::chrono::high_resolution_clock::now();
  diff = t1 - t0;
  std::cout << "Time taken for std::vector: " << diff.count() * 1000.0 << "ms"
            << std::endl;

  // Check the results are the same
  for (std::size_t i = 0; i < in.size(); i++)
    ASSERT_EQ(out_tensor.at({i}), out_vec[i]);
}

TEST(SampleDecoder, checkAPI) {
  using cudaq::qec::float_t;

  std::size_t block_size = 10;
  std::size_t syndrome_size = 4;
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});
  auto H_sparse = cudaq::qec::sparse_binary_matrix(H);
  auto d = cudaq::qec::decoder::get("sample_decoder", H_sparse);
  std::vector<float_t> syndromes(syndrome_size);
  auto dec_result = d->decode(syndromes);
  ASSERT_EQ(dec_result.result.size(), block_size);
  for (auto x : dec_result.result)
    ASSERT_EQ(x, 0.0f);

  // Async test
  dec_result = d->decode_async(syndromes).get();
  ASSERT_EQ(dec_result.result.size(), block_size);
  for (auto x : dec_result.result)
    ASSERT_EQ(x, 0.0f);

  // Test the move constructor and move assignment operator

  // Multi test
  auto dec_results = d->decode_batch({syndromes, syndromes});
  ASSERT_EQ(dec_results.size(), 2);
  for (auto &m : dec_results)
    for (auto x : m.result)
      ASSERT_EQ(x, 0.0f);
}

// A decoder that overrides only the single-argument decode_batch (i.e. every
// decoder written before the overload existed) must still work through the
// two-argument one, and must report no batch-level results.
TEST(SampleDecoder, BatchOptResultsDefaultsToEmpty) {
  constexpr std::size_t block_size = 10;
  constexpr std::size_t syndrome_size = 4;
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("sample_decoder", H);
  ASSERT_NE(d, nullptr);

  std::vector<cudaq::qec::float_t> syndromes(syndrome_size, 0.0);
  std::optional<cudaqx::heterogeneous_map> batch_opt_results;
  auto dec_results = d->decode_batch({syndromes, syndromes}, batch_opt_results);

  EXPECT_EQ(dec_results.size(), 2);
  EXPECT_FALSE(batch_opt_results.has_value());
  for (auto &m : dec_results)
    EXPECT_EQ(m.result.size(), block_size);

  // A stale value in the out-param must be cleared, not left for the caller to
  // mistake for decoder output.
  batch_opt_results = cudaqx::heterogeneous_map();
  batch_opt_results->insert("stale", 1);
  d->decode_batch({syndromes}, batch_opt_results);
  EXPECT_FALSE(batch_opt_results.has_value());
}

TEST(SampleDecoder, RealtimeApiAndDefaultGraphHooks) {
  // CUDAQ_QEC_DEBUG_DECODER enables the base decoder's printf logging paths.
  ScopedEnv debugEnv("CUDAQ_QEC_DEBUG_DECODER", "1");

  constexpr std::size_t block_size = 4;
  constexpr std::size_t syndrome_size = 2;
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});
  // The whole model up front: three observables, and a D mapping three
  // measurement bits onto the two detectors.
  auto decoder = cudaq::qec::decoder::get(
      "sample_decoder", cudaq::qec::decoder_init(
                            cudaq::qec::sparse_binary_matrix(H),
                            cudaq::qec::sparse_binary_matrix::from_nested_csr(
                                3, block_size, {{0}, {}, {2}}),
                            /*error_rates=*/{},
                            cudaq::qec::sparse_binary_matrix::from_nested_csr(
                                syndrome_size, 3, {{0, 1}, {2}})));
  ASSERT_NE(decoder, nullptr);

  // Plain decoders do not support graph dispatch, and their default graph
  // methods should be harmless no-ops.
  EXPECT_FALSE(decoder->supports_graph_dispatch());
  EXPECT_EQ(decoder->capture_decode_graph(), nullptr);
  decoder->release_decode_graph(nullptr);

  decoder->set_decoder_id(7);
  EXPECT_EQ(decoder->get_decoder_id(), 7u);

  // Both sizes come from the construction model; there is no reapplication
  // path, so they cannot be changed under a live decoder.
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 3u);
  EXPECT_EQ(decoder->get_num_observables(), 3u);

  // Three measurement bits fill the D buffer and trigger a decode.
  std::vector<uint8_t> msyn = {1, 0, 1};
  EXPECT_TRUE(decoder->enqueue_syndrome(msyn));
  auto *corrections = decoder->get_obs_corrections();
  ASSERT_NE(corrections, nullptr);
  EXPECT_EQ(corrections[0], 0u);

  decoder->clear_corrections();
  EXPECT_EQ(decoder->get_obs_corrections()[0], 0u);
  decoder->reset_decoder();

  // Longer input than the configured measurement buffer is rejected.
  EXPECT_FALSE(decoder->enqueue_syndrome(msyn.data(), msyn.size() + 1));
}

// An explicit summary is a phase boundary: later summaries must not repeat the
// warmup samples, and destruction must not emit the last phase a second time.
TEST(DecoderStats, ExplicitSummaryStartsFreshAggregate) {
  ScopedLogLevel scoped(cudaq::qec::detail::log_level::info);
  testing::internal::CaptureStdout();
  {
    cudaq::qec::decoder_stats stats;
    for (int phase = 0; phase < 2; ++phase) {
      stats.note_submit(/*first_of_shot=*/true);
      stats.note_decode_complete();
      stats.emit_summary();
    }
  }
  const auto log = testing::internal::GetCapturedStdout();

  const std::string expected = "stats_summary Decodes:1 ";
  std::size_t summaries = 0;
  for (std::size_t at = 0; (at = log.find(expected, at)) != std::string::npos;
       at += expected.size())
    ++summaries;
  EXPECT_EQ(summaries, 2u);
  EXPECT_EQ(log.find("stats_summary Decodes:2 "), std::string::npos);
}

// The summary's percentiles come from a fixed-size log-spaced histogram, so
// they are estimates while the count, mean, min and max stay exact. Feed a
// known ramp so the estimates can be held to a tolerance.
TEST(DecoderStats, LatencySeriesPercentilesTrackKnownDistribution) {
  cudaq::qec::latency_series series;
  EXPECT_EQ(series.count, 0u);
  // An empty series has nothing to report rather than a stale or garbage value.
  EXPECT_EQ(series.percentile_us(0.5), 0.0);

  // A single sample is reported exactly: the bucket it lands in is wider than
  // the value's own precision, so the estimate is clamped to the range seen.
  series.add(7.25);
  EXPECT_DOUBLE_EQ(series.percentile_us(0.5), 7.25);
  EXPECT_DOUBLE_EQ(series.percentile_us(0.99), 7.25);

  cudaq::qec::latency_series ramp;
  for (int us = 1; us <= 1000; us++)
    ramp.add(static_cast<double>(us));

  EXPECT_EQ(ramp.count, 1000u);
  EXPECT_DOUBLE_EQ(ramp.min_us, 1.0);
  EXPECT_DOUBLE_EQ(ramp.max_us, 1000.0);
  EXPECT_NEAR(ramp.average_us(), 500.5, 1e-9);

  // Bucket quantization is proportional to the value, so hold each estimate to
  // 5% of the exact percentile of the ramp.
  EXPECT_NEAR(ramp.percentile_us(0.50), 500.0, 25.0);
  EXPECT_NEAR(ramp.percentile_us(0.90), 900.0, 45.0);
  EXPECT_NEAR(ramp.percentile_us(0.99), 990.0, 49.5);

  // Percentiles are monotonic in the requested fraction and bounded by the
  // exact extremes, whatever the distribution.
  EXPECT_LE(ramp.min_us, ramp.percentile_us(0.50));
  EXPECT_LE(ramp.percentile_us(0.50), ramp.percentile_us(0.90));
  EXPECT_LE(ramp.percentile_us(0.90), ramp.percentile_us(0.99));
  EXPECT_LE(ramp.percentile_us(0.99), ramp.max_us);
}

// A heavy tail is what the percentiles exist to expose: a handful of slow
// samples must move p99 without dragging p50 along with them.
TEST(DecoderStats, LatencySeriesPercentilesSeparateTailFromBody) {
  // 2% slow, so the slow samples start below the 99th percentile rather than
  // straddling it.
  cudaq::qec::latency_series series;
  for (int i = 0; i < 980; i++)
    series.add(2.0);
  for (int i = 0; i < 20; i++)
    series.add(500.0);

  EXPECT_NEAR(series.percentile_us(0.50), 2.0, 0.1);
  EXPECT_NEAR(series.percentile_us(0.90), 2.0, 0.1);
  EXPECT_NEAR(series.percentile_us(0.99), 500.0, 25.0);
  EXPECT_DOUBLE_EQ(series.max_us, 500.0);
}

TEST(DecoderPlugins, SingleErrorLutExample_DecodesSingletonColumnSyndromes) {
  using cudaq::qec::float_t;

  constexpr std::size_t block_size = 3;
  constexpr std::size_t syndrome_size = 2;
  // | 1 1 0 |
  // | 0 1 1 | - single-bit columns are weight-1 syndrome patterns.
  std::vector<uint8_t> H_vec = {1, 1, 0, // row 0
                                0, 1, 1};
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  cudaqx::heterogeneous_map params;
  auto d = cudaq::qec::decoder::get("single_error_lut_example", H, params);

  std::vector<float_t> syndrome0 = {1.0f, 0.0f}; // column 0
  auto r0 = d->decode(syndrome0);
  ASSERT_TRUE(r0.converged);
  EXPECT_FLOAT_EQ(r0.result[0], 1.0f);
  EXPECT_FLOAT_EQ(r0.result[1], 0.0f);
  EXPECT_FLOAT_EQ(r0.result[2], 0.0f);

  std::vector<float_t> syndrome2 = {0.0f, 1.0f}; // column 2
  auto r2 = d->decode(syndrome2);
  ASSERT_TRUE(r2.converged);
  EXPECT_FLOAT_EQ(r2.result[0], 0.0f);
  EXPECT_FLOAT_EQ(r2.result[1], 0.0f);
  EXPECT_FLOAT_EQ(r2.result[2], 1.0f);

  std::vector<float_t> zero(syndrome_size, 0.0f);
  auto rz = d->decode(zero);
  ASSERT_TRUE(rz.converged);
}

TEST(SteaneLutDecoder, checkAPI) {
  using cudaq::qec::float_t;

  // Use Hx from the [7,1,3] Steane code from
  // https://en.wikipedia.org/wiki/Steane_code.
  std::size_t block_size = 7;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  std::vector<uint8_t> H_vec = {0, 0, 0, 1, 1, 1, 1,  // IIIXXXX
                                0, 1, 1, 0, 0, 1, 1,  // IXXIIXX
                                1, 0, 1, 0, 1, 0, 1}; // XIXIXIX
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("single_error_lut", H, custom_args);

  // Run decoding on all possible syndromes.
  const std::size_t num_syndromes_to_check = 1 << syndrome_size;
  bool convergeTrueFound = false;
  bool convergeFalseFound = false;
  assert(syndrome_size <= 64); // Assert due to "1 << bit" below.
  for (std::size_t syn_idx = 0; syn_idx < num_syndromes_to_check; syn_idx++) {
    // Construct a syndrome.
    std::vector<float_t> syndrome(syndrome_size, 0.0);
    for (int bit = 0; bit < syndrome_size; bit++)
      if (syn_idx & (1 << bit))
        syndrome[bit] = 1.0;

    // Perform decoding.
    auto dec_result = d->decode(syndrome);

    // Check results.
    ASSERT_EQ(dec_result.result.size(), block_size);
    const auto printResults = true;
    if (printResults) {
      std::string syndrome_str(syndrome_size, '0');
      for (std::size_t j = 0; j < syndrome_size; j++)
        if (syndrome[j] >= 0.5)
          syndrome_str[j] = '1';
      std::cout << "Syndrome " << syndrome_str
                << " returned: {converged: " << dec_result.converged
                << ", result: {";
      for (std::size_t j = 0; j < block_size; j++) {
        std::cout << dec_result.result[j];
        if (j < block_size - 1)
          std::cout << ",";
        else
          std::cout << "}}\n";
      }
    }
    convergeTrueFound |= dec_result.converged;
    convergeFalseFound |= !dec_result.converged;
  }
  ASSERT_TRUE(convergeTrueFound);
  ASSERT_FALSE(convergeFalseFound);

  std::vector<float_t> boundary_syndrome = {0.5, 0.0, 0.5};
  auto boundary_result = d->decode(boundary_syndrome);
  ASSERT_TRUE(boundary_result.converged);
  ASSERT_EQ(boundary_result.result.size(), block_size);
  EXPECT_EQ(boundary_result.result[4], 1.0);

  // Test opt_results functionality
  // Test case 1: Invalid result type
  cudaqx::heterogeneous_map invalid_args;
  cudaqx::heterogeneous_map invalid_opt_results;
  invalid_opt_results.insert("invalid_type", true);
  invalid_args.insert("opt_results", invalid_opt_results);

  EXPECT_THROW(
      {
        auto d2 = cudaq::qec::decoder::get("single_error_lut", H, invalid_args);
        std::vector<float_t> syndrome(syndrome_size, 0.0);
        d2->decode(syndrome);
      },
      std::runtime_error);

  // Test case 2: Valid result types
  cudaqx::heterogeneous_map valid_args;
  cudaqx::heterogeneous_map valid_opt_results;
  valid_opt_results.insert("error_probability", true);
  valid_opt_results.insert("syndrome_weight", true);
  valid_opt_results.insert("decoding_time", false);
  valid_args.insert("opt_results", valid_opt_results);
  valid_args.insert("lut_error_depth", 2);

  auto d3 = cudaq::qec::decoder::get("multi_error_lut", H, valid_args);
  std::vector<float_t> syndrome(syndrome_size, 0.0);
  // Set syndrome to 101
  syndrome[0] = 1.0;
  syndrome[2] = 1.0;
  auto result = d3->decode(syndrome);

  // Verify opt_results
  ASSERT_TRUE(result.opt_results.has_value());
  ASSERT_TRUE(result.opt_results->contains("error_probability"));
  ASSERT_TRUE(result.opt_results->contains("syndrome_weight"));
  ASSERT_FALSE(
      result.opt_results->contains("decoding_time")); // Was set to false

  // Test case 3: Multiple invalid result types
  cudaqx::heterogeneous_map multi_invalid_args;
  cudaqx::heterogeneous_map multi_invalid_opt_results;

  // Add multiple invalid types to trigger the comma separation logic
  multi_invalid_opt_results.insert("invalid_type1", true);
  multi_invalid_opt_results.insert("invalid_type2", false);
  multi_invalid_opt_results.insert("invalid_type3", 10);
  multi_invalid_args.insert("opt_results", multi_invalid_opt_results);

  // The error message should contain all three invalid types separated by
  // commas
  std::string expected_error =
      "Requested result types not available in LUT decoder: ";
  // Note: The exact order may vary depending on map iteration, but should
  // contain all types

  try {
    auto d4 =
        cudaq::qec::decoder::get("single_error_lut", H, multi_invalid_args);
    FAIL() << "Expected std::runtime_error to be thrown";
  } catch (const std::runtime_error &e) {
    std::string error_msg = e.what();

    // Verify the error message contains the expected prefix
    EXPECT_TRUE(error_msg.find(expected_error) != std::string::npos)
        << "Error message should contain expected prefix. Got: " << error_msg;

    // Verify all three invalid types are mentioned in the error
    EXPECT_TRUE(error_msg.find("invalid_type1") != std::string::npos)
        << "Error message should contain 'invalid_type1'. Got: " << error_msg;
    EXPECT_TRUE(error_msg.find("invalid_type2") != std::string::npos)
        << "Error message should contain 'invalid_type2'. Got: " << error_msg;
    EXPECT_TRUE(error_msg.find("invalid_type3") != std::string::npos)
        << "Error message should contain 'invalid_type3'. Got: " << error_msg;

    // Verify that commas are used to separate the types (testing lines 57-59)
    // Count comma occurrences - should be 2 for 3 items
    std::size_t comma_count = 0;
    std::size_t pos = 0;
    while ((pos = error_msg.find(", ", pos)) != std::string::npos) {
      comma_count++;
      pos += 2;
    }
    EXPECT_EQ(comma_count, 2)
        << "Expected 2 commas for 3 invalid types. Got: " << comma_count
        << " in message: " << error_msg;
  }

  // Test case 4: Test decoding_time=true in lut.cpp
  cudaqx::heterogeneous_map decoding_time_args;
  cudaqx::heterogeneous_map decoding_time_opt_results;
  decoding_time_opt_results.insert("decoding_time", true);
  decoding_time_args.insert("opt_results", decoding_time_opt_results);

  auto d4 = cudaq::qec::decoder::get("single_error_lut", H, decoding_time_args);
  std::vector<float_t> syndrome_dt(syndrome_size, 0.0);
  // Set syndrome to 101
  syndrome_dt[0] = 1.0;
  syndrome_dt[2] = 1.0;
  auto result_dt = d4->decode(syndrome_dt);

  // Verify opt_results contains decoding_time
  ASSERT_TRUE(result_dt.opt_results.has_value());
  ASSERT_TRUE(result_dt.opt_results->contains("decoding_time"));
  // Verify the decoding_time value is the expected 0.0
  ASSERT_GT(result_dt.opt_results->get<double>("decoding_time"), 0.0);
}

void check_pcm_equality(const cudaqx::tensor<uint8_t> &a,
                        const cudaqx::tensor<uint8_t> &b,
                        bool use_assert = true) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::runtime_error("PCM must be a 2D tensor");
  }
  ASSERT_EQ(a.shape(), b.shape());
  auto num_rows = a.shape()[0];
  auto num_cols = a.shape()[1];
  for (std::size_t r = 0; r < num_rows; ++r) {
    for (std::size_t c = 0; c < num_cols; ++c) {
      if (a.at({r, c}) != b.at({r, c})) {
        if (use_assert)
          ASSERT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
        else
          EXPECT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
      }
    }
  }
}

/// This is a parameterized helper function that tests the sliding window
/// decoder by comparing the results of the global decoder and the windowed
/// decoder. The global decoder uses a single decoder for the entire block,
/// while the windowed decoder uses a sliding window of decoders.
/// @param run_batched Whether to run the decoder in batched mode.
/// @param n_rounds The number of rounds in the block.
/// @param n_errs_per_round The number of error mechanisms per round.
/// @param n_syndromes_per_round The number of syndromes per round.
/// @param window_size The size of the sliding window (in rounds).
/// @param step_size The step size for the sliding window (in rounds).
void SlidingWindowDecoderTest(bool run_batched, std::size_t n_rounds,
                              std::size_t n_errs_per_round,
                              std::size_t n_syndromes_per_round,
                              std::size_t window_size, std::size_t step_size) {
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;

  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));
  ASSERT_EQ(pcm.shape()[0], n_rows);
  ASSERT_EQ(pcm.shape()[1], n_cols);
  std::vector<double> weights(n_cols, 0.01);
  auto [simplified_pcm, simplified_weights] =
      cudaq::qec::simplify_pcm(pcm, weights, n_syndromes_per_round);
  ASSERT_TRUE(cudaq::qec::pcm_is_sorted(simplified_pcm, n_syndromes_per_round));

  const std::size_t commit_size = window_size - step_size;
  const std::size_t n_windows = (n_rounds - window_size) / step_size + 1;
  const std::size_t num_syndromes_per_window =
      window_size * n_syndromes_per_round;

  const std::string inner_decoder_name = "single_error_lut";
  cudaqx::heterogeneous_map sliding_window_params;
  sliding_window_params.insert("window_size", window_size);
  sliding_window_params.insert("step_size", step_size);
  sliding_window_params.insert("num_syndromes_per_round",
                               n_syndromes_per_round);
  sliding_window_params.insert("inner_decoder_name", inner_decoder_name);

  cudaqx::heterogeneous_map inner_decoder_params;
  sliding_window_params.insert("inner_decoder_params", inner_decoder_params);

  auto sliding_H = cudaq::qec::sparse_binary_matrix(simplified_pcm);
  auto sliding_O = cudaq::qec::sparse_binary_matrix::from_csr(
      0, sliding_H.num_cols(), {0}, {});
  auto sliding_window_decoder = cudaq::qec::decoder::get(
      "sliding_window",
      cudaq::qec::decoder_init(std::move(sliding_H), std::move(sliding_O),
                               simplified_weights),
      sliding_window_params);

  // Create some random syndromes.
  const int num_syndromes = 1000;
  std::vector<std::vector<cudaq::qec::float_t>> syndromes(num_syndromes);

  // Set a fixed number of error mechanisms to be non-zero. Since we are using
  // "single_error_lut", let's only set 1 error mechanism for now.
  const int num_error_mechanisms_to_set = 1;
  std::uniform_int_distribution<uint32_t> dist(0, n_cols - 1);
  std::mt19937_64 rng(13);
  for (std::size_t i = 0; i < num_syndromes; ++i) {
    syndromes[i] = std::vector<cudaq::qec::float_t>(n_rows, 0.0);
    for (int e = 0; e < num_error_mechanisms_to_set; ++e) {
      auto col = dist(rng);
      // printf("For syndrome %zu, setting error mechanism %d at column %u\n",
      // i, e, col);
      for (std::size_t r = 0; r < n_rows; ++r)
        syndromes[i][r] = pcm.at({r, col});
      // syndromes[i].dump_bits();
    }
  }

  // First decode the syndromes using a global decoder.
  std::vector<std::vector<uint8_t>> global_decoded_results(num_syndromes);
  auto t0 = std::chrono::high_resolution_clock::now();
  {
    printf("Generating global_decoder with PCM dims %zu x %zu\n",
           pcm.shape()[0], pcm.shape()[1]);
    auto global_decoder = cudaq::qec::decoder::get(
        inner_decoder_name, simplified_pcm, inner_decoder_params);
    printf("Done\n");
    if (run_batched) {
      auto dec_results = global_decoder->decode_batch(syndromes);
      for (std::size_t i = 0; i < num_syndromes; ++i) {
        ASSERT_TRUE(dec_results[i].converged);
        cudaq::qec::convert_vec_soft_to_hard(dec_results[i].result,
                                             global_decoded_results[i]);
      }
    } else {
      for (std::size_t i = 0; i < num_syndromes; ++i) {
        // printf("Decoding syndrome %zu\n", i);
        // syndromes[i].dump_bits();
        auto d = global_decoder->decode(syndromes[i]);
        ASSERT_TRUE(d.converged);
        ASSERT_GT(d.result.size(), 0);
        cudaq::qec::convert_vec_soft_to_hard(d.result,
                                             global_decoded_results[i]);
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_global = t1 - t0;
  printf("Global decoder time: %.3f ms, or %.3f us per syndrome\n",
         duration_global.count() * 1000,
         duration_global.count() * 1000 / num_syndromes);

  // Now decode each syndrome using a windowed approach.
  std::vector<std::vector<uint8_t>> windowed_decoded_results(num_syndromes);
  auto t2 = std::chrono::high_resolution_clock::now();
  if (run_batched) {
    printf("Running batched decoding\n");
    auto dec_results = sliding_window_decoder->decode_batch(syndromes);
    ASSERT_EQ(dec_results.size(), num_syndromes);
    for (std::size_t i = 0; i < num_syndromes; ++i) {
      ASSERT_GT(dec_results[i].result.size(), 0);
      ASSERT_TRUE(dec_results[i].converged);
      cudaq::qec::convert_vec_soft_to_hard(dec_results[i].result,
                                           windowed_decoded_results[i]);
    }
  } else {
    for (std::size_t i = 0; i < num_syndromes; ++i) {
      // printf(" ------ Decoding syndrome %zu ------ \n", i);
      auto decoded_result = sliding_window_decoder->decode(syndromes[i]);
      ASSERT_GT(decoded_result.result.size(), 0);
      ASSERT_TRUE(decoded_result.converged);
      cudaq::qec::convert_vec_soft_to_hard(decoded_result.result,
                                           windowed_decoded_results[i]);
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_windowed = t3 - t2;
  printf("Windowed decoder time: %.3f ms, or %.3f us per syndrome\n",
         duration_windowed.count() * 1000,
         duration_windowed.count() * 1000 / num_syndromes);

  // Check that the global and windowed decoders agree.
  auto print_as_bits = [](const std::vector<uint8_t> &v) {
    std::string s;
    s.reserve(v.size());
    for (auto r : v)
      s += (r == 0) ? '.' : '1';
    return s;
  };
  for (std::size_t i = 0; i < num_syndromes; ++i) {
    bool decoder_agreement =
        global_decoded_results[i] == windowed_decoded_results[i];
    EXPECT_EQ(decoder_agreement, true)
        << "Comparison failed for syndrome " << i;
    if (!decoder_agreement) {
      printf("Global   decoder result: %s\n",
             print_as_bits(global_decoded_results[i]).c_str());
      printf("Windowed decoder result: %s\n",
             print_as_bits(windowed_decoded_results[i]).c_str());
    }
  }
}

TEST(SlidingWindowDecoder, SlidingWindowDecoderTestNonBatchedStepSize1) {
  SlidingWindowDecoderTest(false, /*n_rounds=*/8, /*n_errs_per_round=*/30,
                           /*n_syndromes_per_round=*/10, /*window_size=*/3,
                           /*step_size=*/1);
}

TEST(SlidingWindowDecoder, SlidingWindowDecoderTestBatchedStepSize1) {
  SlidingWindowDecoderTest(true, /*n_rounds=*/8, /*n_errs_per_round=*/30,
                           /*n_syndromes_per_round=*/10, /*window_size=*/3,
                           /*step_size=*/1);
}

TEST(SlidingWindowDecoder, SlidingWindowDecoderTestNonBatchedStepSize2) {
  SlidingWindowDecoderTest(false, /*n_rounds=*/13, /*n_errs_per_round=*/30,
                           /*n_syndromes_per_round=*/10, /*window_size=*/3,
                           /*step_size=*/2);
}

TEST(SlidingWindowDecoder, SlidingWindowDecoderTestBatchedStepSize2) {
  SlidingWindowDecoderTest(true, /*n_rounds=*/13, /*n_errs_per_round=*/30,
                           /*n_syndromes_per_round=*/10, /*window_size=*/3,
                           /*step_size=*/2);
}

TEST(SlidingWindowDecoder, EmptyBatchReturnsNoResults) {
  const std::size_t n_rounds = 3;
  const std::size_t n_errs_per_round = 4;
  const std::size_t n_syndromes_per_round = 3;
  auto pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, /*weight=*/2,
      std::mt19937_64(1234));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);

  cudaqx::heterogeneous_map params;
  params.insert("window_size", std::size_t{2});
  params.insert("step_size", std::size_t{1});
  params.insert("num_syndromes_per_round", n_syndromes_per_round);
  params.insert("inner_decoder_name", std::string("single_error_lut"));
  params.insert("inner_decoder_params", cudaqx::heterogeneous_map{});

  auto H = cudaq::qec::sparse_binary_matrix(pcm);
  auto O = cudaq::qec::sparse_binary_matrix::from_csr(0, H.num_cols(), {0}, {});
  auto decoder = cudaq::qec::decoder::get(
      "sliding_window",
      cudaq::qec::decoder_init(std::move(H), std::move(O),
                               std::vector<double>(pcm.shape()[1], 0.1)),
      params);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decoder->decode_batch({}).empty());
}

TEST(SlidingWindowDecoder, PerRoundStreamingUsesRollingWindowUnwrap) {
  const std::size_t n_rounds = 4;
  const std::size_t n_errs_per_round = 3;
  const std::size_t n_syndromes_per_round = 2;
  auto pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, /*weight=*/1,
      std::mt19937_64(2026));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);

  cudaqx::heterogeneous_map params;
  params.insert("window_size", std::size_t{2});
  params.insert("step_size", std::size_t{1});
  params.insert("num_syndromes_per_round", n_syndromes_per_round);
  params.insert("inner_decoder_name", std::string("single_error_lut"));
  params.insert("inner_decoder_params", cudaqx::heterogeneous_map{});

  auto H = cudaq::qec::sparse_binary_matrix(pcm);
  auto O = cudaq::qec::sparse_binary_matrix::from_csr(0, H.num_cols(), {0}, {});
  auto decoder = cudaq::qec::decoder::get(
      "sliding_window",
      cudaq::qec::decoder_init(std::move(H), std::move(O),
                               std::vector<double>(pcm.shape()[1], 0.1)),
      params);
  ASSERT_NE(decoder, nullptr);

  cudaq::qec::decoder_result last_result;
  for (std::size_t r = 0; r < n_rounds; ++r) {
    std::vector<cudaq::qec::float_t> round(n_syndromes_per_round, 0.0);
    last_result = decoder->decode(round);
  }

  // The result is empty until the final streaming round completes all windows.
  EXPECT_TRUE(last_result.converged);
  EXPECT_EQ(last_result.result.size(), pcm.shape()[1]);
}

TEST(AsyncDecoderResultTest, MoveConstructorTransfersFuture) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result original(std::move(future));
  EXPECT_TRUE(original.fut.valid());

  cudaq::qec::async_decoder_result moved(std::move(original));
  EXPECT_TRUE(moved.fut.valid());
  EXPECT_FALSE(original.fut.valid());
}

TEST(AsyncDecoderResultTest, MoveAssignmentTransfersFuture) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result first(std::move(future));
  cudaq::qec::async_decoder_result second = std::move(first);

  EXPECT_TRUE(second.fut.valid());
  EXPECT_FALSE(first.fut.valid());
}

TEST(AsyncDecoderResultTest, ReadyMethod) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result async_result(std::move(future));

  // Initially, the result should not be ready
  EXPECT_FALSE(async_result.ready());

  // Set the promise value to make the future ready
  cudaq::qec::decoder_result result;
  result.converged = true;
  result.result = {0.1f, 0.2f, 0.3f};
  promise.set_value(result);

  // Now the result should be ready
  EXPECT_TRUE(async_result.ready());

  // We can now get the result without blocking
  auto retrieved_result = async_result.get();
  EXPECT_TRUE(retrieved_result.converged);
  EXPECT_EQ(retrieved_result.result.size(), 3);
  EXPECT_FLOAT_EQ(retrieved_result.result[0], 0.1f);
  EXPECT_FLOAT_EQ(retrieved_result.result[1], 0.2f);
  EXPECT_FLOAT_EQ(retrieved_result.result[2], 0.3f);
}

TEST(AsyncDecoderResultTest, ReadyMethodWithException) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result async_result(std::move(future));

  // Initially, the result should not be ready
  EXPECT_FALSE(async_result.ready());

  // Set an exception to make the future ready with an error
  promise.set_exception(
      std::make_exception_ptr(std::runtime_error("Test error")));

  // The future should be ready even though it contains an exception
  EXPECT_TRUE(async_result.ready());

  // Attempting to get the result should throw the exception
  EXPECT_THROW(async_result.get(), std::runtime_error);
}

TEST(DecoderResultTest, DefaultConstructor) {
  cudaq::qec::decoder_result result;
  EXPECT_FALSE(result.converged);
  EXPECT_TRUE(result.result.empty());
  EXPECT_FALSE(result.opt_results.has_value());
}

TEST(DecoderResultTest, OptResultsAssignment) {
  cudaq::qec::decoder_result result;
  cudaqx::heterogeneous_map opt_map;
  opt_map.insert("test_key", 42);
  result.opt_results = opt_map;

  EXPECT_TRUE(result.opt_results.has_value());
  EXPECT_EQ(result.opt_results->get<int>("test_key"), 42);
}

TEST(DecoderResultTest, EqualityOperator) {
  cudaq::qec::decoder_result result1;
  cudaq::qec::decoder_result result2;

  // Test equality with no opt_results
  EXPECT_TRUE(result1 == result2);

  // Test inequality when one has opt_results
  cudaqx::heterogeneous_map opt_map;
  opt_map.insert("test_key", 42);
  result1.opt_results = opt_map;
  EXPECT_FALSE(result1 == result2);

  // Test inequality when both have opt_results (even if same)
  result2.opt_results = opt_map;
  EXPECT_FALSE(result1 == result2);
}

TEST(DecoderResultTest, EqualityOperatorConvergedAndResult) {
  cudaq::qec::decoder_result result1;
  cudaq::qec::decoder_result result2;

  // Test inequality when converged field is different
  result1.converged = true;
  result2.converged = false;
  EXPECT_FALSE(result1 == result2);
  EXPECT_TRUE(result1 != result2);

  // Reset converged fields to be the same
  result1.converged = false;
  result2.converged = false;
  EXPECT_TRUE(result1 == result2);

  // Test inequality when result vector is different
  result1.result = {0.1f, 0.2f, 0.3f};
  result2.result = {0.4f, 0.5f, 0.6f};
  EXPECT_FALSE(result1 == result2);
  EXPECT_TRUE(result1 != result2);

  // Test inequality when result vector sizes are different
  result1.result = {0.1f, 0.2f};
  result2.result = {0.1f, 0.2f, 0.3f};
  EXPECT_FALSE(result1 == result2);
  EXPECT_TRUE(result1 != result2);

  // Test equality when both converged and result are the same
  result1.converged = true;
  result1.result = {0.1f, 0.2f, 0.3f};
  result2.converged = true;
  result2.result = {0.1f, 0.2f, 0.3f};
  EXPECT_TRUE(result1 == result2);
  EXPECT_FALSE(result1 != result2);
}

TEST(DecoderTest, GetWithoutOptionsSetsBlockAndSyndromeSize) {
  std::size_t block_size = 15;
  std::size_t syndrome_size = 8;

  // Create a parity check matrix H with specific dimensions
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});

  // Initialize the tensor with some test data
  for (std::size_t i = 0; i < syndrome_size; ++i) {
    for (std::size_t j = 0; j < block_size; ++j) {
      H.at({i, j}) = (i + j) % 2;
    }
  }

  // Create a decoder instance
  auto H_sparse = cudaq::qec::sparse_binary_matrix(H);
  auto decoder = cudaq::qec::decoder::get("sample_decoder", H_sparse);
  ASSERT_NE(decoder, nullptr);

  // Test get_block_size() returns the correct block size
  EXPECT_EQ(decoder->get_block_size(), block_size);

  // Test get_syndrome_size() returns the correct syndrome size
  EXPECT_EQ(decoder->get_syndrome_size(), syndrome_size);

  // Test with different dimensions
  std::size_t new_block_size = 20;
  std::size_t new_syndrome_size = 12;
  cudaqx::tensor<uint8_t> H2({new_syndrome_size, new_block_size});

  auto H_sparse2 = cudaq::qec::sparse_binary_matrix(H2);
  auto decoder2 = cudaq::qec::decoder::get("sample_decoder", H_sparse2);
  ASSERT_NE(decoder2, nullptr);

  EXPECT_EQ(decoder2->get_block_size(), new_block_size);
  EXPECT_EQ(decoder2->get_syndrome_size(), new_syndrome_size);
}

TEST(StimDemGetDecoder, ConstructsLutDecoderFromStimDemText) {
  const std::string dem_text = R"(error(0.1) D0 L0
error(0.1) D1 L0
error(0.05) D0 D1
)";

  auto d = cudaq::qec::get_decoder("single_error_lut", dem_text);
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(d->get_syndrome_size(), 2u);
  EXPECT_EQ(d->get_block_size(), 3u);

  struct Case {
    std::vector<cudaq::qec::float_t> syndrome;
    std::vector<cudaq::qec::float_t> expected;
  };
  const std::vector<Case> cases = {
      {{0.0, 0.0}, {0.0, 0.0, 0.0}},
      {{1.0, 0.0}, {1.0, 0.0, 0.0}},
      {{0.0, 1.0}, {0.0, 1.0, 0.0}},
      {{1.0, 1.0}, {0.0, 0.0, 1.0}},
  };
  for (const auto &c : cases) {
    auto result = d->decode(c.syndrome);
    EXPECT_TRUE(result.converged)
        << "syndrome {" << c.syndrome[0] << ", " << c.syndrome[1] << "}";
    ASSERT_EQ(result.result.size(), 3u);
    for (std::size_t i = 0; i < 3u; ++i)
      EXPECT_FLOAT_EQ(result.result[i], c.expected[i])
          << "error " << i << " for syndrome {" << c.syndrome[0] << ", "
          << c.syndrome[1] << "}";
  }
}

TEST(StimDemGetDecoder, StaticDecoderGetAcceptsStimDemString) {
  const std::string dem_text = R"(error(0.1) D0 L0
error(0.1) D1 L0
error(0.05) D0 D1
)";

  auto d = cudaq::qec::decoder::get("single_error_lut", dem_text);
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(d->get_syndrome_size(), 2u);
  EXPECT_EQ(d->get_block_size(), 3u);
  auto result = d->decode(std::vector<cudaq::qec::float_t>{1.0, 1.0});
  EXPECT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 3u);
  EXPECT_FLOAT_EQ(result.result[2], 1.0);
}

TEST(StimDemGetDecoder, StillAcceptsParityCheckMatrix) {
  cudaqx::tensor<uint8_t> H({2, 3});
  H.copy(std::vector<uint8_t>{1, 0, 1, 0, 1, 1}.data(), {2, 3});
  auto d = cudaq::qec::get_decoder("single_error_lut", H);
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(d->get_syndrome_size(), 2u);
  EXPECT_EQ(d->get_block_size(), 3u);
}

// The handle a decoder actually receives must describe exactly the matrices the
// materialized model does. Compared through decoder_init rather than the
// internal projection helper, so a mistake wiring the projection into the
// handle cannot pass. The sparse path exists to skip a dense intermediate --
// a distance-13 dense H is ~98 MiB against under 1 MiB sparse -- not to mean
// something different.
TEST(StimDemGetDecoder, DecoderInputsMatchMaterializedModel) {
  const std::string dem_text = R"(error(0.1) D0 L0
error(0.2) D0 D1
error(0.05) D1 D2 L1
error(0.3) D2 D0 L0 L1
error(0.1) D1 D1 D2
)";

  const auto dense = cudaq::qec::dem_from_stim_text(dem_text);
  auto inputs = cudaq::qec::decoder_init::from_stim_dem(dem_text);

  EXPECT_EQ(inputs.num_detectors(), dense.num_detectors());
  EXPECT_EQ(inputs.num_error_mechanisms(), dense.num_error_mechanisms());
  EXPECT_EQ(inputs.num_observables(), dense.num_observables());
  EXPECT_EQ(inputs.error_rates(), dense.error_rates);

  // Content, compared through a common dense view so a layout difference
  // cannot hide a content difference.
  const auto H = inputs.detector_error_matrix().to_dense();
  ASSERT_EQ(H.shape(), dense.detector_error_matrix.shape());
  for (std::size_t r = 0; r < H.shape()[0]; ++r)
    for (std::size_t c = 0; c < H.shape()[1]; ++c)
      EXPECT_EQ(H.at({r, c}), dense.detector_error_matrix.at({r, c}))
          << "H differs at (" << r << ", " << c << ")";

  ASSERT_TRUE(inputs.has_observable_model());
  const auto O = inputs.observable_flips_matrix().to_dense();
  ASSERT_EQ(O.shape(), dense.observables_flips_matrix.shape());
  for (std::size_t r = 0; r < O.shape()[0]; ++r)
    for (std::size_t c = 0; c < O.shape()[1]; ++c)
      EXPECT_EQ(O.at({r, c}), dense.observables_flips_matrix.at({r, c}))
          << "O differs at (" << r << ", " << c << ")";
}

TEST(StimDemGetDecoder, RepeatedDetectorOrObservableTargetsXorFold) {
  const std::string dem_text = R"(error(0.1) D0 D0 D1
error(0.1) L0 L0 D2
)";

  auto dem = cudaq::qec::dem_from_stim_text(dem_text);
  ASSERT_EQ(dem.num_detectors(), 3u);
  ASSERT_EQ(dem.num_observables(), 1u);
  ASSERT_EQ(dem.num_error_mechanisms(), 2u);
  EXPECT_EQ(dem.detector_error_matrix.at({0u, 0u}), 0u)
      << "duplicate D0 in error 0 should XOR-cancel to 0";
  EXPECT_EQ(dem.detector_error_matrix.at({1u, 0u}), 1u) << "D1 survives";
  EXPECT_EQ(dem.observables_flips_matrix.at({0u, 1u}), 0u)
      << "duplicate L0 in error 1 should XOR-cancel to 0";
  EXPECT_EQ(dem.detector_error_matrix.at({2u, 1u}), 1u) << "D2 survives";
}

TEST(StimDemGetDecoder, DemWithoutObservablesDoesNotAddODefault) {
  auto dem = cudaq::qec::dem_from_stim_text("error(0.1) D0\n");
  auto defaults = cudaq::qec::details::dem_defaults_for_missing_keys(
      [](const std::string &) { return false; }, dem);

  EXPECT_EQ(defaults.O, nullptr);
  ASSERT_NE(defaults.error_rate_vec, nullptr);
  EXPECT_EQ(defaults.error_rate_vec->size(), 1u);
}

TEST(StimDemGetDecoder, ThrowsOnProbabilityOutOfRange) {
  const std::string dem_text = "error(1.5) D0\n";
  EXPECT_THROW(cudaq::qec::get_decoder("single_error_lut", dem_text),
               std::runtime_error);
}

TEST(StimDemGetDecoder, ThrowsOnMalformedStimDem) {
  EXPECT_THROW(cudaq::qec::get_decoder("single_error_lut", "not a valid DEM"),
               std::runtime_error);
}

TEST(StimDemGetDecoder, ThrowsOnUnknownDecoderName) {
  const std::string dem_text = "error(0.1) D0 L0\n";
  EXPECT_THROW(cudaq::qec::get_decoder("__no_such_decoder__", dem_text),
               std::runtime_error);
}

TEST(StimDemGetDecoder, ThrowsOnEmptyErrorMechanisms) {
  const std::string dem_text = "detector(0, 0, 0)\n";
  EXPECT_THROW(cudaq::qec::get_decoder("single_error_lut", dem_text),
               std::runtime_error);
}

TEST(StimDemGetDecoder, StimDemTargetCategoriesAreExhaustive) {
  const std::vector<stim::DemTarget> samples = {
      stim::DemTarget::separator(),
      stim::DemTarget::relative_detector_id(0),
      stim::DemTarget::relative_detector_id(42),
      stim::DemTarget::observable_id(0),
      stim::DemTarget::observable_id(7),
  };
  for (const auto &t : samples) {
    const int kinds = static_cast<int>(t.is_separator()) +
                      static_cast<int>(t.is_relative_detector_id()) +
                      static_cast<int>(t.is_observable_id());
    EXPECT_EQ(kinds, 1) << "DemTarget " << t.str() << " matched " << kinds
                        << " predicates; expected exactly 1";
  }
}

TEST(StimDemGetDecoder, UserOptionsAreNotOverwritten) {
  const std::string dem_text = R"(error(0.1) D0 L0
error(0.1) D1 L0
error(0.05) D0 D1
)";
  cudaqx::heterogeneous_map opts;
  opts.insert("error_rate_vec", std::vector<double>{0.5});
  EXPECT_THROW(cudaq::qec::get_decoder("single_error_lut", dem_text, opts),
               std::runtime_error);
}

TEST(StimDemGetDecoder, SeparatorTargetsAreIgnoredByFallbackParser) {
  // The fallback parser must skip Stim's '^' separator while retaining both
  // detector targets and the observable target in the same error mechanism.
  const std::string dem_text = "error(0.25) D0 ^ D1 L0\n";
  auto dem = cudaq::qec::dem_from_stim_text(dem_text);
  ASSERT_EQ(dem.num_error_mechanisms(), 1u);
  ASSERT_EQ(dem.num_detectors(), 2u);
  ASSERT_EQ(dem.num_observables(), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 1u);
}

TEST(StimDemGetDecoder, DecomposeErrorsFalseIgnoresCaret) {
  auto dem = cudaq::qec::dem_from_stim_text(
      "error(0.1) D0 D2 ^ D1 D3\nerror(0.2) D0 D2 D1 D3\n");
  ASSERT_EQ(dem.num_error_mechanisms(), 2u);
  std::size_t expected_num_detectors = 4u;
  for (std::size_t r = 0; r < expected_num_detectors; ++r) {
    EXPECT_EQ(dem.detector_error_matrix.at({r, 0u}), 1u) << "D" << r;
    EXPECT_EQ(dem.detector_error_matrix.at({r, 1u}), 1u) << "D" << r;
  }
  EXPECT_FALSE(dem.error_ids.has_value());
}

TEST(StimDemGetDecoder, DemFromStimTextErrorIds) {
  const std::string dem_text = "error(0.05) D0 D1 L0\n"
                               "error(0.03) D2 L1\n"
                               "error(0.1) D0 D2 ^ D1 D3\n";

  auto dem_no = cudaq::qec::dem_from_stim_text(dem_text);
  EXPECT_FALSE(dem_no.error_ids.has_value());

  auto dem_yes =
      cudaq::qec::dem_from_stim_text(dem_text, /*use_decomp_suggestions=*/true);
  EXPECT_FALSE(dem_yes.error_ids.has_value());
  ASSERT_EQ(dem_yes.error_rates.size(), 4u);
  EXPECT_DOUBLE_EQ(dem_yes.error_rates[0], 0.05);
  EXPECT_DOUBLE_EQ(dem_yes.error_rates[1], 0.03);
  EXPECT_DOUBLE_EQ(dem_yes.error_rates[2], 0.1);
  EXPECT_DOUBLE_EQ(dem_yes.error_rates[3], 0.1);
}

// Verify true: each ^ component becomes a separate column; each component
// inherits the parent instruction's probability; error_ids is nullopt.
TEST(StimDemGetDecoder, DecomposeErrorsTrueSplitsCaretComponents) {
  auto dem = cudaq::qec::dem_from_stim_text("error(0.1) D0 L0 ^ D1\n",
                                            /*use_decomp_suggestions=*/true);
  ASSERT_EQ(dem.num_error_mechanisms(), 2u);
  ASSERT_EQ(dem.num_detectors(), 2u);
  ASSERT_EQ(dem.num_observables(), 1u);

  // Expected result:
  //      E0 E1
  // D0 [[1, 0],
  // D1  [0, 1]]
  // L0 [[1, 0]]
  EXPECT_EQ(dem.detector_error_matrix.at({0u, 0u}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({0u, 1u}), 0u);
  EXPECT_EQ(dem.detector_error_matrix.at({1u, 0u}), 0u);
  EXPECT_EQ(dem.detector_error_matrix.at({1u, 1u}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0u, 0u}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0u, 1u}), 0u);

  // Each component gets the parent probability; error_ids is always nullopt.
  EXPECT_FALSE(dem.error_ids.has_value());
  ASSERT_EQ(dem.error_rates.size(), 2u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.1);
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.1);
}

TEST(StimDemGetDecoder, DecomposeErrorsXorCancelled) {
  auto dem = cudaq::qec::dem_from_stim_text("error(0.1) D0 D0 ^ D1\n",
                                            /*use_decomp_suggestions=*/true);
  ASSERT_EQ(dem.num_error_mechanisms(), 1u)
      << "empty component after XOR cancellation must be dropped";
  EXPECT_EQ(dem.detector_error_matrix.at({0u, 0u}), 0u) << "D0 cancelled";
  EXPECT_EQ(dem.detector_error_matrix.at({1u, 0u}), 1u) << "D1 survives";
}

// ---------------------------------------------------------------------------
// Tests for enqueue_syndrome tagged-result routing
// ---------------------------------------------------------------------------

// Verify that an observable-output decoder session uses its result directly.
TEST(EnqueueSyndrome, ObsFrameDecoderUsesResultDirectly) {
  // H: 2 syndrome measurements, 4 physical errors
  cudaqx::tensor<uint8_t> H_tensor({2, 4});
  H_tensor.at({0, 0}) = 1;
  H_tensor.at({1, 1}) = 1;
  auto H = cudaq::qec::sparse_binary_matrix(H_tensor);
  auto O = cudaq::qec::sparse_binary_matrix::from_nested_csr(
      2, 4, std::vector<std::vector<std::uint32_t>>{{0}, {1}});
  auto dec = cudaq::qec::decoder::get(
      "observable_output_probe",
      // D maps the two enqueued syndrome bits directly to two detector bits.
      cudaq::qec::decoder_init(
          std::move(H), std::move(O), /*error_rates=*/{},
          cudaq::qec::sparse_binary_matrix::from_nested_csr(2, 2, {{0}, {1}})),
      cudaq::qec::decode_result_type::observables);

  bool did_decode = dec->enqueue_syndrome(std::vector<uint8_t>{1, 0});
  EXPECT_TRUE(did_decode);

  const uint8_t *corr = dec->get_obs_corrections();
  EXPECT_EQ(corr[0], 1u);
  EXPECT_EQ(corr[1], 0u);
}

// Verify that corrections XOR-accumulate correctly across multiple shots and
// that clear_corrections() resets them between shots.
TEST(EnqueueSyndrome, ObsFrameMultiShotAccumulation) {
  cudaqx::tensor<uint8_t> H_tensor({2, 4});
  H_tensor.at({0, 0}) = 1;
  H_tensor.at({1, 1}) = 1;
  auto H = cudaq::qec::sparse_binary_matrix(H_tensor);
  auto O = cudaq::qec::sparse_binary_matrix::from_nested_csr(
      2, 4, std::vector<std::vector<std::uint32_t>>{{0}, {1}});
  auto dec = cudaq::qec::decoder::get(
      "observable_output_probe",
      cudaq::qec::decoder_init(
          std::move(H), std::move(O), /*error_rates=*/{},
          cudaq::qec::sparse_binary_matrix::from_nested_csr(2, 2, {{0}, {1}})),
      cudaq::qec::decode_result_type::observables);

  // Shot 1: obs[0]=1, obs[1]=0 -> corrections become [1, 0]
  EXPECT_TRUE(dec->enqueue_syndrome(std::vector<uint8_t>{1, 0}));
  const uint8_t *corr = dec->get_obs_corrections();
  EXPECT_EQ(corr[0], 1u);
  EXPECT_EQ(corr[1], 0u);

  // Shot 2 (no reset): obs[0]=1, obs[1]=1 -> corrections XOR to [0, 1]
  EXPECT_TRUE(dec->enqueue_syndrome(std::vector<uint8_t>{1, 1}));
  corr = dec->get_obs_corrections();
  EXPECT_EQ(corr[0], 0u);
  EXPECT_EQ(corr[1], 1u);

  // After clear, corrections reset to [0, 0]
  dec->clear_corrections();
  corr = dec->get_obs_corrections();
  EXPECT_EQ(corr[0], 0u);
  EXPECT_EQ(corr[1], 0u);
}

// Verify that a tagged observable result is checked against num_observables.
TEST(EnqueueSyndrome, ObsFrameSizeMismatchThrows) {
  cudaqx::tensor<uint8_t> H_tensor({3, 4});
  H_tensor.at({0, 0}) = 1;
  H_tensor.at({1, 1}) = 1;
  H_tensor.at({2, 2}) = 1;
  auto H = cudaq::qec::sparse_binary_matrix(H_tensor);
  auto O = cudaq::qec::sparse_binary_matrix::from_nested_csr(
      2, 4, std::vector<std::vector<std::uint32_t>>{{0}, {1}});
  auto dec = cudaq::qec::decoder::get(
      "observable_output_probe",
      cudaq::qec::decoder_init(
          std::move(H), std::move(O), /*error_rates=*/{},
          cudaq::qec::sparse_binary_matrix::from_nested_csr(3, 3,
                                                            {{0}, {1}, {2}})),
      cudaq::qec::decode_result_type::observables);
  // sample_decoder returns all three detector bits as observables.
  EXPECT_THROW(dec->enqueue_syndrome(std::vector<uint8_t>{1, 0, 1}),
               std::runtime_error);
}

TEST(SlidingWindowDecoder, BaseStreamingCopiesFirstRoundDetectors) {
  // A first-round detector matrix starts with single-syndrome rows; the base
  // enqueue path should decode after one streamed round using a direct copy.
  constexpr std::size_t n_rounds = 3;
  constexpr std::size_t n_errs_per_round = 3;
  constexpr std::size_t n_syndromes_per_round = 2;
  auto pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, /*weight=*/1,
      std::mt19937_64(20260611));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);

  cudaqx::heterogeneous_map params;
  params.insert("window_size", std::size_t{2});
  params.insert("step_size", std::size_t{1});
  params.insert("num_syndromes_per_round", n_syndromes_per_round);
  params.insert("inner_decoder_name", std::string("single_error_lut"));
  params.insert("inner_decoder_params", cudaqx::heterogeneous_map{});

  auto H = cudaq::qec::sparse_binary_matrix(pcm);
  auto O =
      cudaq::qec::sparse_binary_matrix::from_csr(1, H.num_cols(), {0, 1}, {0});
  // D spans every detector the model declares, as the model requires. Only the
  // first-round layer references measurements, so one round of two bits fills
  // the measurement volume and drives the first streamed layer.
  std::vector<std::vector<std::uint32_t>> m2d(H.num_rows());
  m2d[0] = {0};
  m2d[1] = {1};
  auto decoder = cudaq::qec::decoder::get(
      "sliding_window",
      cudaq::qec::decoder_init(
          std::move(H), std::move(O), std::vector<double>(pcm.shape()[1], 0.1),
          cudaq::qec::sparse_binary_matrix::from_nested_csr(
              static_cast<std::uint32_t>(m2d.size()), 2, m2d)),
      cudaq::qec::decode_result_type::observables, params);
  ASSERT_NE(decoder, nullptr);

  std::vector<uint8_t> first_round = {1, 0};
  EXPECT_FALSE(decoder->enqueue_syndrome(first_round))
      << "First-round detector copy runs, but the sliding window is not full "
         "yet so no final correction is committed.";
}

TEST(SlidingWindowDecoder, PreparePcmRejectsBadBoundaryLayout) {
  // The boundary layout is validated during construction, so an inconsistent
  // boundary layout must throw.
  auto params = [](std::size_t S, std::size_t B) {
    cudaqx::heterogeneous_map p;
    p.insert("window_size", std::size_t{1});
    p.insert("step_size", std::size_t{1});
    p.insert("num_syndromes_per_round", S);
    p.insert("num_boundary_syndromes", B);
    p.insert("inner_decoder_name", std::string("single_error_lut"));
    p.insert("inner_decoder_params", cudaqx::heterogeneous_map{});
    return p;
  };

  // Assert the constructor throws with a message identifying the boundary
  // layout as the cause
  auto expectBoundaryThrow = [](const cudaqx::tensor<uint8_t> &H,
                                const cudaqx::heterogeneous_map &p,
                                const std::string &needle) {
    try {
      auto sparse_H = cudaq::qec::sparse_binary_matrix(H);
      auto O = cudaq::qec::sparse_binary_matrix::from_csr(
          0, sparse_H.num_cols(), {0}, {});
      cudaq::qec::decoder::get(
          "sliding_window",
          cudaq::qec::decoder_init(std::move(sparse_H), std::move(O),
                                   std::vector<double>(H.shape()[1], 0.1)),
          p);
      FAIL() << "expected sliding_window construction to throw";
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
          << "unexpected message: " << e.what();
    }
  };

  // Boundary wider than the interior (B > S).
  cudaqx::tensor<uint8_t> H4({4, 2});
  expectBoundaryThrow(
      H4, params(2, 4),
      "num_boundary_syndromes must be <= num_syndromes_per_round");

  // Row count inconsistent with a [B | K*S | B] layout (B < S).
  cudaqx::tensor<uint8_t> H17({17, 2});
  expectBoundaryThrow(H17, params(8, 3), "number of PCM rows is inconsistent");
}

namespace {

int cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    return 0;
  return count;
}

/// Restores the caller's CUDA device on scope exit so the persistent pin
/// made by one test does not leak into the next (gtest shares the process).
class ScopedDeviceRestore {
public:
  ScopedDeviceRestore() {
    if (cudaGetDevice(&prev_) != cudaSuccess)
      prev_ = -1;
  }
  ~ScopedDeviceRestore() {
    if (prev_ >= 0)
      (void)cudaSetDevice(prev_);
  }

private:
  int prev_ = -1;
};

/// A decoder that rejects any construction parameter it does not know,
/// proving decoder::get() strips cuda_device_id before the plugin ctor.
class strict_keys_decoder : public cudaq::qec::decoder {
public:
  strict_keys_decoder(cudaq::qec::decoder_init inputs,
                      cudaq::qec::decode_result_type requested_output,
                      const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    auto invalid =
        cudaq::qec::validate_config_parameters(params, {"decode_to_obs"});
    if (!invalid.empty())
      throw std::runtime_error("strict_keys_decoder: unexpected key " +
                               invalid.front());
  }
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    cudaq::qec::decoder_result r;
    r.converged = true;
    r.result = std::vector<cudaq::qec::float_t>(block_size, 0.0);
    return r;
  }
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      strict_keys_decoder,
      static std::unique_ptr<cudaq::qec::decoder> create(
          cudaq::qec::decoder_init inputs,
          std::optional<cudaq::qec::decode_result_type> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<strict_keys_decoder>(
            std::move(inputs),
            output.value_or(cudaq::qec::decode_result_type::errors), params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(strict_keys_decoder)

cudaq::qec::sparse_binary_matrix make_test_H() {
  cudaqx::tensor<uint8_t> H({std::size_t{4}, std::size_t{10}});
  return cudaq::qec::sparse_binary_matrix(H);
}

/// Records the CUDA device current on the thread that runs decode(), so a
/// test can observe which device an async worker thread actually used.
class device_recording_decoder : public cudaq::qec::decoder {
public:
  std::atomic<int> last_decode_device{-2};
  device_recording_decoder(cudaq::qec::decoder_init inputs,
                           cudaq::qec::decode_result_type requested_output,
                           const cudaqx::heterogeneous_map &)
      : decoder(std::move(inputs), requested_output) {}
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess)
      dev = -1;
    last_decode_device.store(dev);
    cudaq::qec::decoder_result r;
    r.converged = true;
    r.result = std::vector<cudaq::qec::float_t>(block_size, 0.0);
    return r;
  }
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      device_recording_decoder,
      static std::unique_ptr<cudaq::qec::decoder> create(
          cudaq::qec::decoder_init inputs,
          std::optional<cudaq::qec::decode_result_type> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<device_recording_decoder>(
            std::move(inputs),
            output.value_or(cudaq::qec::decode_result_type::errors), params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(device_recording_decoder)

/// Always predicts error mechanism 0 so enqueue_syndrome's decode_to_errs
/// logging records Errors:0.
class one_error_decoder : public cudaq::qec::decoder {
public:
  one_error_decoder(const cudaq::qec::sparse_binary_matrix &H,
                    const cudaqx::heterogeneous_map &)
      : decoder(H) {}
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    cudaq::qec::decoder_result r;
    r.converged = true;
    r.result.assign(block_size, 0.0);
    if (!r.result.empty())
      r.result[0] = 1.0;
    return r;
  }
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      one_error_decoder, static std::unique_ptr<cudaq::qec::decoder> create(
                             const cudaq::qec::decoder_init &init,
                             const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<one_error_decoder>(init, params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(one_error_decoder)

} // namespace

TEST(DecoderCudaDeviceId, AbsentKeyIsNoOp) {
  auto d = cudaq::qec::decoder::get("sample_decoder", make_test_H());
  EXPECT_EQ(d->get_cuda_device_id(), -1);
  std::vector<cudaq::qec::float_t> syndrome(4);
  auto r = d->decode(syndrome);
  EXPECT_EQ(r.result.size(), 10);
}

TEST(DecoderCudaDeviceId, NegativeIdThrows) {
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", -2);
  try {
    auto d = cudaq::qec::decoder::get("sample_decoder", make_test_H(), params);
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("cuda_device_id"), std::string::npos);
  }
}

TEST(DecoderCudaDeviceId, OutOfRangeIdThrows) {
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 1 << 20);
  try {
    auto d = cudaq::qec::decoder::get("sample_decoder", make_test_H(), params);
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("cuda_device_id"), std::string::npos);
  }
}

TEST(DecoderCudaDeviceId, PersistentPinAtConstruction) {
  if (cuda_device_count() < 2)
    GTEST_SKIP() << "needs >= 2 CUDA devices";
  ScopedDeviceRestore restore;
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 1);
  auto d = cudaq::qec::decoder::get("sample_decoder", make_test_H(), params);
  EXPECT_EQ(d->get_cuda_device_id(), 1);
  // The pin is persistent: the constructing thread is still on device 1
  // after decoder::get() returns.
  int cur = -1;
  ASSERT_EQ(cudaGetDevice(&cur), cudaSuccess);
  EXPECT_EQ(cur, 1);
  // Decode entry points need no per-call guard on this thread.
  std::vector<cudaq::qec::float_t> syndrome(4);
  auto r = d->decode(syndrome);
  EXPECT_EQ(r.result.size(), 10);
  ASSERT_EQ(cudaGetDevice(&cur), cudaSuccess);
  EXPECT_EQ(cur, 1);
}

TEST(DecoderCudaDeviceId, KeyStrippedFromPluginParams) {
  if (cuda_device_count() < 1)
    GTEST_SKIP() << "needs >= 1 CUDA device";
  ScopedDeviceRestore restore;
  // Sanity: strict_keys_decoder does reject unknown keys.
  cudaqx::heterogeneous_map bogus;
  bogus.insert("bogus_key", 1);
  EXPECT_THROW(
      cudaq::qec::decoder::get("strict_keys_decoder", make_test_H(), bogus),
      std::runtime_error);
  // cuda_device_id must be consumed by the base and never reach the plugin,
  // while permitted keys (decode_to_obs) must survive the rebuild-and-strip.
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 0);
  params.insert("decode_to_obs", true);
  EXPECT_NO_THROW(
      cudaq::qec::decoder::get("strict_keys_decoder", make_test_H(), params));
}

TEST(DecoderCudaDeviceId, AsyncWorkerPinsItself) {
  if (cuda_device_count() < 2)
    GTEST_SKIP() << "needs >= 2 CUDA devices";
  ScopedDeviceRestore restore;
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 1);
  auto d = cudaq::qec::decoder::get("device_recording_decoder", make_test_H(),
                                    params);
  auto *rec = dynamic_cast<device_recording_decoder *>(d.get());
  ASSERT_NE(rec, nullptr);
  // decode_async spawns a brand-new std::async thread whose current device
  // defaults to 0, NOT the owning thread's device 1. The worker must pin
  // itself to the decoder's device for the duration of the call.
  std::vector<cudaq::qec::float_t> syndrome(4);
  auto r = d->decode_async(syndrome).get();
  EXPECT_EQ(r.result.size(), 10);
  EXPECT_EQ(rec->last_decode_device.load(), 1);
  // The owning (calling) thread's device is untouched by the async call.
  int cur = -1;
  ASSERT_EQ(cudaGetDevice(&cur), cudaSuccess);
  EXPECT_EQ(cur, 1);
}

TEST(DecoderCudaDeviceId, TwoThreadsTwoDevices) {
  if (cuda_device_count() < 2)
    GTEST_SKIP() << "needs >= 2 CUDA devices";
  auto worker = [](int id, int &observed_device, bool &decode_ok) {
    cudaqx::heterogeneous_map params;
    params.insert("cuda_device_id", id);
    auto d = cudaq::qec::decoder::get("sample_decoder", make_test_H(), params);
    std::vector<cudaq::qec::float_t> syndrome(4);
    auto r = d->decode(syndrome);
    decode_ok = (r.result.size() == 10);
    int cur = -1;
    observed_device = (cudaGetDevice(&cur) == cudaSuccess) ? cur : -1;
  };
  int dev0 = -1, dev1 = -1;
  bool ok0 = false, ok1 = false;
  std::thread t0(worker, 0, std::ref(dev0), std::ref(ok0));
  std::thread t1(worker, 1, std::ref(dev1), std::ref(ok1));
  t0.join();
  t1.join();
  EXPECT_TRUE(ok0);
  EXPECT_TRUE(ok1);
  EXPECT_EQ(dev0, 0);
  EXPECT_EQ(dev1, 1);
}

// Zero/NaN clamp into the first histogram bucket; a huge sample clamps high.
TEST(DecoderStats, LatencyHistogramClampsLowAndHigh) {
  cudaq::qec::latency_series series;
  series.add(0.0);
  series.add(std::numeric_limits<double>::quiet_NaN());
  series.add(1.0e9);
  EXPECT_EQ(series.count, 3u);
  EXPECT_EQ(series.min_us, 0.0);
  EXPECT_DOUBLE_EQ(series.max_us, 1.0e9);
  EXPECT_DOUBLE_EQ(series.percentile_us(0.0), 0.0);
  EXPECT_DOUBLE_EQ(series.percentile_us(1.0), 1.0e9);
}

// collect_set_bits clears stale output and reports only the set indices.
TEST(DecoderStats, CollectSetBitsClearsAndReportsSparseIndices) {
  std::vector<uint32_t> out{99, 100};
  const uint8_t bits[] = {0, 1, 0, 1, 1};
  cudaq::qec::collect_set_bits(bits, 5, out);
  EXPECT_EQ(out, (std::vector<uint32_t>{1, 3, 4}));
}

// Fractions outside (0, 1) report the exact min/max; one sample is the whole
// distribution so every rank still equals that sample (the L72 fallback is
// unreachable because every add() increments exactly one bucket).
TEST(DecoderStats, LatencyPercentileFractionsAtBounds) {
  cudaq::qec::latency_series series;
  series.add(12.5);
  series.add(3.5);
  EXPECT_DOUBLE_EQ(series.percentile_us(-0.5), series.min_us);
  EXPECT_DOUBLE_EQ(series.percentile_us(0.0), series.min_us);
  EXPECT_DOUBLE_EQ(series.percentile_us(1.0), series.max_us);
  EXPECT_DOUBLE_EQ(series.percentile_us(2.0), series.max_us);
}

// CUDAQ_QEC_DEBUG_DECODER=y selects printf arrays mode; a frame call is
// therefore emitted.
TEST(DecoderStats, PrintfModeEmitsFrameCall) {
  ScopedEnv debugEnv("CUDAQ_QEC_DEBUG_DECODER", "y");
  testing::internal::CaptureStdout();
  {
    cudaq::qec::decoder_stats stats;
    stats.set_decoder_id(3);
    stats.emit_frame_call("get_obs_corrections", {1, 0, 1});
  }
  const auto log = testing::internal::GetCapturedStdout();
  EXPECT_NE(log.find("get_obs_corrections called"), std::string::npos);
  EXPECT_NE(log.find("ObservableCorrectionsTotal:1,0,1"), std::string::npos);
}

// info -> summary (no per-call line); debug -> arrays; a disabled level is off.
TEST(DecoderStats, DetailFollowsLogLevel) {
  {
    ScopedLogLevel scoped(cudaq::qec::detail::log_level::info);
    cudaq::qec::decoder_stats stats;
    testing::internal::CaptureStdout();
    stats.emit_frame_call("get_obs_corrections", {1});
    EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
    testing::internal::CaptureStdout();
    stats.note_submit(true);
    stats.note_decode_complete();
    stats.emit_summary();
    EXPECT_NE(testing::internal::GetCapturedStdout().find("stats_summary"),
              std::string::npos);
  }
  {
    ScopedLogLevel scoped(cudaq::qec::detail::log_level::debug);
    cudaq::qec::decoder_stats stats;
    testing::internal::CaptureStdout();
    stats.emit_frame_call("reset_decoder", {0});
    EXPECT_NE(
        testing::internal::GetCapturedStdout().find("reset_decoder called"),
        std::string::npos);
  }
  {
    ScopedLogLevel scoped(cudaq::qec::detail::log_level::error);
    cudaq::qec::decoder_stats stats;
    stats.note_submit(true);
    stats.note_decode_complete();
    testing::internal::CaptureStdout();
    stats.emit_summary();
    EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
  }
}

// Submit timing, decode-before-submit no-ops, and reset abandon an open shot.
TEST(DecoderStats, SubmitDecodeResetTiming) {
  ScopedLogLevel scoped(cudaq::qec::detail::log_level::info);
  cudaq::qec::decoder_stats stats;
  EXPECT_DOUBLE_EQ(stats.since_last_submit_us(), 0.0);
  stats.note_decode_complete();
  stats.note_submit(/*first_of_shot=*/true);
  EXPECT_GE(stats.since_last_submit_us(), 0.0);
  stats.note_reset();
  testing::internal::CaptureStdout();
  stats.emit_summary();
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

// Both replay-field overloads render every named field; size mismatch yields
// no flips, equal frames with XOR 1 populate both dense and sparse accessors.
TEST(DecoderStats, ReplayFieldsAndDiffFrame) {
  cudaq::qec::decoder_stats stats;
  stats.diff_frame({0, 1}, {0});
  EXPECT_TRUE(stats.frame_flips().empty());
  EXPECT_TRUE(stats.frame_flip_ids().empty());

  stats.diff_frame({0, 1, 1}, {1, 1, 0});
  EXPECT_EQ(stats.frame_flips(), (std::vector<uint8_t>{1, 0, 1}));
  EXPECT_EQ(stats.frame_flip_ids(), (std::vector<uint32_t>{0, 2}));

  cudaq::qec::sparse_scratch scratch;
  scratch.msyn.assign(1, 1);
  scratch.detectors.assign(1, 2);
  scratch.errors.assign(1, 3);
  const std::vector<uint8_t> total{1, 0, 1};
  std::string from_scratch;
  stats.append_replay_fields(from_scratch, scratch, total);
  EXPECT_NE(from_scratch.find("InputMsyn:1"), std::string::npos);
  EXPECT_NE(from_scratch.find("InputDetectors:2"), std::string::npos);
  EXPECT_NE(from_scratch.find("Errors:3"), std::string::npos);
  EXPECT_NE(from_scratch.find("Observables:0,2"), std::string::npos);
  EXPECT_NE(from_scratch.find("ObservableCorrectionsThisCall:1,0,1"),
            std::string::npos);
  EXPECT_NE(from_scratch.find("ObservableCorrectionsTotal:1,0,1"),
            std::string::npos);

  const std::vector<uint32_t> obs{0};
  const std::vector<uint8_t> this_call{1};
  const std::vector<uint8_t> corr_total{1};
  cudaq::qec::replay_fields fields{scratch.msyn,   scratch.detectors,
                                   scratch.errors, obs,
                                   this_call,      corr_total};
  std::string from_fields;
  stats.append_replay_fields(from_fields, fields);
  EXPECT_NE(from_fields.find("Observables:0"), std::string::npos);
}

TEST(SlidingWindowDecoder, BoundarySyndromeGetter) {
  constexpr std::size_t n_rounds = 4;
  constexpr std::size_t n_errs_per_round = 3;
  constexpr std::size_t n_syndromes_per_round = 2;
  auto pcm = cudaq::qec::generate_random_pcm(n_rounds, n_errs_per_round,
                                             n_syndromes_per_round,
                                             /*weight=*/1, std::mt19937_64(11));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);

  auto make_params = [&](std::optional<std::size_t> boundary) {
    cudaqx::heterogeneous_map p;
    p.insert("window_size", std::size_t{2});
    p.insert("step_size", std::size_t{1});
    p.insert("num_syndromes_per_round", n_syndromes_per_round);
    p.insert("error_rate_vec", std::vector<double>(pcm.shape()[1], 0.1));
    p.insert("inner_decoder_name", std::string("single_error_lut"));
    p.insert("inner_decoder_params", cudaqx::heterogeneous_map{});
    if (boundary)
      p.insert("num_boundary_syndromes", *boundary);
    return p;
  };

  auto defaulted = cudaq::qec::decoder::get("sliding_window", pcm,
                                            make_params(std::nullopt));
  auto *sw_default =
      dynamic_cast<cudaq::qec::sliding_window *>(defaulted.get());
  ASSERT_NE(sw_default, nullptr);
  // A missing/zero boundary width is normalised to the interior round width.
  EXPECT_EQ(sw_default->get_num_boundary_syndromes(), n_syndromes_per_round);

  auto explicit_b = cudaq::qec::decoder::get(
      "sliding_window", pcm, make_params(n_syndromes_per_round));
  auto *sw_explicit =
      dynamic_cast<cudaq::qec::sliding_window *>(explicit_b.get());
  ASSERT_NE(sw_explicit, nullptr);
  EXPECT_EQ(sw_explicit->get_num_boundary_syndromes(), n_syndromes_per_round);
}

// decode_to_errs with a set result bit logs Errors:0 and flips observable 0.
TEST(EnqueueSyndrome, DecodeToErrsLogsSetErrorIndex) {
  ScopedLogLevel scoped(cudaq::qec::detail::log_level::info);
  cudaqx::tensor<uint8_t> H({std::size_t{1}, std::size_t{1}});
  H.at({0, 0}) = 1;
  auto dec = cudaq::qec::decoder::get("one_error_decoder", H);
  dec->set_D_sparse(std::vector<std::vector<uint32_t>>{{0}});
  dec->set_O_sparse(std::vector<std::vector<uint32_t>>{{0}});
  testing::internal::CaptureStdout();
  EXPECT_TRUE(dec->enqueue_syndrome(std::vector<uint8_t>{1}));
  cudaq::qec::detail::flush_logs();
  const auto log = testing::internal::GetCapturedStdout();
  EXPECT_NE(log.find("Errors:0"), std::string::npos);
  EXPECT_NE(log.find("ResultType:errs"), std::string::npos);
  EXPECT_NE(log.find("ObservableCorrectionsThisCall:1"), std::string::npos);
  EXPECT_EQ(dec->get_obs_corrections()[0], 1u);
}

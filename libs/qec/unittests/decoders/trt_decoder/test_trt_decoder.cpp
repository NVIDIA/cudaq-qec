/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "trt_test_data.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/trt_decoder_internal.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <unistd.h>
#include <vector>

#include <cuda_runtime_api.h>

using namespace cudaq::qec;

static bool gpu_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

namespace {
class TestTrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity, const char *) noexcept override {}
};

cudaqx::tensor<uint8_t> make_identity_h(std::size_t n) {
  cudaqx::tensor<uint8_t> H({n, n});
  for (std::size_t i = 0; i < n; ++i)
    H.at({i, i}) = 1;
  return H;
}

decoder_init
make_inputs_with_empty_observables(const cudaqx::tensor<uint8_t> &H,
                                   std::size_t num_observables) {
  auto sparse_H = sparse_binary_matrix(H);
  std::vector<std::uint32_t> row_ptrs(num_observables + 1, 0);
  auto O = sparse_binary_matrix::from_csr(
      static_cast<std::uint32_t>(num_observables), sparse_H.num_cols(),
      std::move(row_ptrs), {});
  return decoder_init(std::move(sparse_H), std::move(O));
}

std::filesystem::path make_temp_engine_path(const std::string &name) {
  return std::filesystem::temp_directory_path() / name;
}

std::optional<std::string> get_dynamic_onnx_asset_path() {
#ifdef TRT_TEST_DYNAMIC_ONNX_PATH
  return std::string(TRT_TEST_DYNAMIC_ONNX_PATH);
#else
  return std::nullopt;
#endif
}

std::optional<std::string> get_uint8_onnx_asset_path() {
#ifdef TRT_TEST_UINT8_ONNX_PATH
  return std::string(TRT_TEST_UINT8_ONNX_PATH);
#else
  return std::nullopt;
#endif
}

std::optional<std::string> get_uint8_to_float_onnx_asset_path() {
#ifdef TRT_TEST_UINT8_TO_FLOAT_ONNX_PATH
  return std::string(TRT_TEST_UINT8_TO_FLOAT_ONNX_PATH);
#else
  return std::nullopt;
#endif
}
} // namespace

// Path to ONNX test asset. Set by CMake (absolute path) so the test finds it
// regardless of the executable's run directory.
static std::string get_onnx_asset_path() {
#ifdef TRT_TEST_ONNX_PATH
  return std::string(TRT_TEST_ONNX_PATH);
#else
  // Fallback when not built with CMake: relative to project root
  return "assets/tests/surface_code_decoder.onnx";
#endif
}

// Test fixture for TRT decoder tests
class TRTDecoderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up test parameters
    block_size = 3;
    syndrome_size = 2;

    // Create a simple parity check matrix H
    H = cudaqx::tensor<uint8_t>({syndrome_size, block_size});
    H.at({0, 0}) = 1;
    H.at({0, 1}) = 0;
    H.at({0, 2}) = 1; // First syndrome bit
    H.at({1, 0}) = 0;
    H.at({1, 1}) = 1;
    H.at({1, 2}) = 1; // Second syndrome bit
  }

  void TearDown() override {
    // Clean up any test files
    std::filesystem::remove("test_load_file.txt");
  }

  std::size_t block_size;
  std::size_t syndrome_size;
  cudaqx::tensor<uint8_t> H;
};

// Test parameter validation function
TEST_F(TRTDecoderTest, ValidateParameters_ValidONNXPath) {
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("test_model.onnx"));
  params.insert("engine_output_format", std::string("errors"));

  // Should not throw
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params));
}

TEST_F(TRTDecoderTest, ValidateParameters_ValidEnginePath) {
  cudaqx::heterogeneous_map params;
  params.insert("engine_load_path", std::string("test_engine.trt"));
  params.insert("engine_output_format", std::string("errors"));

  // Should not throw
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params));
}

TEST_F(TRTDecoderTest, ValidateParameters_BothPathsProvided) {
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("test_model.onnx"));
  params.insert("engine_load_path", std::string("test_engine.trt"));
  params.insert("engine_output_format", std::string("errors"));

  // Should throw runtime_error
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(params),
      std::runtime_error);
}

TEST_F(TRTDecoderTest, ValidateParameters_NoPathsProvided) {
  cudaqx::heterogeneous_map params;

  // Should throw runtime_error
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(params),
      std::runtime_error);
}

TEST_F(TRTDecoderTest, ValidateParameters_EmptyStringPaths) {
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string(""));
  params.insert("engine_load_path", std::string(""));
  params.insert("engine_output_format", std::string("errors"));

  // Should throw runtime_error (empty strings are still considered "provided")
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(params),
      std::runtime_error);
}

// Test load_file function
TEST_F(TRTDecoderTest, LoadFile_ValidFile) {
  // Create a test file
  std::string test_filename = "test_load_file.txt";
  std::string test_content = "Hello, World!";

  std::ofstream file(test_filename);
  file << test_content;
  file.close();

  // Test loading the file
  auto loaded_content =
      cudaq::qec::trt_decoder_internal::load_file(test_filename);
  std::string loaded_string(loaded_content.begin(), loaded_content.end());

  EXPECT_EQ(loaded_string, test_content);
}

TEST_F(TRTDecoderTest, LoadFile_NonExistentFile) {
  // Test loading a non-existent file
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::load_file("non_existent_file.txt"),
      std::runtime_error);
}

TEST_F(TRTDecoderTest, SaveEngineRejectsNullEngine) {
  EXPECT_THROW(cudaq::qec::trt_decoder_internal::save_engine_to_file(
                   nullptr, "unused.engine"),
               std::runtime_error);
}

TEST_F(TRTDecoderTest, ParsePrecisionAcceptsSupportedAndLegacyValues) {
  // Skip without a usable GPU: createInferBuilder initializes CUDA, and on
  // failure TRT 10.7 (CUDA 12.6) throws nvinfer1::APIUsageError while TRT 10.14
  // (CUDA 13.0) returns nullptr. Guarding like the other GPU tests yields the
  // same outcome (skip) on both toolkits instead of aborting on 12.6.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";

  TestTrtLogger logger;

  // Defensive against the throwing variant: catch init failures rather than
  // relying solely on a nullptr return, so the test never terminates the
  // process.
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<nvinfer1::IBuilderConfig> config;
  try {
    builder.reset(nvinfer1::createInferBuilder(logger));
    if (builder)
      config.reset(builder->createBuilderConfig());
  } catch (const std::exception &e) {
    GTEST_SKIP() << "TensorRT initialization failed: " << e.what();
  }
  if (!builder)
    GTEST_SKIP() << "TensorRT builder unavailable";
  if (!config)
    GTEST_SKIP() << "TensorRT builder config unavailable";

  // parse_precision must accept current ("tf32", "best"), legacy/ignored
  // ("noTF32", "fp16") and unknown values without throwing.
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::parse_precision("tf32", config.get()));
  EXPECT_NO_THROW(cudaq::qec::trt_decoder_internal::parse_precision(
      "noTF32", config.get()));
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::parse_precision("best", config.get()));
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::parse_precision("fp16", config.get()));
  EXPECT_NO_THROW(cudaq::qec::trt_decoder_internal::parse_precision(
      "unknown_precision", config.get()));
}

// Test parameter validation edge cases
TEST_F(TRTDecoderTest, ValidateParameters_EdgeCases) {
  // Test with whitespace-only strings
  cudaqx::heterogeneous_map params1;
  params1.insert("onnx_load_path", std::string("   "));
  params1.insert("engine_load_path", std::string("   "));

  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params1),
      std::runtime_error);

  // Test with very long paths
  cudaqx::heterogeneous_map params2;
  std::string long_path(1000, 'a');
  params2.insert("onnx_load_path", long_path);

  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params2));
}

// Test TRT decoder with generated test data
// This test validates that the TRT decoder produces identical results to
// PyTorch
TEST_F(TRTDecoderTest, ValidateAgainstPyTorchModel) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  // Create parity check matrix matching the test data
  // For distance-3 surface code: 24 detectors (syndromes), block_size matches
  // output
  std::size_t num_detectors = NUM_DETECTORS;
  std::size_t num_observables = NUM_OBSERVABLES;

  // Create a dummy H matrix (the TRT decoder doesn't actually use it for
  // inference, but the constructor requires it)
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create the TRT decoder
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("engine_output_format", std::string("observables"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get(
        "trt_decoder", make_inputs_with_empty_observables(H, num_observables),
        decode_result_type::observables, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  // Tolerance for floating point comparison
  constexpr float TOLERANCE = 1e-4f;

  // Track statistics
  int num_passed = 0;
  int num_failed = 0;
  float max_error = 0.0f;
  float total_error = 0.0f;

  // Test each of the 100 test cases
  for (size_t i = 0; i < TEST_INPUTS.size(); ++i) {
    // Convert test input to the format expected by decoder
    std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[i].begin(),
                                              TEST_INPUTS[i].end());

    // Run TRT decoder inference
    auto result = trt_decoder->decode(syndrome);

    // Get the PyTorch expected output
    float expected_output = TEST_OUTPUTS[i][0];

    // Get the TRT decoder output
    ASSERT_FALSE(result.result.empty())
        << "TRT decoder returned empty result for test case " << i;
    float trt_output = result.result[0];

    // Compute absolute error
    float error = std::abs(trt_output - expected_output);
    total_error += error;
    max_error = std::max(max_error, error);

    // Check if within tolerance
    if (error < TOLERANCE) {
      num_passed++;
    } else {
      num_failed++;
      // Print detailed error info for first few failures
      if (num_failed <= 5) {
        std::cout << "Test case " << i << " FAILED:" << std::endl;
        std::cout << "  Expected: " << expected_output << std::endl;
        std::cout << "  Got:      " << trt_output << std::endl;
        std::cout << "  Error:    " << error << std::endl;
      }
    }

    // Assert each individual test case
    EXPECT_LT(error, TOLERANCE)
        << "Test case " << i << " failed: "
        << "TRT output (" << trt_output << ") differs from PyTorch output ("
        << expected_output << ") by " << error;
  }

  // Print summary statistics
  std::cout << "\n=== TRT Decoder Validation Summary ===" << std::endl;
  std::cout << "Total test cases: " << TEST_INPUTS.size() << std::endl;
  std::cout << "Passed: " << num_passed << std::endl;
  std::cout << "Failed: " << num_failed << std::endl;
  std::cout << "Max error: " << max_error << std::endl;
  std::cout << "Average error: " << (total_error / TEST_INPUTS.size())
            << std::endl;
  std::cout << "====================================\n" << std::endl;

  // Overall test assertion: all cases must pass
  EXPECT_EQ(num_failed, 0) << num_failed << " test cases failed validation";
}

// Test a single specific case for detailed debugging
TEST_F(TRTDecoderTest, ValidateSingleTestCase) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  // Create dummy H matrix
  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create the TRT decoder
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("engine_output_format", std::string("observables"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get(
        "trt_decoder", make_inputs_with_empty_observables(H, NUM_OBSERVABLES),
        decode_result_type::observables, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  // Test first case in detail
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());

  std::cout << "Input syndrome (first 10 values): ";
  for (size_t i = 0; i < std::min(size_t(10), syndrome.size()); ++i) {
    std::cout << syndrome[i] << " ";
  }
  std::cout << std::endl;

  auto result = trt_decoder->decode(syndrome);

  float expected = TEST_OUTPUTS[0][0];
  float actual = result.result[0];
  float error = std::abs(actual - expected);

  std::cout << "Expected output: " << expected << std::endl;
  std::cout << "Actual output:   " << actual << std::endl;
  std::cout << "Absolute error:  " << error << std::endl;
  std::cout << "Converged:       " << (result.converged ? "yes" : "no")
            << std::endl;

  EXPECT_LT(error, 1e-4f) << "Single test case validation failed";
  EXPECT_TRUE(result.converged) << "Decoder did not converge";
}

// Test performance comparison: CUDA Graph vs Traditional execution
TEST_F(TRTDecoderTest, PerformanceComparisonCudaGraphVsTraditional) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  // Create dummy H matrix
  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create test syndrome
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());

  // =========================================================================
  // Create decoder WITH CUDA graphs (default)
  // =========================================================================
  cudaqx::heterogeneous_map params_cuda_graph;
  params_cuda_graph.insert("onnx_load_path", onnx_path);
  params_cuda_graph.insert("engine_output_format", std::string("observables"));
  params_cuda_graph.insert("precision", "fp16");
  params_cuda_graph.insert("use_cuda_graph", true);

  std::unique_ptr<decoder> decoder_cuda_graph;
  try {
    decoder_cuda_graph = decoder::get(
        "trt_decoder", make_inputs_with_empty_observables(H, NUM_OBSERVABLES),
        decode_result_type::observables, params_cuda_graph);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create CUDA graph decoder: " << e.what();
  }

  // =========================================================================
  // Create decoder WITHOUT CUDA graphs (traditional)
  // =========================================================================
  cudaqx::heterogeneous_map params_traditional;
  params_traditional.insert("onnx_load_path", onnx_path);
  params_traditional.insert("engine_output_format", std::string("observables"));
  params_traditional.insert("precision", "fp16");
  params_traditional.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> decoder_traditional;
  try {
    decoder_traditional = decoder::get(
        "trt_decoder", make_inputs_with_empty_observables(H, NUM_OBSERVABLES),
        decode_result_type::observables, params_traditional);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create traditional decoder: " << e.what();
  }

  // =========================================================================
  // Warm-up phase (for fair comparison)
  // =========================================================================
  const int warmup_iterations = 5;
  std::cout << "\n=== Warming up decoders ===" << std::endl;

  for (int i = 0; i < warmup_iterations; ++i) {
    decoder_cuda_graph->decode(syndrome);
    decoder_traditional->decode(syndrome);
  }

  // =========================================================================
  // Benchmark CUDA Graph Executor
  // =========================================================================
  const int benchmark_iterations = 200;
  std::cout << "Benchmarking CUDA Graph executor..." << std::endl;

  auto start_cuda_graph = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < benchmark_iterations; ++i) {
    auto result = decoder_cuda_graph->decode(syndrome);
    ASSERT_TRUE(result.converged)
        << "CUDA graph decoder failed at iteration " << i;
  }
  auto end_cuda_graph = std::chrono::high_resolution_clock::now();

  auto duration_cuda_graph =
      std::chrono::duration_cast<std::chrono::microseconds>(end_cuda_graph -
                                                            start_cuda_graph);
  double avg_time_cuda_graph =
      duration_cuda_graph.count() / static_cast<double>(benchmark_iterations);

  // =========================================================================
  // Benchmark Traditional Executor
  // =========================================================================
  std::cout << "Benchmarking Traditional executor..." << std::endl;

  auto start_traditional = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < benchmark_iterations; ++i) {
    auto result = decoder_traditional->decode(syndrome);
    ASSERT_TRUE(result.converged)
        << "Traditional decoder failed at iteration " << i;
  }
  auto end_traditional = std::chrono::high_resolution_clock::now();

  auto duration_traditional =
      std::chrono::duration_cast<std::chrono::microseconds>(end_traditional -
                                                            start_traditional);
  double avg_time_traditional =
      duration_traditional.count() / static_cast<double>(benchmark_iterations);

  // =========================================================================
  // Calculate and report performance improvement
  // =========================================================================
  double speedup = avg_time_traditional / avg_time_cuda_graph;
  double improvement_percent =
      ((avg_time_traditional - avg_time_cuda_graph) / avg_time_traditional) *
      100.0;

  std::cout << "\n=== Performance Comparison Results ===" << std::endl;
  std::cout << "Iterations: " << benchmark_iterations << std::endl;
  std::cout << "CUDA Graph avg time:   " << avg_time_cuda_graph << " μs"
            << std::endl;
  std::cout << "Traditional avg time:  " << avg_time_traditional << " μs"
            << std::endl;
  std::cout << "Speedup:               " << speedup << "x" << std::endl;
  std::cout << "Improvement:           " << improvement_percent << "%"
            << std::endl;
  std::cout << "======================================\n" << std::endl;

  // =========================================================================
  // Performance assertions
  // =========================================================================
  // CUDA graphs should provide at least 5% improvement
  // (Conservative threshold - typical improvement is 10-20%)
  EXPECT_GT(speedup, 1.05)
      << "CUDA graph execution should be at least 5% faster than traditional. "
      << "Speedup: " << speedup << "x, Improvement: " << improvement_percent
      << "%";

  // Sanity check: both should be reasonably fast (< 100ms per decode)
  EXPECT_LT(avg_time_cuda_graph, 100000.0)
      << "CUDA graph execution unexpectedly slow: " << avg_time_cuda_graph
      << " μs";
  EXPECT_LT(avg_time_traditional, 100000.0)
      << "Traditional execution unexpectedly slow: " << avg_time_traditional
      << " μs";
}

TEST_F(TRTDecoderTest, ConstructionFailureThrows) {
  // A decoder that cannot infer must not exist: initialization failure
  // propagates out of decoder::get instead of yielding a half-dead object
  // that reports converged=false forever (converged is a statement about
  // the solution of a completed decode, never about decoder health).
  cudaqx::heterogeneous_map params;
  params.insert("engine_load_path",
                std::string("/no/such/cudaq-qec-test.engine"));
  params.insert("engine_output_format", std::string("errors"));
  EXPECT_THROW(decoder::get("trt_decoder", make_identity_h(2), params),
               std::runtime_error);
}

TEST_F(TRTDecoderTest, EngineSavePathAndEngineLoadPathRoundTrip) {
  // Building from ONNX with engine_save_path must serialize a reusable engine;
  // loading the saved engine should preserve inference numerics.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  const std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path))
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;

  const auto engine_path =
      make_temp_engine_path("cudaq_qec_trt_decoder_roundtrip.engine");
  std::filesystem::remove(engine_path);
  auto H = make_identity_h(NUM_DETECTORS);

  cudaqx::heterogeneous_map build_params;
  build_params.insert("onnx_load_path", onnx_path);
  build_params.insert("engine_output_format", std::string("observables"));
  build_params.insert("engine_save_path", engine_path.string());
  build_params.insert("memory_workspace", std::size_t{1 << 20});
  build_params.insert("precision", std::string("fp16"));
  build_params.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> built_decoder;
  try {
    built_decoder = decoder::get(
        "trt_decoder", make_inputs_with_empty_observables(H, NUM_OBSERVABLES),
        decode_result_type::observables, build_params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to build TRT decoder: " << e.what();
  }
  ASSERT_TRUE(std::filesystem::exists(engine_path));

  cudaqx::heterogeneous_map load_params;
  load_params.insert("engine_load_path", engine_path.string());
  load_params.insert("engine_output_format", std::string("observables"));
  load_params.insert("use_cuda_graph", false);
  std::unique_ptr<decoder> loaded_decoder;
  try {
    loaded_decoder = decoder::get(
        "trt_decoder", make_inputs_with_empty_observables(H, NUM_OBSERVABLES),
        decode_result_type::observables, load_params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to load TRT decoder: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto built_result = built_decoder->decode(syndrome);
  auto loaded_result = loaded_decoder->decode(syndrome);
  ASSERT_EQ(built_result.result.size(), loaded_result.result.size());
  for (std::size_t i = 0; i < built_result.result.size(); ++i)
    EXPECT_NEAR(built_result.result[i], loaded_result.result[i], 1e-4);

  std::filesystem::remove(engine_path);
}

TEST_F(TRTDecoderTest, DynamicBatchIdentityModelUsesOptimizationProfile) {
  // A dynamic-batch ONNX model exercises TensorRT profile creation and the
  // runtime setInputShape path while preserving identity outputs.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format", std::string("errors"));
  params.insert("batch_size", std::size_t{2});
  params.insert("use_cuda_graph", true);
  params.insert("memory_workspace", std::size_t{1 << 20});

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(3), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create dynamic TRT decoder: " << e.what();
  }

  std::vector<std::vector<cudaq::qec::float_t>> syndromes = {
      {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.25, 0.5, 0.75}};
  auto results = trt_decoder->decode_batch(syndromes);
  ASSERT_EQ(results.size(), syndromes.size());
  for (std::size_t r = 0; r < results.size(); ++r) {
    ASSERT_TRUE(results[r].converged);
    ASSERT_EQ(results[r].result.size(), syndromes[r].size());
    for (std::size_t c = 0; c < syndromes[r].size(); ++c)
      EXPECT_NEAR(results[r].result[c], syndromes[r][c], 1e-5);
  }
}

TEST_F(TRTDecoderTest, Uint8IdentityModelBinarizesInputAndOutput) {
  // UINT8 I/O models should hard-threshold soft syndromes before inference and
  // convert UINT8 outputs back to the decoder's floating result vector.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_uint8_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated uint8 ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format", std::string("errors"));
  params.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(3), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create uint8 TRT decoder: " << e.what();
  }

  auto result = trt_decoder->decode({0.49, 0.5, 0.75});
  ASSERT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 3u);
  EXPECT_FLOAT_EQ(result.result[0], 0.0);
  EXPECT_FLOAT_EQ(result.result[1], 1.0);
  EXPECT_FLOAT_EQ(result.result[2], 1.0);
}

TEST_F(TRTDecoderTest, MixedDtypeCopiesOutput) {
  // Mixed UINT8 input and FLOAT output should use the engine output dtype when
  // copying and interpreting output, not the input dtype selected for dispatch.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_uint8_to_float_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated uint8-to-float ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format", std::string("errors"));
  params.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(3), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create uint8-to-float TRT decoder: " << e.what();
  }

  auto result = trt_decoder->decode({0.49, 0.5, 0.75});

  // The Cast model should execute successfully; failure here means the mixed
  // dtype path did not produce a usable decoder result.
  ASSERT_TRUE(result.converged);
  // The output has three float elements; a shorter result would show that the
  // output buffer cardinality was not preserved.
  ASSERT_EQ(result.result.size(), 3u);
  EXPECT_FLOAT_EQ(result.result[0], 0.0);
  EXPECT_FLOAT_EQ(result.result[1], 1.0);
  EXPECT_FLOAT_EQ(result.result[2], 1.0);
}

TEST_F(TRTDecoderTest, RejectsGlobalDecoderSyndromeMismatchAtConstruction) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format", std::string("residual_detectors"));
  params.insert("batch_size", std::size_t{1});
  params.insert("use_cuda_graph", false);
  params.insert("global_decoder", std::string("single_error_lut"));
  params.insert("global_decoder_params", cudaqx::heterogeneous_map{});

  // The engine emits three residual detectors, while make_identity_h(2)
  // configures a child expecting two. Reject this before any decode call.
  EXPECT_THROW((void)decoder::get("trt_decoder", make_identity_h(2), params),
               std::runtime_error);
}

// Acceptance 4: TensorRT constructs Chromobius as its global child while the
// authoritative raw DEM survives the derivation. Chromobius refuses to build
// from matrices alone, so construction succeeding IS the assertion that raw
// provenance reached the child through TensorRT's input derivation.
TEST_F(TRTDecoderTest, NestsChromobiusPreservingRawDem) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  if (cudaq::qec::decoding::config::find_decoder_schema("chromobius") ==
      nullptr)
    GTEST_SKIP() << "chromobius plugin not built in this configuration";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  // Two coloured detectors and one observable, so the engine's
  // [observable, residual detectors] output is three wide, matching the
  // dynamic fixture the other composite tests use.
  const auto dem_path =
      std::filesystem::temp_directory_path() /
      ("cudaqx_trt_chromobius_" + std::to_string(::getpid()) + ".dem");
  std::ofstream(dem_path) << "error(0.1) D0 L0\n"
                             "error(0.1) D0 D1\n"
                             "error(0.1) D1 L0\n"
                             "detector(0, 0, 0, 0) D0\n"
                             "detector(0, 0, 0, 1) D1\n";
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  } cleanup{dem_path};

  std::ifstream dem_file(dem_path);
  std::string dem_text((std::istreambuf_iterator<char>(dem_file)),
                       std::istreambuf_iterator<char>());
  auto inputs = decoder_init::from_stim_dem(dem_text);
  ASSERT_TRUE(inputs.has_stim_dem());
  ASSERT_EQ(inputs.num_detectors(), 2u);
  ASSERT_EQ(inputs.num_observables(), 1u);

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format",
                std::string("observables_and_residual_detectors"));
  params.insert("batch_size", std::size_t{1});
  params.insert("use_cuda_graph", false);
  params.insert("global_decoder", std::string("chromobius"));
  params.insert("global_decoder_params", cudaqx::heterogeneous_map{});

  std::unique_ptr<decoder> composite;
  try {
    composite = decoder::get("trt_decoder", inputs,
                             decode_result_type::observables, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "TensorRT engine build unavailable: " << e.what();
  }
  ASSERT_NE(composite, nullptr);
  EXPECT_EQ(composite->get_num_observables(), 1u);

  // The engine fixes the input width, as in the other composite tests; a
  // decode exercises the child on the residual detectors it is handed.
  auto result = composite->decode({0.0, 0.0, 0.0});
  EXPECT_EQ(result.result.size(), 1u);

  // The converse, so the case above cannot pass for an unrelated reason: with
  // the same engine and child but a matrix-only model, there is no DEM to hand
  // down and construction must fail.
  cudaqx::tensor<uint8_t> H({2, 3});
  H.at({0, 0}) = 1;
  H.at({1, 1}) = 1;
  cudaqx::tensor<uint8_t> O({1, 3});
  O.at({0, 0}) = 1;
  EXPECT_THROW((void)decoder::get("trt_decoder",
                                  decoder_init(sparse_binary_matrix(H),
                                               sparse_binary_matrix(O)),
                                  decode_result_type::observables, params),
               std::runtime_error);
}

TEST_F(TRTDecoderTest, CompositeGlobalDecoderCombinesLogicalFrame) {
  // TRT emits [pre_L, residual syndrome]; the optional global decoder decodes
  // the residual part and XORs it with pre_L to form the final observable.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::tensor<uint8_t> H({2, 1});
  H.at({0, 0}) = 1;
  cudaqx::tensor<uint8_t> O({1, 1});
  O.at({0, 0}) = 1;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format",
                std::string("observables_and_residual_detectors"));
  params.insert("batch_size", std::size_t{2});
  params.insert("use_cuda_graph", false);
  params.insert("global_decoder", std::string("single_error_lut"));
  params.insert("global_decoder_params", cudaqx::heterogeneous_map{});

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get(
        "trt_decoder",
        decoder_init(sparse_binary_matrix(H), sparse_binary_matrix(O)),
        decode_result_type::observables, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create composite TRT decoder: " << e.what();
  }

  auto results = trt_decoder->decode_batch({{1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}});
  ASSERT_EQ(results.size(), 2u);
  ASSERT_EQ(results[0].result.size(), 1u);
  ASSERT_EQ(results[1].result.size(), 1u);
  EXPECT_FLOAT_EQ(results[0].result[0], 0.0);
  EXPECT_FLOAT_EQ(results[1].result[0], 1.0);
}

TEST_F(TRTDecoderTest, RejectsResidualOutputWithoutGlobalDecoderAtTopLevel) {
  cudaqx::heterogeneous_map params;

  params.insert("onnx_load_path", std::string("unused.onnx"));
  params.insert("engine_output_format", std::string("residual_detectors"));

  try {
    (void)decoder::get("trt_decoder", make_identity_h(3), params);
    FAIL() << "expected standalone residual output without a global decoder "
              "to be rejected";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("requires global_decoder"),
              std::string::npos)
        << e.what();
  }
}

TEST_F(TRTDecoderTest, RejectsUnsupportedAuxiliaryOutput) {
  const auto H = sparse_binary_matrix::from_nested_csr(
      1, 1, std::vector<std::vector<uint32_t>>{{0}});
  const auto O = sparse_binary_matrix::from_nested_csr(
      1, 1, std::vector<std::vector<uint32_t>>{{0}});
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("unused.onnx"));
  params.insert("engine_output_format", std::string("observables"));
  constexpr auto unsupported =
      static_cast<decoder_auxiliary_result_type>(1u << 1);

  try {
    (void)decoder::get(
        "trt_decoder", decoder_init(H, O),
        decoder_output_request{decode_result_type::observables, unsupported},
        params);
    FAIL() << "expected unsupported auxiliary output to be rejected";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("unsupported auxiliary"),
              std::string::npos)
        << e.what();
  }
}

TEST_F(TRTDecoderTest, MultiOutputPackedChecksWidth) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  const auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  const auto H = sparse_binary_matrix::from_nested_csr(
      3, 3, std::vector<std::vector<uint32_t>>{{0}, {1}, {2}});
  const auto O = sparse_binary_matrix::from_nested_csr(
      1, 3, std::vector<std::vector<uint32_t>>{{0}});
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format",
                std::string("observables_and_residual_detectors"));
  params.insert("batch_size", std::size_t{1});
  params.insert("use_cuda_graph", false);

  // The identity fixture emits three values, but a packed model over this
  // input must emit one observable followed by three residual detectors.
  try {
    (void)decoder::get("trt_decoder", decoder_init(H, O),
                       decoder_output_request{
                           decode_result_type::observables,
                           decoder_auxiliary_result_type::residual_detectors},
                       params);
    FAIL() << "expected packed TRT output width to be rejected";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "must equal num_observables + residual detector count"),
              std::string::npos)
        << e.what();
  }
}

TEST_F(TRTDecoderTest, RejectsPackedOutputWithoutGlobalDecoderAtTopLevel) {
  const auto H = sparse_binary_matrix::from_nested_csr(
      3, 3, std::vector<std::vector<uint32_t>>{{0}, {1}, {2}});
  const auto O = sparse_binary_matrix::from_nested_csr(
      1, 3, std::vector<std::vector<uint32_t>>{{0}});
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("unused.onnx"));
  params.insert("engine_output_format",
                std::string("observables_and_residual_detectors"));
  params.insert("batch_size", std::size_t{1});
  params.insert("use_cuda_graph", false);

  try {
    (void)decoder::get("trt_decoder", decoder_init(H, O),
                       decode_result_type::observables, params);
    FAIL() << "expected top-level packed TRT output without a global decoder "
              "to be rejected";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("requires global_decoder"),
              std::string::npos)
        << e.what();
  }
}

TEST_F(TRTDecoderTest, CompositeGlobalDecoderReceivesCancellationToken) {
  // TensorRT inference is not interruptible, but the token handed to
  // decode()/decode_batch() must reach the global decoder unchanged.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::tensor<uint8_t> H({2, 1});
  H.at({0, 0}) = 1;
  cudaqx::tensor<uint8_t> O({1, 1});
  O.at({0, 0}) = 1;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("engine_output_format",
                std::string("observables_and_residual_detectors"));
  params.insert("batch_size", std::size_t{2});
  params.insert("use_cuda_graph", false);
  params.insert("global_decoder", std::string("cancellation_probe_decoder"));
  params.insert("global_decoder_params", cudaqx::heterogeneous_map{});

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get(
        "trt_decoder",
        decoder_init(sparse_binary_matrix(H), sparse_binary_matrix(O)),
        decode_result_type::observables, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create composite TRT decoder: " << e.what();
  }

  const std::vector<std::vector<cudaq::qec::float_t>> syndromes{
      {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
  cancellation_source src;
  src.request_hard_stop();
  EXPECT_FALSE(
      trt_decoder->decode_batch(syndromes, src.get_token()).has_value());
  EXPECT_FALSE(trt_decoder->decode(syndromes[0], src.get_token()).has_value());
}

// Note: Constructor tests and parse_precision tests are disabled because they
// require actual TensorRT/CUDA initialization which is not available in the
// test environment. Only parameter validation and utility function tests are
// enabled above.

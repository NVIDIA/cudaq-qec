/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "realtime_decoding.h"
#include "../hardware_guards.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

// Optional syndrome capture callback for --save_syndrome feature
namespace {
using SyndromeCaptureCallback = void (*)(const uint8_t *, size_t);
SyndromeCaptureCallback g_syndrome_capture_callback = nullptr;
} // namespace

std::vector<std::unique_ptr<cudaq::qec::decoder>> g_decoders;

// Helper to pack syndrome bits into bytes (8 bits per byte, MSB first for
// readability)
static std::vector<uint8_t> pack_syndrome_bits(const uint8_t *syndromes,
                                               size_t length) {
  size_t num_bytes = (length + 7) / 8; // Round up
  std::vector<uint8_t> packed(num_bytes, 0);

  for (size_t i = 0; i < length; i++) {
    if (syndromes[i]) {
      size_t byte_idx = i / 8;
      size_t bit_idx = 7 - (i % 8); // MSB first
      packed[byte_idx] |= (1 << bit_idx);
    }
  }

  return packed;
}

namespace cudaq::qec::decoding::host {

/// Read the DEM a decoder entry names.
std::string read_stim_dem(const std::filesystem::path &dem_path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(dem_path, ec))
    throw std::runtime_error(fmt::format(
        "stim_dem_path is not a readable file: {}", dem_path.string()));
  std::ifstream file(dem_path);
  if (!file)
    throw std::runtime_error(fmt::format(
        "stim_dem_path could not be opened: {}", dem_path.string()));
  try {
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  } catch (const std::exception &error) {
    throw std::runtime_error(
        fmt::format("stim_dem_path could not be read: {} ({})",
                    dem_path.string(), error.what()));
  }
}

cudaqx::heterogeneous_map prepare_decoder_params(
    const cudaq::qec::decoding::config::decoder_config &decoder_config) {
  auto params = decoder_config.decoder_custom_args_to_heterogeneous_map();
  // Placement is common factory policy and is consumed by decoder::get().
  if (decoder_config.cuda_device_id.has_value())
    params.insert("cuda_device_id", decoder_config.cuda_device_id.value());
  return params;
}

namespace {

/// Build D in GF(2)-canonical form from the flat -1-terminated encoding. A
/// repeated index in a row cancels under the realtime detector XOR, so
/// canonicalizing here puts that rule in the model rather than leaving each
/// consumer to interpret duplicates its own way.
cudaq::qec::sparse_binary_matrix
canonical_measurement_to_detectors(const std::vector<std::int64_t> &d_sparse) {
  std::vector<std::vector<std::uint32_t>> detector_rows;
  std::vector<std::uint32_t> row;
  // -1 terminates a row; every other value must be a measurement index that,
  // with its exclusive upper bound, fits the sparse matrix index type.
  // Narrowing an out-of-range value would silently alias it onto a real
  // measurement.
  constexpr std::int64_t max_measurement_index =
      static_cast<std::int64_t>(
          std::numeric_limits<
              cudaq::qec::sparse_binary_matrix::index_type>::max()) -
      1;
  for (std::int64_t entry : d_sparse) {
    if (entry == -1) {
      detector_rows.push_back(std::move(row));
      row.clear();
      continue;
    }
    if (entry < -1 || entry > max_measurement_index)
      throw std::runtime_error(fmt::format(
          "Value in D_sparse vector is out of range: {} (expected -1 as a row "
          "terminator, or a measurement index in [0, {}])",
          entry, max_measurement_index));
    row.push_back(static_cast<std::uint32_t>(entry));
  }
  if (!row.empty())
    detector_rows.push_back(std::move(row));

  // Width is taken before cancellation, so a trailing measurement referenced
  // only by a cancelling pair still counts toward the per-decode width. The
  // bound above makes column + 1 safe.
  std::uint32_t num_measurements = 0;
  for (const auto &r : detector_rows)
    for (auto column : r)
      num_measurements = std::max(num_measurements, column + 1);

  return cudaq::qec::sparse_binary_matrix::from_nested_csr(
             static_cast<std::uint32_t>(detector_rows.size()), num_measurements,
             detector_rows)
      .canonicalize();
}

void validate_sparse_indices(const std::vector<std::int64_t> &sparse,
                             std::uint64_t num_columns, const char *name) {
  for (auto value : sparse)
    if (value < -1 ||
        (value >= 0 && static_cast<std::uint64_t>(value) >= num_columns))
      throw std::runtime_error(
          fmt::format("Value in {} vector is out of range: {}", name, value));
}

void validate_detector_rows(const std::vector<std::int64_t> &d_sparse,
                            std::int64_t id) {
  if (!d_sparse.empty() && d_sparse.front() == -1)
    throw std::runtime_error(
        fmt::format("D_sparse row is empty for decoder {}", id));
  for (std::size_t i = 0; i + 1 < d_sparse.size(); ++i)
    if (d_sparse.at(i) == -1 && d_sparse.at(i + 1) == -1)
      throw std::runtime_error(
          fmt::format("D_sparse row is empty for decoder {}", id));
}

} // namespace

cudaq::qec::decoder_init resolve_decoder_init(
    const cudaq::qec::decoding::config::decoder_config &config_in,
    const std::filesystem::path &base_dir) {
  if (!config_in.stim_dem_path.empty() && config_in.dem_chunks.has_value())
    throw std::runtime_error(
        "dem_chunks and stim_dem_path must not both be set for decoder " +
        std::to_string(config_in.id) +
        "; each describes the whole DEM on its own");

  // A chunk-form configuration names its model one round at a time. Expand it
  // into the same flat construction contract used by every matrix decoder,
  // including the per-fault priors carried only by the closed DEM.
  auto expanded_config = config_in;
  cudaq::qec::decoding::config::expand_dem_chunks(expanded_config);
  const auto &decoder_config = expanded_config;
  const auto error_rate_vec = decoder_config.effective_error_rate_vec();

  if (decoder_config.D_sparse.empty())
    throw std::runtime_error(
        "D_sparse must be provided in decoder configuration");
  validate_detector_rows(decoder_config.D_sparse, decoder_config.id);
  auto D = canonical_measurement_to_detectors(decoder_config.D_sparse);

  const bool dem_source = !decoder_config.stim_dem_path.empty();
  if (dem_source) {
    // The matrix keys are a competing representation of the same model, not
    // assertions about it, so supplying both leaves no single authority.
    if (!decoder_config.H_sparse.empty() || !decoder_config.O_sparse.empty() ||
        !error_rate_vec.empty())
      throw std::runtime_error(
          "stim_dem_path is mutually exclusive with H_sparse, O_sparse and "
          "error_rate_vec; supply exactly one model source");

    // Absolute, not merely normalized: base_dir may itself be relative (a
    // server started with `configs/decoders.yml`), and a stored relative path
    // stops resolving once the working directory changes.
    std::filesystem::path dem_path(decoder_config.stim_dem_path);
    if (dem_path.is_relative())
      dem_path =
          std::filesystem::absolute(base_dir / dem_path).lexically_normal();
    std::string dem_text = read_stim_dem(dem_path);

    // The model is identified by path, so editing a DEM in place leaves the
    // configuration byte-identical and a reload keeps serving the old model.
    // Change the path to change the model.

    auto inputs = cudaq::qec::decoder_init::from_stim_dem(std::move(dem_text),
                                                          std::move(D));

    // The DEM defines the detector basis; a supplied syndrome_size is only an
    // assertion about it.
    if (decoder_config.syndrome_size != 0 &&
        decoder_config.syndrome_size != inputs.num_detectors())
      throw std::runtime_error(fmt::format(
          "syndrome_size ({}) does not match the detector count of {} ({})",
          decoder_config.syndrome_size, decoder_config.stim_dem_path,
          inputs.num_detectors()));
    if (decoder_config.block_size != 0 &&
        decoder_config.block_size != inputs.num_error_mechanisms())
      throw std::runtime_error(fmt::format(
          "block_size ({}) does not match the error-mechanism count of {} "
          "({}). The derived value is the column count of the flattened matrix "
          "projection of the DEM, which need not equal a count reported using "
          "a different decomposition.",
          decoder_config.block_size, decoder_config.stim_dem_path,
          inputs.num_error_mechanisms()));
    return inputs;
  }

  // Matrix source. These dimensions are needed to interpret the flat sparse
  // encodings, so they stay required on this branch.
  if (decoder_config.syndrome_size == 0 || decoder_config.block_size == 0)
    throw std::runtime_error(
        "block_size and syndrome_size are required for a matrix decoder model");
  const auto num_H_rows = std::count(decoder_config.H_sparse.begin(),
                                     decoder_config.H_sparse.end(), -1);
  if (static_cast<std::uint64_t>(num_H_rows) != decoder_config.syndrome_size)
    throw std::runtime_error(fmt::format(
        "Number of rows in H_sparse vector is not equal to syndrome_size: {} "
        "!= {}",
        num_H_rows, decoder_config.syndrome_size));
  validate_sparse_indices(decoder_config.H_sparse, decoder_config.block_size,
                          "H_sparse");
  // The realtime path exists to return observable corrections, so a model that
  // supplies no observable mapping cannot serve it. Without this the decoder
  // constructs and silently decodes to a zero-length observable frame.
  if (decoder_config.O_sparse.empty())
    throw std::runtime_error(
        "O_sparse is required: the decoding server constructs every decoder "
        "for observable output, which needs an observable mapping");
  validate_sparse_indices(decoder_config.O_sparse, decoder_config.block_size,
                          "O_sparse");
  if (!error_rate_vec.empty() &&
      error_rate_vec.size() != decoder_config.block_size)
    throw std::runtime_error(
        fmt::format("error_rate_vec size is not equal to block_size: {} != {}",
                    error_rate_vec.size(), decoder_config.block_size));
  if (static_cast<std::uint64_t>(D.num_rows()) != decoder_config.syndrome_size)
    throw std::runtime_error(
        fmt::format("Number of rows in D_sparse vector is not equal to "
                    "syndrome_size: {} != {}",
                    D.num_rows(), decoder_config.syndrome_size));

  auto pcm = cudaq::qec::pcm_from_sparse_vec(decoder_config.H_sparse,
                                             decoder_config.syndrome_size,
                                             decoder_config.block_size);
  const auto num_observables = std::count(decoder_config.O_sparse.begin(),
                                          decoder_config.O_sparse.end(), -1);
  auto observable_matrix = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.O_sparse, num_observables, decoder_config.block_size);
  return cudaq::qec::decoder_init(std::move(pcm), std::move(observable_matrix),
                                  error_rate_vec, std::move(D));
}

std::unique_ptr<cudaq::qec::decoder> create_realtime_decoder(
    const cudaq::qec::decoding::config::decoder_config &decoder_config,
    cudaq::qec::decoder_init inputs) {
  if (decoder_config.id < 0 || static_cast<std::uint64_t>(decoder_config.id) >
                                   std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Decoder ID is outside the uint32_t range: " +
                                std::to_string(decoder_config.id));

  auto t0 = std::chrono::high_resolution_clock::now();
  CUDA_QEC_INFO("Creating decoder {} of type {}", decoder_config.id,
                decoder_config.type);

  auto decoder =
      cudaq::qec::get_decoder(decoder_config.type, std::move(inputs),
                              cudaq::qec::decode_result_type::observables,
                              prepare_decoder_params(decoder_config));
  decoder->set_decoder_id(decoder_config.id);

  // Force plugin initialization before the caller publishes the decoder for
  // realtime work. This preserves configure_decoders()'s existing behavior.
  auto t1 = std::chrono::high_resolution_clock::now();
  // Size the dry run from the constructed decoder, not the configuration: a
  // DEM-sourced config leaves syndrome_size unset and derives it from the
  // model.
  std::vector<cudaq::qec::float_t> syndrome(decoder->get_syndrome_size(), 0.0);
  decoder->decode(syndrome);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> creation_duration = t1 - t0;
  std::chrono::duration<double> initialization_duration = t2 - t1;
  CUDA_QEC_INFO(
      "Done initializing decoder {} in {:.6f} seconds (creation: {:.6f}s, "
      "initial decoding dry run: {:.6f}s)",
      decoder_config.id,
      creation_duration.count() + initialization_duration.count(),
      creation_duration.count(), initialization_duration.count());

  return decoder;
}

int configure_decoders(
    cudaq::qec::decoding::config::multi_decoder_config &config,
    const std::filesystem::path &base_dir) {
  CUDA_QEC_INFO("Initializing decoders...");

  const auto &decoder_configs = config.decoders;

  // First validate that the there are no duplicate decoder IDs.
  std::set<int64_t> decoder_ids;
  auto min_decoder_id = std::numeric_limits<int64_t>::max();
  auto max_decoder_id = std::numeric_limits<int64_t>::min();
  for (auto &decoder_config : decoder_configs) {
    if (decoder_ids.count(decoder_config.id) > 0) {
      CUDA_QEC_WARN("Duplicate decoder ID found: {}", decoder_config.id);
      return 1;
    }
    decoder_ids.insert(decoder_config.id);
    min_decoder_id = std::min(min_decoder_id, decoder_config.id);
    max_decoder_id = std::max(max_decoder_id, decoder_config.id);
  }

  // Then check that the maximum decoder ID is less than the number of decoders.
  if (max_decoder_id >= decoder_configs.size()) {
    CUDA_QEC_WARN(
        "Maximum decoder ID is greater than the number of decoders: {} >= {}",
        max_decoder_id, decoder_configs.size());
    return 2;
  }
  if (min_decoder_id < 0) {
    CUDA_QEC_WARN("Minimum decoder ID is less than 0: {}", min_decoder_id);
    return 3;
  }

  // Resolve every model before touching any process state. Resolution reads
  // and parses model files and performs all model validation, so a bad
  // configuration fails here, with the previously active decoders intact.
  // Resolution errors propagate as exceptions rather than becoming a status
  // code, preserving the behavior callers already see for invalid models.
  const auto absolute_base =
      std::filesystem::absolute(base_dir).lexically_normal();

  std::vector<cudaq::qec::decoder_init> resolved;
  resolved.reserve(config.decoders.size());
  // The absolute form of each model path, applied to the caller's
  // configuration only once the whole configuration has been applied. Rewriting
  // as we go would leave a caller's config partly rewritten when a later entry
  // fails to resolve, so a retry against a different base directory would
  // silently keep the first one.
  std::vector<std::string> absolute_model_paths(config.decoders.size());
  for (std::size_t i = 0; i < config.decoders.size(); ++i) {
    const auto &decoder_config = config.decoders[i];
    resolved.push_back(resolve_decoder_init(decoder_config, absolute_base));
    if (!decoder_config.stim_dem_path.empty()) {
      std::filesystem::path model(decoder_config.stim_dem_path);
      absolute_model_paths[i] =
          model.is_relative() ? std::filesystem::absolute(absolute_base / model)
                                    .lexically_normal()
                                    .string()
                              : model.lexically_normal().string();
    }
  }

  // Construction allocates, so replacements are built in place rather than
  // alongside the old set: a constructor failure can still leave the decoder
  // set empty. Overlapping both sets would double peak decoder memory, which
  // is not an acceptable cost here.
  try {
    g_decoders.clear();
    g_decoders.resize(max_decoder_id + 1);
    for (std::size_t i = 0; i < decoder_configs.size(); ++i) {
      g_decoders[decoder_configs[i].id] =
          create_realtime_decoder(decoder_configs[i], std::move(resolved[i]));
    }
  } catch (const std::exception &e) {
    CUDA_QEC_WARN("Error initializing decoders: {}", e.what());
    return 4;
  }

  // The configuration is now in effect. Make its model paths absolute so the
  // copy that gets cached, published and re-read by the session registry
  // resolves without knowing the base directory used here.
  for (std::size_t i = 0; i < config.decoders.size(); ++i)
    if (!absolute_model_paths[i].empty())
      config.decoders[i].stim_dem_path = absolute_model_paths[i];
  return 0;
}

void finalize_decoders() {
  CUDA_QEC_INFO("Finalizing the realtime decoding library.");
  g_decoders.clear();
}

__attribute__((visibility("default"))) void
_set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t)) {
  g_syndrome_capture_callback = callback;
}

__attribute__((visibility("default"))) void (*_get_syndrome_capture_callback())(
    const uint8_t *, size_t) {
  return g_syndrome_capture_callback;
}

void enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                       std::uint64_t syndrome_length, std::uint64_t tag) {
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  if (syndrome_length == 0) {
    throw std::invalid_argument("syndrome_length must be greater than 0");
  }
  if (!syndromes) {
    throw std::invalid_argument("syndromes buffer is null");
  }
  const auto max_syndromes = decoder->get_num_msyn_per_decode();
  if (max_syndromes == 0) {
    throw std::invalid_argument(
        "Decoder has no measurement syndromes configured");
  }
  if (syndrome_length > max_syndromes) {
    throw std::invalid_argument(
        fmt::format("syndrome_length ({}) exceeds configured measurement count "
                    "({})",
                    syndrome_length, max_syndromes));
  }

  const auto capture_syndromes = [&] {
    // --save_syndrome feature: record what is actually submitted for decode.
    if (g_syndrome_capture_callback) {
      auto packed_syndrome = pack_syndrome_bits(syndromes, syndrome_length);
      g_syndrome_capture_callback(packed_syndrome.data(),
                                  packed_syndrome.size());
    }
  };

  // Direct-call path: this caller thread runs the decode, but
  // configure_decoders() constructed every decoder sequentially on one thread,
  // leaving the LAST decoder's device current. Point the thread at this
  // decoder's pinned device before decoding (set-if-different; throws on
  // failure) -- and before the capture callback, so a pin failure cannot
  // record a round that was never decoded.
  cudaq::qec::detail_affinity::pin_decode_device(*decoder);
  capture_syndromes();

  std::vector<uint8_t> syndrome_u8(syndrome_length);
  bool did_decode = false;
  for (std::size_t i = 0; i < syndrome_length; i++) {
    syndrome_u8[i] = syndromes[i];
  }
  std::chrono::duration<double> duration{};
  auto t0 = std::chrono::high_resolution_clock::now();
  did_decode =
      decoder->enqueue_syndrome(syndrome_u8.data(), syndrome_u8.size());
  auto t1 = std::chrono::high_resolution_clock::now();
  duration = t1 - t0;

  // Consider demoting this to a lower log level.
  // Also consider logging the syndrome (at a lower log level).
  CUDA_QEC_INFO("[decoder={}][tag={}] enqueue_syndrome took {:.3f} us, "
                "syndrome_length={}, did_decode={}",
                decoder_id, tag, duration.count() * 1e6, syndrome_length,
                did_decode ? 'Y' : 'N');
}

void get_corrections(std::size_t decoder_id, uint8_t *corrections,
                     std::uint64_t correction_length, bool reset) {
  CUDA_QEC_INFO("Entered get_corrections function decoder_id={}, "
                "correction_length={}, reset={}",
                decoder_id, correction_length, reset);
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  const auto num_observables = decoder->get_num_observables();
  if (correction_length == 0) {
    throw std::invalid_argument("correction_length must be greater than 0");
  }
  if (!corrections) {
    throw std::invalid_argument("corrections buffer is null");
  }
  if (correction_length != num_observables) {
    throw std::invalid_argument(
        fmt::format("correction_length ({}) does not match number of "
                    "observables ({})",
                    correction_length, num_observables));
  }

  // clear_corrections may touch device memory in some plugins.
  cudaq::qec::detail_affinity::pin_decode_device(*decoder);
  auto ret = decoder->get_obs_corrections();
  for (std::size_t i = 0; i < correction_length; ++i) {
    corrections[i] = ret[i];
  }
  if (reset)
    decoder->clear_corrections();
}

void reset_decoder(std::size_t decoder_id) {
  CUDA_QEC_INFO("Entered reset_decoder for decoder_id={}", decoder_id);
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }

  cudaq::qec::detail_affinity::pin_decode_device(*decoder);
  decoder->reset_decoder();
}

} // namespace cudaq::qec::decoding::host

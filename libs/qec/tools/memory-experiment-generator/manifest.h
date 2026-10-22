/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::qec::memory_experiment::manifest {

// Tool-local metadata, not a decoding-server configuration or installed SDK.
struct experiment_metadata {
  std::string code_family;
  std::uint64_t distance = 0;
  std::uint64_t rounds = 0;
  std::string basis;
  std::uint64_t logical_qubits = 0;
};

struct layout_metadata {
  std::string convention;
  std::uint64_t regular_measurements = 0;
  std::uint64_t final_measurements = 0;
  std::uint64_t input_rounds = 0;
  std::uint64_t raw_measurements = 0;
  std::uint64_t physical_qubits_per_patch = 0;
};

struct delivery_metadata {
  std::uint64_t batch_rounds = 0;
  std::uint64_t regular_batches = 0;
  std::uint64_t regular_batch_measurements = 0;
  std::uint64_t terminal_extraction_rounds = 0;
  std::uint64_t terminal_batch_measurements = 0;
};

struct model_metadata {
  std::string path;
  std::string sha256;
  std::uint64_t characterization_rounds = 0;
  std::uint64_t validation_rounds = 0;
};

struct normalization_metadata {
  bool decompose_errors = false;
  std::string correlations;
};

struct noise_metadata {
  std::string kind;
  double probability = 0;
};

struct replay_file_metadata {
  std::uint64_t patch = 0;
  std::uint64_t shot = 0;
  std::uint64_t seed = 0;
  std::string path;
  std::string sha256;
  std::string raw_logical;
};

struct replay_metadata {
  std::string format;
  std::uint64_t shots_per_patch = 0;
  std::uint64_t base_seed = 0;
  std::vector<replay_file_metadata> files;
  bool decoded = false;
};

struct provenance_metadata {
  std::string qec_version;
  std::string qec_repository_version;
};

struct validation_metadata {
  bool phase_model_roundtrip = false;
  bool graph_generated = false;
  bool graph_constructed = false;
  bool replayed = false;
};

struct document {
  std::uint64_t schema_version = 0;
  std::string kind;
  std::string status;
  experiment_metadata experiment;
  layout_metadata layout;
  delivery_metadata delivery;
  model_metadata model;
  normalization_metadata normalization;
  noise_metadata noise;
  std::vector<std::vector<std::uint64_t>> final_m2o;
  std::optional<replay_metadata> replay;
  provenance_metadata provenance;
  validation_metadata validation;
};

/// Serialize typed metadata through LLVM YAML, preserving exact probabilities
/// and string-valued logical bit patterns. Inactive optional replay is omitted.
std::string to_yaml(document metadata);

/// Parse exactly one YAML mapping document. Required, unknown and duplicate
/// fields are checked by the same LLVM mapping traits used for serialization.
/// File integrity and model semantics are checked by validate_model_bundle.
document from_yaml(std::string_view yaml);

} // namespace cudaq::qec::memory_experiment::manifest

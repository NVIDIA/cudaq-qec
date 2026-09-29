/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::qec {

/// @brief Authoritative representation from which a decoder model originates.
///
/// Matrix and Stim sources are implemented. This is the entry point for a
/// compact chunked DEM: that source would be added here with a new enumerator
/// plus its typed constructor and accessor, so a decoder that consumes chunks
/// reads them directly instead of the handle first flattening them into
/// matrices. Adding one changes neither the `decoder_init` object layout nor
/// the decoder factory signature.
enum class decoder_model_source : std::uint8_t {
  matrices,
  stim_dem,
};

/// @brief Stable, owning input contract shared by offline and server decoders.
///
/// This is a small immutable value handle. Copies share the same model state;
/// the decoder factory takes the handle by value and the decoder base retains
/// it. Source-specific data is authoritative and the common matrix accessors
/// expose the projection stored when the handle is constructed. Model matrices
/// are stored sparsely instead of composing detector_error_model, whose matrix
/// fields are dense tensors.
class decoder_init {
public:
  /// @brief Construct an H-only matrix model.
  explicit decoder_init(sparse_binary_matrix detector_error_matrix);

  /// Raw Stim DEM text enters through from_stim_dem(), which parses and
  /// projects it. Deleted so the older spelling fails here rather than
  /// through overload resolution somewhere less obvious.
  explicit decoder_init(std::string) = delete;

  /// @brief Construct a materialized matrix model.
  /// @param detector_error_matrix H, with shape detectors x error mechanisms.
  /// @param observable_flips_matrix O, with shape observables x error
  /// mechanisms. Supplying it establishes an observable model; its row count is
  /// retained even when a row has no nonzeros, so a zero-row O is a supplied
  /// model rather than an absent one.
  /// @param error_rates Optional rate per error mechanism.
  /// @param measurement_to_detectors Optional D, with shape detectors x raw
  /// measurements.
  /// @param error_ids Optional correlation ID per error mechanism.
  decoder_init(
      sparse_binary_matrix detector_error_matrix,
      std::optional<sparse_binary_matrix> observable_flips_matrix,
      std::vector<double> error_rates = {},
      std::optional<sparse_binary_matrix> measurement_to_detectors =
          std::nullopt,
      std::optional<std::vector<std::size_t>> error_ids = std::nullopt);

  /// @brief Construct from the existing materialized detector-error model.
  explicit decoder_init(detector_error_model model,
                        std::optional<sparse_binary_matrix>
                            measurement_to_detectors = std::nullopt);

  /// @brief Construct from authoritative raw Stim DEM text.
  ///
  /// Matrix accessors expose the common lossy projection produced by
  /// `dem_from_stim_text`; DEM-native decoders should consume `stim_dem()`.
  static decoder_init
  from_stim_dem(std::string stim_dem_text,
                std::optional<sparse_binary_matrix> measurement_to_detectors =
                    std::nullopt);

  decoder_init(const decoder_init &) noexcept;
  /// @brief Move construction leaves the source valid only for destruction or
  /// assignment.
  decoder_init(decoder_init &&) noexcept;
  decoder_init &operator=(const decoder_init &) noexcept;
  /// @brief Move assignment leaves the source valid only for destruction or
  /// assignment.
  decoder_init &operator=(decoder_init &&) noexcept;
  ~decoder_init();

  /// @brief The authoritative representation. Consumers that only need to
  /// know whether raw DEM text is available should ask %has_stim_dem(); this
  /// discriminator is what a future compact source would extend.
  decoder_model_source source() const noexcept;

  /// @brief Return the stored common H projection.
  const sparse_binary_matrix &detector_error_matrix() const;

  /// @brief Whether this model supplies an observable mapping at all.
  ///
  /// Distinct from `%num_observables() == 0`: a supplied O with zero rows is an
  /// observable model, an H-only input is not. Construction-time validation of
  /// an observable-output request depends on this distinction.
  bool has_observable_model() const noexcept;

  /// @brief Return the stored common O projection.
  /// @throws std::logic_error if this model supplies no observable mapping.
  const sparse_binary_matrix &observable_flips_matrix() const;

  const std::vector<double> &error_rates() const;
  const std::optional<std::vector<std::size_t>> &error_ids() const;

  /// @brief Return D, or nullptr when input syndromes are already detectors.
  const sparse_binary_matrix *measurement_to_detectors() const noexcept;

  /// @brief Return the same inputs without D, for a decoder that is fed
  /// detectors rather than a raw measurement stream. Everything else,
  /// including the authoritative source, is preserved.
  decoder_init decoder_init_without_d() const;

  /// @brief Return the same inputs with H in GF(2)-canonical CSC form.
  ///
  /// Sorts indices within each compressed group and XOR-merges duplicates,
  /// leaving column identity, ordering and dimensions unchanged. O and D are
  /// passed through untouched, and the authoritative source is retained.
  /// Consumers that need a canonical H should ask for it here rather than
  /// rebuilding a matrix-authoritative handle by hand.
  decoder_init canonicalize_H() const;

  bool has_stim_dem() const noexcept;

  /// @throws std::logic_error if the authoritative source is not a Stim DEM.
  const std::string &stim_dem() const;

  /// Dimensions are stored as source metadata so these accessors never need to
  /// request H or O. For matrix sources they intentionally duplicate the O(1)
  /// matrix shape values in preparation for compact source alternatives.
  std::size_t num_detectors() const noexcept;
  std::size_t num_error_mechanisms() const noexcept;
  std::size_t num_observables() const noexcept;

private:
  struct impl;
  static std::shared_ptr<const impl> make_matrix_state(
      decoder_model_source source, sparse_binary_matrix detector_error_matrix,
      std::optional<sparse_binary_matrix> observable_flips_matrix,
      std::vector<double> error_rates,
      std::optional<std::vector<std::size_t>> error_ids,
      std::optional<sparse_binary_matrix> measurement_to_detectors,
      std::optional<std::string> raw_stim_dem = std::nullopt);
  explicit decoder_init(std::shared_ptr<const impl> state);
  std::shared_ptr<const impl> state_;
};

} // namespace cudaq::qec

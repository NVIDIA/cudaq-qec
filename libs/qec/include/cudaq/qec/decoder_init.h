/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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

struct dem_chunks_spec;

/// @brief Authoritative representation from which a decoder model originates.
///
/// A chunked DEM is the compact source: a decoder that consumes chunks reads
/// them through `dem_chunks()`, and the handle carries the closed matrices only
/// when built with `from_dem_chunks_closed()`. A further source is added the
/// same way, with a new enumerator plus its typed constructor and accessor,
/// changing neither the `decoder_init` object layout nor the decoder factory
/// signature.
enum class decoder_model_source : std::uint8_t {
  matrices,
  stim_dem,
  dem_chunks,
};

/// @brief Stable, owning input contract shared by offline and server decoders.
///
/// This is a small immutable value handle. Copies share the same model state;
/// the decoder factory takes the handle by value and the decoder base retains
/// it. Source-specific data is authoritative and the common matrix accessors
/// expose its projection, stored when the handle is constructed. A chunked
/// source built by `from_dem_chunks()` has none (see `has_matrices()`). Model
/// matrices
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

  /// @brief Construct from an authoritative chunked DEM, without its closed
  /// model.
  ///
  /// For a decoder that consumes `dem_chunks()`. The dimensions are those of
  /// `dem_close_all` over the spec's chunks, computed from its phases alone,
  /// and nothing proportional to the shot's error mechanisms is stored. The
  /// matrix accessors throw; a decoder that needs them takes
  /// `from_dem_chunks_closed()` instead. Projecting an error frame onto
  /// observables does not need them: `dem_chunks_to_o_chunked()` reads the
  /// mapping straight out of `dem_chunks()`.
  ///
  /// @param spec Chunked DEM; a repeating phase needs `num_rounds`.
  /// @param measurement_to_detectors Optional D, one row per detector.
  /// @throws std::invalid_argument If the spec does not validate or expand,
  ///   or D has the wrong row count.
  static decoder_init
  from_dem_chunks(dem_chunks_spec spec,
                  std::optional<sparse_binary_matrix> measurement_to_detectors =
                      std::nullopt);

  /// @brief Construct from an authoritative chunked DEM together with its
  /// closed model.
  ///
  /// As `from_dem_chunks()`, and H, O and the error rates are the model
  /// `dem_close_all` closes the chunks to, built sparsely here. Any decoder
  /// accepts this handle; a chunk-native one still reads `dem_chunks()`.
  ///
  /// @throws std::invalid_argument As `from_dem_chunks()`.
  static decoder_init from_dem_chunks_closed(
      dem_chunks_spec spec,
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
  /// @throws std::logic_error if `%has_matrices()` is false.
  const sparse_binary_matrix &detector_error_matrix() const;

  /// @brief Whether this model supplies an observable mapping at all.
  ///
  /// Distinct from `%num_observables() == 0`: a supplied O with zero rows is an
  /// observable model, an H-only input is not. Construction-time validation of
  /// an observable-output request depends on this distinction. A chunked
  /// source has one when its chunks carry observables, matrices or not.
  bool has_observable_model() const noexcept;

  /// @brief Return the stored common O projection.
  /// @throws std::logic_error if this model supplies no observable mapping,
  ///   or `%has_matrices()` is false.
  const sparse_binary_matrix &observable_flips_matrix() const;

  /// @throws std::logic_error if `%has_matrices()` is false.
  const std::vector<double> &error_rates() const;
  const std::optional<std::vector<std::size_t>> &error_ids() const;

  /// @brief Return D, or nullptr when input syndromes are already detectors.
  const sparse_binary_matrix *measurement_to_detectors() const noexcept;

  /// @brief Return the same inputs without D, for a decoder that is fed
  /// detectors rather than a raw measurement stream. Everything else,
  /// including the authoritative source, is preserved.
  decoder_init decoder_init_without_d() const;

  /// @brief Return the same inputs without an observable model, for a decoder
  /// that must report errors rather than observables. Chunks lose their
  /// observable rows and keep their form; a Stim source, whose text names the
  /// observables, becomes a matrix source.
  decoder_init without_observables() const;

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

  bool has_dem_chunks() const noexcept;

  /// @throws std::logic_error if the authoritative source is not chunked.
  const dem_chunks_spec &dem_chunks() const;

  /// @brief Whether H, O and the error rates are available. False only for a
  /// chunked source built by `from_dem_chunks()`, whose matrix accessors
  /// throw std::logic_error.
  bool has_matrices() const noexcept;

  /// Dimensions are stored as source metadata so these accessors never need to
  /// request H or O. For matrix sources they intentionally duplicate the O(1)
  /// matrix shape values in preparation for compact source alternatives.
  std::size_t num_detectors() const noexcept;
  std::size_t num_error_mechanisms() const noexcept;
  std::size_t num_observables() const noexcept;

private:
  struct impl;
  static std::shared_ptr<impl> make_matrix_state(
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

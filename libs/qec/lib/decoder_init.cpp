/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder_init.h"
#include "sparse_dem_from_stim_text.h"
#include "cudaq/qec/extended_dem.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cudaq::qec {

struct decoder_init::impl {
  decoder_model_source source = decoder_model_source::matrices;
  std::size_t num_detectors = 0;
  std::size_t num_error_mechanisms = 0;
  std::size_t num_observables = 0;
  /// False only for a chunked source built without its closed model, which
  /// leaves H, O and rates empty.
  bool has_matrices = true;
  sparse_binary_matrix H;
  /// Absent when the model supplies no observable mapping. A present but
  /// zero-row O is a supplied model, not an absent one.
  std::optional<sparse_binary_matrix> O;
  std::vector<double> rates;
  std::optional<std::vector<std::size_t>> ids;
  std::optional<sparse_binary_matrix> D;
  std::optional<std::string> raw_stim_dem;
  std::shared_ptr<const dem_chunks_spec> chunks;

  const impl &matrices(const char *accessor) const {
    if (!has_matrices)
      throw std::logic_error(
          std::string("decoder_init::") + accessor +
          ": this dem_chunks source was built without its closed model; "
          "construct it with decoder_init::from_dem_chunks_closed()");
    return *this;
  }
};

namespace {

void validate_model(const sparse_binary_matrix &H,
                    const std::optional<sparse_binary_matrix> &O,
                    const std::vector<double> &rates,
                    const std::optional<std::vector<std::size_t>> &ids,
                    const std::optional<sparse_binary_matrix> &D) {
  if (O && O->num_cols() != H.num_cols())
    throw std::invalid_argument(
        "decoder_init: O column count must match H column count");
  if (!rates.empty() && rates.size() != H.num_cols())
    throw std::invalid_argument(
        "decoder_init: error_rates size must match H column count");
  if (ids && ids->size() != H.num_cols())
    throw std::invalid_argument(
        "decoder_init: error_ids size must match H column count");
  if (D && D->num_rows() != H.num_rows())
    throw std::invalid_argument(
        "decoder_init: D row count must match H row count");
}

} // namespace

std::shared_ptr<decoder_init::impl> decoder_init::make_matrix_state(
    decoder_model_source source, sparse_binary_matrix H,
    std::optional<sparse_binary_matrix> O, std::vector<double> rates,
    std::optional<std::vector<std::size_t>> ids,
    std::optional<sparse_binary_matrix> D,
    std::optional<std::string> raw_stim_dem) {
  H = H.to_csc();
  if (O)
    *O = O->to_csr();
  if (D)
    *D = D->to_csr();
  validate_model(H, O, rates, ids, D);

  auto state = std::make_shared<decoder_init::impl>();
  state->source = source;
  state->num_detectors = H.num_rows();
  state->num_error_mechanisms = H.num_cols();
  state->num_observables = O ? O->num_rows() : 0;
  state->H = std::move(H);
  state->O = std::move(O);
  state->rates = std::move(rates);
  state->ids = std::move(ids);
  state->D = std::move(D);
  state->raw_stim_dem = std::move(raw_stim_dem);
  return state;
}

decoder_init::decoder_init(sparse_binary_matrix H)
    : decoder_init(make_matrix_state(decoder_model_source::matrices,
                                     std::move(H), std::nullopt, {},
                                     std::nullopt, std::nullopt)) {}

decoder_init::decoder_init(
    sparse_binary_matrix H, std::optional<sparse_binary_matrix> O,
    std::vector<double> error_rates,
    std::optional<sparse_binary_matrix> measurement_to_detectors,
    std::optional<std::vector<std::size_t>> error_ids)
    : decoder_init(make_matrix_state(
          decoder_model_source::matrices, std::move(H), std::move(O),
          std::move(error_rates), std::move(error_ids),
          std::move(measurement_to_detectors))) {}

decoder_init::decoder_init(
    detector_error_model model,
    std::optional<sparse_binary_matrix> measurement_to_detectors)
    : decoder_init(make_matrix_state(
          decoder_model_source::matrices,
          sparse_binary_matrix(model.detector_error_matrix),
          sparse_binary_matrix(model.observables_flips_matrix),
          std::move(model.error_rates), std::move(model.error_ids),
          std::move(measurement_to_detectors))) {}

decoder_init decoder_init::from_stim_dem(
    std::string stim_dem_text,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  // Project straight to sparse. Going through the materialized
  // detector_error_model would allocate a dense detectors x mechanisms tensor
  // only to scan it back out again: ~98 MiB for a distance-13 model whose
  // sparse form is under 1 MiB, and wasted entirely for a DEM-native decoder.
  auto [H, O, error_rates] = details::sparse_dem_from_stim_text(stim_dem_text);
  return decoder_init(make_matrix_state(
      decoder_model_source::stim_dem, std::move(H), std::move(O),
      std::move(error_rates), std::nullopt, std::move(measurement_to_detectors),
      std::move(stim_dem_text)));
}

decoder_init decoder_init::from_dem_chunks(
    dem_chunks_spec spec,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  spec.validate();
  const seam_id from_seam = spec.seam.from_seam;
  const seam_id to_seam = spec.seam.to_seam;

  // Closing the compact chain runs the chain checks dem_close_all() would,
  // and each elided round is one more bulk chunk, adding its interior rows,
  // its outgoing seam and its faults.
  const auto chain = dem_chunks_to_compact_chain(spec);
  const auto &chunks = chain.chunks;
  const std::uint64_t extra = chain.elided_rounds();
  const std::size_t bulk = chain.repeating;
  const auto closed = dem_chunks_to_pcm(chunks, from_seam, to_seam);
  std::size_t detectors = closed.num_rows();
  std::size_t mechanisms = closed.num_cols();
  if (extra > 0) {
    detectors += extra * (chunks[bulk].num_interior_rows() +
                          chunks[bulk].get_seam(from_seam).num_rows());
    mechanisms += extra * chunks[bulk].num_faults();
  }
  const std::size_t observables = chunks.front().num_observables();

  if (measurement_to_detectors) {
    *measurement_to_detectors = measurement_to_detectors->to_csr();
    if (measurement_to_detectors->num_rows() != detectors)
      throw std::invalid_argument(
          "decoder_init: D row count must match the dem_chunks detector "
          "count");
  }

  auto state = std::make_shared<impl>();
  state->source = decoder_model_source::dem_chunks;
  state->num_detectors = detectors;
  state->num_error_mechanisms = mechanisms;
  state->num_observables = observables;
  state->D = std::move(measurement_to_detectors);
  state->has_matrices = false;
  state->chunks = std::make_shared<const dem_chunks_spec>(std::move(spec));
  return decoder_init(std::move(state));
} // end - decoder_init::from_dem_chunks()

decoder_init decoder_init::from_dem_chunks_closed(
    dem_chunks_spec spec,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  const auto chunked = from_dem_chunks(std::move(spec));
  const auto &kept = *chunked.state_->chunks;
  const auto expanded = dem_chunks_from_spec(kept);
  auto H = dem_chunks_to_pcm(expanded, kept.seam.from_seam, kept.seam.to_seam);
  if (H.num_rows() != chunked.num_detectors() ||
      H.num_cols() != chunked.num_error_mechanisms())
    throw std::logic_error("decoder_init: dem_chunks closed model is " +
                           std::to_string(H.num_rows()) + " x " +
                           std::to_string(H.num_cols()) + ", not " +
                           std::to_string(chunked.num_detectors()) + " x " +
                           std::to_string(chunked.num_error_mechanisms()));
  std::optional<sparse_binary_matrix> O;
  if (chunked.num_observables() > 0)
    O = sparse_binary_matrix::from_nested_csr(
        static_cast<std::uint32_t>(chunked.num_observables()),
        static_cast<std::uint32_t>(chunked.num_error_mechanisms()),
        dem_chunks_to_o_sparse(expanded));
  std::vector<double> rates;
  rates.reserve(chunked.num_error_mechanisms());
  for (const auto &chunk : expanded)
    rates.insert(rates.end(), chunk.error_rates.begin(),
                 chunk.error_rates.end());

  auto state = make_matrix_state(decoder_model_source::dem_chunks, std::move(H),
                                 std::move(O), std::move(rates), std::nullopt,
                                 std::move(measurement_to_detectors));
  state->chunks = chunked.state_->chunks;
  return decoder_init(std::move(state));
} // end - decoder_init::from_dem_chunks_closed()

decoder_init::decoder_init(std::shared_ptr<const impl> state)
    : state_(std::move(state)) {}

decoder_init::decoder_init(const decoder_init &) noexcept = default;
decoder_init::decoder_init(decoder_init &&) noexcept = default;
decoder_init &decoder_init::operator=(const decoder_init &) noexcept = default;
decoder_init &decoder_init::operator=(decoder_init &&) noexcept = default;
decoder_init::~decoder_init() = default;

decoder_model_source decoder_init::source() const noexcept {
  return state_->source;
}

const sparse_binary_matrix &decoder_init::detector_error_matrix() const {
  return state_->matrices("detector_error_matrix").H;
}

bool decoder_init::has_observable_model() const noexcept {
  // A chunked source maps observables exactly when its chunks carry any.
  return state_->has_matrices ? state_->O.has_value()
                              : state_->num_observables > 0;
}

const sparse_binary_matrix &decoder_init::observable_flips_matrix() const {
  if (!has_observable_model())
    throw std::logic_error("decoder_init: no observable mapping was supplied");
  return *state_->matrices("observable_flips_matrix").O;
}

const std::vector<double> &decoder_init::error_rates() const {
  return state_->matrices("error_rates").rates;
}

const std::optional<std::vector<std::size_t>> &decoder_init::error_ids() const {
  return state_->ids;
}

const sparse_binary_matrix *
decoder_init::measurement_to_detectors() const noexcept {
  return state_->D ? &*state_->D : nullptr;
}

decoder_init decoder_init::canonicalize_H() const {
  // dem_chunks_to_pcm() already projects a canonical H.
  if (state_->chunks)
    return *this;
  auto H = state_->H.canonicalize().to_csc();
  return decoder_init(make_matrix_state(state_->source, std::move(H), state_->O,
                                        state_->rates, state_->ids, state_->D,
                                        state_->raw_stim_dem));
}

decoder_init decoder_init::decoder_init_without_d() const {
  auto state = std::make_shared<impl>(*state_);
  state->D.reset();
  return decoder_init(std::move(state));
}

decoder_init decoder_init::without_observables() const {
  auto state = std::make_shared<impl>(*state_);
  state->O.reset();
  state->num_observables = 0;
  if (state->chunks) {
    auto chunks = *state->chunks;
    for (auto &phase : chunks.phases) {
      phase.spec.O_sparse.clear();
      for (auto &seam : phase.spec.seam_specs)
        seam.spec.O_sparse.clear();
    }
    state->chunks = std::make_shared<const dem_chunks_spec>(std::move(chunks));
  }
  if (state->raw_stim_dem) {
    state->raw_stim_dem.reset();
    state->source = decoder_model_source::matrices;
  }
  return decoder_init(std::move(state));
}

bool decoder_init::has_stim_dem() const noexcept {
  return state_->raw_stim_dem.has_value();
}

const std::string &decoder_init::stim_dem() const {
  if (!state_->raw_stim_dem)
    throw std::logic_error(
        "decoder_init: authoritative source is not a Stim DEM");
  return *state_->raw_stim_dem;
}

bool decoder_init::has_dem_chunks() const noexcept {
  return state_->chunks != nullptr;
}

const dem_chunks_spec &decoder_init::dem_chunks() const {
  if (!state_->chunks)
    throw std::logic_error(
        "decoder_init: authoritative source is not a chunked DEM");
  return *state_->chunks;
}

bool decoder_init::has_matrices() const noexcept {
  return state_->has_matrices;
}

std::size_t decoder_init::num_detectors() const noexcept {
  return state_->num_detectors;
}

std::size_t decoder_init::num_error_mechanisms() const noexcept {
  return state_->num_error_mechanisms;
}

std::size_t decoder_init::num_observables() const noexcept {
  return state_->num_observables;
}

} // namespace cudaq::qec

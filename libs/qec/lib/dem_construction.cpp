/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Constructs a T-round code-capacity detector_error_model from CSS generator
// matrices and a depolarizing noise model. Each data-qubit fault maps to a
// DEM column whose detector rows span two consecutive round bands (the round
// of the fault and the next), except for faults in the final round which
// span only that round's band. Observable rows are the same for every round
// since the logical measurement is taken once at the end of the experiment.

#include "cudaq/qec/dem_construction.h"
#include "dem_construction_utils.h"

#include <stdexcept>
#include <vector>

namespace cudaq::qec {

using namespace cudaq::qec::detail;

detector_error_model dem_from_css_matrices(const css_code_matrices &code,
                                           const css_noise_params &noise,
                                           std::size_t num_rounds) {
  if (num_rounds == 0)
    throw std::invalid_argument("num_rounds must be >= 1");

  // Checked before the n == 0 early-out below so that a malformed rate is
  // reported even when the code matrices describe no qubits to apply it to.
  validate_noise_rates(noise);

  detector_error_model result;
  const std::size_t n = resolve_num_qubits(code);

  if (n == 0) {
    // All matrices have zero columns. A matrix with rows but zero columns
    // indicates a malformed input (e.g. 2×0 hz) rather than a trivially
    // empty code, so reject it rather than silently returning an empty DEM.
    auto check_zero_rows = [](const sparse_binary_matrix &m, const char *name) {
      if (m.num_rows() > 0)
        throw std::invalid_argument(
            std::string(name) + " has " + std::to_string(m.num_rows()) +
            " rows but zero columns; all matrices must be empty when "
            "num_qubits cannot be determined");
    };
    check_zero_rows(code.hz, "hz");
    check_zero_rows(code.hx, "hx");
    check_zero_rows(code.lz, "lz");
    check_zero_rows(code.lx, "lx");
    if (!noise.px_per_qubit.empty() || !noise.py_per_qubit.empty() ||
        !noise.pz_per_qubit.empty())
      throw std::invalid_argument(
          "per-qubit noise vectors are non-empty but all code matrices have "
          "zero columns");
    return result;
  }

  if (n != 0) {
    // Validate hz even though n may have been resolved from hz itself:
    // if n came from a different matrix while hz has rows but zero columns,
    // this catches the inconsistency that would otherwise produce an all-empty
    // Z-type detector block with no error or exception.
    check_num_cols(code.hz, n, "hz");
    check_num_cols(code.hx, n, "hx");
    check_num_cols(code.lz, n, "lz");
    check_num_cols(code.lx, n, "lx");
    check_per_qubit_size(noise.px_per_qubit, n, "px_per_qubit");
    check_per_qubit_size(noise.py_per_qubit, n, "py_per_qubit");
    check_per_qubit_size(noise.pz_per_qubit, n, "pz_per_qubit");

    // Per-qubit CSC column lists, padded so hz_csc[q] etc. are always valid.
    // hz_csc[q] = Z-check rows triggered by an X fault on qubit q.
    // hx_csc[q] = X-check rows triggered by a Z fault on qubit q.
    // lz_csc[q] = Z-obs rows flipped by an X fault on qubit q.
    // lx_csc[q] = X-obs rows flipped by a Z fault on qubit q.
    const auto hz_csc = padded_nested_csc(code.hz, n);
    const auto hx_csc = padded_nested_csc(code.hx, n);
    const auto lz_csc = padded_nested_csc(code.lz, n);
    const auto lx_csc = padded_nested_csc(code.lx, n);

    const std::size_t nz = static_cast<std::size_t>(code.hz.num_rows());
    const std::size_t nx = static_cast<std::size_t>(code.hx.num_rows());
    const std::size_t kz = static_cast<std::size_t>(code.lz.num_rows());
    const std::size_t kx = static_cast<std::size_t>(code.lx.num_rows());
    const std::size_t d = nz + nx; // detector rows per round = total checks

    // pm_per_check must have length d (= nz + nx) when non-empty.
    check_rate_vector_size(noise.pm_per_check, d, "pm_per_check", "n_checks");

    const std::size_t n_detectors = num_rounds * d;
    const std::size_t n_observables = kz + kx;

    const auto x_qubits = active_qubits(noise.px, noise.px_per_qubit, n);
    const auto z_qubits = active_qubits(noise.pz, noise.pz_per_qubit, n);
    const auto y_qubits = active_qubits(noise.py, noise.py_per_qubit, n);
    // Measurement errors: one column per active check (0..d-1) per round.
    // Check index k < nz is a Z-type check; k >= nz is an X-type check.
    const auto m_checks = active_checks(noise.pm, noise.pm_per_check, d);

    const std::size_t per_round =
        x_qubits.size() + z_qubits.size() + y_qubits.size() + m_checks.size();
    const std::size_t n_errors = num_rounds * per_round;

    // Always size the output tensors correctly so that consumers can derive
    // syndrome_size and num_observables from shape even when all rates are
    // zero.
    result.detector_error_matrix =
        cudaqx::tensor<uint8_t>({n_detectors, n_errors});
    result.observables_flips_matrix =
        cudaqx::tensor<uint8_t>({n_observables, n_errors});
    result.error_rates.reserve(n_errors);

    if (n_errors != 0) {

      std::size_t col = 0;

      for (std::size_t r = 0; r < num_rounds; ++r) {
        const std::size_t r_off = r * d; // detector row base for round r
        const std::size_t r1_off =
            (r + 1) * d; // detector row base for round r+1
        const bool has_next = (r + 1 < num_rounds);

        // X faults: Z-type detectors at round r (and r+1). Traverse hz_csc[q]
        // once, writing to both row bands in the same pass.
        for (const std::size_t q : x_qubits) {
          for (auto row : hz_csc[q]) {
            result.detector_error_matrix.at({r_off + row, col}) ^= 1;
            if (has_next)
              result.detector_error_matrix.at({r1_off + row, col}) ^= 1;
          }
          for (auto row : lz_csc[q])
            result.observables_flips_matrix.at({row, col}) ^= 1;
          result.error_rates.push_back(
              qubit_rate(noise.px, noise.px_per_qubit, q));
          ++col;
        }

        // Z faults: X-type detectors (offset nz) at rounds r and r+1.
        for (const std::size_t q : z_qubits) {
          for (auto row : hx_csc[q]) {
            result.detector_error_matrix.at({r_off + nz + row, col}) ^= 1;
            if (has_next)
              result.detector_error_matrix.at({r1_off + nz + row, col}) ^= 1;
          }
          for (auto row : lx_csc[q])
            result.observables_flips_matrix.at({kz + row, col}) ^= 1;
          result.error_rates.push_back(
              qubit_rate(noise.pz, noise.pz_per_qubit, q));
          ++col;
        }

        // Y faults: both detector bands at rounds r and r+1.
        for (const std::size_t q : y_qubits) {
          for (auto row : hz_csc[q]) {
            result.detector_error_matrix.at({r_off + row, col}) ^= 1;
            if (has_next)
              result.detector_error_matrix.at({r1_off + row, col}) ^= 1;
          }
          for (auto row : hx_csc[q]) {
            result.detector_error_matrix.at({r_off + nz + row, col}) ^= 1;
            if (has_next)
              result.detector_error_matrix.at({r1_off + nz + row, col}) ^= 1;
          }
          for (auto row : lz_csc[q])
            result.observables_flips_matrix.at({row, col}) ^= 1;
          for (auto row : lx_csc[q])
            result.observables_flips_matrix.at({kz + row, col}) ^= 1;
          result.error_rates.push_back(
              qubit_rate(noise.py, noise.py_per_qubit, q));
          ++col;
        }

        // Measurement error faults: one column per active check per round.
        // A measurement error on check k flips syndrome[r][k] without any data
        // error. detector[r][k] = syndrome[r][k] XOR syndrome[r-1][k] fires,
        // and so does detector[r+1][k] = syndrome[r+1][k] XOR syndrome[r][k]
        // (if the next round measures syndrome[r+1][k] correctly and sees
        // syndrome[r][k] was wrong). No logical observable is flipped.
        for (const std::size_t k : m_checks) {
          result.detector_error_matrix.at({r_off + k, col}) ^= 1;
          if (has_next)
            result.detector_error_matrix.at({r1_off + k, col}) ^= 1;
          result.error_rates.push_back(
              check_rate(noise.pm, noise.pm_per_check, k));
          ++col;
        }
      } // end - for(r)
    } // end - if (n_errors != 0)
  } // end - if (n != 0)

  return result;
} // end - dem_from_css_matrices()

} // namespace cudaq::qec

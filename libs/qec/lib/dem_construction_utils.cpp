/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Implements helpers declared in dem_construction_utils.h for validating CSS
// code matrices / noise params and extracting active fault supports.

#include "dem_construction_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::detail {

std::size_t resolve_num_qubits(const css_code_matrices &code) {
  if (code.hz.num_cols() > 0)
    return static_cast<std::size_t>(code.hz.num_cols());
  if (code.hx.num_cols() > 0)
    return static_cast<std::size_t>(code.hx.num_cols());
  if (code.lz.num_cols() > 0)
    return static_cast<std::size_t>(code.lz.num_cols());
  if (code.lx.num_cols() > 0)
    return static_cast<std::size_t>(code.lx.num_cols());
  return 0;
}

void check_num_cols(const sparse_binary_matrix &m, std::size_t n,
                    const char *label) {
  if (m.num_rows() != 0 && static_cast<std::size_t>(m.num_cols()) != n)
    throw std::invalid_argument(
        std::string(label) + " num_cols (" + std::to_string(m.num_cols()) +
        ") does not match n_qubits (" + std::to_string(n) + ")");
}

void check_rate_vector_size(const std::vector<double> &rates, std::size_t n,
                            const char *label, const char *size_label) {
  if (!rates.empty() && rates.size() != n)
    throw std::invalid_argument(std::string(label) + " has " +
                                std::to_string(rates.size()) + " entries but " +
                                size_label + "=" + std::to_string(n));
}

void check_per_qubit_size(const std::vector<double> &per_qubit, std::size_t n,
                          const char *label) {
  check_rate_vector_size(per_qubit, n, label, "n_qubits");
}

void check_probability(double p, const std::string &label) {
  if (!(p >= 0.0 && p <= 1.0))
    throw std::invalid_argument(label + " must be a probability in [0, 1]," +
                                " got " + std::to_string(p));
}

void validate_noise_rates(const css_noise_params &noise) {
  check_probability(noise.px, "px");
  check_probability(noise.py, "py");
  check_probability(noise.pz, "pz");
  check_probability(noise.pm, "pm");

  const auto check_each = [](const std::vector<double> &rates,
                             const char *label) {
    for (std::size_t i = 0; i < rates.size(); ++i)
      check_probability(rates[i],
                        std::string(label) + "[" + std::to_string(i) + "]");
  };
  check_each(noise.px_per_qubit, "px_per_qubit");
  check_each(noise.py_per_qubit, "py_per_qubit");
  check_each(noise.pz_per_qubit, "pz_per_qubit");
  check_each(noise.pm_per_check, "pm_per_check");
} // end - validate_noise_rates()

bool has_any_noise(const css_noise_params &noise) {
  const auto any_nonzero = [](const std::vector<double> &rates) {
    for (const double p : rates)
      if (p != 0.0)
        return true;
    return false;
  };
  return noise.px != 0.0 || noise.py != 0.0 || noise.pz != 0.0 ||
         noise.pm != 0.0 || any_nonzero(noise.px_per_qubit) ||
         any_nonzero(noise.py_per_qubit) || any_nonzero(noise.pz_per_qubit) ||
         any_nonzero(noise.pm_per_check);
} // end - has_any_noise()

col_list padded_nested_csc(const sparse_binary_matrix &m, std::size_t n) {
  auto cols = m.to_nested_csc();
  cols.resize(n);
  return cols;
}

double qubit_rate(double uniform, const std::vector<double> &per_qubit,
                  std::size_t q) {
  return per_qubit.empty() ? uniform : per_qubit[q];
}

std::vector<std::size_t> active_qubits(double uniform,
                                       const std::vector<double> &per_qubit,
                                       std::size_t n) {
  std::vector<std::size_t> out;
  out.reserve(n);
  for (std::size_t q = 0; q < n; ++q)
    if (qubit_rate(uniform, per_qubit, q) > 0.0)
      out.push_back(q);
  return out;
} // end - active_qubits()

double check_rate(double uniform_pm, const std::vector<double> &pm_per_check,
                  std::size_t k) {
  return pm_per_check.empty() ? uniform_pm : pm_per_check[k];
}

std::vector<std::size_t> active_checks(double uniform_pm,
                                       const std::vector<double> &pm_per_check,
                                       std::size_t n_checks) {
  std::vector<std::size_t> out;
  out.reserve(n_checks);
  for (std::size_t k = 0; k < n_checks; ++k)
    if (check_rate(uniform_pm, pm_per_check, k) > 0.0)
      out.push_back(k);
  return out;
} // end - active_checks()

} // namespace cudaq::qec::detail

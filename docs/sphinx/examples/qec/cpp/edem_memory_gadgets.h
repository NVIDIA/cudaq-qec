/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
#pragma once
#include "edem_example_utils.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

// Hand-authored linear fixtures, not compiler-derived gadget EDEMs. They give
// d_i = f_i XOR f_(i+1), and readout flip = f_prep XOR f_mz.
namespace edem_memory_demo {
inline constexpr uint64_t scalar_offsets[]{0, 1};
inline constexpr uint32_t scalar_rows[]{0};
inline constexpr uint64_t identity_offsets[]{0, 1, 2};
inline constexpr uint32_t identity_rows[]{0, 1};
inline constexpr uint64_t select_offsets[]{0, 1, 1};
inline constexpr cudaq_gf2_csc_view one{1, 1, 1, scalar_offsets, scalar_rows};
inline constexpr cudaq_gf2_csc_view logical_from_fault{2, 1, 1, scalar_offsets,
                                                       scalar_rows};
inline constexpr cudaq_gf2_csc_view logical_identity{2, 2, 2, identity_offsets,
                                                     identity_rows};
inline constexpr cudaq_gf2_csc_view logical_x{1, 2, 1, select_offsets,
                                              scalar_rows};
inline constexpr char name[] = "monolithic-demo";
inline constexpr cudaq_edem_code_presentation code{name, sizeof(name) - 1, 1,
                                                   1};

inline cudaq::edem::PackedEDEM prep(double probability = .01) {
  uint32_t code_id = 0, fault_id = 0;
  cudaq_edem_fault_group group{CUDAQ_EDEM_INDEPENDENT, 1, &fault_id,
                               &probability};
  cudaq_edem_fault_distribution fault{1, 1, &group};
  cudaq_edem_block blocks[]{
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_FAULT, 0}, one},
      {{CUDAQ_EDEM_LOGICAL, 0}, {CUDAQ_EDEM_FAULT, 0}, logical_from_fault}};
  cudaq_edem_description d{{1, 0, 1, 1, 0, 0, 0, 2},
                           &code,
                           nullptr,
                           &code_id,
                           &fault,
                           nullptr,
                           nullptr,
                           nullptr,
                           blocks};
  return edem_example::construct(d);
}

inline cudaq::edem::PackedEDEM sec(double probability = .02) {
  uint32_t code_id = 0, fault_id = 0, detector_size = 1;
  cudaq_edem_fault_group group{CUDAQ_EDEM_INDEPENDENT, 1, &fault_id,
                               &probability};
  cudaq_edem_fault_distribution fault{1, 1, &group};
  cudaq_edem_block blocks[]{
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_FAULT, 0}, one},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, one},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_SYNDROME, 0}, one},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_FAULT, 0}, one},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_SYNDROME, 0}, one},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_SYNDROME, 0}, one},
      {{CUDAQ_EDEM_LOGICAL, 0}, {CUDAQ_EDEM_LOGICAL, 0}, logical_identity}};
  cudaq_edem_description d{
      {1, 1, 1, 1, 1, 0, 0, 7}, &code,   &code_id, &code_id, &fault,
      &detector_size,           nullptr, nullptr,  blocks};
  return edem_example::construct(d);
}

inline cudaq::edem::PackedEDEM mz(double probability = .03) {
  uint32_t code_id = 0, fault_id = 0, size = 1;
  cudaq_edem_fault_group group{CUDAQ_EDEM_INDEPENDENT, 1, &fault_id,
                               &probability};
  cudaq_edem_fault_distribution fault{1, 1, &group};
  cudaq_edem_block blocks[]{
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_FAULT, 0}, one},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, one},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_SYNDROME, 0}, one},
      {{CUDAQ_EDEM_OBSERVABLE, 0}, {CUDAQ_EDEM_FAULT, 0}, one},
      {{CUDAQ_EDEM_OBSERVABLE, 0}, {CUDAQ_EDEM_LOGICAL, 0}, logical_x}};
  cudaq_edem_description d{{1, 1, 0, 1, 1, 1, 0, 5},
                           &code,
                           &code_id,
                           nullptr,
                           &fault,
                           &size,
                           &size,
                           nullptr,
                           blocks};
  return edem_example::construct(d);
}

// The experiment recipe belongs to this fixture, independently of composition.
// Returned views borrow the supplied definitions until the composer evaluates
// them. The same recipe also accepts definitions loaded from packed EDEM files.
inline std::vector<cudaq_edem_view>
memory_experiment(const cudaq::edem::PackedEDEM &prep,
                  const cudaq::edem::PackedEDEM &sec,
                  const cudaq::edem::PackedEDEM &mz, std::size_t rounds = 3) {
  auto require_shape = [](const cudaq::edem::PackedEDEM &value, uint32_t inputs,
                          uint32_t outputs, const char *message) {
    cudaq_edem_shape shape;
    edem_example::check(cudaq_edem_get_shape(&value.view(), &shape));
    if (shape.input_code_blocks != inputs ||
        shape.output_code_blocks != outputs)
      throw std::invalid_argument(message);
  };
  require_shape(prep, 0, 1,
                "prep must have no input and one output code block");
  require_shape(sec, 1, 1, "SEC must have one input and one output code block");
  require_shape(mz, 1, 0, "mz must have one input and no output code block");
  if (rounds > UINT32_MAX - 2)
    throw std::invalid_argument("too many SEC occurrences");
  std::vector<cudaq_edem_view> occurrences;
  occurrences.reserve(rounds + 2);
  occurrences.push_back(prep.view());
  for (std::size_t i = 0; i < rounds; ++i)
    occurrences.push_back(sec.view());
  occurrences.push_back(mz.view());
  return occurrences;
}
} // namespace edem_memory_demo

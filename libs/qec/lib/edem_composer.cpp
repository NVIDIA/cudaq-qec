/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
#include "cudaq/qec/edem_composer.h"
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace cudaq::qec {
namespace {
void check(cudaq_edem_status status) {
  if (status == CUDAQ_EDEM_ALLOCATION_FAILED)
    throw std::bad_alloc();
  if (status != CUDAQ_EDEM_OK)
    throw std::invalid_argument(cudaq_edem_status_string(status));
}

void check(cudaq_gf2_status status) {
  if (status == CUDAQ_GF2_ALLOCATION_FAILED)
    throw std::bad_alloc();
  if (status != CUDAQ_GF2_OK)
    throw std::invalid_argument("invalid projected EDEM map");
}

size_t append(size_t total, size_t size) {
  if (size > std::numeric_limits<size_t>::max() - total)
    throw std::length_error("DEM dimensions overflow");
  return total + size;
}

std::vector<size_t> row_offsets(const cudaq_edem_view &value, uint32_t kind,
                                uint32_t count) {
  std::vector<size_t> offsets{0};
  offsets.reserve(size_t{count} + 1);
  for (uint32_t i = 0; i < count; ++i) {
    cudaq_gf2_wire wire;
    uint32_t width;
    uint64_t offset;
    check(cudaq_edem_resolve_endpoint(&value, 1, {kind, i}, &wire));
    check(cudaq_gf2_map_get_wire(&value.map, 1, wire, &width, &offset));
    offsets.push_back(append(offsets.back(), width));
  }
  return offsets;
}

cudaqx::tensor<uint8_t> matrix(size_t rows, size_t columns) {
  if (columns && rows > std::numeric_limits<size_t>::max() / columns)
    throw std::length_error("dense DEM matrix dimensions overflow");
  cudaqx::tensor<uint8_t> result({rows, columns});
  if (result.size())
    std::fill_n(result.data(), result.size(), uint8_t{0});
  return result;
}

} // namespace

detector_error_model dem_from_edem(const cudaq_edem_dem_view &projected) {
  const auto &value = projected.edem;
  uint32_t closed;
  check(cudaq_edem_is_dem(&value, &closed));
  if (!closed)
    throw std::invalid_argument(
        "QEC conversion requires a closed projected DEM");
  cudaq_edem_shape shape;
  check(cudaq_edem_get_shape(&value, &shape));
  auto detectors =
      row_offsets(value, CUDAQ_EDEM_DETECTOR, shape.detector_ports);
  auto observables =
      row_offsets(value, CUDAQ_EDEM_OBSERVABLE, shape.observable_ports);
  auto successes = row_offsets(value, CUDAQ_EDEM_SUCCESS, shape.success_ports);
  if (successes.back())
    throw std::invalid_argument("QEC DEM has no success outputs; choose a "
                                "success/postselection policy first");
  std::vector<size_t> columns{0};
  columns.reserve(size_t{shape.fault_ports} + 1);
  for (uint32_t p = 0; p < shape.fault_ports; ++p) {
    uint32_t size, groups;
    check(cudaq_edem_get_fault(&value, p, &size, &groups));
    columns.push_back(append(columns.back(), size));
  }

  detector_error_model result;
  result.detector_error_matrix = matrix(detectors.back(), columns.back());
  result.observables_flips_matrix = matrix(observables.back(), columns.back());
  result.error_rates.resize(columns.back());
  result.error_ids.emplace(columns.back());
  size_t next_id = 0;
  for (uint32_t p = 0; p < shape.fault_ports; ++p) {
    uint32_t size, groups;
    check(cudaq_edem_get_fault(&value, p, &size, &groups));
    for (uint32_t g = 0; g < groups; ++g) {
      cudaq_edem_fault_group group;
      check(cudaq_edem_get_group(&value, p, g, &group));
      for (uint32_t i = 0; i < group.size; ++i) {
        size_t column = columns[p] + group.fault_ids[i];
        result.error_rates[column] = group.probabilities[i];
        // QEC's decoder/sampler input uses independent mechanisms. Retaining
        // exclusive alternatives at their marginal rates approximates their
        // joint law; composition and native projection remain exact.
        (*result.error_ids)[column] = next_id++;
      }
    }
  }
  cudaq_gf2_map_shape map;
  check(cudaq_gf2_map_get_shape(&value.map, &map));
  for (uint64_t b = 0; b < map.block_count; ++b) {
    cudaq_gf2_map_block block;
    cudaq_edem_endpoint input, output;
    check(cudaq_gf2_map_get_block(&value.map, b, &block));
    check(cudaq_edem_describe_wire(&value, 0, block.input, &input));
    check(cudaq_edem_describe_wire(&value, 1, block.output, &output));
    auto &target = output.kind == CUDAQ_EDEM_DETECTOR
                       ? result.detector_error_matrix
                       : result.observables_flips_matrix;
    auto &offsets =
        output.kind == CUDAQ_EDEM_DETECTOR ? detectors : observables;
    for (uint32_t c = 0; c < block.matrix.columns; ++c)
      for (uint64_t i = block.matrix.column_offsets[c];
           i < block.matrix.column_offsets[c + 1]; ++i)
        target.at({offsets[output.index] + block.matrix.row_indices[i],
                   columns[input.index] + c}) = 1;
  }
  return result;
}
} // namespace cudaq::qec

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
#pragma once
#include "cudaq/edem/composition.hpp"
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>

// Application-local conveniences for these examples, not a QEC ownership API.
// PackedEDEM is compiled into this consumer; only C views cross into QEC.
namespace edem_example {
inline void check(cudaq_edem_status status) {
  if (status == CUDAQ_EDEM_ALLOCATION_FAILED)
    throw std::bad_alloc();
  if (status != CUDAQ_EDEM_OK)
    throw std::invalid_argument(cudaq_edem_status_string(status));
}

inline cudaq::edem::PackedEDEM construct(const cudaq_edem_description &d) {
  cudaq::edem::PackedEDEM result;
  check(cudaq::edem::PackedEDEM::construct(d, result));
  return result;
}

inline std::span<const std::byte> bytes(const cudaq::edem::PackedEDEM &model) {
  return {static_cast<const std::byte *>(model.data()),
          static_cast<std::size_t>(model.size())};
}

// Serialized inputs may be unaligned. Copy into a default-allocator buffer,
// then adopt only after validation; release the C record even on failure.
inline cudaq::edem::PackedEDEM from_bytes(std::span<const std::byte> bytes) {
  if (bytes.empty())
    throw std::invalid_argument("empty packed EDEM");
  auto allocator = cudaq_gf2_default_allocator();
  void *data = allocator.allocate(allocator.context, bytes.size(),
                                  CUDAQ_EDEM_PACKED_ALIGNMENT);
  if (!data)
    throw std::bad_alloc();
  std::memcpy(data, bytes.data(), bytes.size());
  cudaq_gf2_owned_buffer owner{data, bytes.size(), allocator.deallocate,
                               allocator.context};
  cudaq::edem::PackedEDEM result;
  auto status = cudaq::edem::PackedEDEM::adopt(owner, result);
  cudaq_gf2_owned_buffer_release(&owner);
  check(status);
  return result;
}
} // namespace edem_example

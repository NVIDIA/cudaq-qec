/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::detail {

// The replay producer writes a headerless sequence of little-endian uint32
// frame widths. Parse it once at graph construction, never in the decode path.
inline std::vector<std::size_t> read_measurement_frame_lengths(
    const std::string &path,
    std::optional<std::size_t> measurement_count = std::nullopt) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::invalid_argument("cannot read measurement frame lengths: " +
                                path);
  std::vector<std::size_t> lengths;
  std::size_t total = 0;
  for (;;) {
    unsigned char bytes[4];
    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (input.gcount() == 0 && input.eof())
      break;
    if (input.gcount() != sizeof(bytes))
      throw std::invalid_argument("truncated measurement frame lengths: " +
                                  path);
    const auto width = static_cast<std::uint32_t>(bytes[0]) |
                       (static_cast<std::uint32_t>(bytes[1]) << 8) |
                       (static_cast<std::uint32_t>(bytes[2]) << 16) |
                       (static_cast<std::uint32_t>(bytes[3]) << 24);
    const auto limit =
        measurement_count.value_or(std::numeric_limits<std::size_t>::max());
    if (width == 0 || width > limit - total)
      throw std::invalid_argument("invalid measurement frame width: " + path);
    total += width;
    lengths.push_back(width);
  }
  if (!input.eof() || lengths.empty() ||
      (measurement_count && total != *measurement_count))
    throw std::invalid_argument("measurement frame lengths do not cover D: " +
                                path);
  return lengths;
}

} // namespace cudaq::qec::detail

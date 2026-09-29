/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/decoder.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct reference_shot {
  std::string name;
  std::vector<cudaq::qec::float_t> syndrome;
  std::vector<cudaq::qec::float_t> expected_observables;
};

std::filesystem::path reference_data_dir() {
  return std::filesystem::path{CHROMOBIUS_REFERENCE_DATA_DIR};
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Failed to open " + path.string());
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

std::vector<cudaq::qec::float_t> parse_bits(const std::string &bits) {
  std::vector<cudaq::qec::float_t> values;
  values.reserve(bits.size());
  for (char bit : bits) {
    if (bit != '0' && bit != '1')
      throw std::runtime_error("Invalid reference bit: " + std::string(1, bit));
    values.push_back(bit == '1' ? 1.0 : 0.0);
  }
  return values;
}

std::vector<reference_shot> read_reference_shots() {
  const auto path = reference_data_dir() / "basic_reference.tsv";
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Failed to open " + path.string());

  std::vector<reference_shot> shots;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(file, line)) {
    ++line_no;
    const auto first_non_space = line.find_first_not_of(" \t");
    if (first_non_space == std::string::npos || line[first_non_space] == '#')
      continue;

    std::istringstream stream(line);
    std::string name;
    std::string syndrome_bits;
    std::string expected_observable_bits;
    std::string trailing;
    if (!(stream >> name >> syndrome_bits >> expected_observable_bits) ||
        stream >> trailing) {
      throw std::runtime_error("Malformed reference row " + path.string() +
                               ":" + std::to_string(line_no));
    }

    shots.push_back(reference_shot{name, parse_bits(syndrome_bits),
                                   parse_bits(expected_observable_bits)});
  }
  return shots;
}

} // namespace

TEST(ChromobiusCorrectness, matchesUpstreamPredictReference) {
  const auto dem = read_text_file(reference_data_dir() / "basic_reference.dem");
  const auto shots = read_reference_shots();
  ASSERT_FALSE(shots.empty());

  const auto num_detectors = shots.front().syndrome.size();
  const auto num_observables = shots.front().expected_observables.size();
  auto decoder = cudaq::qec::decoder::get("chromobius", std::string_view{dem});

  for (const auto &shot : shots) {
    SCOPED_TRACE(shot.name);
    ASSERT_EQ(shot.syndrome.size(), num_detectors);
    ASSERT_EQ(shot.expected_observables.size(), num_observables);

    auto result = decoder->decode(shot.syndrome);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.result, shot.expected_observables);
  }
}

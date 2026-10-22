/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
// clang-format off
// nvq++ -std=c++20 -lcudaq-qec -lcudaq-qec-decoders -lcudaq-edem -lcudaq-gf2 edem_monolithic.cpp
// clang-format on
#include "edem_memory_gadgets.h"
#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/edem_composer.h"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <vector>

namespace {
cudaq::edem::PackedEDEM load(const char *path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error(std::string("cannot open EDEM: ") + path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
  if (file.bad())
    throw std::runtime_error(std::string("cannot read EDEM: ") + path);
  return edem_example::from_bytes(std::as_bytes(std::span(bytes)));
}

template <class T>
T number(const char *text) {
  T value;
  std::string_view input(text);
  auto [end, error] =
      std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size())
    throw std::invalid_argument(std::string("invalid count or seed: ") + text);
  return value;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 1 && (argc < 4 || argc > 6)) {
      std::cerr << "Usage: " << argv[0]
                << " [prep.edem sec.edem mz.edem [shots [seed]]]\n";
      return 1;
    }
    auto prep = argc == 1 ? edem_memory_demo::prep() : load(argv[1]);
    auto sec = argc == 1 ? edem_memory_demo::sec() : load(argv[2]);
    auto mz = argc == 1 ? edem_memory_demo::mz() : load(argv[3]);
    size_t shots = argc >= 5 ? number<size_t>(argv[4]) : 1024;
    unsigned seed = argc >= 6 ? number<unsigned>(argv[5]) : 13;
    if (argc == 1)
      std::cout << "Using hand-authored demonstration EDEMs.\n";

    // [Begin Documentation]
    auto experiment =
        edem_memory_demo::memory_experiment(prep, sec, mz, /*rounds=*/3);
    // Reference composition evaluates basis columns today. Sparse block
    // composition in cudaq-edem will improve this offline build path.
    cudaq::edem::PackedEDEM composed, projected;
    edem_example::check(cudaq::edem::compose_chain(
        experiment.data(), static_cast<uint32_t>(experiment.size()), composed));
    uint32_t closed;
    edem_example::check(cudaq_edem_is_dem(&composed.view(), &closed));
    std::cout << "is_DEM: " << std::boolalpha << bool(closed) << '\n';
    if (!closed)
      throw std::runtime_error("prep -> SEC x3 -> mz did not close naturally");
    edem_example::check(cudaq::edem::project_DEM(composed, projected));
    auto dem = cudaq::qec::dem_from_edem({projected.view()});

    // QEC conversion uses independent marginal probabilities. This is exact
    // for these independent demo faults; exclusive source laws are
    // approximated.
    auto [detectors, errors] = cudaq::qec::dem_sampler::cpu::sample_dem(
        dem.detector_error_matrix, shots, dem.error_rates, seed);
    auto observables = errors.dot(dem.observables_flips_matrix.transpose()) % 2;
    // [End Documentation]
    size_t detector_flips = 0, observable_flips = 0;
    for (size_t i = 0; i < detectors.size(); ++i)
      detector_flips += detectors.data()[i];
    for (size_t i = 0; i < observables.size(); ++i)
      observable_flips += observables.data()[i];
    std::cout << "H: " << dem.num_detectors() << " x "
              << dem.num_error_mechanisms() << "\nO: " << dem.num_observables()
              << " x " << dem.num_error_mechanisms() << "\nshots: " << shots
              << " (seed " << seed << ")\n";
    std::cout << "detector flips: " << detector_flips
              << "\nobservable flips: " << observable_flips << '\n';
    for (size_t shot = 0; shot < std::min(shots, size_t{8}); ++shot) {
      std::cout << "shot " << shot << " detectors=";
      for (size_t row = 0; row < dem.num_detectors(); ++row)
        std::cout << unsigned(detectors.at({shot, row}));
      std::cout << " observable_flips=";
      for (size_t row = 0; row < dem.num_observables(); ++row)
        std::cout << unsigned(observables.at({shot, row}));
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EDEM example: " << error.what() << '\n';
    return 1;
  }
}

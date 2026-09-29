/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// This example shows how to use decoders from decoder plugins
//
// Compile and run with
// nvq++ -lcudaq-qec decoder_plugins_demo.cpp -o
// decoder_plugins_demo
// ./decoder_plugins_demo

#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cudaq.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"

int main() {
  auto steane = cudaq::qec::get_code("steane");
  auto Hz = steane->get_parity_z();
  std::vector<size_t> t_shape = Hz.shape();

  std::cout << "Hz.shape():\n";
  for (size_t elem : t_shape)
    std::cout << elem << " ";
  std::cout << "\n";

  std::cout << "Hz:\n";
  Hz.dump();

  auto Lz = steane->get_observables_x();
  std::cout << "Lz:\n";
  Lz.dump();

  double p = 0.2;
  size_t nShots = 5;

  // Check for available decoders
  for (auto &name : cudaq::qec::decoder::get_registered())
    printf("Decoder: %s\n", name.c_str());
  // create a decoder from the plugins
  auto lut_decoder = cudaq::qec::get_decoder("single_error_lut_example", Hz);

  std::cout << "nShots: " << nShots << "\n";

  // May want a order-2 tensor of syndromes
  // access tensor by stride to write in an entire syndrome
  cudaqx::tensor<uint8_t> syndrome({Hz.shape()[0]});

  int nErrors = 0;
  for (size_t shot = 0; shot < nShots; ++shot) {
    std::cout << "shot: " << shot << "\n";
    auto shot_data = cudaq::qec::generate_random_bit_flips(Hz.shape()[1], p);
    std::cout << "shot data\n";
    shot_data.dump();

    auto observable_z_data = Lz.dot(shot_data);
    observable_z_data = observable_z_data % 2;
    std::cout << "Data Lz state:\n";
    observable_z_data.dump();

    auto syndrome = Hz.dot(shot_data);
    syndrome = syndrome % 2;
    std::cout << "syndrome:\n";
    syndrome.dump();

    auto results = lut_decoder->decode(syndrome);
    auto v_result = results.result;
    cudaqx::tensor<uint8_t> result_tensor;
    // v_result is a std::vector<float_t>, of soft information. We'll convert
    // this to hard information and store as a tensor<uint8_t>.
    cudaq::qec::convert_vec_soft_to_tensor_hard(v_result, result_tensor);
    std::cout << "decode result:\n";
    result_tensor.dump();

    // check observable result
    auto decoded_observable_z = Lz.dot(result_tensor);
    std::cout << "decoded observable:\n";
    decoded_observable_z.dump();

    // check how many observable operators were decoded correctly
    // observable_z_data == decoded_observable_z This maps onto element wise
    // addition (mod 2)
    auto observable_flips = decoded_observable_z + observable_z_data;
    observable_flips = observable_flips % 2;
    std::cout << "Logical errors:\n";
    observable_flips.dump();
    std::cout << "\n";

    // shot counts as a observable error unless all observables are correct
    if (observable_flips.any()) {
      nErrors++;
    }
  }
  std::cout << "Total logical errors: " << nErrors << "\n";

  // Full data gen in function call
  auto [syn, data] = cudaq::qec::sample_code_capacity(Hz, nShots, p);
  std::cout << "Numerical experiment:\n";
  std::cout << "Data:\n";
  data.dump();
  std::cout << "Syn:\n";
  syn.dump();
}

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/decoding_config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {
struct byte_span {
  std::uint8_t *buffer;
  std::uint64_t length;
};
} // namespace

extern "C" {
void enqueue_syndromes(std::uint64_t decoder_id, std::uint64_t counter,
                       std::uint64_t syndrome_mapping_id,
                       byte_span syndrome_bits);
void get_corrections(std::uint64_t decoder_id, byte_span corrections,
                     bool reset);
void reset_decoder(std::uint64_t decoder_id);
}

namespace {

cudaq::qec::decoding::config::multi_decoder_config
make_lut(std::uint64_t bits) {
  cudaq::qec::decoding::config::multi_decoder_config cfg;
  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "single_error_lut";
  dc.block_size = bits;
  dc.syndrome_size = bits;
  dc.H_sparse.clear();
  dc.O_sparse = {0, -1};
  dc.D_sparse.clear();
  for (std::uint64_t i = 0; i < bits; ++i) {
    dc.H_sparse.push_back(static_cast<std::int64_t>(i));
    dc.H_sparse.push_back(-1);
    dc.D_sparse.push_back(static_cast<std::int64_t>(i));
    dc.D_sparse.push_back(-1);
  }
  cfg.decoders.push_back(dc);
  return cfg;
}

TEST(SimulationCqrHost, PacksAndUnpacksOneBitAndMultiByteSyndromes) {
  auto one = make_lut(1);
  ASSERT_EQ(cudaq::qec::decoding::config::configure_decoders(one), 0);
  std::uint8_t packed = 1;
  byte_span syn{&packed, 1};
  enqueue_syndromes(0, 0, 0, syn);
  std::uint8_t corr_buf = 0;
  byte_span corr{&corr_buf, 1};
  get_corrections(0, corr, true);
  reset_decoder(0);
  cudaq::qec::decoding::config::finalize_decoders();

  auto nine = make_lut(9);
  ASSERT_EQ(cudaq::qec::decoding::config::configure_decoders(nine), 0);
  std::uint8_t packed9[2] = {0xff, 0x01};
  byte_span syn9{packed9, 9};
  enqueue_syndromes(0, 0, 0, syn9);
  std::uint8_t corr9[2] = {0, 0};
  byte_span corr_span{corr9, 1};
  get_corrections(0, corr_span, false);
  reset_decoder(0);
  cudaq::qec::decoding::config::finalize_decoders();
}

} // namespace

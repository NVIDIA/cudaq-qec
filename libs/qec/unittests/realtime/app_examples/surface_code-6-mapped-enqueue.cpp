/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// NAMING A SYNDROME_MAPPING ROW FROM A KERNEL
// ===========================================
//
// `enqueue_syndromes_mapped` is the entry point a kernel uses to say which
// syndrome_mapping row a round belongs to.  On the Quantinuum backend it
// travels as its own wire function -- `enqueue_syndromes_mapped_ui64`, five
// uint64 fields, function id derived from the name -- rather than over the
// four-field `enqueue_syndromes_ui64`.
//
// That wire has no compile-time contract.  The sending client, the UDP packer
// and the receiving shim each spell the field order out by hand, and
// shim_function.cpp's own comment says a mismatch produces "no errors or
// warnings, other than bad crashes".  This example exists to give that wire a
// caller.
//
// Why a wrong field order cannot pass here
// ----------------------------------------
// The kernel names row 0, the pass-through row every configuration has.  The
// five fields carry (decoder_id=0, syndrome_size=3, syndrome, tag=1, row=0),
// so row 0 is the ONLY field whose value is zero.  Any rotation or swap
// therefore delivers a non-zero value into the row slot, no row declares it,
// and the enqueue throws rather than decoding.  A misordering that somehow
// kept the row at zero would still scramble the syndrome and fail the
// correction check below.
//
// What the example does
// ---------------------
// One pymatching decoder (id 0) with a 3-bit identity syndrome map.  The
// kernel prepares the syndrome {0,1,0} by flipping one qubit, enqueues it
// naming row 0, and reads the corrections back.  Expected: a correction at
// exactly bit 1.
//
// Run:
//   ./surface_code-6-mapped-enqueue

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace config = cudaq::qec::decoding::config;

constexpr std::uint64_t kBlockSize = 3;
constexpr std::uint64_t kSyndromeSize = 3;
constexpr double kUniformErrorRate = 0.1;

// The one syndrome bit the kernel sets, and so the one correction bit the
// decoder must return.
constexpr std::size_t kFlippedBit = 1;

std::vector<std::int64_t> make_identity_sparse_matrix() {
  std::vector<std::int64_t> sparse;
  sparse.reserve(2 * kSyndromeSize);
  for (std::uint64_t column = 0; column < kSyndromeSize; ++column) {
    sparse.push_back(static_cast<std::int64_t>(column));
    sparse.push_back(-1);
  }
  return sparse;
}

config::multi_decoder_config make_config() {
  config::multi_decoder_config multi;
  const auto identity = make_identity_sparse_matrix();
  config::decoder_config dc;
  dc.id = 0;
  dc.type = "pymatching";
  dc.dispatch = config::DecoderDispatch::host;
  dc.block_size = kBlockSize;
  dc.syndrome_size = kSyndromeSize;
  dc.H_sparse = identity;
  dc.O_sparse = identity;
  dc.D_sparse = identity;
  cudaqx::heterogeneous_map pm_args;
  pm_args.insert("error_rate_vec",
                 std::vector<double>(kBlockSize, kUniformErrorRate));
  pm_args.insert("merge_strategy", "smallest_weight");
  dc.decoder_custom_args = pm_args;
  multi.decoders.push_back(dc);
  return multi;
}

} // namespace

// Prepares syndrome {0,1,0} on real qubits -- enqueue_syndromes_mapped takes
// measure_results, so the bits have to come from a measurement rather than a
// hand-built vector.  Returns the corrections packed one per bit.
__qpu__ std::int64_t mapped_enqueue_kernel() {
  constexpr std::uint64_t kKernelSyndromeSize = 3;
  constexpr std::uint64_t kKernelBlockSize = 3;
  constexpr std::size_t kKernelFlippedBit = 1;

  cudaq::qvector q(kKernelSyndromeSize);
  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/0);
  x(q[kKernelFlippedBit]);
  auto syndrome = mz(q);

  // Row 0 is the pass-through BULK row.  Naming it explicitly is what puts
  // this round on the five-field wire.
  cudaq::qec::decoding::enqueue_syndromes_mapped(/*decoder_id=*/0, syndrome,
                                                 /*tag=*/1,
                                                 /*syndrome_mapping_id=*/0);

  auto corrections = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/0, /*return_size=*/kKernelBlockSize, /*reset=*/true);

  std::int64_t packed = 0;
  for (std::size_t i = 0; i < kKernelBlockSize; ++i)
    if (corrections[i])
      packed |= (static_cast<std::int64_t>(1) << i);
  return packed;
}

// `--save_config <path>` writes the decoder configuration and exits, so the
// server process can be started against the same decoders the kernel targets.
// Without it the run is self-contained and decodes in-process.
int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--save_config") == 0 && i + 1 < argc) {
      std::ofstream out(argv[i + 1]);
      if (!out) {
        std::printf("FAIL: cannot write %s\n", argv[i + 1]);
        return 1;
      }
      out << make_config().to_yaml_str(/*width=*/200);
      std::printf("wrote %s\n", argv[i + 1]);
      return 0;
    }
  }

  auto decoder_config = make_config();
  if (config::configure_decoders(decoder_config) != 0) {
    std::printf("FAIL: configure_decoders failed\n");
    return 1;
  }

  const auto results = cudaq::run(/*shots=*/1, mapped_enqueue_kernel);
  config::finalize_decoders();

  if (results.empty()) {
    std::printf("FAIL: no shot results\n");
    return 1;
  }

  const std::int64_t expected = static_cast<std::int64_t>(1) << kFlippedBit;
  const std::int64_t got = results[0];
  std::printf("corrections=0x%llx expected=0x%llx\n",
              static_cast<unsigned long long>(got),
              static_cast<unsigned long long>(expected));
  if (got != expected) {
    std::printf("FAIL: enqueue_syndromes_mapped did not decode the syndrome "
                "the kernel sent\n");
    return 1;
  }

  std::printf("PASS: enqueue_syndromes_mapped round-tripped row 0\n");
  return 0;
}

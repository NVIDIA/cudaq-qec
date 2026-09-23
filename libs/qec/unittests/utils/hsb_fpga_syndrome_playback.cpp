/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include "cudaq/qec/realtime/decoding_config.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/timeout.hpp>

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

namespace {

// ============================================================================
// Playback BRAM Constants
// ============================================================================
constexpr std::uint32_t PLAYER_ADDR = 0x5000'0000;
constexpr std::uint32_t RAM_ADDR = 0x5010'0000;
constexpr std::uint32_t PLAYER_TIMER_OFFSET = 0x0008;
constexpr std::uint32_t PLAYER_WINDOW_SIZE_OFFSET = 0x000C;
constexpr std::uint32_t PLAYER_WINDOW_NUMBER_OFFSET = 0x0010;
constexpr std::uint32_t PLAYER_ENABLE_OFFSET = 0x0004;
constexpr std::uint32_t RAM_NUM = 16;
constexpr std::uint32_t RAM_DEPTH = 512;

constexpr std::uint32_t PLAYER_ENABLE =
    0x0000'000D; // enable + single-pass + ptp_bram_ena
constexpr std::uint32_t PLAYER_DISABLE = 0x0000'0000;

// Sensor TX streaming threshold register. The Host→FPGA path buffers
// incoming data until this byte threshold is met before streaming to the
// sensor (ILA). Setting this to 0 (value 0x5) causes data to stream
// immediately, which is required when the total capture is small enough
// to sit entirely within the default buffer.
constexpr std::uint32_t SIF_TX_THRESHOLD_ADDR = 0x0120'0000;
constexpr std::uint32_t SIF_TX_THRESHOLD_IMMEDIATE = 0x0000'0005;

constexpr std::uint32_t METADATA_PACKET_ADDR = 0x102C;

constexpr std::uint32_t DEFAULT_TIMER_SPACING_US = 10;
constexpr std::uint32_t RF_SOC_TIMER_SCALE = 322;

constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("mock_decode");

// ============================================================================
// ILA Capture Block Constants — from spec_sif_tx.json
//
// The SIF TX capture block at 0x4000_0000 records the full 512-bit incoming
// data bus.  Each captured sample is 585 bits wide:
//   bits [511:0]   sif_tx_axis_tdata_0  — 512-bit raw payload
//   bit  512       sif_tx_axis_tvalid_0 — valid indicator
//   bit  513       sif_tx_axis_tlast_0  — last beat of ethernet packet
//   bits [520:514] sif_ila_wr_tcnt_0    — valid byte count in this beat
//   bits [584:521] current_ptp_timestamp {sec[31:0], nsec[31:0]}
//
// Samples are captured only on clock cycles where valid data is present.
// The status register at base+0x84 holds the current sample write address,
// allowing us to poll for the expected number of captured frames rather than
// waiting for the full buffer to fill.
// ============================================================================
constexpr std::uint32_t ILA_BASE_ADDR = 0x4000'0000;
constexpr std::uint32_t ILA_CTRL_OFFSET = 0x0000;
constexpr std::uint32_t ILA_STATUS_OFFSET = 0x0080;
constexpr std::uint32_t ILA_SAMPLE_ADDR_OFFSET = 0x0084;
constexpr std::uint32_t ILA_W_DATA = 585;
constexpr std::uint32_t ILA_DEPTH = 8192;
constexpr std::uint32_t ILA_NUM_RAM = (ILA_W_DATA + 31) / 32; // 19
constexpr std::uint32_t ILA_W_ADDR = 13;                      // log2(8192)
constexpr std::uint32_t ILA_W_RAM = 5;                        // ceil(log2(19))

constexpr std::uint32_t ILA_CTRL_ENABLE = 0x0000'0001;
constexpr std::uint32_t ILA_CTRL_RESET = 0x0000'0002;
constexpr std::uint32_t ILA_CTRL_DISABLE = 0x0000'0000;
constexpr std::uint32_t ILA_STATUS_DONE = 0x2;

// Signal bit positions within the 521-bit captured word.
constexpr std::uint32_t ILA_TVALID_BIT = 512;
constexpr std::uint32_t ILA_TLAST_BIT = 513;
constexpr std::uint32_t ILA_WR_TCNT_LSB = 514;
constexpr std::uint32_t ILA_WR_TCNT_WIDTH = 7;

// ============================================================================
// Helper Functions
// ============================================================================

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::uint32_t load_le_u32(const std::uint8_t *bytes) {
  std::uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

std::uint64_t parse_scalar(const std::string &content,
                           const std::string &field_name) {
  std::size_t pos = content.find(field_name + ":");
  if (pos == std::string::npos)
    return 0;

  pos = content.find(':', pos);
  if (pos == std::string::npos)
    return 0;

  std::size_t end_pos = content.find_first_of("\n[", pos + 1);
  if (end_pos == std::string::npos)
    end_pos = content.length();

  std::string value_str = content.substr(pos + 1, end_pos - pos - 1);
  value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
  value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);

  try {
    return std::stoull(value_str);
  } catch (...) {
    return 0;
  }
}

/// @brief Derive num_measurements from D_sparse in the config.
///
/// D_sparse encodes the detector-measurement matrix in row-major order with
/// -1 as the row delimiter.  The column indices are measurement indices, so
/// max(column indices) + 1 = num_measurements.  Returns 0 if D_sparse is
/// absent or empty.
std::size_t derive_num_measurements(const std::vector<std::int64_t> &d_sparse) {
  std::size_t span = 0;
  for (const auto value : d_sparse) {
    if (value < 0)
      continue;
    const auto index = static_cast<std::uint64_t>(value);
    if (index >= std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("D_sparse measurement index is too large");
    span = std::max(span, static_cast<std::size_t>(index + 1));
  }
  return span;
}

// ============================================================================
// Syndrome Data
// ============================================================================

struct SyndromeEntry {
  std::vector<std::uint8_t> measurements; ///< flat concat of all rounds
  /// Per-round measurement slices (from ROUND_START markers).  Used by the
  /// --per-round playback mode to emit one enqueue frame per round.  Empty (or
  /// a single slice) when the syndrome file has no per-round structure.
  std::vector<std::vector<std::uint8_t>> per_round;
  std::uint8_t expected_correction;
};

std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                          std::size_t syndrome_size) {
  std::vector<SyndromeEntry> entries;
  std::ifstream file(path);
  if (!file.good())
    return entries;

  std::string line;
  std::vector<std::uint8_t> current_shot;  // flat (all rounds)
  std::vector<std::uint8_t> current_round; // bits since ROUND_START
  std::vector<std::vector<std::uint8_t>> current_rounds; // rounds of this shot
  std::vector<std::vector<std::uint8_t>> shots;
  std::vector<std::vector<std::vector<std::uint8_t>>> shots_rounds;
  std::vector<std::uint8_t> corrections;
  bool reading_shot = false;
  bool reading_corrections = false;
  bool saw_round_marker = false;

  auto close_round = [&]() {
    if (!current_round.empty()) {
      current_rounds.push_back(current_round);
      current_round.clear();
    }
  };
  auto close_shot = [&]() {
    close_round();
    if (reading_shot && !current_shot.empty()) {
      shots.push_back(current_shot);
      shots_rounds.push_back(current_rounds);
    }
    current_shot.clear();
    current_rounds.clear();
  };

  while (std::getline(file, line)) {
    if (line.find("SHOT_START") == 0) {
      close_shot();
      reading_shot = true;
      reading_corrections = false;
      continue;
    }
    if (line == "CORRECTIONS_START") {
      close_shot();
      reading_shot = false;
      reading_corrections = true;
      continue;
    }
    if (line == "CORRECTIONS_END") {
      break;
    }
    if (line.find("NUM_DATA") == 0 || line.find("NUM_LOGICAL") == 0) {
      continue;
    } else if (line.find("ROUND_START") == 0) {
      // A new round within the current shot: close the prior round slice.
      saw_round_marker = true;
      close_round();
      continue;
    } else if (reading_shot) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        current_shot.push_back(static_cast<std::uint8_t>(bit));
        current_round.push_back(static_cast<std::uint8_t>(bit));
      } catch (...) {
      }
    } else if (reading_corrections) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        corrections.push_back(static_cast<std::uint8_t>(bit));
      } catch (...) {
      }
    }
  }
  close_shot();
  (void)saw_round_marker;

  for (std::size_t i = 0; i < shots.size(); ++i) {
    if (shots[i].size() < syndrome_size) {
      shots[i].resize(syndrome_size, 0);
    } else if (shots[i].size() > syndrome_size) {
      shots[i].resize(syndrome_size);
    }
    SyndromeEntry entry{};
    entry.measurements = std::move(shots[i]);
    entry.per_round = (i < shots_rounds.size())
                          ? std::move(shots_rounds[i])
                          : std::vector<std::vector<std::uint8_t>>{};
    // No per-round markers: treat the whole shot as a single round.
    if (entry.per_round.empty())
      entry.per_round.push_back(entry.measurements);
    entry.expected_correction = (i < corrections.size()) ? corrections[i] : 0;
    entries.push_back(std::move(entry));
  }

  return entries;
}

/// Build an RPC request message.
/// Layout: [RPCHeader (24 bytes, ptp_timestamp zeroed)][syndrome
/// measurements...] The FPGA overwrites header bytes 16-23 (ptp_timestamp
/// field) with the PTP send timestamp at transmit time.
std::vector<std::uint8_t>
build_rpc_payload(const std::vector<std::uint8_t> &measurements,
                  std::uint32_t function_id, std::uint32_t request_id = 0) {
  std::size_t arg_len = measurements.size();
  std::vector<std::uint8_t> payload(
      sizeof(cudaq::realtime::RPCHeader) + arg_len, 0);

  auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(payload.data());
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = function_id;
  header->arg_len = static_cast<std::uint32_t>(arg_len);
  header->request_id = request_id;
  header->ptp_timestamp = 0;

  std::memcpy(payload.data() + sizeof(cudaq::realtime::RPCHeader),
              measurements.data(), measurements.size());
  return payload;
}

/// Build a per-round `enqueue_syndromes` RPC frame for the device-graph
/// scheduler (decoder_server_runtime.md#enqueue_syndromes):
///   [RPCHeader (fid=enqueue, arg_len)][EnqueueRequestPayload (32B)]
///   [ceil(n/8) LSB-first bit-packed syndrome bytes, no pad].
/// One input byte (0/1) per syndrome bit.  ptp_timestamp left 0 (FPGA fills
/// it).
std::vector<std::uint8_t>
build_enqueue_frame(const std::vector<std::uint8_t> &round_bits,
                    std::uint32_t request_id, std::int64_t decoder_id,
                    std::int64_t counter) {
  namespace rpc = cudaq::qec::decoding::rpc;
  const std::uint64_t n = round_bits.size();
  const std::size_t packed = rpc::bit_packed_bytes(n);
  const std::size_t arg_len = sizeof(rpc::EnqueueRequestPayload) + packed;
  std::vector<std::uint8_t> payload(
      sizeof(cudaq::realtime::RPCHeader) + arg_len, 0);
  auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(payload.data());
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = rpc::kEnqueueSyndromesFunctionId;
  header->arg_len = static_cast<std::uint32_t>(arg_len);
  header->request_id = request_id;
  header->ptp_timestamp = 0;

  auto *body = reinterpret_cast<rpc::EnqueueRequestPayload *>(
      payload.data() + sizeof(cudaq::realtime::RPCHeader));
  body->decoder_id = decoder_id;
  body->counter = counter;
  body->syndrome_mapping_id = 0;
  body->num_syndromes = static_cast<std::int64_t>(n);

  std::uint8_t *bits = payload.data() + sizeof(cudaq::realtime::RPCHeader) +
                       sizeof(rpc::EnqueueRequestPayload);
  for (std::uint64_t i = 0; i < n; ++i)
    if (round_bits[i] & 0x1u)
      bits[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
  return payload;
}

/// Build a `get_corrections` RPC frame (decoder_server_runtime.md):
///   [RPCHeader (fid=get_corrections, arg_len=17)]
///   [GetCorrectionsRequestPayload{decoder_id, return_size, reset}].
std::vector<std::uint8_t> build_get_corrections_frame(std::uint32_t request_id,
                                                      std::int64_t decoder_id,
                                                      std::int64_t return_size,
                                                      std::uint8_t reset) {
  namespace rpc = cudaq::qec::decoding::rpc;
  const std::size_t arg_len = sizeof(rpc::GetCorrectionsRequestPayload);
  std::vector<std::uint8_t> payload(
      sizeof(cudaq::realtime::RPCHeader) + arg_len, 0);
  auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(payload.data());
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = rpc::kGetCorrectionsFunctionId;
  header->arg_len = static_cast<std::uint32_t>(arg_len);
  header->request_id = request_id;
  header->ptp_timestamp = 0;

  auto *body = reinterpret_cast<rpc::GetCorrectionsRequestPayload *>(
      payload.data() + sizeof(cudaq::realtime::RPCHeader));
  body->decoder_id = decoder_id;
  body->return_size = return_size;
  body->reset = reset;
  return payload;
}

// ============================================================================
// Command-Line Options
// ============================================================================

struct Options {
  std::string hsb_ip;
  std::string data_dir;
  std::string config_file; // explicit config path (overrides data_dir default)
  std::string
      syndromes_file; // explicit syndromes path (overrides data_dir default)
  std::string
      function_name; // RPC function name (overrides default mock_decode)
  std::optional<std::size_t> num_shots;
  bool verify = false;

  // Per-round mode (device-graph scheduler): emit N enqueue_syndromes frames
  // (one per round) followed by 1 get_corrections frame per shot, instead of a
  // single full-window request.  request_id = shot*(rounds+1)+local_frame.
  bool per_round = false;

  // When set, bypass BOOTP enumeration and connect directly to this UDP port.
  // Used with the FPGA emulator which doesn't support BOOTP.
  std::optional<std::uint16_t> control_port;
  std::uint32_t vp_address = 0x1000;  // VP register base (sensor 0 default)
  std::uint32_t hif_address = 0x0800; // HIF register base

  // RDMA target configuration (from bridge tool output).
  // When all three are provided, the FPGA SIF registers are configured
  // so playback data is sent via RDMA to the bridge tool's GPU buffers.
  std::optional<std::uint32_t> qp_number;
  std::optional<std::uint32_t> rkey;
  std::optional<std::uint64_t> buffer_addr;
  std::optional<std::uint32_t> rdma_page_size; // ring buffer slot size (bytes)
  std::optional<std::uint32_t> rdma_num_pages; // number of ring buffer slots
  std::uint32_t spacing_us =
      DEFAULT_TIMER_SPACING_US; // per-frame playback pacing (us): the FPGA
                                // timer fires once per BRAM window (= one
                                // frame), so a shot spans
                                // frames-per-shot x spacing
};

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " --hsb-ip <ip> --data-dir <path> [options]\n\n"
      << "Required:\n"
      << "  --hsb-ip <ip>       FPGA or emulator IP address\n"
      << "  --data-dir <path>     Path to syndrome data directory\n"
      << "                        (expects config_multi_err_lut.yml and\n"
      << "                         syndromes_multi_err_lut.txt inside)\n\n"
      << "Optional:\n"
      << "  --config <path>       Explicit config YAML path (overrides "
         "--data-dir default)\n"
      << "  --syndromes <path>    Explicit syndromes file path (overrides "
         "--data-dir default)\n"
      << "  --function-name <s>   RPC function name for dispatch "
         "(default: mock_decode)\n"
      << "  --num-shots <n>       Number of shots to play back (default: all)\n"
      << "  --spacing <us>        Per-frame playback pacing in microseconds "
         "(default: 10;\n"
         "                        the FPGA timer fires once per frame, so a "
         "shot spans\n"
         "                        frames-per-shot x spacing)\n"
      << "  --verify              Capture and verify correction responses "
         "via ILA\n"
      << "  --per-round           Per-round protocol (device-graph scheduler): "
         "send\n"
      << "                        N enqueue_syndromes frames (one per "
         "ROUND_START\n"
      << "                        slice) + 1 get_corrections frame per shot. "
         "Note:\n"
      << "                        multiplies frames by rounds+1, so the 512-"
         "cycle\n"
      << "                        BRAM limits --num-shots accordingly.\n\n"
      << "Emulator mode (bypass BOOTP enumeration):\n"
      << "  --control-port <n>   UDP control port of the emulator\n"
      << "  --vp-address <n>     VP register base (default: 0x1000)\n"
      << "  --hif-address <n>    HIF register base (default: 0x0800)\n\n"
      << "RDMA target (from bridge tool output):\n"
      << "  --qp-number <n>      Destination QP number (hex or decimal)\n"
      << "  --rkey <n>            Remote key\n"
      << "  --buffer-addr <n>    GPU buffer address (hex or decimal)\n"
      << "  --page-size <n>      Ring buffer slot size in bytes (default: "
         "256)\n"
      << "  --num-pages <n>      Number of ring buffer slots (default: 64)\n"
      << "\n"
      << "When --qp-number, --rkey, and --buffer-addr are all provided, the\n"
      << "FPGA SIF registers are configured to send playback data via RDMA\n"
      << "to the bridge tool's GPU buffers.\n";
}

Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--hsb-ip" && i + 1 < argc) {
      options.hsb_ip = argv[++i];
    } else if (arg == "--data-dir" && i + 1 < argc) {
      options.data_dir = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      options.config_file = argv[++i];
    } else if (arg == "--syndromes" && i + 1 < argc) {
      options.syndromes_file = argv[++i];
    } else if (arg == "--function-name" && i + 1 < argc) {
      options.function_name = argv[++i];
    } else if (arg == "--num-shots" && i + 1 < argc) {
      options.num_shots = std::stoull(argv[++i]);
    } else if (arg == "--spacing" && i + 1 < argc) {
      options.spacing_us =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--verify") {
      options.verify = true;
    } else if (arg == "--per-round") {
      options.per_round = true;
    } else if (arg == "--qp-number" && i + 1 < argc) {
      options.qp_number =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--rkey" && i + 1 < argc) {
      options.rkey =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--buffer-addr" && i + 1 < argc) {
      options.buffer_addr = std::stoull(argv[++i], nullptr, 0);
    } else if (arg == "--page-size" && i + 1 < argc) {
      options.rdma_page_size =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--num-pages" && i + 1 < argc) {
      options.rdma_num_pages =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--control-port" && i + 1 < argc) {
      options.control_port =
          static_cast<std::uint16_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--vp-address" && i + 1 < argc) {
      options.vp_address =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--hif-address" && i + 1 < argc) {
      options.hif_address =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
  }
  return options;
}

// ============================================================================
// BRAM Write
// ============================================================================

/// Compute the address-width parameter for the playback BRAM (log2 of depth).
std::uint32_t bram_w_sample_addr() {
  std::uint32_t w = 0;
  while ((1u << w) < RAM_DEPTH)
    ++w;
  return w;
}

void write_bram(hololink::Hololink &hsb,
                const std::vector<std::vector<std::uint8_t>> &windows,
                std::size_t bytes_per_window) {
  if (bytes_per_window % 64 != 0)
    throw std::runtime_error("bytes_per_window must be a multiple of 64");

  std::size_t cycles = bytes_per_window / 64;
  if (cycles == 0)
    throw std::runtime_error("bytes_per_window is too small");

  if (windows.size() * cycles > RAM_DEPTH) {
    std::ostringstream msg;
    msg << "Requested " << windows.size() << " windows with " << cycles
        << " cycles each exceeds RAM depth " << RAM_DEPTH;
    throw std::runtime_error(msg.str());
  }

  const std::uint32_t w_sample_addr = bram_w_sample_addr();

  // HSB WR_BLOCK packet has 6-byte header + 8 bytes per (addr, value)
  // pair. With CONTROL_PACKET_SIZE=1472, max pairs = (1472-6)/8 = 183.
  constexpr std::size_t kBatchWrites = 180;
  hololink::Hololink::WriteData write_data;

  for (std::size_t w = 0; w < windows.size(); ++w) {
    const auto &window = windows[w];
    for (std::size_t s = 0; s < cycles; ++s) {
      for (std::size_t i = 0; i < RAM_NUM; ++i) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t value = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size()) {
          value = load_le_u32(window.data() + byte_offset);
        }

        auto ram_addr = static_cast<std::uint32_t>(i << (w_sample_addr + 2));
        auto sample_addr = static_cast<std::uint32_t>((s + (w * cycles)) * 0x4);
        std::uint32_t address = RAM_ADDR + ram_addr + sample_addr;

        write_data.queue_write_uint32(address, value);
        if (write_data.size() >= kBatchWrites) {
          if (!hsb.write_uint32(write_data))
            throw std::runtime_error("Failed to write BRAM batch");
          write_data = hololink::Hololink::WriteData();
        }
      }
    }
  }

  if (write_data.size() > 0) {
    if (!hsb.write_uint32(write_data))
      throw std::runtime_error("Failed to write BRAM batch");
  }
}

// ============================================================================
// Chunked block read helper
// ============================================================================

/// HSB RD_BLOCK packets have a 6-byte header plus 8 bytes per address.
/// With CONTROL_PACKET_SIZE=1472, the maximum number of contiguous 32-bit
/// registers that can be read in one RD_BLOCK is (1472-6)/8 = 183.
/// This helper splits a large block read into multiple chunks.
constexpr std::uint32_t kMaxBlockReadCount = 183;

std::tuple<bool, std::vector<std::uint32_t>>
chunked_read_uint32(hololink::Hololink &hl, std::uint32_t base_addr,
                    std::uint32_t count,
                    std::shared_ptr<hololink::Timeout> timeout =
                        hololink::Timeout::default_timeout()) {
  std::vector<std::uint32_t> result;
  result.reserve(count);
  std::uint32_t remaining = count;
  std::uint32_t offset = 0;
  while (remaining > 0) {
    std::uint32_t chunk = std::min(remaining, kMaxBlockReadCount);
    auto [ok, data] = hl.read_uint32(base_addr + offset * 4, chunk, timeout);
    if (!ok)
      return {false, {}};
    result.insert(result.end(), data.begin(), data.end());
    offset += chunk;
    remaining -= chunk;
  }
  return {true, result};
}

// ============================================================================
// BRAM Readback Verification
// ============================================================================

/// Read back playback BRAM using block reads (one per RAM bank) and compare
/// against what was written.  Returns true if all words match.
bool verify_bram(hololink::Hololink &hsb,
                 const std::vector<std::vector<std::uint8_t>> &windows,
                 std::size_t bytes_per_window) {
  const std::size_t cycles = bytes_per_window / 64;
  const auto total_cycles = static_cast<std::uint32_t>(windows.size() * cycles);
  const std::uint32_t w_sample_addr = bram_w_sample_addr();

  bool all_ok = true;
  std::size_t mismatches = 0;

  for (std::uint32_t i = 0; i < RAM_NUM; ++i) {
    std::uint32_t bank_base = RAM_ADDR + (i << (w_sample_addr + 2));
    auto [ok, readback] = chunked_read_uint32(hsb, bank_base, total_cycles);
    if (!ok) {
      std::cerr << "BRAM readback: failed to read bank " << i << "\n";
      return false;
    }

    for (std::size_t w = 0; w < windows.size(); ++w) {
      const auto &window = windows[w];
      for (std::size_t s = 0; s < cycles; ++s) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t expected = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size())
          expected = load_le_u32(window.data() + byte_offset);

        std::size_t sample_idx = w * cycles + s;
        std::uint32_t actual = readback[sample_idx];

        if (actual != expected) {
          if (mismatches < 10) {
            std::cerr << "  BRAM mismatch: bank=" << i
                      << " sample=" << sample_idx << " expected=0x" << std::hex
                      << expected << " got=0x" << actual << std::dec << "\n";
          }
          all_ok = false;
          ++mismatches;
        }
      }
    }
  }

  if (mismatches > 10) {
    std::cerr << "  ... and " << (mismatches - 10) << " more mismatches\n";
  }

  return all_ok;
}

// ============================================================================
// ILA Capture Functions
// ============================================================================

void ila_reset(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_RESET))
    throw std::runtime_error("ILA reset write failed");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable-after-reset write failed");
}

void ila_enable(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_ENABLE))
    throw std::runtime_error("ILA enable write failed");
}

void ila_disable(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable write failed");
}

/// Read the current ILA sample write address (number of samples captured).
std::uint32_t ila_sample_count(hololink::Hololink &hl) {
  return hl.read_uint32(ILA_BASE_ADDR + ILA_SAMPLE_ADDR_OFFSET);
}

/// Poll the ILA sample count register until at least @p expected samples have
/// been captured, or @p timeout_ms elapses.
bool ila_wait_for_samples(hololink::Hololink &hl,
                          std::uint32_t expected_samples, int timeout_ms) {
  int polls = std::max(1, timeout_ms / 100);
  for (int i = 0; i < polls; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::uint32_t count = ila_sample_count(hl);
    if (count >= expected_samples)
      return true;
  }
  return false;
}

/// Read @p num_samples captured ILA samples using block reads (one per bank).
/// Returns vector of samples; each sample is ILA_NUM_RAM uint32 words ordered
/// LSW-first (word[0] = bits [31:0], word[1] = bits [63:32], ...).
std::vector<std::vector<std::uint32_t>> ila_dump(hololink::Hololink &hl,
                                                 std::uint32_t num_samples) {
  constexpr std::uint32_t ctrl_switch = 1u << (ILA_W_ADDR + 2 + ILA_W_RAM);
  auto timeout = hololink::Timeout::default_timeout();

  // Read each bank using chunked block reads.
  std::vector<std::vector<std::uint32_t>> bank_data(ILA_NUM_RAM);
  for (std::uint32_t y = 0; y < ILA_NUM_RAM; ++y) {
    std::uint32_t bank_base =
        ILA_BASE_ADDR + ctrl_switch + (y << (ILA_W_ADDR + 2));
    auto [ok, data] = chunked_read_uint32(hl, bank_base, num_samples, timeout);
    if (!ok)
      throw std::runtime_error("Failed to read ILA bank " + std::to_string(y));
    bank_data[y] = std::move(data);
  }

  // Transpose bank_data[bank][sample] → samples[sample][bank].
  std::vector<std::vector<std::uint32_t>> samples(
      num_samples, std::vector<std::uint32_t>(ILA_NUM_RAM));
  for (std::uint32_t i = 0; i < num_samples; ++i) {
    for (std::uint32_t y = 0; y < ILA_NUM_RAM; ++y) {
      samples[i][y] = bank_data[y][i];
    }
  }
  return samples;
}

// ============================================================================
// ILA Sample Field Extraction
// ============================================================================

/// Extract a single bit from a wide sample (array of uint32 words, LSW first).
bool extract_bit(const std::vector<std::uint32_t> &sample,
                 std::uint32_t bit_pos) {
  std::uint32_t word_idx = bit_pos / 32;
  std::uint32_t bit_idx = bit_pos % 32;
  return (sample.at(word_idx) >> bit_idx) & 1;
}

/// Extract a multi-bit field (up to 32 bits) from a wide sample.
std::uint32_t extract_field(const std::vector<std::uint32_t> &sample,
                            std::uint32_t lsb, std::uint32_t width) {
  std::uint32_t result = 0;
  for (std::uint32_t b = 0; b < width && b < 32; ++b) {
    std::uint32_t bit_pos = lsb + b;
    std::uint32_t word_idx = bit_pos / 32;
    std::uint32_t bit_idx = bit_pos % 32;
    if ((sample.at(word_idx) >> bit_idx) & 1)
      result |= (1u << b);
  }
  return result;
}

/// Extract raw bytes from the 512-bit data field (bits [511:0]) of a captured
/// ILA sample.  The data bus occupies words[0] through words[15].
std::vector<std::uint8_t>
extract_tdata_bytes(const std::vector<std::uint32_t> &sample,
                    std::size_t num_bytes) {
  std::vector<std::uint8_t> bytes(num_bytes, 0);
  for (std::size_t i = 0; i < num_bytes && i < 64; ++i) {
    std::uint32_t word_idx = static_cast<std::uint32_t>(i / 4);
    std::uint32_t byte_idx = static_cast<std::uint32_t>(i % 4);
    bytes[i] =
        static_cast<std::uint8_t>((sample[word_idx] >> (byte_idx * 8)) & 0xFF);
  }
  return bytes;
}

// ============================================================================
// PTP Timestamp Helpers
// ============================================================================

/// Extract the 64-bit current_ptp_timestamp from ILA bits [584:521].
std::uint64_t
extract_ila_ptp_timestamp(const std::vector<std::uint32_t> &sample) {
  uint64_t raw = 0;
  for (int b = 0; b < 64; ++b) {
    uint32_t bit_pos = 521 + b;
    uint32_t w = bit_pos / 32;
    uint32_t off = bit_pos % 32;
    if ((sample[w] >> off) & 1)
      raw |= (uint64_t(1) << b);
  }
  return raw;
}

/// Extract the echoed PTP send timestamp from RPCResponse.ptp_timestamp.
std::uint64_t
extract_echoed_ptp_timestamp(const cudaq::realtime::RPCResponse &resp) {
  return resp.ptp_timestamp;
}

struct PtpTimestamp {
  uint32_t sec;
  uint32_t nsec;
};

PtpTimestamp decode_ptp(uint64_t raw) {
  return {static_cast<uint32_t>(raw >> 32),
          static_cast<uint32_t>(raw & 0xFFFF'FFFF)};
}

int64_t ptp_delta_ns(PtpTimestamp send, PtpTimestamp recv) {
  int64_t d_sec = static_cast<int64_t>(recv.sec) - send.sec;
  int64_t d_nsec = static_cast<int64_t>(recv.nsec) - send.nsec;
  return d_sec * 1'000'000'000LL + d_nsec;
}

struct LatencySample {
  uint32_t request_id; ///< echoed request_id of the frame
  uint32_t shot;       ///< request_id / frames_per_shot (per-round)
  uint32_t local;      ///< frame index within the shot (per-round); else 0
  bool is_corr;        ///< true = get_corrections frame, false = enqueue ACK
  uint32_t send_sec, send_nsec;
  uint32_t recv_sec, recv_nsec;
  int64_t delta_ns;
};

// ============================================================================
// Correction Verification
// ============================================================================

struct VerifyResult {
  std::size_t total_samples = 0;
  std::size_t unique_shots_verified = 0;
  std::size_t responses_matched = 0;
  std::size_t header_errors = 0;
  std::size_t correction_errors = 0;
  std::size_t rpc_responses = 0;
  std::size_t non_rpc_frames = 0;
  std::size_t tvalid_zero = 0;
  std::size_t enqueue_acks = 0; ///< per-round: result_len==0 ACK responses
  std::vector<LatencySample> latency_samples;
};

/// Scan captured ILA samples for RPC correction responses and compare each
/// against the expected values from the syndromes file.
///
/// Each captured 512-bit data word should contain a complete RPCResponse:
///   bytes [0:3]   RPCResponse.magic         = 0x43555153 (RPC_MAGIC_RESPONSE)
///   bytes [4:7]   RPCResponse.status        = 0 (success)
///   bytes [8:11]  RPCResponse.result_len    = num_observables
///   bytes [12:15] RPCResponse.request_id    = shot index (echoed from request)
///   bytes [16:23] RPCResponse.ptp_timestamp = echoed PTP send timestamp
///   byte  [24]    correction value
VerifyResult verify_captured_responses(
    const std::vector<std::vector<std::uint32_t>> &samples,
    const std::vector<SyndromeEntry> &syndromes, std::size_t num_expected,
    bool per_round = false, std::size_t frames_per_shot = 1,
    std::size_t ring_depth = 64) {
  VerifyResult result;
  result.total_samples = samples.size();
  std::set<std::uint32_t> shots_seen;
  // DIAGNOSTIC census: count how many RPC responses carry each request_id, so a
  // duplicate-processed frame (count==2) and a dropped frame (count==0) are
  // identified exactly -- and we can check whether they alias the same ring
  // slot (rid % num_pages).
  std::map<std::uint32_t, int> rid_count;

  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto &sample = samples[i];

    bool tvalid = extract_bit(sample, ILA_TVALID_BIT);

    if (!tvalid) {
      result.tvalid_zero++;
      continue;
    }

    constexpr std::size_t kResponseSize =
        sizeof(cudaq::realtime::RPCResponse) + 1;
    auto data_bytes = extract_tdata_bytes(sample, kResponseSize);

    cudaq::realtime::RPCResponse resp{};
    std::memcpy(&resp, data_bytes.data(), sizeof(resp));
    std::uint8_t correction_byte =
        data_bytes[sizeof(cudaq::realtime::RPCResponse)];

    // The ILA captures the FPGA-bound (TX) response path only, so every RPC
    // frame here carries RPC_MAGIC_RESPONSE; anything else is a non-RPC frame
    // (e.g. a continuation beat of a multi-beat AXI transfer).
    if (resp.magic != cudaq::realtime::RPC_MAGIC_RESPONSE) {
      result.non_rpc_frames++;
      continue;
    }
    result.rpc_responses++;
    if (per_round)
      rid_count[resp.request_id]++;

    if (resp.status != 0) {
      result.header_errors++;
      continue;
    }

    // Frame identity (per-round): shot, local frame index within the shot, and
    // whether this is the get_corrections frame (result_len>0) vs an enqueue
    // ACK (result_len==0).
    const std::size_t frames = frames_per_shot ? frames_per_shot : 1;
    const std::uint32_t shot_index =
        per_round ? static_cast<std::uint32_t>(resp.request_id / frames)
                  : resp.request_id;
    const std::uint32_t local =
        per_round ? static_cast<std::uint32_t>(resp.request_id % frames) : 0;
    const bool is_corr = (!per_round) || (resp.result_len != 0);

    // PTP round-trip latency for EVERY response -- enqueue ACKs AND
    // get_corrections.  The FPGA injects a send timestamp into every playback
    // frame and the ILA captures a recv timestamp for every captured sample,
    // so all frames carry valid PTP.  Recorded here (before the ACK skip
    // below) so the CSV has one row per captured frame, not just one per shot.
    {
      uint64_t send_raw = extract_echoed_ptp_timestamp(resp);
      uint64_t recv_raw = extract_ila_ptp_timestamp(sample);
      if (send_raw != 0 && recv_raw != 0) {
        auto send_ts = decode_ptp(send_raw);
        auto recv_ts = decode_ptp(recv_raw);
        int64_t delta = ptp_delta_ns(send_ts, recv_ts);
        result.latency_samples.push_back({resp.request_id, shot_index, local,
                                          is_corr, send_ts.sec, send_ts.nsec,
                                          recv_ts.sec, recv_ts.nsec, delta});
      }
    }

    // Per-round: enqueue_syndromes responses are empty ACKs (result_len==0);
    // only get_corrections responses (result_len>0) carry corrections.
    if (per_round && resp.result_len == 0) {
      result.enqueue_acks++;
      // DIAGNOSTIC: an empty response at the get_corrections frame position
      // (local index == rounds) would be a get_corrections that came back
      // result_len==0 (the old 99/100 anomaly).  Dump what we know about it.
      const std::size_t rounds = frames ? frames - 1 : 0;
      if (local == rounds) {
        std::cout << "  [ANOMALY] sample " << i << ": result_len==0 at "
                  << "get_corrections position; request_id=" << resp.request_id
                  << " shot=" << shot_index << " status=" << resp.status
                  << " magic=0x" << std::hex << resp.magic << std::dec
                  << " corr_byte=" << static_cast<int>(correction_byte) << "\n";
      }
      continue;
    }

    if (shot_index >= syndromes.size()) {
      std::cout << "  Sample " << i << ": request_id=" << resp.request_id
                << " -> shot=" << shot_index
                << " out of range (num_shots=" << syndromes.size()
                << ") [WARN]\n";
      result.correction_errors++;
      continue;
    }

    // Per-round get_corrections returns bit-packed observables (LSB-first);
    // compare observable bit 0 to the expected logical correction.  The
    // shot-based path compares the raw correction byte as before.
    std::uint8_t got = per_round
                           ? static_cast<std::uint8_t>(correction_byte & 0x1u)
                           : correction_byte;
    std::uint8_t expected = syndromes[shot_index].expected_correction;
    if (got == expected) {
      result.responses_matched++;
    } else {
      std::cout << "  Sample " << i << " request_id=" << resp.request_id
                << " shot=" << shot_index << ": got=" << static_cast<int>(got)
                << " expected=" << static_cast<int>(expected) << " [FAIL]\n";
      result.correction_errors++;
    }

    shots_seen.insert(shot_index);
  }

  result.unique_shots_verified = shots_seen.size();
  std::cout << "  Unique shots verified:  " << shots_seen.size() << " of "
            << num_expected << "\n";

  // DIAGNOSTIC: list any shots that never produced a get_corrections response.
  if (shots_seen.size() < num_expected) {
    std::cout << "  [ANOMALY] missing shots (no correction response):";
    for (std::uint32_t s = 0; s < num_expected; ++s)
      if (!shots_seen.count(s))
        std::cout << " " << s;
    std::cout << "\n";
  }

  // DIAGNOSTIC: per-request_id census -- report duplicated (count>1) and
  // dropped (count==0) frames with their (shot, local-frame) and ring slot, to
  // confirm a slot-aliasing race between the scheduler's flag clear and the
  // HSB RX kernel refilling reused slots.
  if (per_round && frames_per_shot) {
    const std::uint32_t total_frames =
        static_cast<std::uint32_t>(num_expected * frames_per_shot);
    auto describe = [&](std::uint32_t rid) {
      const std::uint32_t local = rid % frames_per_shot;
      const std::uint32_t shot = rid / frames_per_shot;
      const char *kind =
          (local + 1 == frames_per_shot) ? "get_corrections" : "enqueue";
      std::cout << "    rid=" << rid << " (shot=" << shot << " local=" << local
                << " " << kind << " slot%" << ring_depth << "="
                << (ring_depth ? rid % ring_depth : 0) << ")";
    };
    bool any = false;
    for (std::uint32_t rid = 0; rid < total_frames; ++rid) {
      int c = rid_count.count(rid) ? rid_count[rid] : 0;
      if (c != 1) {
        if (!any) {
          std::cout << "  [ANOMALY] request_id census (expected each seen "
                       "exactly once):\n";
          any = true;
        }
        describe(rid);
        std::cout << " seen " << c << " times\n";
      }
    }
  }

  return result;
}

} // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  Options options = parse_args(argc, argv);
  // --data-dir is required unless both --config and --syndromes are given
  bool has_explicit_files =
      !options.config_file.empty() && !options.syndromes_file.empty();
  if (options.hsb_ip.empty() ||
      (options.data_dir.empty() && !has_explicit_files)) {
    print_usage(argv[0]);
    return 1;
  }

  // ------------------------------------------------------------------
  // Load configuration and syndrome data
  // ------------------------------------------------------------------
  std::string config_path = options.config_file.empty()
                                ? options.data_dir + "/config_multi_err_lut.yml"
                                : options.config_file;
  std::string syndromes_path =
      options.syndromes_file.empty()
          ? options.data_dir + "/syndromes_multi_err_lut.txt"
          : options.syndromes_file;

  std::ifstream config_file(config_path);
  if (!config_file.good()) {
    std::cerr << "Could not open config file: " << config_path << "\n";
    return 1;
  }
  std::string config_content((std::istreambuf_iterator<char>(config_file)),
                             std::istreambuf_iterator<char>());

  std::size_t syndrome_size = 0;
  std::size_t num_measurements = 0;
  std::size_t num_observables = 0;
  if (options.per_round) {
    cudaq::qec::decoding::config::multi_decoder_config config;
    try {
      config =
          cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
              config_content);
    } catch (const std::exception &error) {
      std::cerr << "Invalid config file " << config_path << ": " << error.what()
                << "\n";
      return 1;
    }
    if (config.decoders.empty()) {
      std::cerr << "Config file contains no decoders\n";
      return 1;
    }

    const auto &decoder_config = config.decoders.front();
    if (decoder_config.syndrome_size >
        std::numeric_limits<std::size_t>::max()) {
      std::cerr << "syndrome_size is too large\n";
      return 1;
    }
    syndrome_size = static_cast<std::size_t>(decoder_config.syndrome_size);
    try {
      num_measurements = derive_num_measurements(decoder_config.D_sparse);
    } catch (const std::exception &error) {
      std::cerr << "Invalid config file " << config_path << ": " << error.what()
                << "\n";
      return 1;
    }
    num_observables = static_cast<std::size_t>(std::count(
        decoder_config.O_sparse.begin(), decoder_config.O_sparse.end(), -1));
  } else {
    // The predecoder playback path intentionally accepts a minimal YAML file
    // containing only syndrome_size; it is not a decoder configuration.
    const auto parsed_size = parse_scalar(config_content, "syndrome_size");
    if (parsed_size > std::numeric_limits<std::size_t>::max()) {
      std::cerr << "syndrome_size is too large\n";
      return 1;
    }
    syndrome_size = static_cast<std::size_t>(parsed_size);
  }

  if (syndrome_size == 0) {
    std::cerr << "Invalid syndrome_size in config file\n";
    return 1;
  }

  // num_measurements is the number of raw measurement bits per shot (used as
  // RPC payload size).  Derived from D_sparse (max column index + 1).
  // Non-per-round predecoder configs have no D matrix because their input
  // bits are already syndromes, so syndrome_size is their payload size.
  if (num_measurements == 0)
    num_measurements = syndrome_size;

  if (options.per_round && num_observables == 0) {
    std::cerr << "Per-round mode requires O_sparse in config file\n";
    return 1;
  }
  if (options.per_round && num_observables != 1) {
    std::cerr << "Per-round playback verification currently supports exactly "
                 "one observable (found "
              << num_observables << " from O_sparse)\n";
    return 1;
  }

  auto syndromes = load_syndromes(syndromes_path, num_measurements);
  if (syndromes.empty()) {
    std::cerr << "No syndrome data loaded from " << syndromes_path << "\n";
    return 1;
  }

  std::size_t num_shots = options.num_shots
                              ? std::min(*options.num_shots, syndromes.size())
                              : syndromes.size();
  if (num_shots == 0) {
    std::cerr << "No shots to play back\n";
    return 1;
  }

  // ------------------------------------------------------------------
  // Build RPC payloads and pad to 64-byte alignment
  // ------------------------------------------------------------------
  std::uint32_t function_id =
      options.function_name.empty()
          ? MOCK_DECODE_FUNCTION_ID
          : cudaq::realtime::fnv1a_hash(options.function_name.c_str());
  std::cout << "RPC function: "
            << (options.function_name.empty() ? "mock_decode"
                                              : options.function_name)
            << " (id=0x" << std::hex << function_id << std::dec << ")\n";

  // Per-round shape (device-graph scheduler protocol): each shot becomes
  // `rounds` enqueue_syndromes frames + 1 get_corrections frame.  rounds is the
  // number of ROUND_START slices in the syndrome file (1 if unstructured).
  const std::size_t rounds =
      options.per_round
          ? std::max<std::size_t>(1, syndromes.front().per_round.size())
          : 0;
  const std::size_t frames_per_shot = options.per_round ? rounds + 1 : 1;

  std::vector<std::vector<std::uint8_t>> windows;
  windows.reserve(num_shots * frames_per_shot);
  if (options.per_round) {
    for (std::size_t s = 0; s < num_shots; ++s) {
      const auto &pr = syndromes[s].per_round;
      for (std::size_t r = 0; r < rounds; ++r) {
        const std::vector<std::uint8_t> &bits =
            (r < pr.size()) ? pr[r] : syndromes[s].measurements;
        auto rid = static_cast<std::uint32_t>(s * frames_per_shot + r);
        windows.push_back(build_enqueue_frame(bits, rid, /*decoder_id=*/0,
                                              /*counter=*/rid));
      }
      auto grid = static_cast<std::uint32_t>(s * frames_per_shot + rounds);
      windows.push_back(build_get_corrections_frame(
          grid, /*decoder_id=*/0, num_observables, /*reset=*/1));
    }
    std::cout << "Per-round mode: " << rounds << " enqueue + 1 get_corrections "
              << "per shot (" << frames_per_shot << " frames/shot)\n";
  } else {
    for (std::size_t i = 0; i < num_shots; ++i)
      windows.push_back(build_rpc_payload(syndromes[i].measurements,
                                          function_id,
                                          static_cast<std::uint32_t>(i)));
  }

  // Frames may differ in size (per-round); pad all to the largest, 64-aligned.
  std::size_t payload_size = 0;
  for (const auto &w : windows)
    payload_size = std::max(payload_size, w.size());
  std::size_t bytes_per_window = align_up(payload_size, 64);

  for (auto &window : windows)
    window.resize(bytes_per_window, 0);

  std::size_t cycles_per_window = bytes_per_window / 64;
  if (cycles_per_window == 0)
    cycles_per_window = 1;

  // Per-round multiplies frames by (rounds+1), so all shots rarely fit the
  // 512-cycle playback BRAM.  Auto-clamp num_shots to what fits (and truncate
  // the already-built frames) so the run works without a hand-tuned
  // --num-shots; the >120-decode guarantee is owned by the Stage A
  // surface_code run, and hardware >120 would need multi-load chunking.
  if (options.per_round) {
    const std::size_t per_shot_cycles = frames_per_shot * cycles_per_window;
    const std::size_t max_shots =
        RAM_DEPTH / std::max<std::size_t>(1, per_shot_cycles);
    if (max_shots == 0) {
      std::cerr << "A single shot's " << frames_per_shot << " frames x "
                << cycles_per_window << " cycles exceed the " << RAM_DEPTH
                << "-cycle playback BRAM\n";
      return 1;
    }
    if (num_shots > max_shots) {
      std::cout << "WARNING: per-round playback is BRAM-limited to "
                << max_shots << " shots (requested " << num_shots
                << "); truncating.\n";
      num_shots = max_shots;
      windows.resize(num_shots * frames_per_shot);
    }
  }

  const std::size_t num_windows = windows.size();
  if (num_windows * cycles_per_window > RAM_DEPTH) {
    std::cerr << "Data exceeds playback BRAM capacity: " << num_windows
              << " frames x " << cycles_per_window
              << " cycles = " << (num_windows * cycles_per_window) << " > "
              << RAM_DEPTH << " depth.  Reduce --num-shots"
              << (options.per_round ? " (per-round multiplies frames by "
                                      "rounds+1)"
                                    : "")
              << ".\n";
    return 1;
  }

  std::cout << "Loaded " << num_shots << " shots / " << num_windows
            << " frames (syndrome_size=" << syndrome_size
            << ", payload=" << payload_size
            << " bytes, padded=" << bytes_per_window << " bytes, "
            << cycles_per_window << " cycles/frame)\n";

  // ------------------------------------------------------------------
  // Connect to HSB (or emulator) and reset
  // ------------------------------------------------------------------
  hololink::Metadata channel_metadata;
  bool using_emulator = options.control_port.has_value();

  if (using_emulator) {
    // Direct connection to emulator — bypass BOOTP enumeration.
    // Construct synthetic metadata with the required fields.
    std::cout << "Using direct connection to emulator at " << options.hsb_ip
              << ":" << *options.control_port << "\n";
    channel_metadata["peer_ip"] = options.hsb_ip;
    channel_metadata["control_port"] =
        static_cast<std::int64_t>(*options.control_port);
    channel_metadata["serial_number"] = std::string("emulator");
    channel_metadata["sequence_number_checking"] = static_cast<std::int64_t>(0);
    channel_metadata["hsb_ip_version"] =
        static_cast<std::int64_t>(0x2501); // minimum required by DataChannel
    channel_metadata["fpga_uuid"] = std::string("emulator");
    channel_metadata["vp_mask"] = static_cast<std::int64_t>(0x1);
    channel_metadata["data_plane"] = static_cast<std::int64_t>(0);
    channel_metadata["sensor"] = static_cast<std::int64_t>(0);
    channel_metadata["sif_address"] = static_cast<std::int64_t>(0);
    channel_metadata["vp_address"] =
        static_cast<std::int64_t>(options.vp_address);
    channel_metadata["hif_address"] =
        static_cast<std::int64_t>(options.hif_address);
  } else {
    channel_metadata = hololink::Enumerator::find_channel(options.hsb_ip);
    hololink::DataChannel::use_sensor(channel_metadata, 0);
  }

  hololink::DataChannel hsb_channel(channel_metadata);
  auto hsb = hsb_channel.hololink();

  hsb->start();
  if (!using_emulator) {
    hsb->reset();
  }

  // ------------------------------------------------------------------
  // Configure FPGA SIF registers for RDMA target (if provided)
  // ------------------------------------------------------------------
  std::uint32_t rdma_num_pages = options.rdma_num_pages.value_or(64);
  if (options.qp_number && options.rkey && options.buffer_addr) {
    std::uint32_t rdma_page_size = options.rdma_page_size.value_or(256);

    std::cout << "Configuring FPGA SIF for RDMA target:\n"
              << "  QP number:    0x" << std::hex << *options.qp_number
              << std::dec << "\n"
              << "  RKEY:         " << *options.rkey << "\n"
              << "  Buffer addr:  0x" << std::hex << *options.buffer_addr
              << std::dec << "\n"
              << "  Page size:    " << rdma_page_size << " bytes\n"
              << "  Num pages:    " << rdma_num_pages << "\n"
              << "  Frame size:   " << bytes_per_window << " bytes\n";

    hsb_channel.authenticate(*options.qp_number, *options.rkey);
    // RoCE v2 uses IANA-assigned UDP destination port 4791. The FPGA
    // embeds this in outgoing RoCE packets; the host NIC silently drops
    // packets that arrive on any other port.
    constexpr std::uint32_t ROCEV2_UDP_PORT = 4791;
    hsb_channel.configure_roce(*options.buffer_addr, bytes_per_window,
                               rdma_page_size, rdma_num_pages, ROCEV2_UDP_PORT);

    std::cout << "FPGA SIF registers configured for RDMA" << std::endl;
  }

  // ------------------------------------------------------------------
  // Disable player, configure, and write BRAM
  // ------------------------------------------------------------------
  if (!hsb->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET, PLAYER_DISABLE))
    throw std::runtime_error("Failed to disable player");

  hololink::Hololink::WriteData config_write;
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_SIZE_OFFSET,
                                  static_cast<std::uint32_t>(bytes_per_window));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_NUMBER_OFFSET,
                                  static_cast<std::uint32_t>(num_windows));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_TIMER_OFFSET,
                                  RF_SOC_TIMER_SCALE * options.spacing_us);
  if (!hsb->write_uint32(config_write))
    throw std::runtime_error("Failed to configure player");

  std::cout << "Writing " << num_windows << " windows to playback BRAM..."
            << std::endl;
  try {
    write_bram(*hsb, windows, bytes_per_window);
    std::cout << "BRAM write completed successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "BRAM write FAILED: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Verifying playback BRAM contents..." << std::endl;
  try {
    if (!verify_bram(*hsb, windows, bytes_per_window)) {
      std::cerr << "BRAM readback verification FAILED\n";
      return 1;
    } else {
      std::cout << "BRAM readback verification PASSED\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "BRAM readback error: " << e.what() << std::endl;
    return 1;
  }

  // ------------------------------------------------------------------
  // Arm ILA capture (before playback) if --verify
  // ------------------------------------------------------------------
  if (options.verify) {
    std::cout << "\n=== Arming ILA capture (SIF TX at 0x" << std::hex
              << ILA_BASE_ADDR << std::dec << ") ===\n";
    ila_disable(*hsb);
    ila_reset(*hsb);
    ila_enable(*hsb);
    std::cout << "ILA: armed for capture\n";
  }

  // ------------------------------------------------------------------
  // Disable metadata packet (required for bitfile 0x0227+)
  // Needed for FPGA bitfile 0x0227+; comment out for older bitfiles (e.g.
  // 0x2601).
  // ------------------------------------------------------------------
  {
    std::uint32_t val = hsb->read_uint32(METADATA_PACKET_ADDR);
    if (!hsb->write_uint32(METADATA_PACKET_ADDR, val | (1u << 16)))
      throw std::runtime_error("Failed to disable metadata packet");
  }

  // ------------------------------------------------------------------
  // Set sensor TX streaming threshold to zero so captured responses
  // stream to the ILA immediately (required for small capture counts).
  // ------------------------------------------------------------------
  if (!hsb->write_uint32(SIF_TX_THRESHOLD_ADDR, SIF_TX_THRESHOLD_IMMEDIATE))
    throw std::runtime_error("Failed to set SIF TX streaming threshold");

  // ------------------------------------------------------------------
  // Enable playback
  // ------------------------------------------------------------------
  if (!hsb->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET, PLAYER_ENABLE))
    throw std::runtime_error("Failed to enable player");

  std::cout << "Playback enabled: " << num_shots << " shots / " << num_windows
            << " frames on hsb " << options.hsb_ip << "\n";

  // ------------------------------------------------------------------
  // ILA capture and correction verification
  // ------------------------------------------------------------------
  if (options.verify) {
    std::cout << "\n=== ILA Capture & Verification ===\n";

    constexpr int kPollIntervalMs = 500;
    constexpr int kVerifyTimeoutMs = 30000;
    const std::uint32_t expected_samples = static_cast<std::uint32_t>(
        std::min<std::size_t>(num_windows, ILA_DEPTH));
    std::cout << "Waiting for ILA capture to reach " << expected_samples
              << " samples (timeout " << kVerifyTimeoutMs << " ms)...\n";

    int elapsed = 0;
    while (elapsed < kVerifyTimeoutMs) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      elapsed += kPollIntervalMs;
      const std::uint32_t count = ila_sample_count(*hsb);
      if (count >= expected_samples)
        break;
    }

    std::uint32_t actual_samples = ila_sample_count(*hsb);
    ila_disable(*hsb);

    if (actual_samples < expected_samples) {
      std::cerr << "ILA: captured " << actual_samples << " of "
                << expected_samples << " expected samples (timeout "
                << kVerifyTimeoutMs << " ms)\n";
      return 1;
    }
    std::cout << "ILA: captured " << actual_samples << " samples\n";

    // Read captured data from ILA RAM banks.
    std::cout << "Reading ILA data RAM...\n";
    auto samples = ila_dump(*hsb, actual_samples);
    std::cout << "Read " << samples.size() << " samples from ILA\n";

    // Verify correction responses against expected values.
    auto vr = verify_captured_responses(samples, syndromes, num_shots,
                                        options.per_round, frames_per_shot,
                                        rdma_num_pages);

    // In per-round mode the response frames split into enqueue ACKs
    // (result_len==0) and get_corrections frames (result_len>0); only the
    // latter carry corrections.
    const std::size_t corrections_returned = vr.rpc_responses - vr.enqueue_acks;
    std::cout << "\n=== Verification Summary ===\n"
              << "  ILA samples captured:   " << actual_samples << "\n"
              << "  tvalid=0 (idle):        " << vr.tvalid_zero << "\n"
              << "  RPC response frames:    " << vr.rpc_responses << "\n";
    if (options.per_round)
      std::cout << "  Enqueue ACKs:           " << vr.enqueue_acks << "\n"
                << "  get_corrections frames: " << corrections_returned << "\n";
    std::cout << "  Non-RPC frames:         " << vr.non_rpc_frames << "\n"
              << "  Unique shots verified:  " << vr.unique_shots_verified
              << "\n"
              << "  Corrections matched:    " << vr.responses_matched << "\n"
              << "  Header errors:          " << vr.header_errors << "\n"
              << "  Correction errors:      " << vr.correction_errors << "\n"
              << "  Expected shots:         " << num_shots << "\n";
    if (!vr.latency_samples.empty()) {
      int64_t lat_min = std::numeric_limits<int64_t>::max();
      int64_t lat_max = std::numeric_limits<int64_t>::min();
      int64_t lat_sum = 0;
      for (auto &s : vr.latency_samples) {
        lat_sum += s.delta_ns;
        if (s.delta_ns < lat_min)
          lat_min = s.delta_ns;
        if (s.delta_ns > lat_max)
          lat_max = s.delta_ns;
      }
      double lat_avg = static_cast<double>(lat_sum) / vr.latency_samples.size();

      // Print first 5 samples for diagnostic
      for (std::size_t k = 0; k < 5 && k < vr.latency_samples.size(); ++k) {
        auto &s = vr.latency_samples[k];
        std::cout << "  rid " << std::setw(3) << s.request_id << " (shot "
                  << s.shot << " local " << s.local << " "
                  << (s.is_corr ? "get_corrections" : "enqueue") << ")"
                  << ": send={sec=" << s.send_sec << ", nsec=" << s.send_nsec
                  << "} recv={sec=" << s.recv_sec << ", nsec=" << s.recv_nsec
                  << "} delta=" << s.delta_ns << " ns\n";
      }

      std::cout << "\n=== PTP Round-Trip Latency ===\n"
                << "  Samples:  " << vr.latency_samples.size()
                << " (all captured frames)\n"
                << "  Min:      " << lat_min << " ns\n"
                << "  Max:      " << lat_max << " ns\n"
                << "  Avg:      " << std::fixed << std::setprecision(1)
                << lat_avg << " ns\n";

      // One row per captured frame.  `kind` is enqueue|get_corrections;
      // `local` is the frame index within its shot (per-round).
      const std::string csv_path = "ptp_latency.csv";
      std::ofstream csv(csv_path);
      if (csv.is_open()) {
        csv << "request_id,shot,local,kind,send_sec,send_nsec,recv_sec,"
               "recv_nsec,delta_ns\n";
        for (auto &s : vr.latency_samples)
          csv << s.request_id << "," << s.shot << "," << s.local << ","
              << (s.is_corr ? "get_corrections" : "enqueue") << ","
              << s.send_sec << "," << s.send_nsec << "," << s.recv_sec << ","
              << s.recv_nsec << "," << s.delta_ns << "\n";
        csv.close();
        std::cout << "  CSV written: " << csv_path << " ("
                  << vr.latency_samples.size() << " rows)\n";
      }
    } else {
      std::cout << "\n  PTP latency: no valid timestamps found\n";
    }

    if (vr.correction_errors > 0 || vr.header_errors > 0) {
      std::cout << "  RESULT: FAIL\n";
      return 1;
    }
    if (vr.responses_matched == 0) {
      std::cout << "  RESULT: FAIL (no valid responses found)\n";
      return 1;
    }
    if (vr.unique_shots_verified < num_shots) {
      std::cout << "  RESULT: FAIL (verified only " << vr.unique_shots_verified
                << " of " << num_shots << " expected shots)\n";
      return 1;
    }
    std::cout << "  RESULT: PASS\n";
  }

  return 0;
}

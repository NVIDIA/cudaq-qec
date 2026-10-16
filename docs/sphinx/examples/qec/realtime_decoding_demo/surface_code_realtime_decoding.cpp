/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The realtime_decoding_demo example source (a copy of the in-tree
// unittests/realtime/app_examples/surface_code-1.cpp, adapted in this header,
// the usage line, and the --save_syndrome capture, which emits true-width
// groups so the FPGA playback tool replays exact per-round measurement
// counts). CMakeLists.txt builds it two ways against the installed SDK: a
// plain generator (writes the decoder config and, for the FPGA source, the
// playback syndrome file) and a lowered `-cqr` kernel (-frealtime-lowering
// -DQEC_APP_CQR) that streams syndromes to the delivered decoding_server over
// UDP. run_realtime_decoding.sh drives both.
//
// This demo drives a surface-code memory experiment with realtime decoding:
//
//   * The detector error model and measurement-to-detector map are obtained
//     from `decoder_context_from_memory_circuit`.
//   * The QPU kernel loops over logical qubits, calling a per-logical
//     `memory_circuit` kernel that preps the state, streams every round's
//     syndrome plus the final data readout to that logical's decoder, and
//     returns the data measurements.
//   * Circuit-level two-qubit depolarizing noise on the stabilizer-extraction
//     CNOTs is used for both characterization and simulation.
//
// The same source runs on the local (Stim) target and, compiled with
// `--target quantinuum --emulate`, on the Quantinuum QIR emulator: in both
// cases the syndromes are generated on the QPU and decoded live through the
// realtime decoding API.
//
// The host-side flow (see demo_circuit_host): configure the decoders
// (setup_decoders), then take one of two paths --
//   * --load_syndrome : replay_syndrome_file feeds a captured file to the
//     decoders and verifies the corrections (no circuit is run);
//   * otherwise       : run_experiment runs the shots, reports the logical
//     error rate, and optionally captures the syndrome stream
//     (--save_syndrome).

#include "cudaq.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/noise_model.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef QEC_APP_CQR
// cqr build variant: this same application compiled with -frealtime-lowering
// and linked against the simulation-cqr client wrappers, so every
// cudaq::qec::decoding::* device_call crosses the cudaq-realtime wire instead
// of resolving to the in-process trampolines. The decoding is served either by
// the in-process decoding-server-cqr service
// (CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch) or by a standalone
// decoding_server (QEC_DECODING_SERVER_PORT=<port>). The wire to the
// server defaults to udp loopback; set QEC_DECODING_SERVER_TRANSPORT=cpu_roce
// to use the CPU RoCE RDMA channel instead (works over SoftRoCE/rdma_rxe; the
// RDMA topology comes from the same CUDAQ_CPU_ROCE_TEST_* env vars as CUDA-Q's
// CpuRoceChannelTester).
#include "cudaq/realtime.h"

// In-process service self-check hook (defined in decoding-server-cqr):
// non-zero only if device_calls actually traversed the host-dispatch ring.
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();

namespace {
std::string env_or(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return (value && *value) ? std::string(value) : fallback;
}

void initialize_realtime_channel(const char *prog) {
  std::vector<std::string> args = {prog};
  if (const char *port = std::getenv("QEC_DECODING_SERVER_PORT");
      port && *port) {
    const std::string transport =
        env_or("QEC_DECODING_SERVER_TRANSPORT", "udp");
    if (transport == "cpu_roce") {
      // The RDMA ring geometry (slots x slot-size) is part of the cpu_roce
      // wire contract: the channel writes requests directly into the server's
      // rings, so these must match decoding_server's --num-slots /
      // --slot-size defaults (8 x 256).
      args.push_back("--cudaq-device-call=cpu_roce");
      args.push_back("--cudaq-device-call-slots=8");
      args.push_back("--cudaq-device-call-slot-size=256");
      args.push_back("ib-device=" +
                     env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE", "mlx5_0"));
      args.push_back("local-ip=" +
                     env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_IP", "10.0.0.1"));
      args.push_back("rendezvous-host=" +
                     env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_IP", "10.0.0.2"));
      args.push_back(std::string("rendezvous-port=") + port);
    } else {
      args.push_back("--cudaq-device-call=udp");
      args.push_back("udp-host=127.0.0.1");
      args.push_back(std::string("udp-port=") + port);
    }
  }
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
}
} // namespace
#endif

// Host-side decoding API (for syndrome capture)
namespace cudaq::qec::decoding::host {
void _set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t));
}

namespace {

// Command-line options, parsed in main() and threaded through as a bundle.
struct run_options {
  int distance = 5;
  int num_shots = 10;
  double p_cnot = 0.001; // two-qubit depolarizing rate on the CNOTs
  int num_logical = 1;
  int num_rounds = -1; // defaults to distance
  int seed = 42;       // simulator seed; < 0 leaves the simulator unseeded
  std::string decoder_type = "multi_error_lut";
  int sw_window_size =
      -1; // sliding_window size in rounds; defaults to distance
  int sw_step_size = 1;
  bool save_dem = false;
  bool load_dem = false;
  std::string dem_filename;
  bool save_syndrome = false;
  bool load_syndrome = false;
  std::string syndrome_filename;
};

// The per-experiment inputs derived once from the code + options: register
// sizes, the stabilizer support vectors, and the logical observable. The
// syndrome stream is, per logical qubit, `num_rounds` interior groups of
// `num_ancx + num_ancz` bits followed by one data group of `num_data` bits.
struct experiment {
  std::size_t num_data = 0, num_ancx = 0, num_ancz = 0;
  std::size_t num_rounds = 0, num_logical = 0, num_obs = 0;
  std::vector<std::size_t> x_vec, z_vec, obs_flat;
};

// --save_syndrome capture state. The library callback is a bare
// `void(*)(const uint8_t *, size_t)` with no user-data pointer, so the
// captureless callback below reaches this through a global.
struct syndrome_capture_state {
  std::ofstream file;
  std::mutex mutex;
  int count = 0;
  int syndromes_per_shot = 0; // groups written per shot
  // True per-group bit widths, used to trim the byte-padded packed bytes the
  // callback receives back to the exact measurement counts. The FPGA playback
  // tool sets each replayed frame's num_syndromes from the group's line
  // count, and the decoder's per-shot measurement capacity is the exact sum
  // (num_rounds * interior + data) -- byte-padded groups would overrun it.
  int groups_per_logical = 0; // num_rounds interior groups + 1 data group
  int interior_width = 0;     // num_ancx + num_ancz bits
  int data_width = 0;         // num_data bits
};
syndrome_capture_state g_capture;

} // namespace

// Build one decoder config per logical qubit from the decoder_inputs produced
// by `decoder_context_from_memory_circuit(...).full_component()`. All decoders
// share the same DEM (`H_sparse`), observables (`O_sparse`) and
// measurement-to-detector map (`D_sparse`); only the decoder type/parameters
// differ. Per-decoder args use the declarative `heterogeneous_map` schema
// (`decoder_custom_args`) validated by `multi_decoder_config`.
static cudaq::qec::decoding::config::multi_decoder_config
build_multi_decoder_config(const cudaq::qec::decoder_inputs &inputs,
                           std::size_t num_syndromes_per_round,
                           std::size_t num_boundary_syndromes,
                           const run_options &opts) {
  namespace config = cudaq::qec::decoding::config;
  const auto &dem = inputs.dem;
  const auto d_sparse = cudaq::qec::d_sparse(inputs.m2d);

  config::multi_decoder_config multi_config;
  for (int i = 0; i < opts.num_logical; i++) {
    config::decoder_config dc;
    dc.id = i;
    dc.block_size = dem.num_error_mechanisms();
    dc.syndrome_size = dem.num_detectors();
    dc.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
    dc.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);
    dc.D_sparse = d_sparse;
    dc.error_rate_vec = dem.error_rates;

    if (opts.decoder_type == "nv-qldpc-decoder") {
      dc.type = "nv-qldpc-decoder";
      cudaqx::heterogeneous_map nv_args;
      nv_args.insert("use_sparsity", true);
      nv_args.insert("max_iterations", 50);
      nv_args.insert("bp_method", 3);   // min-sum + dmem (required for relay)
      nv_args.insert("composition", 1); // sequential relay
      nv_args.insert("gamma0", 0.0);
      nv_args.insert("clip_value", 200.0);
      nv_args.insert("repeatable", true);
      cudaqx::heterogeneous_map srelay_args;
      srelay_args.insert("pre_iter", std::size_t{5});
      srelay_args.insert("num_sets", std::size_t{10});
      srelay_args.insert("stopping_criterion", "All");
      srelay_args.insert("stop_nconv", std::size_t{1});
      nv_args.insert("srelay_config", srelay_args);
      nv_args.insert("gamma_dist", std::vector<double>{0.1, 0.2});
      dc.decoder_custom_args = nv_args;
    } else if (opts.decoder_type == "sliding_window") {
      dc.type = "sliding_window";
      cudaqx::heterogeneous_map sw_args;
      sw_args.insert("window_size", opts.sw_window_size);
      sw_args.insert("step_size", opts.sw_step_size);
      sw_args.insert("num_syndromes_per_round", num_syndromes_per_round);
      sw_args.insert("num_boundary_syndromes", num_boundary_syndromes);
      sw_args.insert("straddle_start_round", false);
      sw_args.insert("straddle_end_round", true);
      sw_args.insert("inner_decoder_name", "multi_error_lut");
      cudaqx::heterogeneous_map inner_lut_args;
      inner_lut_args.insert("lut_error_depth", 2);
      sw_args.insert("inner_decoder_params", inner_lut_args);
      dc.decoder_custom_args = sw_args;
    } else if (opts.decoder_type == "pymatching") {
      dc.type = "pymatching";
      cudaqx::heterogeneous_map pm_args;
      pm_args.insert("merge_strategy", "smallest_weight");
      dc.decoder_custom_args = pm_args;
    } else {
      dc.type = "multi_error_lut";
      cudaqx::heterogeneous_map lut_args;
      lut_args.insert("lut_error_depth", 2);
      dc.decoder_custom_args = lut_args;
    }

    multi_config.decoders.push_back(dc);
  }
  return multi_config;
}

namespace cudaq::qec::qpu {

// Per-logical memory circuit kernel. Preps the logical qubit, streams
// `num_rounds` syndrome groups plus the final data readout to the decoder, and
// returns that final data measurement. The decoder computes detectors from the
// measurement-to-detector map (D_sparse).
__qpu__ std::vector<cudaq::measure_result> memory_circuit(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const cudaq::qec::code::stabilizer_round &stab_round,
    const cudaq::qec::code::one_qubit_encoding &state_prep,
    std::size_t num_rounds, const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers, std::int64_t decoder_id) {
  patch logical(data, xstab_anc, zstab_anc);
  state_prep(logical);

  // Lock-in round (round 0)
  auto syndrome = stab_round(logical, x_stabilizers, z_stabilizers);
  cudaq::qec::decoding::enqueue_syndromes(decoder_id, syndrome);

  // Rounds 1..num_rounds-1: stream raw syndromes
  for (std::size_t r = 1; r < num_rounds; r++) {
    auto s = stab_round(logical, x_stabilizers, z_stabilizers);
    cudaq::qec::decoding::enqueue_syndromes(decoder_id, s);
  }

  // Final data readout
  auto data_meas = mz(data);
  cudaq::qec::decoding::enqueue_syndromes(decoder_id, data_meas);
  return data_meas;
}

// Shot kernel: run an independent surface-code memory experiment on each of
// `num_logical` logical qubits and return, per logical qubit, its logical
// observable value updated by the decoder's correction
//
// Return layout: the low 32 bits hold, per logical qubit,
// bit i = logical qubit i's corrected observable (so num_logical <= 32); the
// high 32 bits hold how many corrections the decoders applied this shot.
__qpu__ std::int64_t demo_circuit_qpu(
    const cudaq::qec::code::stabilizer_round &stab_round,
    const cudaq::qec::code::one_qubit_encoding &state_prep,
    std::size_t num_data, std::size_t num_ancx, std::size_t num_ancz,
    std::size_t num_rounds, std::size_t num_logical,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, const std::vector<std::int64_t> &decoder_ids) {
  cudaq::qvector data(num_logical * num_data),
      xstab_anc(num_logical * num_ancx), zstab_anc(num_logical * num_ancz);

  for (std::size_t i = 0; i < num_logical; i++)
    cudaq::qec::decoding::reset_decoder(decoder_ids[i]);

  std::uint64_t num_corrections = 0;
  std::uint64_t ret = 0;
  for (std::size_t i = 0; i < num_logical; i++) {
    // Stream this logical's rounds + data readout to its decoder, keeping the
    // single data measurement it returns.
    auto data_meas = memory_circuit(
        data.slice(i * num_data, num_data),
        xstab_anc.slice(i * num_ancx, num_ancx),
        zstab_anc.slice(i * num_ancz, num_ancz), stab_round, state_prep,
        num_rounds, x_stabilizers, z_stabilizers, decoder_ids[i]);

    // Raw logical observable (row 0 of the observable matrix) from that
    // readout.
    bool obs = false;
    for (std::size_t q = 0; q < num_data; q++)
      if (obs_matrix_flat[q] != 0 && data_meas[q])
        obs = !obs;

    // Update it by the decoder's predicted observable flip.
    auto correction = cudaq::qec::decoding::get_corrections(
        static_cast<std::uint64_t>(decoder_ids[i]), num_observables,
        /*reset=*/false);
    if (correction[0]) {
      obs = !obs;
      num_corrections++;
    }

    if (obs)
      ret |= std::uint64_t{1} << i;
  }
  // Corrections in the high 32 bits
  ret |= num_corrections << 32;
  return ret;
}
} // namespace cudaq::qec::qpu

namespace {

// Gather the experiment inputs (register sizes, stabilizer supports, logical
// observable) from the code once.
experiment make_experiment(const cudaq::qec::code &code,
                           cudaq::qec::operation state_prep,
                           const run_options &opts) {
  experiment exp;
  exp.num_data = code.get_num_data_qubits();
  exp.num_ancx = code.get_num_ancilla_x_qubits();
  exp.num_ancz = code.get_num_ancilla_z_qubits();
  exp.num_rounds = static_cast<std::size_t>(opts.num_rounds);
  exp.num_logical = static_cast<std::size_t>(opts.num_logical);

  auto schedule_x = code.get_stabilizer_schedule_x();
  auto schedule_z = code.get_stabilizer_schedule_z();
  exp.x_vec.assign(schedule_x.data(), schedule_x.data() + schedule_x.size());
  exp.z_vec.assign(schedule_z.data(), schedule_z.data() + schedule_z.size());

  const bool is_z_prep = state_prep == cudaq::qec::operation::prep0 ||
                         state_prep == cudaq::qec::operation::prep1;
  auto obs_matrix =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  exp.num_obs = obs_matrix.shape()[0];
  exp.obs_flat.assign(obs_matrix.data(), obs_matrix.data() + obs_matrix.size());
  return exp;
}

// Configure the decoders for the run. Returns true if they are ready and the
// caller should proceed to run shots; false if the run is complete (--save_dem
// wrote the config and exited) or a file could not be opened.
//
// --load_dem and --save_dem are the two ends of the same YAML round-trip a
// standalone decoding_server uses:
//   --load_dem <file> : read a saved config (the decoders may live in a
//                       separate decoding_server fed the same YAML).
//   --save_dem <file> : characterize the DEM, write the config YAML, and exit.
//   (default)         : characterize and configure the decoders in-process.
bool setup_decoders(const cudaq::qec::code &code,
                    cudaq::qec::operation state_prep, const run_options &opts,
                    cudaq::noise_model &noise) {
  namespace config = cudaq::qec::decoding::config;

  if (opts.load_dem) {
    std::ifstream config_file(opts.dem_filename);
    if (!config_file)
      throw std::runtime_error("Could not open dem config file: " +
                               opts.dem_filename);
    std::stringstream config_text;
    config_text << config_file.rdbuf();
    auto cfg = config::multi_decoder_config::from_yaml_str(config_text.str());
    config::configure_decoders(cfg);
    printf("Loaded decoder config from %s (%zu decoders)\n",
           opts.dem_filename.c_str(), cfg.decoders.size());
    return true;
  }

  // Characterize the DEM and build the decoder configuration. full_component()
  // canonicalizes both stabilizer types (boundary-aware) into decoder_inputs.
  const bool decompose_errors = (opts.decoder_type == "pymatching");
  auto ctx = cudaq::qec::decoder_context_from_memory_circuit(
      code, state_prep, opts.num_rounds, noise, decompose_errors);
  const auto inputs = ctx.full_component();
  printf("DEM: %ld detectors x %ld error mechanisms\n",
         inputs.dem.num_detectors(), inputs.dem.num_error_mechanisms());

  const bool is_z_prep = state_prep == cudaq::qec::operation::prep0 ||
                         state_prep == cudaq::qec::operation::prep1;
  const std::size_t num_x_stabilizers = code.get_num_ancilla_x_qubits();
  const std::size_t num_z_stabilizers = code.get_num_ancilla_z_qubits();
  const std::size_t num_syndromes_per_round =
      num_x_stabilizers + num_z_stabilizers;
  const std::size_t num_boundary_syndromes =
      is_z_prep ? num_z_stabilizers : num_x_stabilizers;
  auto cfg = build_multi_decoder_config(inputs, num_syndromes_per_round,
                                        num_boundary_syndromes, opts);

  if (opts.save_dem) {
    // Serialize the config to YAML -- the inverse of the decoding server's
    // multi_decoder_config::from_yaml_str -- and exit; no shots are run.
    std::ofstream(opts.dem_filename) << cfg.to_yaml_str(200);
    printf("Saved decoder config to %s\n", opts.dem_filename.c_str());
    return false;
  }

  config::configure_decoders(cfg);
  return true;
}

// --save_syndrome: register a callback that tees every syndrome the decoder
// receives during the live run into `filename`, so it can be replayed offline.
void begin_syndrome_capture(const std::string &filename,
                            const experiment &exp) {
  g_capture.file.open(filename, std::ios::out | std::ios::trunc);
  if (!g_capture.file)
    throw std::runtime_error("Could not open syndrome file for writing: " +
                             filename);
  g_capture.count = 0;
  g_capture.syndromes_per_shot =
      static_cast<int>(exp.num_logical * (exp.num_rounds + 1));
  g_capture.groups_per_logical = static_cast<int>(exp.num_rounds + 1);
  g_capture.interior_width = static_cast<int>(exp.num_ancx + exp.num_ancz);
  g_capture.data_width = static_cast<int>(exp.num_data);

  printf("Syndrome capture enabled: saving to %s\n", filename.c_str());
  g_capture.file << "NUM_DATA " << exp.num_data << "\n";
  g_capture.file << "NUM_LOGICAL " << exp.num_logical << "\n";
  g_capture.file.flush();

  cudaq::qec::decoding::host::_set_syndrome_capture_callback(
      [](const uint8_t *data, size_t len) {
        std::lock_guard<std::mutex> lock(g_capture.mutex);
        if (!g_capture.file.is_open())
          return;

        if (g_capture.count % g_capture.syndromes_per_shot == 0)
          g_capture.file << "SHOT_START "
                         << (g_capture.count / g_capture.syndromes_per_shot)
                         << "\n";
        g_capture.file << "ROUND_START " << g_capture.count << "\n";

        // Emit exactly the group's true bit count (the packed bytes are
        // byte-padded): the first num_rounds groups per logical are interior
        // syndrome rounds, the last is the data readout.
        const int group_in_logical =
            g_capture.count % g_capture.groups_per_logical;
        const size_t width = static_cast<size_t>(
            group_in_logical < g_capture.groups_per_logical - 1
                ? g_capture.interior_width
                : g_capture.data_width);
        // Unpack MSB-first, one bit per line.
        for (size_t k = 0; k < width && k < len * 8; k++)
          g_capture.file << ((data[k / 8] >> (7 - (k % 8))) & 1) << "\n";
        g_capture.file.flush();
        g_capture.count++;
      });
}

// Append the per-shot corrections so a later --load_syndrome run can verify the
// replayed corrections against them, then close the capture file.
void finish_syndrome_capture(const std::vector<std::int64_t> &run_result,
                             const std::string &filename) {
  if (!g_capture.file.is_open())
    return;
  cudaq::qec::decoding::host::_set_syndrome_capture_callback(nullptr);
  g_capture.file << "CORRECTIONS_START\n";
  for (auto shot : run_result)
    g_capture.file << ((shot >> 32) > 0 ? 1 : 0) << "\n";
  g_capture.file << "CORRECTIONS_END\n";
  g_capture.file.close();
  printf("Syndrome data saved to: %s\n", filename.c_str());
}

// --load_syndrome: replay a captured syndrome file through the decoders (no
// circuit is run) and check the resulting corrections match the ones saved
// alongside the syndromes.
void replay_syndrome_file(const std::string &filename, const experiment &exp) {
  printf("\n=== Syndrome Replay Mode ===\n");
  printf("Loading syndromes from: %s\n", filename.c_str());

  std::ifstream syndrome_file(filename);
  if (!syndrome_file)
    throw std::runtime_error("Could not open syndrome file: " + filename);

  // Parse the header, then per shot a list of ROUND_START-delimited groups
  // (each a byte-aligned, MSB-first bit dump), then the per-shot corrections.
  std::size_t file_num_data = 0, file_num_logical = 0;
  std::vector<uint8_t> saved_corrections;
  std::vector<std::vector<std::vector<bool>>>
      saved_shots; // shot -> group -> bits
  std::string line;
  while (std::getline(syndrome_file, line)) {
    std::istringstream iss(line);
    std::string tag;
    iss >> tag;
    if (tag == "NUM_DATA") {
      iss >> file_num_data;
    } else if (tag == "NUM_LOGICAL") {
      iss >> file_num_logical;
    } else if (tag == "SHOT_START") {
      saved_shots.emplace_back();
    } else if (tag == "ROUND_START") {
      if (!saved_shots.empty())
        saved_shots.back().emplace_back();
    } else if (tag == "CORRECTIONS_START") {
      while (std::getline(syndrome_file, line)) {
        if (line.find("CORRECTIONS_END") == 0)
          break;
        saved_corrections.push_back(static_cast<uint8_t>(std::stoi(line)));
      }
      break;
    } else if (!tag.empty() && !saved_shots.empty() &&
               !saved_shots.back().empty()) {
      // A bit line belonging to the current group.
      saved_shots.back().back().push_back(std::stoi(tag) != 0);
    }
  }

  if (file_num_data != exp.num_data || file_num_logical != exp.num_logical)
    throw std::runtime_error(
        "Syndrome file mismatch: file has num_data=" +
        std::to_string(file_num_data) +
        " num_logical=" + std::to_string(file_num_logical) +
        ", expected num_data=" + std::to_string(exp.num_data) +
        " num_logical=" + std::to_string(exp.num_logical));

  printf("Read %zu shots with syndromes\n", saved_shots.size());
  printf("Feeding %zu shots of saved syndromes to decoder...\n",
         saved_shots.size());

  int matched = 0, mismatched = 0;
  // Per logical qubit the stream is num_rounds interior groups of
  // (num_ancx + num_ancz) bits, then one data group of num_data bits. Each
  // stored group is byte-padded, so trim it back to this true width.
  const std::size_t groups_per_logical = exp.num_rounds + 1;
  const std::size_t interior_width = exp.num_ancx + exp.num_ancz;
  for (std::size_t shot = 0; shot < saved_shots.size(); shot++) {
    for (std::size_t l = 0; l < exp.num_logical; l++)
      cudaq::qec::decoding::reset_decoder(l);

    // Feed each group to the logical qubit it belongs to.
    const auto &groups = saved_shots[shot];
    for (std::size_t g = 0; g < groups.size(); g++) {
      const std::size_t logical_idx = g / groups_per_logical;
      if (logical_idx >= exp.num_logical)
        break;
      const std::size_t group_in_logical = g % groups_per_logical;
      const std::size_t width =
          group_in_logical < exp.num_rounds ? interior_width : exp.num_data;
      std::vector<bool> bits(groups[g].begin(),
                             groups[g].begin() +
                                 std::min(width, groups[g].size()));
      cudaq::qec::decoding::enqueue_syndromes_test(logical_idx, bits);
    }

    uint8_t correction_bit = 0;
    for (std::size_t l = 0; l < exp.num_logical; l++) {
      auto corrections = cudaq::qec::decoding::get_corrections(l, 1, false);
      for (auto c : corrections)
        if (c)
          correction_bit = 1;
    }

    if (shot < saved_corrections.size()) {
      if (correction_bit == saved_corrections[shot]) {
        matched++;
      } else {
        mismatched++;
        if (mismatched <= 10)
          printf("  Shot %zu: mismatch! Replayed=%u, Saved=%u\n", shot,
                 correction_bit, saved_corrections[shot]);
      }
    }
  }

  printf("Replay complete: %zu shots processed\n", saved_shots.size());
  if (!saved_corrections.empty()) {
    printf("Correction verification: %d matched, %d mismatched\n", matched,
           mismatched);
    if (mismatched > 0)
      throw std::runtime_error(
          "Replay correction mismatch: " + std::to_string(mismatched) + " of " +
          std::to_string(matched + mismatched) + " shots did not match");
    printf("SUCCESS: All corrections match!\n");
  }
}

// Run `num_shots` shots of the experiment. The kernel generates each round's
// syndrome on the QPU (simulated locally, or emulated on the hardware target)
// and streams it to the decoder in real time; the logical correction comes
// back before readout. A remote platform applies its own noise, so the local
// model is not attached there.
std::vector<std::int64_t>
run_shots(std::size_t num_shots, cudaq::noise_model &noise,
          const cudaq::qec::code::stabilizer_round &stab_round,
          const cudaq::qec::code::one_qubit_encoding &prep,
          const experiment &exp, const std::vector<std::int64_t> &decoder_ids) {
  return cudaq::get_platform().is_remote()
             ? cudaq::run(num_shots, cudaq::qec::qpu::demo_circuit_qpu,
                          stab_round, prep, exp.num_data, exp.num_ancx,
                          exp.num_ancz, exp.num_rounds, exp.num_logical,
                          exp.x_vec, exp.z_vec, exp.obs_flat, exp.num_obs,
                          decoder_ids)
             : cudaq::run(num_shots, noise, cudaq::qec::qpu::demo_circuit_qpu,
                          stab_round, prep, exp.num_data, exp.num_ancx,
                          exp.num_ancz, exp.num_rounds, exp.num_logical,
                          exp.x_vec, exp.z_vec, exp.obs_flat, exp.num_obs,
                          decoder_ids);
}

// Decode the per-shot results and print the logical error rate.
void report_results(const std::vector<std::int64_t> &run_result,
                    std::size_t num_logical) {
  printf("Result size: %ld\n", run_result.size());
  int num_non_zero_values = 0;
  std::int64_t num_corrections = 0;
  for (auto shot : run_result) {
    // High 32 bits: number of corrections the decoder applied this shot.
    num_corrections += (shot >> 32);
    // Low 32 bits: each logical qubit's observable, already updated by the
    // decoder's correction. A set bit is a residual logical error.
    for (std::size_t j = 0; j < num_logical; j++)
      if ((shot >> j) & 1)
        num_non_zero_values++;
  }
  printf("Number of non-zero values measured : %d\n", num_non_zero_values);
  printf("Number of corrections decoder found: %ld\n", num_corrections);
}

// The live-run path: (optionally) start capturing the syndrome stream, run the
// shots and report the logical error rate, then (optionally) persist the
// per-shot corrections for a later --load_syndrome.
void run_experiment(const cudaq::qec::code &code,
                    cudaq::qec::operation state_prep, const run_options &opts,
                    cudaq::noise_model &noise, const experiment &exp,
                    const std::vector<std::int64_t> &decoder_ids) {
  if (opts.save_syndrome)
    begin_syndrome_capture(opts.syndrome_filename, exp);

  auto &prep =
      code.get_operation<cudaq::qec::code::one_qubit_encoding>(state_prep);
  auto &stab_round = code.get_operation<cudaq::qec::code::stabilizer_round>(
      cudaq::qec::operation::stabilizer_round);

  // Seed the simulator so the sampled shots -- and thus the reported logical
  // error / correction counts (and any captured syndromes) -- are reproducible.
  if (opts.seed >= 0)
    cudaq::set_random_seed(static_cast<std::size_t>(opts.seed));
  auto run_result =
      run_shots(opts.num_shots, noise, stab_round, prep, exp, decoder_ids);
  report_results(run_result, exp.num_logical);

  if (opts.save_syndrome)
    finish_syndrome_capture(run_result, opts.syndrome_filename);
}

} // namespace

// Host-side driver: configure the decoders, then either replay a saved syndrome
// file or run the circuit and report the logical error rate.
void demo_circuit_host(const cudaq::qec::code &code, const run_options &opts) {
  const auto state_prep = cudaq::qec::operation::prep0;

  // Circuit-level two-qubit depolarizing noise on the stabilizer-extraction
  // CNOTs, shared by DEM characterization and shot simulation.
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x",
                              cudaq::qec::two_qubit_depolarization(opts.p_cnot),
                              /*num_controls=*/1);

  const experiment exp = make_experiment(code, state_prep, opts);

  // Configure the decoders. --save_dem stops here (config written, no shots).
  if (!setup_decoders(code, state_prep, opts, noise))
    return;

  // One decoder id per logical qubit, matching config ids 0..num_logical-1.
  std::vector<std::int64_t> decoder_ids(exp.num_logical);
  std::iota(decoder_ids.begin(), decoder_ids.end(), 0);

  // Two paths: replay a captured syndrome file, or run the live circuit.
  if (opts.load_syndrome)
    replay_syndrome_file(opts.syndrome_filename, exp);
  else
    run_experiment(code, state_prep, opts, noise, exp, decoder_ids);
}

void show_help() {
  printf("Usage: surface_code_realtime_decoding [options]\n");
  printf("Options:\n");
  printf("  --distance <int>      Distance of the surface code. Default: 5\n");
  printf("  --num_shots <int>     Number of shots. Default: 10\n");
  printf("  --p_cnot <double>     Two-qubit depolarizing rate on CNOT gates. "
         "Range[0, 1]. Default: 0.001\n");
  printf("  --num_logical <int>   Number of logical qubits. Default: 1\n");
  printf("  --num_rounds <int>    Number of measurement rounds. Default: "
         "distance\n");
  printf("  --seed <int>          Simulator seed for reproducible shots; "
         "negative leaves it unseeded. Default: 42\n");
  printf("  --decoder_type <string> Decoder type: 'multi_error_lut', "
         "'nv-qldpc-decoder', 'sliding_window', or 'pymatching'. Default: "
         "multi_error_lut\n");
  printf("  --sw_window_size <int>  Sliding window size (only for "
         "sliding_window decoder). Default: distance\n");
  printf("  --sw_step_size <int>    Sliding window step size. Default: 1\n");
  printf("  --save_dem <string> Characterize the DEM, write the decoder config "
         "YAML to a file, and exit (to configure a standalone "
         "decoding_server).\n");
  printf("  --load_dem <string> Load the decoder config from a YAML file "
         "instead of characterizing in-process.\n");
  printf("  --save_syndrome <string> Save syndrome data to a file for later "
         "replay.\n");
  printf("  --load_syndrome <string> Load and replay syndrome data from a "
         "file.\n");
  printf("  --help              Show this help message\n");
}

int main(int argc, char **argv) {
  run_options opts;

  // A value-taking option must not read past argv, and a malformed number
  // must exit with a message instead of unwinding out of main.
  try {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      auto next_value = [&]() -> const char * {
        if (i + 1 >= argc)
          throw std::runtime_error(arg + " requires a value");
        return argv[++i];
      };
      if (arg == "--distance") {
        opts.distance = std::stoi(next_value());
      } else if (arg == "--num_shots") {
        opts.num_shots = std::stoi(next_value());
      } else if (arg == "--p_cnot") {
        opts.p_cnot = std::stod(next_value());
      } else if (arg == "--help" || arg == "-h") {
        show_help();
        return 0;
      } else if (arg == "--num_logical") {
        opts.num_logical = std::stoi(next_value());
      } else if (arg == "--num_rounds") {
        opts.num_rounds = std::stoi(next_value());
      } else if (arg == "--seed") {
        opts.seed = std::stoi(next_value());
      } else if (arg == "--decoder_type") {
        opts.decoder_type = next_value();
      } else if (arg == "--sw_window_size") {
        opts.sw_window_size = std::stoi(next_value());
      } else if (arg == "--sw_step_size") {
        opts.sw_step_size = std::stoi(next_value());
      } else if (arg == "--save_dem") {
        opts.save_dem = true;
        opts.dem_filename = next_value();
      } else if (arg == "--load_dem") {
        opts.load_dem = true;
        opts.dem_filename = next_value();
      } else if (arg == "--save_syndrome") {
        opts.save_syndrome = true;
        opts.syndrome_filename = next_value();
      } else if (arg == "--load_syndrome") {
        opts.load_syndrome = true;
        opts.syndrome_filename = next_value();
      } else {
        printf("Unknown argument: %s\n", arg.c_str());
        show_help();
        return 1;
      }
    }
  } catch (const std::invalid_argument &) {
    printf("Error: invalid numeric value on the command line\n");
    show_help();
    return 1;
  } catch (const std::exception &e) {
    printf("Error: %s\n", e.what());
    show_help();
    return 1;
  }

  if (opts.save_syndrome && opts.load_syndrome) {
    printf("Error: Cannot use both --save_syndrome and --load_syndrome "
           "together\n");
    return 1;
  }

  if (opts.save_dem && opts.load_dem) {
    printf("Error: Cannot use both --save_dem and --load_dem together\n");
    return 1;
  }

  // --save_dem only characterizes the DEM and writes the config; it never runs
  // shots, so it can't be combined with the syndrome capture/replay options.
  if (opts.save_dem && (opts.save_syndrome || opts.load_syndrome)) {
    printf("Error: --save_dem cannot be combined with --save_syndrome or "
           "--load_syndrome\n");
    return 1;
  }

  if (opts.num_rounds == -1)
    opts.num_rounds = opts.distance;
  if (opts.sw_window_size == -1)
    opts.sw_window_size = opts.distance;

  if (opts.decoder_type != "multi_error_lut" &&
      opts.decoder_type != "sliding_window" &&
      opts.decoder_type != "nv-qldpc-decoder" &&
      opts.decoder_type != "pymatching") {
    printf("Error: --decoder_type must be 'multi_error_lut', "
           "'nv-qldpc-decoder', 'sliding_window', or 'pymatching'\n");
    return 1;
  }

  if (opts.num_rounds < opts.distance) {
    printf("Error: num_rounds (%d) must be at least equal to distance (%d)\n",
           opts.num_rounds, opts.distance);
    return 1;
  }

  if (opts.num_logical > 32) {
    printf("num_logical > 32 is not supported.\n");
    return 1;
  }

  printf("Running with p_cnot = %f, distance = %d, num_shots = %d, "
         "num_rounds = %d\n",
         opts.p_cnot, opts.distance, opts.num_shots, opts.num_rounds);
  auto code = cudaq::qec::get_code(
      "surface_code", cudaqx::heterogeneous_map{{"distance", opts.distance}});

#ifdef QEC_APP_CQR
  // Bring up the cudaq-realtime device-call channel for the shots' decoding
  // device_calls. The --save_dem pass only characterizes the DEM (no shots, no
  // device_calls), so it needs no channel.
  if (!opts.save_dem)
    initialize_realtime_channel(argv[0]);
#endif

  try {
    demo_circuit_host(*code, opts);
  } catch (const std::exception &e) {
    printf("Error: %s\n", e.what());
    cudaq::qec::decoding::config::finalize_decoders();
    return 1;
  }

#ifdef QEC_APP_CQR
  if (!opts.save_dem) {
    // With CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch this proves the shots'
    // device_calls crossed the ring to the in-process decoding server (it stays
    // 0 if they bypassed to a trampoline, or a udp channel routed them to an
    // external server instead).
    printf("CQR service dispatch count: %llu\n",
           static_cast<unsigned long long>(
               cudaqx_qec_device_call_dispatch_count()));
    cudaq::realtime::finalize();
  }
#endif

  cudaq::qec::decoding::config::finalize_decoders();
  return 0;
}

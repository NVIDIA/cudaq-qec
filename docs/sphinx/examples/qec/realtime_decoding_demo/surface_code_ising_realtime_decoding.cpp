/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// surface_code_ising_realtime_decoding.cpp
//
// Companion source to surface_code_realtime_decoding.cpp, dedicated to the
// Ising decoder profile: a TensorRT neural-network predecoder plus a
// PyMatching global decoder (decoder type `trt_decoder`), configured from a
// locally prepared Ising artifact directory.
//
// This app follows the surface_code-4 lineage rather than the surface_code-1
// lineage of its sibling, because the Ising model's artifacts are bound to
// that lineage's conventions:
//   - code orientation XV (the predecoder's training orientation),
//   - per-round measurement layout [X-ancillas..., Z-ancillas...],
//   - SPAM (single-qubit depolarizing) noise at the model's trained operating
//     point (p_spam = 0.01 for the published d7/T7 checkpoint).
// DO NOT reorder the measurement/enqueue schedule in the kernels below: the
// artifact directory's D_sparse.txt maps Ising detector rows onto THIS exact
// live measurement stream.
//
// The artifact directory (--ising-artifacts-dir, or QEC_ISING_ARTIFACTS_DIR)
// holds six files, none of which ship with CUDA-Q QEC:
//   model.onnx    the exported Ising predecoder (from the gated Hugging Face
//                 SafeTensors checkpoint via the NVIDIA/Ising-Decoding repo)
//   H_csr.bin     detector-error matrix, Ising detector order (binary CSR)
//   O_csr.bin     observables-flips matrix (binary CSR)
//   priors.bin    per-error-mechanism priors (error_rate_vec)
//   metadata.txt  key=value contract: basis=Z, code_rotation=XV, distance,
//                 n_rounds
//   D_sparse.txt  each Ising detector as a parity over THIS app's live
//                 measurement buffer (generated with the shipped
//                 prepare_ising_artifacts.py from this app's printed
//                 CNOT schedules)
// See the example documentation for the full preparation recipe.
//
// Two build variants (see CMakeLists.txt):
//   surface_code_ising_realtime_decoding      generator: writes the decoder
//       config YAML (--save_dem) embedding the Ising matrices + ONNX path,
//       and captures/replays syndrome files (--save_syndrome/--load_syndrome).
//   surface_code_ising_realtime_decoding-cqr  lowered kernel
//       (-frealtime-lowering -DQEC_APP_CQR): streams the live syndromes to
//       the delivered decoding_server over udp or cpu_roce.

#include "cudaq.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/codes/surface_code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#ifdef QEC_APP_CQR
#include "cudaq/realtime.h"
#endif
#include <algorithm>
#include <cmath>
#include <common/NoiseModel.h>
#include <cstdint>
#include <cstdlib>
#include <cudaq/algorithms/dem.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

namespace {
std::string env_or(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return (value && *value) ? std::string(value) : fallback;
}
} // namespace

#ifdef QEC_APP_CQR
// cqr build variant: this same application compiled with -frealtime-lowering
// and linked against the simulation-cqr client wrappers, so every
// cudaq::qec::decoding::* device_call crosses the cudaq-realtime wire to a
// standalone decoding_server (QEC_DECODING_SERVER_PORT=<port>). The wire
// defaults to udp loopback; set QEC_DECODING_SERVER_TRANSPORT=cpu_roce to use
// the CPU RoCE RDMA channel instead (works over loopback-cabled ConnectX ports
// or SoftRoCE/rdma_rxe; the RDMA topology comes from the same
// CUDAQ_CPU_ROCE_TEST_* env vars as CUDA-Q's CpuRoceChannelTester).

// In-process service self-check hook (defined in decoding-server-cqr):
// non-zero only if device_calls actually traversed the host-dispatch ring.
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();

namespace {

class realtime_channel_guard {
public:
  realtime_channel_guard() = default;
  realtime_channel_guard(const realtime_channel_guard &) = delete;
  realtime_channel_guard &operator=(const realtime_channel_guard &) = delete;

  ~realtime_channel_guard() {
    if (active_)
      cudaq::realtime::finalize();
  }

  void initialize(const char *program) {
    const char *port = std::getenv("QEC_DECODING_SERVER_PORT");
    if (!port || port[0] == '\0')
      throw std::runtime_error(
          "QEC_DECODING_SERVER_PORT is required for external decoding");

    std::vector<std::string> args = {program};
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
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (auto &arg : args)
      argv.push_back(arg.data());
    argv.push_back(nullptr);
    int argc = static_cast<int>(args.size());
    cudaq::realtime::initialize(argc, argv.data());
    active_ = true;
  }

private:
  bool active_ = false;
};

} // namespace
#endif

// Host-side decoding API (for syndrome capture)
namespace cudaq::qec::decoding::host {
void _set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t));
}

// Global syndrome capture state for --save_syndrome option.
//
// The live path enqueues a HETEROGENEOUS stream per shot: `num_rounds` uniform
// syndrome rounds of `g_syndrome_bits_per_round` bits each (the prep round plus
// the num_rounds-1 paired rounds), followed by ONE final DATA round of
// `g_data_bits` bits (numData). The capture callback fires once per enqueue, in
// chronological order, so it uses the per-shot enqueue index
// (g_syndrome_count % g_enqueues_per_shot) to decide how many bits to record:
// the syndrome rounds emit g_syndrome_bits_per_round, the last enqueue emits
// g_data_bits.
static std::ofstream g_syndrome_output_file;
static std::mutex g_syndrome_file_mutex;
static int g_syndrome_count = 0;
static int g_enqueues_per_shot = 0;
static int g_syndrome_bits_per_round = 0;
static int g_data_bits = 0;

// ---------------------------------------------------------------------------
// Ising artifact interop (trt_decoder path)
//
// The Ising d/T/Z artifacts (generate_test_data.py) ship H_csr.bin/O_csr.bin/
// priors.bin in *Ising detector order*, plus a D_sparse.txt we generate that
// expresses each Ising detector as a parity over the *cudaqx* live measurement
// buffer. With these, the trt+Ising config carries Ising's exact H/O/priors and
// a D_sparse aligned to Ising's detector rows while reading the cudaqx stream.
//
// Geometry: cudaqx's surface code at orientation XV is identical to the Ising
// geometry (Ising code_rotation string "XV" == first_bulk X, rotated_type V,
// logical_direction XH) under the IDENTITY data and X-ancilla mapping; only the
// Z-ancillas are permuted (a fixed bijection). D_sparse.txt encodes exactly
// that: it takes Ising's detector->measurement map and translates each
// measurement index from Ising's buffer order into cudaqx's buffer order
// (X-ancilla and data identity, Z-ancilla permuted), so every D row reproduces
// one of cudaqx's own detector bits, now in Ising's detector order.
// ---------------------------------------------------------------------------

// The six files every prepared artifact directory must contain. Checked up
// front so a partial preparation fails with the complete list of gaps instead
// of a low-level read error.
static void require_ising_artifact_files(const std::string &dir) {
  const char *required[] = {"model.onnx", "H_csr.bin",    "O_csr.bin",
                            "priors.bin", "metadata.txt", "D_sparse.txt"};
  std::string missing;
  for (const char *name : required) {
    std::ifstream f(dir + "/" + name, std::ios::binary);
    if (!f || f.peek() == std::ifstream::traits_type::eof())
      missing += std::string(" ") + name;
  }
  if (!missing.empty())
    throw std::runtime_error(
        "Ising artifact directory '" + dir +
        "' is incomplete; missing or empty:" + missing +
        " (see the realtime_decoding_demo documentation for the "
        "preparation recipe)");
}

// Read a binary-CSR file (rows:u32, cols:u32, nnz:u32, indptr[(rows+1)*i32],
// indices[nnz*i32]) and return it as a -1-terminated sparse row vector
// (each row lists its non-zero column indices, terminated by -1) -- exactly the
// shape pcm_to_sparse_vec produces. Also returns the row/col counts.
static std::vector<std::int64_t>
read_csr_bin_to_sparse_vec(const std::string &path, std::uint32_t &rows,
                           std::uint32_t &cols) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Could not open Ising CSR file: " + path);
  std::uint32_t nnz = 0;
  f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  f.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  f.read(reinterpret_cast<char *>(&nnz), sizeof(nnz));
  std::vector<std::int32_t> indptr(rows + 1), indices(nnz);
  f.read(reinterpret_cast<char *>(indptr.data()),
         static_cast<std::streamsize>(indptr.size() * sizeof(std::int32_t)));
  f.read(reinterpret_cast<char *>(indices.data()),
         static_cast<std::streamsize>(indices.size() * sizeof(std::int32_t)));
  if (!f)
    throw std::runtime_error("Truncated Ising CSR file: " + path);
  std::vector<std::int64_t> sparse;
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::int32_t k = indptr[r]; k < indptr[r + 1]; ++k)
      sparse.push_back(static_cast<std::int64_t>(indices[k]));
    sparse.push_back(-1);
  }
  return sparse;
}

// Read priors.bin (n:u32, n*float64) -> error_rate_vec.
static std::vector<double> read_priors_bin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Could not open Ising priors file: " + path);
  std::uint32_t n = 0;
  f.read(reinterpret_cast<char *>(&n), sizeof(n));
  std::vector<double> priors(n);
  f.read(reinterpret_cast<char *>(priors.data()),
         static_cast<std::streamsize>(priors.size() * sizeof(double)));
  if (!f)
    throw std::runtime_error("Truncated Ising priors file: " + path);
  return priors;
}

// Read D_sparse.txt -- a whitespace-separated, -1-terminated sparse detector
// matrix. One row per Ising detector; entries are cudaqx live-buffer
// measurement indices (so each row reproduces a cudaqx detector bit, in Ising's
// detector order). Returns the flat -1-terminated vector and the row count.
static std::vector<std::int64_t> read_D_sparse_txt(const std::string &path,
                                                   std::size_t &numRows) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Could not open Ising D_sparse.txt file: " + path);
  std::vector<std::int64_t> D;
  std::int64_t v;
  numRows = 0;
  while (f >> v) {
    D.push_back(v);
    if (v == -1)
      ++numRows;
  }
  return D;
}

// Read the artifact directory's metadata.txt (key=value lines) and enforce
// that it matches THIS experiment: basis Z, code_rotation XV, and the same
// distance / n_rounds. This is the semantic guard the dimensional checks
// cannot provide -- so the artifacts' basis and orientation match the
// experiment.
static void enforce_ising_metadata(const std::string &bundle, int distance,
                                   std::size_t numRounds) {
  std::ifstream f(bundle + "/metadata.txt");
  if (!f)
    throw std::runtime_error("Ising artifacts missing metadata.txt: " + bundle +
                             "/metadata.txt");
  std::vector<std::pair<std::string, std::string>> kv;
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
      line.pop_back();
    auto eq = line.find('=');
    if (eq != std::string::npos)
      kv.emplace_back(line.substr(0, eq), line.substr(eq + 1));
  }
  auto require = [&](const std::string &key, const std::string &want) {
    for (const auto &p : kv)
      if (p.first == key) {
        if (p.second != want)
          throw std::runtime_error("Ising artifacts " + bundle +
                                   ": metadata.txt " + key + "='" + p.second +
                                   "' != required '" + want + "'");
        return;
      }
    throw std::runtime_error("Ising artifacts " + bundle +
                             ": metadata.txt missing key '" + key + "'");
  };
  require("basis", "Z");
  require("code_rotation", "XV");
  require("distance", std::to_string(distance));
  require("n_rounds", std::to_string(numRounds));
}

// Flatten cudaqx's m2d into the -1-terminated sparse detector matrix, in
// cudaqx detector order (D_sparse[i] == m2d.rows[i]). This is the
// self-consistent D for the cudaqx-native pymatching entry, which carries
// cudaqx's own dem_gen_circuit H/O. It references the same chronological
// measurement indices the live path enqueues (385 bits for d7/T7), including
// the final 49 data measurements used by the boundary detectors.
static std::vector<std::int64_t>
build_cudaqx_D_sparse(const cudaq::M2DSparseMatrix &m2d) {
  std::vector<std::int64_t> D;
  for (const auto &row : m2d.rows) {
    for (auto meas : row)
      D.push_back(static_cast<std::int64_t>(meas));
    D.push_back(-1);
  }
  return D;
}

// Write the single-decoder config. `pymatching` carries cudaqx's own DEM;
// `trt_decoder` carries the Ising artifacts: H/O/priors in Ising detector
// order plus the Ising-ordered D_sparse over the cudaqx live buffer, and the
// ONNX predecoder path (<artifacts>/model.onnx) so the decoding server's TRT
// plugin builds its engine from the same model after reload.
void save_dem_to_file(const std::vector<cudaq::qec::detector_error_model> &dems,
                      std::string dem_filename,
                      const std::vector<std::string> &decoder_types,
                      const cudaq::M2DSparseMatrix &m2d,
                      const std::string &ising_artifacts_dir, int distance,
                      std::size_t numRounds) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  for (uint64_t i = 0; i < decoder_types.size(); i++) {
    const std::string &decoder_type = decoder_types[i];
    const auto &edem = dems[i];
    cudaq::qec::decoding::config::decoder_config config;
    config.id = i;
    config.type = decoder_type;
    config.block_size = edem.num_error_mechanisms();
    config.syndrome_size = edem.num_detectors();
    config.H_sparse = cudaq::qec::pcm_to_sparse_vec(edem.detector_error_matrix);
    config.O_sparse =
        cudaq::qec::pcm_to_sparse_vec(edem.observables_flips_matrix);
    config.error_rate_vec = edem.error_rates;
    // Default D == cudaqx's m2d (cudaqx detector order), self-consistent with
    // the dem_gen_circuit H/O above and with the full measurement stream the
    // live path enqueues. The trt+Ising branch below overrides this with the
    // Ising-ordered D_sparse.txt to match the Ising H/O.
    config.D_sparse = build_cudaqx_D_sparse(m2d);

    if (decoder_type == "pymatching") {
      cudaqx::heterogeneous_map pm_args;
      pm_args.insert("merge_strategy", "smallest_weight");
      config.decoder_custom_args = pm_args;
    } else if (decoder_type == "trt_decoder") {
      if (ising_artifacts_dir.empty())
        throw std::runtime_error(
            "trt_decoder requires an Ising artifact directory "
            "(--ising-artifacts-dir or QEC_ISING_ARTIFACTS_DIR)");
      // Enforce the artifacts' semantics match this experiment (basis Z,
      // code_rotation XV, same d / n_rounds) before trusting the matrices.
      require_ising_artifact_files(ising_artifacts_dir);
      enforce_ising_metadata(ising_artifacts_dir, distance, numRounds);

      cudaqx::heterogeneous_map trt_args;
      // The TensorRT predecoder model is supplied through the saved decoder
      // config so the same ONNX path is used after reload.
      trt_args.insert("onnx_load_path", ising_artifacts_dir + "/model.onnx");
      trt_args.insert("batch_size", std::size_t{1});
      trt_args.insert("use_cuda_graph", true);
      trt_args.insert("engine_output_format",
                      std::string("observables_and_residual_detectors"));
      trt_args.insert("global_decoder", "pymatching");

      cudaqx::heterogeneous_map pm_args;
      pm_args.insert("merge_strategy", "smallest_weight");

      // trt+Ising path: carry the Ising d/T/Z model. H/O/priors come from
      // the artifacts (Ising detector order); D_sparse (D_sparse.txt)
      // expresses each Ising detector as a parity over the cudaqx live
      // measurement buffer, so the live stream feeds Ising's detectors in
      // Ising's row order.
      std::uint32_t hRows = 0, hCols = 0, oRows = 0, oCols = 0;
      config.H_sparse = read_csr_bin_to_sparse_vec(
          ising_artifacts_dir + "/H_csr.bin", hRows, hCols);
      config.O_sparse = read_csr_bin_to_sparse_vec(
          ising_artifacts_dir + "/O_csr.bin", oRows, oCols);
      auto priors = read_priors_bin(ising_artifacts_dir + "/priors.bin");
      std::size_t dRows = 0;
      config.D_sparse =
          read_D_sparse_txt(ising_artifacts_dir + "/D_sparse.txt", dRows);

      if (hRows != m2d.rows.size())
        throw std::runtime_error("Ising H rows (" + std::to_string(hRows) +
                                 ") != cudaqx m2d detectors (" +
                                 std::to_string(m2d.rows.size()) + ")");
      if (dRows != m2d.rows.size())
        throw std::runtime_error("D_sparse.txt rows (" + std::to_string(dRows) +
                                 ") != cudaqx m2d detectors (" +
                                 std::to_string(m2d.rows.size()) + ")");
      if (hCols != oCols || hCols != priors.size())
        throw std::runtime_error("Ising H/O/priors column counts disagree");

      config.syndrome_size = hRows;
      config.block_size = hCols;
      config.error_rate_vec = priors;
      printf("trt+Ising: loaded Ising artifacts '%s' (H %ux%u, O %u rows, "
             "priors %zu); D_sparse from D_sparse.txt (%zu detectors)\n",
             ising_artifacts_dir.c_str(), hRows, hCols, oRows, priors.size(),
             dRows);
      trt_args.insert("global_decoder_params", pm_args);

      config.decoder_custom_args = trt_args;
    }

    multi_config.decoders.push_back(config);
  }
  std::string config_str = multi_config.to_yaml_str(200);
  std::ofstream config_file(dem_filename);
  config_file << config_str;
  config_file.close();
  if (!config_file)
    throw std::runtime_error("failed to write decoder config: " + dem_filename);
  std::string types_str;
  for (std::size_t i = 0; i < decoder_types.size(); ++i)
    types_str += (i ? "," : "") + decoder_types[i];
  printf("Saved %s config to file: %s\n", types_str.c_str(),
         dem_filename.c_str());
  return;
}

// Parse and validate a saved decoder config: geometry guards, per-id types,
// and the reconstructed DEM. Does NOT construct decoders -- instantiation is
// the serving side's job (this process for in-process serving; the decoding
// server for a server-served run).
cudaq::qec::decoding::config::multi_decoder_config
load_decoder_config(const std::string &dem_filename,
                    cudaq::qec::detector_error_model &dem, uint64_t numLogical,
                    uint64_t &measSpan,
                    std::vector<std::string> &decoder_types_out) {
  printf("load_decoder_config: Loading dem from file: %s\n",
         dem_filename.c_str());
  // Read dem_filename into a std::string
  std::ifstream dem_file(dem_filename);
  if (!dem_file)
    throw std::runtime_error("could not open decoder config file: " +
                             dem_filename);
  std::string dem_str((std::istreambuf_iterator<char>(dem_file)),
                      std::istreambuf_iterator<char>());
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          dem_str);
  if (numLogical != config.decoders.size())
    throw std::runtime_error("numLogical [" + std::to_string(numLogical) +
                             "] != config.decoders.size() [" +
                             std::to_string(config.decoders.size()) + "]");
  auto span_of = [](const std::vector<std::int64_t> &sparse) {
    std::int64_t m = -1;
    for (auto v : sparse)
      if (v > m)
        m = v;
    return static_cast<uint64_t>(m + 1);
  };
  const auto &d0 = config.decoders[0];
  const uint64_t span0 = span_of(d0.D_sparse);
  decoder_types_out.assign(config.decoders.size(), std::string());
  for (size_t k = 0; k < config.decoders.size(); ++k) {
    const auto &dc = config.decoders[k];
    if (dc.id < 0 || static_cast<size_t>(dc.id) >= config.decoders.size() ||
        !decoder_types_out[dc.id].empty())
      throw std::runtime_error(
          "loaded config decoder ids must be unique and dense in [0, " +
          std::to_string(config.decoders.size()) + "); entry " +
          std::to_string(k) + " has id " + std::to_string(dc.id));
    decoder_types_out[dc.id] = dc.type;
    const size_t nobs = std::count(dc.O_sparse.begin(), dc.O_sparse.end(), -1);
    if (nobs != 1)
      throw std::runtime_error(
          "loaded config decoder " + std::to_string(k) + " has " +
          std::to_string(nobs) +
          " observables; expected 1 (one observable per single-patch "
          "surface-code decoder)");
    if (dc.syndrome_size != d0.syndrome_size || span_of(dc.D_sparse) != span0)
      throw std::runtime_error(
          "loaded config decoder " + std::to_string(k) +
          " geometry (detectors " + std::to_string(dc.syndrome_size) +
          ", measurement span " + std::to_string(span_of(dc.D_sparse)) +
          ") differs from decoder 0 (detectors " +
          std::to_string(d0.syndrome_size) + ", span " + std::to_string(span0) +
          "); all decoders must share one experiment geometry");
  }

  auto decoder_config = d0;
  dem.detector_error_matrix = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.H_sparse, decoder_config.syndrome_size,
      decoder_config.block_size);
  size_t num_observables = std::count(decoder_config.O_sparse.begin(),
                                      decoder_config.O_sparse.end(), -1);
  dem.observables_flips_matrix = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.O_sparse, num_observables, decoder_config.block_size);
  // The runtime decodes once the enqueued measurement buffer is full; that span
  // is max(D_sparse)+1. Surface it so the caller can bind the config to this
  // experiment's geometry (see the load_dem check in demo_circuit_host).
  measSpan = span0;

  std::string types_str;
  for (std::size_t i = 0; i < decoder_types_out.size(); ++i)
    types_str += (i ? "," : "") + decoder_types_out[i];
  printf("Loaded %s config from file: %s\n", types_str.c_str(),
         dem_filename.c_str());
  return config;
}

std::vector<size_t> get_stab_cnot_schedule(char stab_type, int distance) {
  // Build the stabilizer CNOT schedule from an XV-oriented grid (the
  // predecoder's training orientation).
  cudaq::qec::surface_code::stabilizer_grid grid(
      distance, cudaq::qec::surface_code::sc_orientation::XV);
  if (stab_type != 'X' && stab_type != 'Z') {
    throw std::runtime_error(
        "get_stab_cnot_schedule: Invalid stabilizer type. Must be 'X' or 'Z'.");
  }
  // CNOT pairs ordered by timestep within each stabilizer, so that mid-round
  // ancilla (hook) errors land perpendicular to the logical operators.
  // Stabilizer indices match the sorted parity-matrix rows and hence the
  // ancilla indexing.
  return stab_type == 'X' ? grid.get_cnot_schedule_pairs_x()
                          : grid.get_cnot_schedule_pairs_z();
}

// Per-stabilizer data-qubit supports, ordered to match the ancilla measurement
// order produced by get_stab_cnot_schedule(stab_type, ...). The supports are
// derived from the same schedule matrix as the CNOT pairs (row s = ancilla s),
// so support[s] lines up with ancilla[s] in the extraction kernels'
// measurement vector by construction rather than via a parallel sort. This is
// what the Ising MemoryCircuit boundary detectors pair against (a stabilizer's
// data support XOR-ed with that stabilizer's last ancilla measurement).
// Returns a flat vector of data-qubit indices plus per-stabilizer offsets
// (offsets has size num_stabs+1; support s spans [offsets[s], offsets[s+1])),
// the same flat+offset pattern as cnot_schedZ_flat.
void get_stab_data_supports(char stab_type, int distance,
                            std::vector<std::size_t> &supports_flat,
                            std::vector<std::size_t> &supports_offsets) {
  cudaq::qec::surface_code::stabilizer_grid grid(
      distance, cudaq::qec::surface_code::sc_orientation::XV);
  if (stab_type != 'X' && stab_type != 'Z') {
    throw std::runtime_error(
        "get_stab_data_supports: Invalid stabilizer type. Must be 'X' or 'Z'.");
  }
  auto sched = stab_type == 'X' ? grid.get_cnot_schedule_x()
                                : grid.get_cnot_schedule_z();
  supports_flat.clear();
  supports_offsets.clear();
  supports_offsets.push_back(0);
  for (std::size_t s = 0; s < sched.shape()[0]; ++s) {
    for (std::size_t d = 0; d < sched.shape()[1]; ++d) {
      if (sched.at({s, d}) != 0)
        supports_flat.push_back(d);
    }
    supports_offsets.push_back(supports_flat.size());
  }
}

namespace cudaq::qec::qpu {

__qpu__ void spam_error(cudaq::qec::patch logicalQubit, double p_spam_data,
                        double p_spam_ancx, double p_spam_ancz) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_data,
                                               logicalQubit.data[i]);
  }
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_ancx,
                                               logicalQubit.ancx[i]);
  }
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_ancz,
                                               logicalQubit.ancz[i]);
  }
}

__qpu__ std::vector<cudaq::measure_result>
extract_z_syndrome(cudaq::qec::patch logicalQubit,
                   const std::vector<std::size_t> &cnot_sched) {
  for (std::size_t i = 0; i < cnot_sched.size(); i += 2) {
    cudaq::x<cudaq::ctrl>(logicalQubit.data[cnot_sched[i + 1]],
                          logicalQubit.ancz[cnot_sched[i]]);
  }
  auto results = mz(logicalQubit.ancz);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);
  return results;
}

__qpu__ std::vector<cudaq::measure_result>
extract_x_syndrome(cudaq::qec::patch logicalQubit,
                   const std::vector<std::size_t> &cnot_sched) {
  h(logicalQubit.ancx);
  for (std::size_t i = 0; i < cnot_sched.size(); i += 2) {
    cudaq::x<cudaq::ctrl>(logicalQubit.ancx[cnot_sched[i]],
                          logicalQubit.data[cnot_sched[i + 1]]);
  }
  h(logicalQubit.ancx);
  auto results = mz(logicalQubit.ancx);
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  return results;
}

__qpu__ void
measure_syndrome_round(cudaq::qec::patch logicalQubit,
                       const std::vector<std::size_t> &cnot_schedX_flat,
                       const std::vector<std::size_t> &cnot_schedZ_flat,
                       std::vector<cudaq::measure_result> &combined_syndrome) {
  // Measure X-ancillas then Z-ancillas (combined layout [X..., Z...]). The
  // shared helper keeps the DEM and the live enqueue path on the same ordering.
  auto syndrome_x = extract_x_syndrome(logicalQubit, cnot_schedX_flat);
  auto syndrome_z = extract_z_syndrome(logicalQubit, cnot_schedZ_flat);
  int i = 0;
  for (auto s : syndrome_x)
    combined_syndrome[i++] = s;
  for (auto s : syndrome_z)
    combined_syndrome[i++] = s;
}

// Run ONE syndrome-extraction round on one patch and (optionally) enqueue it
// to that patch's decoder. The round loop lives in demo_circuit_qpu so the
// patches can be interleaved round-major. round_counter is the per-patch
// monotonic enqueue index (0 = lock-in, 1..T-1 paired, T = data round):
// logging-only on the in-process path, the round counter on the
// decoding-server wire.
__qpu__ void
syndrome_round_once(cudaq::qview<> data, cudaq::qview<> xstab_anc,
                    cudaq::qview<> zstab_anc,
                    const std::vector<std::size_t> &cnot_schedX_flat,
                    const std::vector<std::size_t> &cnot_schedZ_flat,
                    bool do_spam, double p_spam, bool do_enqueue,
                    int logical_qubit_idx, std::uint64_t round_counter) {
  patch logical(data, xstab_anc, zstab_anc);
  std::vector<cudaq::measure_result> combined_syndrome(xstab_anc.size() +
                                                       zstab_anc.size());
  // Inject errors BEFORE the measurement, matching the DEM kernel's
  // spam-then-measure placement.
  if (do_spam)
    spam_error(logical, p_spam, 0.0, 0.0);
  measure_syndrome_round(logical, cnot_schedX_flat, cnot_schedZ_flat,
                         combined_syndrome);
  if (do_enqueue)
    cudaq::qec::decoding::enqueue_syndromes(
        /*decoder_id=*/logical_qubit_idx, combined_syndrome, round_counter);
}

__qpu__ std::vector<std::uint64_t> demo_circuit_qpu(
    bool allow_device_calls,
    const cudaq::qec::code::one_qubit_encoding &statePrep, std::size_t numData,
    std::size_t numAncx, std::size_t numAncz, std::size_t numRounds,
    std::size_t numLogical, const std::vector<std::size_t> &cnot_schedX_flat,
    const std::vector<std::size_t> &cnot_schedZ_flat,
    const std::vector<double> &p_spam_per_patch, bool apply_corrections) {
  // ret[i] = patch i's final data bits (numData bits, enforced < 64 in
  // main()); ret[numLogical] = per-patch correction bitmask (bit i set iff
  // decoder i predicted a logical flip).
  std::vector<std::uint64_t> ret(numLogical + 1);
  for (std::size_t k = 0; k <= numLogical; k++)
    ret[k] = 0;

  // Reset the decoder
  if (allow_device_calls) {
    for (int i = 0; i < numLogical; i++) {
      cudaq::qec::decoding::reset_decoder(/*decoder_id=*/i);
    }
  }

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numLogical * numData), xstab_anc(numLogical * numAncx),
      zstab_anc(numLogical * numAncz);

  // Call state prep
  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
    auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);
    patch logical(subData, subXstab_anc, subZstab_anc);
    statePrep(logical);
  }

  // Round 0 (lock-in) for every patch: no spam; pins the stabilizer
  // eigenvalues after prep.
  for (int i = 0; i < numLogical; i++) {
    syndrome_round_once(
        data.slice(i * numData, numData), xstab_anc.slice(i * numAncx, numAncx),
        zstab_anc.slice(i * numAncz, numAncz), cnot_schedX_flat,
        cnot_schedZ_flat, /*do_spam=*/false, p_spam_per_patch[i],
        /*do_enqueue=*/allow_device_calls, i,
        /*round_counter=*/0);
  }

  // Paired rounds, ROUND-MAJOR: every patch measures round r before any patch
  // measures round r+1. Each patch's own stream (spam BEFORE each paired
  // round, then measure) is identical to the DEM kernel's schedule. This
  // example decodes ONE volume of num_rounds rounds (no sliding windows), so
  // each syndrome is enqueued EXACTLY ONCE.
  for (std::size_t round = 0; round + 1 < numRounds; round++) {
    for (int i = 0; i < numLogical; i++) {
      syndrome_round_once(data.slice(i * numData, numData),
                          xstab_anc.slice(i * numAncx, numAncx),
                          zstab_anc.slice(i * numAncz, numAncz),
                          cnot_schedX_flat, cnot_schedZ_flat, /*do_spam=*/true,
                          p_spam_per_patch[i],
                          /*do_enqueue=*/allow_device_calls, i,
                          /*round_counter=*/round + 1);
    }
  }

  // Final data round per patch: measure the data qubits in the logical (Z)
  // basis and enqueue them as the LAST enqueue of that patch's stream,
  // completing the num_rounds*(numAncx+numAncz)+numData-bit buffer the DEM was
  // built under (385 at d7/T7). The decoder's boundary detectors XOR these
  // data bits against the final-round Z-ancilla measurements.
  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subMeas = mz(subData);
    if (allow_device_calls)
      cudaq::qec::decoding::enqueue_syndromes(/*decoder_id=*/i, subMeas,
                                              /*round_counter=*/numRounds);
    ret[i] = cudaq::to_integer(cudaq::to_bools(subMeas));
  }

  // Apply each decoder's correction classically: a transversal X on the data
  // flips every measured data bit, so XOR the low-numData mask into that
  // patch's word when its decoder predicts a logical flip. This matches the
  // physical-X-then-measure result but lets us enqueue the (uncorrected) data
  // bits first.
  if (allow_device_calls && apply_corrections) {
    for (int i = 0; i < numLogical; i++) {
      auto correction_result = cudaq::qec::decoding::get_corrections(
          /*decoder_id=*/i, /*return_size=*/1, /*reset=*/false);
      if (correction_result[0] != 0) {
        ret[numLogical] |= (1ull << i);
        std::uint64_t mask = (1ull << numData) - 1;
        ret[i] = ret[i] ^ mask;
      }
    }
  }
  return ret;
}

// DEM-generation kernel. The example uses two kernels that share one syndrome-
// extraction helper (measure_syndrome_round): dem_gen_circuit (this one) is
// sampled by dem_from_kernel to build the DEM, so it is a pure detector-
// annotated circuit with no decoder calls; demo_circuit_qpu runs the live
// experiment and makes the decoder RPC calls (enqueue_syndromes /
// get_corrections). The two kernels need to stay in lockstep on the measurement
// schedule (same rounds, X-then-Z order, spam-then-measure placement).
//
// dem_gen_circuit declares detectors; the layout is basis Z, orientation XV:
//   - Block 0 (numAncz prep singles): one single-term detector per round-0
//     Z-ancilla (deterministic after prep0 |0>_L).
//   - Blocks 1+2 (paired): `pairedRounds` rounds, each pairing this round's
//     full syndrome [X..., Z...] against the previous round's, via
//     cudaq::detectors(prev, curr) (numAncx + numAncz detectors per round).
//   - Block 3 (numAncz boundary): one detector per Z-stabilizer s, XOR-ing the
//     final data measurements in that stabilizer's support with the
//     last-round Z-ancilla measurement for that stabilizer.
__qpu__ void
dem_gen_circuit(const cudaq::qec::code::one_qubit_encoding &statePrep,
                std::size_t numData, std::size_t numAncx, std::size_t numAncz,
                std::size_t pairedRounds,
                const std::vector<std::size_t> &cnot_schedX_flat,
                const std::vector<std::size_t> &cnot_schedZ_flat, double p_spam,
                const std::vector<std::size_t> &z_logical_indices,
                const std::vector<std::size_t> &z_supports_flat,
                const std::vector<std::size_t> &z_supports_offsets) {
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);
  patch logical(data, xstab_anc, zstab_anc);

  statePrep(logical);

  std::vector<cudaq::measure_result> prev(numAncz + numAncx);
  measure_syndrome_round(logical, cnot_schedX_flat, cnot_schedZ_flat, prev);

  // Block 0: prep singles on the round-0 Z-ancillas. Post-flip the combined
  // syndrome is [X(numAncx), Z(numAncz)], so the Z-ancillas are
  // prev[numAncx..].
  for (std::size_t k = 0; k < numAncz; ++k)
    cudaq::detector(prev[numAncx + k]);

  // Blocks 1+2: paired cross-round detectors. Exactly `pairedRounds` rounds
  // (the middle rounds plus the final round), giving
  // pairedRounds * (numAncx + numAncz) detectors.
  for (std::size_t round = 0; round < pairedRounds; ++round) {
    spam_error(logical, /*p_spam_data=*/p_spam, /*p_spam_ancx=*/0.0,
               /*p_spam_ancz=*/0.0);
    std::vector<cudaq::measure_result> curr(numAncz + numAncx);
    measure_syndrome_round(logical, cnot_schedX_flat, cnot_schedZ_flat, curr);
    cudaq::detectors(prev, curr);
    prev = curr;
  }

  auto dataM = mz(data);

  // Block 3: boundary detectors. For each Z-stabilizer s, XOR its data-qubit
  // support (from the final data measurement) with that stabilizer's last-round
  // Z-ancilla measurement (prev still holds the final round syndrome).
  for (std::size_t s = 0; s + 1 < z_supports_offsets.size(); ++s) {
    std::vector<cudaq::measure_result> stab_data(z_supports_offsets[s + 1] -
                                                 z_supports_offsets[s]);
    std::size_t j = 0;
    for (std::size_t t = z_supports_offsets[s]; t < z_supports_offsets[s + 1];
         ++t)
      stab_data[j++] = dataM[z_supports_flat[t]];
    cudaq::detector(stab_data, prev[numAncx + s]);
  }

  std::vector<cudaq::measure_result> zlog(z_logical_indices.size());
  for (std::size_t k = 0; k < z_logical_indices.size(); ++k)
    zlog[k] = dataM[z_logical_indices[k]];
  cudaq::logical_observable(zlog, /*observable_index=*/0);
}
} // namespace cudaq::qec::qpu

void demo_circuit_host(const cudaq::qec::code &code, int distance,
                       const std::vector<double> &p_spam_per_patch,
                       cudaq::qec::operation statePrep, std::size_t numShots,
                       std::size_t numRounds, std::size_t numLogical,
                       std::string dem_filename, bool save_dem, bool load_dem,
                       const std::vector<std::string> &decoder_types,
                       bool save_syndrome = false, bool load_syndrome = false,
                       std::string syndrome_filename = "",
                       std::string ising_artifacts_dir = "") {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep =
      code.get_operation<cudaq::qec::code::one_qubit_encoding>(statePrep);

  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  auto cnot_schedX_flat = get_stab_cnot_schedule('X', distance);
  auto cnot_schedZ_flat = get_stab_cnot_schedule('Z', distance);
  std::size_t numSyndromesPerRound = 0;

  // Print the CNOT schedules: prepare_ising_artifacts.py consumes these
  // lines to build the artifact directory's D_sparse.txt against THIS app's
  // measurement layout (its prepare mode runs this app itself; its d-sparse
  // mode takes them from a saved sched.txt).
  printf("cnot_schedX_flat: ");
  for (std::size_t i = 0; i < cnot_schedX_flat.size(); i += 2)
    printf("%lu %lu, ", cnot_schedX_flat[i], cnot_schedX_flat[i + 1]);
  printf("\n");
  printf("cnot_schedZ_flat: ");
  for (std::size_t i = 0; i < cnot_schedZ_flat.size(); i += 2)
    printf("%lu %lu, ", cnot_schedZ_flat[i], cnot_schedZ_flat[i + 1]);
  printf("\n");

  cudaq::noise_model noise;

  // First get the DEM.
  cudaq::qec::detector_error_model dem;
  // Per-id decoder types read back from the YAML (run phase), for the
  // per-decoder report lines.
  std::vector<std::string> loaded_types;
  if (load_dem) {
    uint64_t measSpan = 0;
    auto decoder_config = load_decoder_config(dem_filename, dem, numLogical,
                                              measSpan, loaded_types);
    const auto numDetectors = dem.detector_error_matrix.shape()[0];
    const auto fullSyndromesPerRound = numAncx + numAncz;
    // Bind the loaded config to this experiment's geometry: the detector count
    // and the measurement-buffer span (max(D_sparse)+1) must both match the
    // distance and num_rounds being run.
    const auto expectedDetectors =
        static_cast<std::size_t>(numRounds) * fullSyndromesPerRound;
    const auto expectedSpan = expectedDetectors + numData;
    if (numDetectors != expectedDetectors)
      throw std::runtime_error(
          "Loaded DEM detector count (" + std::to_string(numDetectors) +
          ") does not match this experiment (d=" + std::to_string(distance) +
          ", num_rounds=" + std::to_string(numRounds) + " -> expected " +
          std::to_string(expectedDetectors) + ")");
    if (measSpan != expectedSpan)
      throw std::runtime_error(
          "Loaded config measurement-buffer span max(D_sparse)+1 (" +
          std::to_string(measSpan) +
          ") does not match this experiment's enqueued bits (expected " +
          std::to_string(expectedSpan) +
          " = num_rounds*(numAncx+numAncz)+numData for d=" +
          std::to_string(distance) +
          ", num_rounds=" + std::to_string(numRounds) +
          "); the YAML was generated for a different geometry");
    numSyndromesPerRound = fullSyndromesPerRound;
    // Construct local decoders only for the in-process executable. The external
    // executable still performs all geometry validation above, but the server
    // owns the decoder instances described by this same configuration.
#ifdef QEC_APP_CQR
    printf("External decoding server owns all configured decoder instances\n");
#else
    int rc = cudaq::qec::decoding::config::configure_decoders(decoder_config);
    if (rc != 0)
      throw std::runtime_error("configure_decoders failed (status " +
                               std::to_string(rc) + ")");
#endif
  } else {
    for (std::size_t i = 0; i < p_spam_per_patch.size(); ++i)
      if (p_spam_per_patch[i] == 0.0)
        throw std::runtime_error("--p_spam must be > 0 to generate a DEM");

    // Left-column data qubits are the Z logical for XV. This must track the
    // code's Z observable (code.get_observables_z()); if the orientation
    // changes to XH/ZV the Z logical moves to the top row.
    std::vector<std::size_t> z_logical_indices;
    for (int i = 0; i < distance; ++i)
      z_logical_indices.push_back(static_cast<std::size_t>(i) * distance);

    // Z-stabilizer data-qubit supports, aligned to the Z-ancilla measurement
    // order (support[s] <-> Z-ancilla[s]). Used to build the boundary
    // detectors (basis Z).
    std::vector<std::size_t> z_supports_flat, z_supports_offsets;
    get_stab_data_supports('Z', distance, z_supports_flat, z_supports_offsets);

    // The DEM detector layout is:
    //   numAncz prep singles + pairedRounds * (numAncx + numAncz) paired
    //   + numAncz boundary.
    // This example decodes ONE volume of num_rounds rounds. Ising's n_rounds
    // counts the state-prep round and the final logical-measurement round, so
    // it has (n_rounds - 1) paired transitions. Here the round-0 syndrome is
    // the prep round and the volume covers num_rounds syndrome rounds total, so
    // there are (num_rounds - 1) paired rounds (the 5 middle + 1 final at d=7,
    // num_rounds=7).
    const std::size_t pairedRounds = static_cast<std::size_t>(numRounds) - 1;

    // Both supported decoder types (pymatching, and trt_decoder's global
    // pymatching stage) are matching decoders, so the DEM is always sampled
    // with decomposition suggestions.
    const bool decompose_errors = true;
    cudaq::M2DSparseMatrix m2d;
    std::vector<cudaq::qec::detector_error_model> patch_dems;
    patch_dems.reserve(numLogical);
    for (std::size_t patch = 0; patch < numLogical; ++patch) {
      cudaq::M2DSparseMatrix patch_m2d;
      cudaq::M2OSparseMatrix patch_m2o;
      const std::string dem_text = cudaq::dem_from_kernel(
          cudaq::qec::qpu::dem_gen_circuit, &noise,
          cudaq::dem_options{.decompose_errors = decompose_errors}, patch_m2d,
          patch_m2o, prep, numData, numAncx, numAncz, pairedRounds,
          cnot_schedX_flat, cnot_schedZ_flat, p_spam_per_patch[patch],
          z_logical_indices, z_supports_flat, z_supports_offsets);
      if (patch == 0)
        m2d = patch_m2d;
      else if (patch_m2d.rows != m2d.rows)
        throw std::runtime_error(
            "per-patch DEMs produced different measurement mappings");

      patch_dems.push_back(cudaq::qec::dem_from_stim_text(
          dem_text, /*use_decomp_suggestions=*/decompose_errors));
    }
    dem = patch_dems.front();

    numSyndromesPerRound = numAncx + numAncz;
    printf("numSyndromesPerRound: %ld\n", numSyndromesPerRound);

    if (save_dem) {
      save_dem_to_file(patch_dems, dem_filename, decoder_types, m2d,
                       ising_artifacts_dir, distance, numRounds);
      return;
    }
  }

  // Detector count for one volume of num_rounds rounds:
  //   numAncz (prep singles) + (num_rounds-1)*(numAncx+numAncz) (paired)
  //   + numAncz (boundary).
  // This is divisible by numSyndromesPerRound (= numAncx+numAncz) into exactly
  // num_rounds rounds ONLY because numAncx == numAncz for the rotated surface
  // code (so the prep block numAncz and the boundary block numAncz together
  // make one full numAncx+numAncz round).
  if (dem.detector_error_matrix.shape()[0] % numSyndromesPerRound != 0) {
    throw std::runtime_error("Num syndromes per round is not a divisor of "
                             "the number of syndrome measurements");
  }
  size_t numRoundsOfSyndromData =
      dem.detector_error_matrix.shape()[0] / numSyndromesPerRound;

  if (numRoundsOfSyndromData != static_cast<std::size_t>(numRounds)) {
    throw std::runtime_error("Num rounds of syndrome data [" +
                             std::to_string(numRoundsOfSyndromData) +
                             "] is not equal to num_rounds [" +
                             std::to_string(numRounds) + "]");
  }

  // Setup syndrome capture if requested (--save_syndrome option)
  if (save_syndrome) {
    if (syndrome_filename.empty()) {
      printf("Error: --save_syndrome requires a filename argument\n");
      return;
    }

    g_syndrome_output_file.open(syndrome_filename,
                                std::ios::out | std::ios::trunc);
    if (!g_syndrome_output_file) {
      printf("Error: Could not open syndrome file for writing: %s\n",
             syndrome_filename.c_str());
      return;
    }

    // Per-shot enqueue structure: num_rounds uniform syndrome rounds
    // (numSyndromesPerRound bits each) + 1 final DATA round (numData bits). The
    // capture callback fires once per enqueue in chronological order, so the
    // per-shot enqueue index distinguishes the heterogeneous final round.
    g_enqueues_per_shot = static_cast<int>(numRounds) + 1;
    g_syndrome_bits_per_round = static_cast<int>(numSyndromesPerRound);
    g_data_bits = static_cast<int>(numData);
    g_syndrome_count = 0;

    printf("Syndrome capture enabled: saving to %s\n",
           syndrome_filename.c_str());
    printf(
        "Will capture %d enqueues per shot (%ld syndrome rounds of %d bits + "
        "1 data round of %d bits)\n",
        g_enqueues_per_shot, numRounds, g_syndrome_bits_per_round, g_data_bits);

    // Write metadata to file header
    g_syndrome_output_file << "NUM_DATA " << numData << "\n";
    g_syndrome_output_file << "NUM_LOGICAL " << numLogical << "\n";
    g_syndrome_output_file.flush();

    // Register capture callback with decoder library
    cudaq::qec::decoding::host::_set_syndrome_capture_callback(
        [](const uint8_t *data, size_t len) {
          std::lock_guard<std::mutex> lock(g_syndrome_file_mutex);
          if (!g_syndrome_output_file.is_open())
            return;

          // Position of this enqueue within the current shot.
          int enqueue_idx = g_syndrome_count % g_enqueues_per_shot;

          // Write shot boundary marker at the start of each shot
          if (enqueue_idx == 0) {
            int shot_num = g_syndrome_count / g_enqueues_per_shot;
            g_syndrome_output_file << "SHOT_START " << shot_num << "\n";
          }

          // The last enqueue of each shot is the final DATA round (numData
          // bits); all others are uniform syndrome rounds. Record the true bit
          // width on the ROUND_START line so replay can chunk it back exactly.
          int bits_this_round = (enqueue_idx == g_enqueues_per_shot - 1)
                                    ? g_data_bits
                                    : g_syndrome_bits_per_round;

          g_syndrome_output_file << "ROUND_START " << g_syndrome_count << " "
                                 << bits_this_round << "\n";

          // Unpack syndrome data - each byte contains 8 bits (packed format,
          // MSB first). Emit exactly bits_this_round bits.
          int bits_written = 0;
          for (size_t i = 0; i < len && bits_written < bits_this_round; i++) {
            uint8_t byte = data[i];
            for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
              if (bits_written >= bits_this_round)
                break;
              int bit = (byte >> bit_idx) & 1;
              g_syndrome_output_file << bit << "\n";
              bits_written++;
            }
          }
          g_syndrome_output_file.flush();

          g_syndrome_count++;
        });
  }

  // Either run quantum simulation OR replay syndromes from file
  std::vector<std::vector<std::uint64_t>> run_result;

  if (load_syndrome) {
    // Syndrome replay mode
    if (syndrome_filename.empty())
      throw std::runtime_error("--load_syndrome requires a filename argument");

    printf("\n=== Syndrome Replay Mode ===\n");
    printf("Loading syndromes from: %s\n", syndrome_filename.c_str());

    std::ifstream syndrome_file(syndrome_filename);
    if (!syndrome_file)
      throw std::runtime_error("Could not open syndrome file: " +
                               syndrome_filename);

    // Read header and syndrome data. Each shot is stored as a list of rounds,
    // and each round keeps its own bit count (the final DATA round has numData
    // bits, the syndrome rounds have numSyndromesPerRound). The per-round width
    // is read from the "ROUND_START <idx> <nbits>" header that the capture
    // writes, so replay reconstructs the heterogeneous stream exactly.
    std::size_t file_numData = 0;
    std::size_t file_numLogical = 0;
    std::vector<uint8_t> saved_corrections;
    std::vector<std::vector<std::vector<uint8_t>>> saved_shots;
    std::string line;

    bool reading_syndromes = false;
    bool saw_corrections_start = false;
    bool saw_corrections_end = false;
    while (std::getline(syndrome_file, line)) {
      if (line.find("NUM_DATA") == 0) {
        std::istringstream iss(line);
        std::string tag;
        iss >> tag >> file_numData;
      } else if (line.find("NUM_LOGICAL") == 0) {
        std::istringstream iss(line);
        std::string tag;
        iss >> tag >> file_numLogical;
      } else if (line.find("CORRECTIONS_START") == 0) {
        saw_corrections_start = true;
        while (std::getline(syndrome_file, line)) {
          if (line.find("CORRECTIONS_END") == 0) {
            saw_corrections_end = true;
            break;
          }
          uint8_t correction_bit = static_cast<uint8_t>(std::stoi(line));
          saved_corrections.push_back(correction_bit);
        }
        printf("Read %zu saved corrections\n", saved_corrections.size());
        break;
      } else if (line.find("SHOT_START") == 0) {
        saved_shots.emplace_back();
        reading_syndromes = true;
      } else if (line.find("ROUND_START") == 0) {
        // Start a new round, read greedily until the next marker. The resulting
        // per-shot round count and widths are validated against the known
        // geometry after parsing, which is what catches a truncated/corrupt
        // capture.
        if (!saved_shots.empty())
          saved_shots.back().emplace_back();
        continue;
      } else if (reading_syndromes) {
        try {
          int bit = std::stoi(line);
          saved_shots.back().back().push_back(static_cast<uint8_t>(bit));
        } catch (...) {
          break;
        }
      }
    }

    printf("Read %zu shots with syndromes\n", saved_shots.size());

    // Require a STRUCTURALLY COMPLETE capture so a truncated/corrupt file fails
    // loudly instead of silently "passing" with no verification.
    if (saved_shots.empty())
      throw std::runtime_error(
          "no shots parsed from syndrome file (empty or corrupt): " +
          syndrome_filename);
    if (!saw_corrections_start || !saw_corrections_end)
      throw std::runtime_error(
          "syndrome file is missing the CORRECTIONS_START/CORRECTIONS_END "
          "footer (truncated or incomplete capture): " +
          syndrome_filename);
    if (file_numData != numData || file_numLogical != numLogical)
      throw std::runtime_error(
          "syndrome file parameters (numData=" + std::to_string(file_numData) +
          ", numLogical=" + std::to_string(file_numLogical) +
          ") do not match this run (numData=" + std::to_string(numData) +
          ", numLogical=" + std::to_string(numLogical) + ")");
    if (saved_corrections.size() != saved_shots.size())
      throw std::runtime_error(
          "syndrome file has " + std::to_string(saved_corrections.size()) +
          " corrections for " + std::to_string(saved_shots.size()) +
          " shots; expected exactly one correction per shot (truncated "
          "capture)");
    // Each shot must be the full enqueue stream: num_rounds syndrome rounds of
    // numSyndromesPerRound bits + one final data round of numData bits.
    const std::size_t expectedRounds = static_cast<std::size_t>(numRounds) + 1;
    for (std::size_t s = 0; s < saved_shots.size(); ++s) {
      const auto &rounds = saved_shots[s];
      if (rounds.size() != expectedRounds)
        throw std::runtime_error(
            "syndrome file shot " + std::to_string(s) + " has " +
            std::to_string(rounds.size()) + " rounds; expected " +
            std::to_string(expectedRounds) +
            " (num_rounds+1); truncated or wrong-geometry capture");
      for (std::size_t r = 0; r < rounds.size(); ++r) {
        const std::size_t expectedBits = (r + 1 == expectedRounds)
                                             ? static_cast<std::size_t>(numData)
                                             : numSyndromesPerRound;
        if (rounds[r].size() != expectedBits)
          throw std::runtime_error(
              "syndrome file shot " + std::to_string(s) + " round " +
              std::to_string(r) + " has " + std::to_string(rounds[r].size()) +
              " bits; expected " + std::to_string(expectedBits) +
              " (truncated or corrupt capture)");
      }
    }

    syndrome_file.close();

    // Process saved syndromes through decoder
    printf("Feeding %zu shots of saved syndromes to decoder...\n",
           saved_shots.size());

    int corrections_matched = 0;
    int corrections_mismatched = 0;

    for (size_t shot_idx = 0; shot_idx < saved_shots.size(); shot_idx++) {
      // Reset decoder for new shot
      for (size_t logical_idx = 0; logical_idx < numLogical; logical_idx++) {
        cudaq::qec::decoding::reset_decoder(logical_idx);
      }

      // Feed syndromes to the decoder one captured round at a time, each at its
      // recorded width. This mirrors the live enqueue cadence exactly, so the
      // boundary detectors -- which XOR the final data bits against the
      // last-round Z-ancillas -- see the same data round the live path
      // enqueued.
      const auto &shot_rounds = saved_shots[shot_idx];
      for (const auto &round_bits : shot_rounds) {
        // Replay path: raw syndrome bits read from a saved file. Use the
        // test-only `enqueue_syndromes_test` API since these bits have no
        // measurement-event identity to preserve.
        std::vector<bool> syndrome_round;
        syndrome_round.reserve(round_bits.size());
        for (uint8_t b : round_bits)
          syndrome_round.push_back(static_cast<bool>(b));

        // Enqueue this round for all logical qubits
        for (size_t logical_idx = 0; logical_idx < numLogical; logical_idx++) {
          cudaq::qec::decoding::enqueue_syndromes_test(logical_idx,
                                                       syndrome_round);
        }
      }

      // Get logical corrections from decoder
      uint8_t correction_bit = 0;
      for (size_t logical_idx = 0; logical_idx < numLogical; logical_idx++) {
        auto corrections =
            cudaq::qec::decoding::get_corrections(logical_idx, 1, false);
        if (!corrections.empty() && corrections[0]) {
          correction_bit = 1;
        }
      }

      // Compare with saved correction if available
      if (shot_idx < saved_corrections.size()) {
        if (correction_bit == saved_corrections[shot_idx]) {
          corrections_matched++;
        } else {
          corrections_mismatched++;
          if (corrections_mismatched <= 10) {
            printf("  Shot %zu: mismatch! Replayed=%u, Saved=%u\n", shot_idx,
                   correction_bit, saved_corrections[shot_idx]);
          }
        }
      }
    }

    printf("Replay complete: %zu shots processed\n", saved_shots.size());
    if (!saved_corrections.empty()) {
      printf("Correction verification: %d matched, %d mismatched\n",
             corrections_matched, corrections_mismatched);
      if (corrections_mismatched != 0)
        throw std::runtime_error(
            "replay correction mismatch: " +
            std::to_string(corrections_mismatched) + " of " +
            std::to_string(corrections_matched + corrections_mismatched) +
            " shots differ from the captured run");
      printf("SUCCESS: All corrections match!\n");
    }
    return;

  } else {
    // Normal quantum simulation mode
    printf("\n=== Quantum Simulation Mode ===\n");

    // If this is a remote platform (not local sim nor emulation), don't use the
    // noise model.
    run_result =
        cudaq::get_platform().is_remote()
            ? cudaq::run(numShots, cudaq::qec::qpu::demo_circuit_qpu,
                         /*allow_device_calls=*/true, prep, numData, numAncx,
                         numAncz, numRounds, numLogical, cnot_schedX_flat,
                         cnot_schedZ_flat, p_spam_per_patch,
                         /*apply_corrections=*/true)
            : cudaq::run(numShots, noise, cudaq::qec::qpu::demo_circuit_qpu,
                         /*allow_device_calls=*/true, prep, numData, numAncx,
                         numAncz, numRounds, numLogical, cnot_schedX_flat,
                         cnot_schedZ_flat, p_spam_per_patch,
                         /*apply_corrections=*/true);
  }
  printf("Result size: %ld\n", run_result.size());
  std::vector<std::vector<uint8_t>> logical_results;
  auto obs_matrix = code.get_observables_z();
  int num_non_zero_values = 0;
  std::int64_t num_corrections = 0;
  for (int i = 0; i < run_result.size(); i++) {
    const auto &shot = run_result[i];
    const std::uint64_t corrections_mask = shot[numLogical];
    logical_results.emplace_back();
    for (int j = 0; j < numLogical; j++) {
      if ((corrections_mask >> j) & 1)
        num_corrections++;
      std::vector<double> result_vec(numData);
      for (int l = 0; l < numData; l++) {
        result_vec[l] = (shot[j] & (1ull << l)) ? 1.0 : 0.0;
      }
      cudaqx::tensor<uint8_t> result_tensor;
      cudaq::qec::convert_vec_soft_to_tensor_hard(result_vec, result_tensor);
      // Calculate the logical observable for each logical qubit
      uint8_t logical_result = (obs_matrix.dot(result_tensor) % 2).at({0});
      logical_results.back().push_back(logical_result);
      if (logical_result != 0)
        num_non_zero_values++;
    }
  }
  printf("Number of non-zero values measured : %d\n", num_non_zero_values);
  printf("Number of corrections decoder found: %ld\n", num_corrections);

  // Save corrections to file if syndrome capture was enabled
  if (save_syndrome && g_syndrome_output_file.is_open()) {
    // Disable callback to stop capturing
    cudaq::qec::decoding::host::_set_syndrome_capture_callback(nullptr);

    // Save logical corrections for each shot (for verification during replay)
    g_syndrome_output_file << "CORRECTIONS_START\n";
    for (size_t i = 0; i < logical_results.size(); i++) {
      uint8_t any_correction = run_result[i][numLogical] != 0 ? 1 : 0;
      g_syndrome_output_file << static_cast<int>(any_correction) << "\n";
    }
    g_syndrome_output_file << "CORRECTIONS_END\n";
    g_syndrome_output_file.close();
    printf("Syndrome data saved to: %s\n", syndrome_filename.c_str());
  }
}

void show_help() {
  printf("Usage: surface_code_ising_realtime_decoding [options]\n");
  printf("Options:\n");
  printf("  --distance <int>    Distance of the surface code. Default: 7\n");
  printf("  --num_shots <int>   Number of shots. Default: 10\n");
  printf(
      "  --p_spam <double>   SPAM probability. Range[0, 1]. Default: 0.01\n");
  printf("  --num_rounds <int>  Number of measurement rounds. Default: "
         "distance\n");
  printf("  --seed <int>        Simulator seed; negative = unseeded. "
         "Default: 42\n");
  printf("  --decoder_type <string> Decoder for the generated config (with "
         "--save_dem): 'trt_decoder' (TensorRT Ising predecoder + PyMatching "
         "global decoder; the default) or 'pymatching'.\n");
  printf("  --ising-artifacts-dir <dir> Prepared Ising artifact directory "
         "containing model.onnx, H_csr.bin, O_csr.bin, priors.bin, "
         "metadata.txt, and D_sparse.txt. Required with --decoder_type "
         "trt_decoder and --save_dem; QEC_ISING_ARTIFACTS_DIR is honored "
         "when the flag is absent. Nothing in this directory ships with "
         "CUDA-Q QEC -- see the example docs for the preparation recipe.\n");
  printf("  --save_dem <string> Generate the DEM + decoder config and save to "
         "a YAML file (generation phase).\n");
  printf(
      "  --yaml <string>     Run realtime decoding from a YAML config; the "
      "decoder is read from the file. Do not combine with --decoder_type.\n");
  printf("  --load_dem <string> Alias of --yaml.\n");
  printf("  --save_syndrome <string> Save syndrome data to a file for later "
         "replay (and FPGA playback).\n");
  printf("  --load_syndrome <string> Load and replay syndrome data from a "
         "file.\n");
  printf("  --help              Show this help message\n");
}

int main(int argc, char **argv) {
#ifdef QEC_APP_CQR
  realtime_channel_guard realtime_channel;
#endif
  int num_shots = 10;
  int distance = 7;
  double p_spam = 0.01;
  int num_rounds = -1; // Will be set to distance if not specified
  int seed = 42;       // negative = unseeded
  bool save_dem = false;
  bool load_dem = false;
  std::string dem_filename;

  // Decoder type selection. This is a GENERATION-phase knob (used with
  // --save_dem); with --yaml the decoder is read from the file.
  std::string decoder_type = "trt_decoder"; // Default
  bool decoder_type_explicit = false;
  bool yaml_mode = false;
  // The Ising artifact directory (six files; see the header comment). Flag
  // overrides the QEC_ISING_ARTIFACTS_DIR environment variable.
  std::string ising_artifacts_dir = env_or("QEC_ISING_ARTIFACTS_DIR", "");

  // Syndrome save/load options
  bool save_syndrome = false;
  bool load_syndrome = false;
  std::string syndrome_filename;

  // This example is single-patch by design (one logical qubit, one decoder):
  // the Ising artifacts describe one code patch.
  const int num_logical = 1;

  // Parse the command line arguments. Value-taking flags read the next argv
  // entry through require_value, which errors out (rather than reading past the
  // end of argv) when a flag is given with no following value; numeric flags go
  // through require_int/require_double, which reject malformed values.
  int i;
  auto require_value = [&](const char *flag) -> std::string {
    if (i + 1 >= argc) {
      printf("Error: %s requires a value.\n", flag);
      std::exit(1);
    }
    return argv[++i];
  };
  auto require_int = [&](const char *flag) -> int {
    const std::string v = require_value(flag);
    try {
      std::size_t pos = 0;
      int r = std::stoi(v, &pos);
      if (pos != v.size())
        throw std::invalid_argument(v);
      return r;
    } catch (const std::exception &) {
      printf("Error: %s expects an integer, got '%s'.\n", flag, v.c_str());
      std::exit(1);
    }
  };
  auto require_double = [&](const char *flag) -> double {
    const std::string v = require_value(flag);
    try {
      std::size_t pos = 0;
      double r = std::stod(v, &pos);
      if (pos != v.size())
        throw std::invalid_argument(v);
      return r;
    } catch (const std::exception &) {
      printf("Error: %s expects a number, got '%s'.\n", flag, v.c_str());
      std::exit(1);
    }
  };
  for (i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--distance") {
      distance = require_int("--distance");
    } else if (arg == "--num_shots") {
      num_shots = require_int("--num_shots");
    } else if (arg == "--p_spam") {
      p_spam = require_double("--p_spam");
    } else if (arg == "--help" || arg == "-h") {
      show_help();
      return 0;
    } else if (arg == "--num_rounds") {
      num_rounds = require_int("--num_rounds");
    } else if (arg == "--seed") {
      seed = require_int("--seed");
    } else if (arg == "--decoder_type") {
      decoder_type = require_value("--decoder_type");
      decoder_type_explicit = true;
    } else if (arg == "--ising-artifacts-dir" ||
               arg == "--ising_artifacts_dir") {
      ising_artifacts_dir = require_value("--ising-artifacts-dir");
    } else if (arg == "--save_dem") {
      save_dem = true;
      dem_filename = require_value("--save_dem");
    } else if (arg == "--yaml" || arg == "--load_dem") {
      // Realtime phase: the decoder is read from the YAML (authoritative).
      load_dem = true;
      yaml_mode = true;
      dem_filename = require_value("--yaml");
    } else if (arg == "--save_syndrome") {
      save_syndrome = true;
      syndrome_filename = require_value("--save_syndrome");
    } else if (arg == "--load_syndrome") {
      load_syndrome = true;
      syndrome_filename = require_value("--load_syndrome");
    } else {
      printf("Unknown argument: %s\n", arg.c_str());
      show_help();
      return 1;
    }
  }

  if (!load_dem && !save_dem && !load_syndrome) {
    printf("Neither --save_dem nor --load_dem nor --load_syndrome was "
           "specified. This is not a valid use case for this program.\n");
    show_help();
    return 1;
  }

  // Validate syndrome save/load options
  if (save_syndrome && load_syndrome) {
    printf("Error: Cannot use both --save_syndrome and --load_syndrome "
           "together\n");
    return 1;
  }
  if (save_syndrome && save_dem) {
    printf("Error: Cannot use --save_syndrome with --save_dem\n");
    printf("       --save_dem returns early without running simulation.\n");
    return 1;
  }

  // Reject geometries the rotated surface-code kernel cannot build: the
  // distance must be an odd integer >= 3, and each patch's final data bits
  // are returned in one 64-bit word, so numData = d^2 must fit (d <= 7).
  if (distance < 3 || distance % 2 == 0) {
    printf("Error: distance must be an odd integer >= 3 (got %d).\n", distance);
    return 1;
  }
  const std::int64_t num_data_bits =
      static_cast<std::int64_t>(distance) * distance;
  if (num_data_bits >= 64) {
    printf("Error: distance %d has %lld data qubits; only numData < 64 "
           "(d <= 7) is supported.\n",
           distance, static_cast<long long>(num_data_bits));
    return 1;
  }
  if (num_shots < 1) {
    printf("Error: --num_shots must be >= 1 (got %d).\n", num_shots);
    return 1;
  }
  if (!std::isfinite(p_spam) || p_spam < 0.0 || p_spam > 1.0) {
    printf("Error: --p_spam must be a finite value in [0, 1] (got %g).\n",
           p_spam);
    return 1;
  }
  const std::vector<double> p_spam_per_patch(num_logical, p_spam);

  // Set defaults if not specified
  if (num_rounds == -1)
    num_rounds = distance;
  // --yaml is authoritative for the decoder; a co-passed --decoder_type would
  // be ambiguous, so reject the combination.
  if (yaml_mode && decoder_type_explicit) {
    printf(
        "Error: --decoder_type only applies to --save_dem (generation). "
        "With --yaml the decoder is read from the file; do not pass both.\n");
    return 1;
  }
  // --save_dem (generation) and --yaml (realtime) are separate phases that
  // write and read the same config file.
  if (yaml_mode && save_dem) {
    printf("Error: --save_dem (generation) and --yaml (realtime) are separate "
           "phases; do not pass both.\n");
    return 1;
  }

  if (decoder_type != "trt_decoder" && decoder_type != "pymatching") {
    printf("Error: --decoder_type must be 'trt_decoder' or 'pymatching' "
           "(got '%s')\n",
           decoder_type.c_str());
    return 1;
  }
  const std::vector<std::string> decoder_types(num_logical, decoder_type);

  if (num_rounds < 2) {
    printf("Error: num_rounds (%d) must be >= 2 (a memory experiment needs at "
           "least one cross-round detector).\n",
           num_rounds);
    return 1;
  }
  if (num_rounds < distance)
    printf("Warning: num_rounds (%d) < distance (%d): decodable but not "
           "fault-tolerant (fewer rounds than the code distance).\n",
           num_rounds, distance);

  // Syndrome replay feeds saved syndromes through a configured decoder, which
  // only exists under --yaml.
  if (load_syndrome && !yaml_mode) {
    printf("Error: --load_syndrome requires --yaml (the decoder to replay "
           "through is configured from the loaded config).\n");
    return 1;
  }

  // The trt+Ising generation path needs the prepared artifact directory. Fail
  // up front with the complete gap list rather than deep inside DEM
  // generation. (The run phase reads everything from the YAML; the ONNX path
  // embedded there is validated by the serving side.)
  if (save_dem && decoder_type == "trt_decoder") {
    if (ising_artifacts_dir.empty()) {
      printf("Error: --decoder_type trt_decoder requires an Ising artifact "
             "directory: pass --ising-artifacts-dir <dir> or set "
             "QEC_ISING_ARTIFACTS_DIR.\n"
             "The six artifact files are prepared locally (gated Hugging Face "
             "weights + the NVIDIA/Ising-Decoding repository); see the "
             "realtime_decoding_demo documentation for the recipe.\n");
      return 1;
    }
    try {
      require_ising_artifact_files(ising_artifacts_dir);
      enforce_ising_metadata(ising_artifacts_dir, distance,
                             static_cast<std::size_t>(num_rounds));
    } catch (const std::exception &e) {
      printf("Error: %s\n", e.what());
      return 1;
    }
  }

  if (seed >= 0) {
    cudaq::set_random_seed(static_cast<std::size_t>(seed));
    printf("Set RNG seed to %d for deterministic results\n", seed);
  }

  printf("Running with p_spam = %g, distance = %d, num_shots = %d, "
         "num_rounds = %d\n",
         p_spam, distance, num_shots, num_rounds);

  // Build the code at code_rotation XV (the predecoder's training orientation),
  // which sets the geometry and observable basis.
  try {
#ifdef QEC_APP_CQR
    if (load_dem)
      realtime_channel.initialize(argv[0]);
#endif

    auto code = cudaq::qec::get_code(
        "surface_code",
        cudaqx::heterogeneous_map{{"distance", distance},
                                  {"orientation", std::string("XV")}});

    demo_circuit_host(*code, distance, p_spam_per_patch,
                      cudaq::qec::operation::prep0, num_shots, num_rounds,
                      num_logical, dem_filename, save_dem, load_dem,
                      decoder_types, save_syndrome, load_syndrome,
                      syndrome_filename, ising_artifacts_dir);
  } catch (const std::exception &e) {
    // Configuration, channel, and geometry failures surface as a clean error
    // rather than an uncaught-exception abort.
    printf("Error: %s\n", e.what());
    cudaq::qec::decoding::config::finalize_decoders();
    return 1;
  }

#ifdef QEC_APP_CQR
  if (load_dem) {
    // Proof the shots' device_calls crossed the wire to the EXTERNAL server:
    // this in-process counter stays 0 unless the in-process host-dispatch
    // service (not the external decoding_server) served the calls.
    printf("CQR service dispatch count: %llu\n",
           static_cast<unsigned long long>(
               cudaqx_qec_device_call_dispatch_count()));
  }
#endif

  // Ensure clean shutdown
  cudaq::qec::decoding::config::finalize_decoders();

  return 0;
}

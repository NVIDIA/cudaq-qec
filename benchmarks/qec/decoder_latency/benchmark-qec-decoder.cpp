/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// General-purpose latency and throughput benchmark for any registered
// CUDA-QX QEC decoder plugin.  Generates rotated surface-code memory circuits
// and sweeps over distances, round counts, and noise rates, running both the
// batch decode() and streaming enqueue_syndrome() paths for each named decoder.
// Results are printed in a single flat table that includes every sweep
// dimension and a per-decoder integer ID.
//
// A sweep point can also be run on several concurrent decoder instances
// (--decoder_instances): each instance is an independent decoder driven by its
// own thread, which is how a real deployment decodes several logical qubits at
// once.  Latency percentiles are pooled over all instances of a point, so they
// describe the machine while every instance is loaded; rounds/s sums the rate
// each instance held over its own window.  --shots counts timed shots *per
// instance*, so --decoder_instances 1 reproduces the single-decoder numbers
// exactly and higher counts add work rather than dividing it.
//
// Run with --help for the available knobs.
//
// Examples, where <decoder> is any name the plugin registry resolves:
//   benchmark-qec-decoder --decoders <decoder>,<decoder>
//   benchmark-qec-decoder \
//     --decoders <decoder>,<decoder> \
//     --distances 3,5,7,9 --rounds 5,10 --noises 0.001,0.005 \
//     --param num_threads=4
//   benchmark-qec-decoder --decoders <decoder> --decoder_instances 1,2,4,8
//   benchmark-qec-decoder --decoders <decoder> \
//     --decoder_instances 1,2,4,8 --pin_instances
//   benchmark-qec-decoder --decoders <decoder> --param num_threads=4 \
//     --decoder_instances 4 --instance_core_base 8 --instance_core_width 4
//
// Values for --param are auto-typed: "true"/"false" -> bool,
// digits -> uint64, decimal -> double, otherwise -> string.

#include "benchmark_instance_pool.h"
#include "stim.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using index_type = cudaq::qec::sparse_binary_matrix::index_type;
using cudaq::qec::benchmark::allowed_cpus;
using cudaq::qec::benchmark::concat_instance_samples;
using cudaq::qec::benchmark::core_layout;
using cudaq::qec::benchmark::divide_cpus;
using cudaq::qec::benchmark::effective_stride;
using cudaq::qec::benchmark::first_instance_error;
using cudaq::qec::benchmark::instance_pool_result;
using cudaq::qec::benchmark::instance_result;
using cudaq::qec::benchmark::instance_sample_kind;
using cudaq::qec::benchmark::instance_shot_offset;
using cudaq::qec::benchmark::instance_work;
using cudaq::qec::benchmark::resolve_instance_cpus;
using cudaq::qec::benchmark::run_instances;
using cudaq::qec::benchmark::sum_instance_results;
using cudaq::qec::benchmark::worker_threads_per_instance;

// ── Options
// ───────────────────────────────────────────────────────────────────

struct options {
  std::vector<std::size_t> distances = {5};
  // Empty rounds_list means "use distance" for each configuration.
  std::vector<std::size_t> rounds_list;
  std::vector<double> noises;
  // Timed shots per decoder instance, so the instance count scales total work
  // rather than splitting a fixed pool.
  std::size_t shots = 500;
  std::size_t warmup = 20;
  std::vector<std::string> decoders;
  // Concurrent decoder instances per sweep point. Empty means one instance.
  std::vector<std::size_t> instances_list;
  // CPU placement for the instances. A negative base disables pinning,
  // matching the core_pinning convention in realtime/pipeline.h.
  core_layout cores;
  // Divide the available CPUs among each row's instances instead of taking a
  // hand-written base/width/stride.
  bool auto_pin = false;
  // 0 = unset: the key is withheld so the decoder picks its own schedule.
  // Otherwise forwarded as uint64 to decoders whose schema lists
  // block_leaf_size.
  std::size_t block_leaf_size = 0;
  std::chrono::nanoseconds round_interval{0};
  bool run_batch = true;
  bool run_stream = true;
  bool emit_csv = false;
  std::vector<std::pair<std::string, std::string>> extra_params;
};

// CPU placement for a row running @p instances instances.
//
// Automatic placement depends on the instance count, so it cannot be resolved
// once at startup: a --decoder_instances 1,2,4 sweep divides the machine three
// different ways.
core_layout layout_for(const options &opts, std::size_t instances) {
  if (!opts.auto_pin)
    return opts.cores;
  return divide_cpus(instances, allowed_cpus());
}

void print_usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --decoders NAME[,...]      plugins to compare (default: "
         "pymatching)\n"
      << "  --distance N               single distance (default 5)\n"
      << "  --distances N[,N,...]      distances to sweep\n"
      << "  --rounds N[,N,...]         round counts to sweep "
         "(default=distance)\n"
      << "  --noise P                  single noise value (default 0.001)\n"
      << "  --noises P[,P,...]         noise values to sweep\n"
      << "  --shots N                  timed shots per instance (default "
         "500)\n"
      << "  --warmup N                 untimed warmup shots (default 20)\n"
      << "  --decoder_instances N[,...]\n"
      << "                             concurrent whole decoders per point,\n"
      << "                             one thread each (default 1)\n"
      << "  --pin_instances            pin instances, dividing the available\n"
      << "                             CPUs evenly among each row's\n"
      << "                             instances (default: unpinned)\n"
      << "  --instance_core_base N     place instances by hand: instance i on\n"
      << "                             CPU N+i*stride\n"
      << "  --instance_core_width W    CPUs per instance (default 1). Its\n"
      << "                             decoder's own threads inherit the\n"
      << "                             mask, so match a threaded decoder\n"
      << "  --instance_core_stride S   CPUs between instances (default: same\n"
      << "                             as width). Use 2 to skip SMT siblings\n"
      << "  --block_leaf_size N        brickwall leaf height; omit to let the\n"
      << "                             decoder choose its own schedule\n"
      << "  --round_interval_us T      pace streaming rounds by T "
         "microseconds\n"
      << "  --mode M                   batch, stream, or both (default both)\n"
      << "  --param KEY=VALUE          extra param forwarded to all decoders\n"
      << "                             (repeatable; auto-typed)\n"
      << "  --csv                      emit machine-readable CSV rows\n"
      << "  --help                     show this message\n";
}

// ── Argument parsing
// ──────────────────────────────────────────────────────────

std::vector<std::string> split_csv_str(const std::string &text) {
  std::vector<std::string> out;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const auto comma = text.find(',', pos);
    const auto piece = text.substr(pos, comma - pos);
    if (!piece.empty())
      out.push_back(piece);
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

std::vector<std::size_t> parse_size_list(const std::string &text) {
  std::vector<std::size_t> out;
  for (const auto &s : split_csv_str(text))
    out.push_back(static_cast<std::size_t>(std::stoull(s)));
  return out;
}

std::vector<double> parse_double_list(const std::string &text) {
  std::vector<double> out;
  for (const auto &s : split_csv_str(text))
    out.push_back(std::stod(s));
  return out;
}

bool parse_args(int argc, char **argv, options &opts, int &exit_code) {
  // No value of this CLI starts with "--", so a token that does is the next
  // option rather than this one's value. Catching that here turns
  // "--shots --csv" into an error instead of a std::stoull throw, and stops a
  // misspelled valueless flag from swallowing the option after it.
  auto need_value = [&](int i) -> bool {
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
      return true;
    std::cerr << "error: " << argv[i]
              << " requires a value (run --help to list the options)\n";
    exit_code = 1;
    return false;
  };

  for (int i = 1; i < argc; ++i) {
    // Accept either spelling of an option name: every flag here is written
    // with underscores, but the binary itself is hyphenated, which makes
    // --pin-instances an easy slip for --pin_instances.
    std::string arg = argv[i];
    if (arg.rfind("--", 0) == 0)
      std::replace(arg.begin() + 2, arg.end(), '-', '_');

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      exit_code = 0;
      return false;
    }
    if (arg == "--csv") {
      opts.emit_csv = true;
      continue;
    }
    if (arg == "--pin_instances") {
      opts.auto_pin = true;
      continue;
    }
    if (!need_value(i))
      return false;
    const char *const typed = argv[i];
    const std::string val = argv[++i];

    if (arg == "--decoders")
      opts.decoders = split_csv_str(val);
    else if (arg == "--distance") {
      opts.distances.clear();
      opts.distances.push_back(std::stoull(val));
    } else if (arg == "--distances")
      opts.distances = parse_size_list(val);
    else if (arg == "--rounds")
      opts.rounds_list = parse_size_list(val);
    else if (arg == "--noise") {
      opts.noises.clear();
      opts.noises.push_back(std::stod(val));
    } else if (arg == "--noises")
      opts.noises = parse_double_list(val);
    else if (arg == "--shots")
      opts.shots = std::stoull(val);
    else if (arg == "--warmup")
      opts.warmup = std::stoull(val);
    else if (arg == "--decoder_instances")
      opts.instances_list = parse_size_list(val);
    else if (arg == "--instance_core_base")
      opts.cores.base = std::stoi(val);
    else if (arg == "--instance_core_width")
      opts.cores.width = std::stoull(val);
    else if (arg == "--instance_core_stride")
      opts.cores.stride = std::stoull(val);
    else if (arg == "--block_leaf_size")
      opts.block_leaf_size = std::stoull(val);
    else if (arg == "--round_interval_us")
      opts.round_interval = std::chrono::nanoseconds(
          static_cast<int64_t>(std::llround(std::stod(val) * 1e3)));
    else if (arg == "--mode") {
      opts.run_batch = val == "batch" || val == "both";
      opts.run_stream = val == "stream" || val == "both";
      if (!opts.run_batch && !opts.run_stream) {
        std::cerr << "error: --mode must be batch, stream, or both\n";
        exit_code = 1;
        return false;
      }
    } else if (arg == "--param") {
      const auto eq = val.find('=');
      if (eq == std::string::npos) {
        std::cerr << "error: --param requires KEY=VALUE format\n";
        exit_code = 1;
        return false;
      }
      opts.extra_params.push_back({val.substr(0, eq), val.substr(eq + 1)});
    } else {
      std::cerr << "error: unknown option " << typed << "\n";
      print_usage(argv[0]);
      exit_code = 1;
      return false;
    }
  } // end - for(i)

  if (opts.decoders.empty())
    opts.decoders.push_back("pymatching");
  if (opts.noises.empty())
    opts.noises.push_back(0.001);

  if (opts.distances.empty()) {
    std::cerr << "error: --distances needs at least one value\n";
    exit_code = 1;
    return false;
  }
  for (auto d : opts.distances) {
    if (d < 3 || d % 2 == 0) {
      std::cerr << "error: all distances must be odd and >= 3\n";
      exit_code = 1;
      return false;
    }
  }
  if (opts.decoders.empty()) {
    std::cerr << "error: --decoders needs at least one name\n";
    exit_code = 1;
    return false;
  }
  if (opts.shots == 0) {
    std::cerr << "error: --shots must be positive\n";
    exit_code = 1;
    return false;
  }
  if (opts.instances_list.empty())
    opts.instances_list.push_back(1);
  for (auto n : opts.instances_list) {
    if (n == 0) {
      std::cerr << "error: --decoder_instances values must be positive\n";
      exit_code = 1;
      return false;
    }
  }
  // --pin_instances computes the placement, so a hand-written one alongside it
  // would be silently ignored.
  if (opts.auto_pin &&
      (opts.cores.base >= 0 || opts.cores.width > 1 || opts.cores.stride > 0)) {
    std::cerr << "error: --pin_instances places instances automatically; drop "
                 "it to use --instance_core_base/_width/_stride\n";
    exit_code = 1;
    return false;
  }
  // Width and stride only mean something next to a base; accepting them alone
  // would silently discard the placement the user asked for.
  if (opts.cores.base < 0 && (opts.cores.width > 1 || opts.cores.stride > 0)) {
    std::cerr << "error: --instance_core_width and --instance_core_stride "
                 "need --instance_core_base\n";
    exit_code = 1;
    return false;
  }
  // Reject an unhonorable pin now: the widest instance count decides how many
  // CPUs the run needs, and finding out mid-sweep would throw away every point
  // measured so far.
  try {
    for (const std::size_t n : opts.instances_list)
      (void)resolve_instance_cpus(n - 1, layout_for(opts, n),
                                  std::thread::hardware_concurrency());
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    exit_code = 1;
    return false;
  }
  if (opts.noises.empty()) {
    std::cerr << "error: --noises needs at least one value\n";
    exit_code = 1;
    return false;
  }
  for (auto n : opts.noises) {
    if (n <= 0.0 || n >= 1.0) {
      std::cerr << "error: noise values must be in (0, 1)\n";
      exit_code = 1;
      return false;
    }
  }
  return true;
} // end - parse_args()

// ── Circuit / DEM helpers
// ─────────────────────────────────────────────────────

struct benchmark_data {
  stim::Circuit circuit;
  stim::DetectorErrorModel dem;
  std::vector<std::vector<cudaq::qec::float_t>> soft_syndromes;
  std::vector<std::vector<uint8_t>> hard_syndromes;
  std::vector<uint64_t> observable_masks;
};

benchmark_data generate_data(const options &opts, std::size_t distance,
                             std::size_t rounds, double noise) {
  stim::CircuitGenParameters gen(rounds, distance, "rotated_memory_z");
  gen.after_clifford_depolarization = noise;
  gen.after_reset_flip_probability = noise;
  gen.before_measure_flip_probability = noise;
  gen.before_round_data_depolarization = noise;

  benchmark_data data;
  data.circuit = stim::generate_surface_code_circuit(gen).circuit;

  const std::size_t total = opts.warmup + opts.shots;
  std::mt19937_64 rng(0);
  auto sampled = stim::sample_batch_detection_events<stim::MAX_BITWORD_WIDTH>(
      data.circuit, total, rng);
  const auto &dets = sampled.first;
  const auto &obs = sampled.second;

  const std::size_t nd = data.circuit.count_detectors();
  const std::size_t no = data.circuit.count_observables();
  data.soft_syndromes.resize(total);
  data.hard_syndromes.resize(total);
  data.observable_masks.assign(total, 0);
  for (std::size_t s = 0; s < total; ++s) {
    data.soft_syndromes[s].assign(nd, 0.0);
    data.hard_syndromes[s].assign(nd, 0);
    for (std::size_t d = 0; d < nd; ++d)
      if (dets[d][s]) {
        data.soft_syndromes[s][d] = 1.0;
        data.hard_syndromes[s][d] = 1;
      }
    for (std::size_t o = 0; o < no && o < 64; ++o)
      if (obs[o][s])
        data.observable_masks[s] ^= uint64_t{1} << o;
  }

  data.dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      data.circuit, /*decompose_errors=*/true, /*fold_loops=*/true,
      /*allow_gauge_detectors=*/false,
      /*approximate_disjoint_errors_threshold=*/0,
      /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
  return data;
} // end - generate_data()

struct dem_matrices {
  cudaq::qec::sparse_binary_matrix H;
  cudaqx::tensor<uint8_t> O;
  std::vector<double> priors;
};

// Extract H, O, and priors from the decomposed DEM without depending on
// PyMatching internals.  Stim's decompose_errors=true splits hyperedges
// into graphlike components separated by ^ (is_separator()) targets; each
// component becomes one column of H with 1 or 2 non-zero rows.
dem_matrices make_matrices(const stim::DetectorErrorModel &dem) {
  struct edge {
    double p = 0.0;
    std::vector<std::size_t> detectors;
    std::vector<std::size_t> observables;
  };

  std::vector<edge> edges;
  const stim::DetectorErrorModel flat = dem.flattened();
  for (const stim::DemInstruction &ins : flat.instructions) {
    if (ins.type != stim::DemInstructionType::DEM_ERROR)
      continue;
    const double p = ins.arg_data[0];
    edge cur;
    cur.p = p;
    for (const stim::DemTarget &t : ins.target_data) {
      if (t.is_separator()) {
        edges.push_back(cur);
        cur = edge{};
        cur.p = p;
      } else if (t.is_relative_detector_id()) {
        cur.detectors.push_back(static_cast<std::size_t>(t.val()));
      } else if (t.is_observable_id()) {
        cur.observables.push_back(static_cast<std::size_t>(t.val()));
      }
    }
    edges.push_back(cur);
  }

  const std::size_t nd = dem.count_detectors();
  const std::size_t no = dem.count_observables();
  const std::size_t ne = edges.size();

  dem_matrices m;
  m.priors.assign(ne, 0.0);
  m.O = cudaqx::tensor<uint8_t>({no, ne});
  std::vector<std::vector<index_type>> H_nested(ne);
  for (std::size_t col = 0; col < ne; ++col) {
    m.priors[col] = edges[col].p;
    for (auto det : edges[col].detectors)
      H_nested[col].push_back(static_cast<index_type>(det));
    for (auto o : edges[col].observables)
      m.O.at({o, col}) = 1;
  }
  m.H = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      static_cast<index_type>(nd), static_cast<index_type>(ne), H_nested);
  return m;
} // end - make_matrices()

std::vector<int32_t> detector_round_map(const stim::Circuit &circuit) {
  std::set<uint64_t> all;
  for (uint64_t d = 0; d < circuit.count_detectors(); ++d)
    all.insert(d);
  std::vector<int32_t> dr(circuit.count_detectors(), 0);
  for (const auto &e : circuit.get_detector_coordinates(all)) {
    if (!e.second.empty())
      dr[e.first] = static_cast<int32_t>(std::llround(e.second.back()));
  }
  return dr;
}

std::vector<std::size_t> detectors_per_round(const std::vector<int32_t> &dr) {
  const auto max_r = *std::max_element(dr.begin(), dr.end());
  std::vector<std::size_t> widths(static_cast<std::size_t>(max_r) + 1, 0);
  for (auto r : dr)
    ++widths[static_cast<std::size_t>(r)];
  return widths;
}

cudaq::qec::sparse_binary_matrix identity_measurement_map(std::size_t nd) {
  std::vector<std::vector<uint32_t>> d_sparse(nd);
  for (std::size_t i = 0; i < nd; ++i)
    d_sparse[i].push_back(static_cast<uint32_t>(i));
  return cudaq::qec::sparse_binary_matrix::from_nested_csr(nd, nd, d_sparse);
}

// ── Parameter helpers
// ─────────────────────────────────────────────────────────

void insert_typed(cudaqx::heterogeneous_map &m, const std::string &key,
                  const std::string &val) {
  if (val == "true") {
    m.insert(key, true);
    return;
  }
  if (val == "false") {
    m.insert(key, false);
    return;
  }
  try {
    std::size_t pos = 0;
    const uint64_t u = std::stoull(val, &pos);
    if (pos == val.size()) {
      m.insert(key, u);
      return;
    }
  } catch (...) {
  }
  try {
    std::size_t pos = 0;
    const double d = std::stod(val, &pos);
    if (pos == val.size()) {
      m.insert(key, d);
      return;
    }
  } catch (...) {
  }
  m.insert(key, val);
}

// True for a --param value that means numeric zero, the sentinel decoders use
// for "size this yourself".
bool spells_zero(const std::string &val) {
  return !val.empty() && val.find_first_not_of('0') == std::string::npos;
}

// Whether the user asked a decoder to size its thread pool from the machine.
bool wants_machine_sized_threads(const options &opts) {
  for (const auto &kv : opts.extra_params)
    if (kv.first == "num_threads" && spells_zero(kv.second))
      return true;
  return false;
}

// Thread count to substitute for a num_threads=0 request on a row running
// @p instances instances, or 0 to leave the request untouched.
std::size_t auto_thread_share(const options &opts, std::size_t instances) {
  const core_layout layout = layout_for(opts, instances);

  // A single unpinned instance genuinely does have the machine to itself, so
  // defer to the decoder's own sentinel handling and keep its numbers intact.
  if (instances <= 1 && layout.base < 0)
    return 0;
  // Short of the instance's whole share: the substituted pool has to leave the
  // instance thread a CPU, the way a decoder's own sentinel path does.
  return worker_threads_per_instance(instances, layout,
                                     std::thread::hardware_concurrency());
}

// Build a params map tailored to one decoder, in tiers:
//
//   detector_round -- vector<int32_t> has no corresponding param_kind so it
//                 cannot appear in any schema.  nv-fusion-decoder reads it
//                 directly, while decoders with strict schema validation
//                 reject unknown keys. Passed only to nv-fusion-decoder by
//                 name; add other decoder names here if a future decoder also
//                 needs temporal round information.
//
//   num_threads -- a request of 0 ("size yourself from the machine") is
//                 rewritten to the caller's auto_threads, since a decoder
//                 resolves that sentinel with no idea how many other
//                 instances are running. 0 for auto_threads leaves it alone.
//
//   Schema-gated -- decoder-specific knobs and user --param values are
//                 included only when the decoder's registered schema
//                 explicitly lists them.  This silently skips a knob for every
//                 decoder that does not declare it, without warnings or
//                 validation failures.
cudaqx::heterogeneous_map build_decoder_params(
    const std::string &name, const std::vector<int32_t> &dr,
    std::size_t block_leaf,
    const std::vector<std::pair<std::string, std::string>> &extra,
    std::size_t auto_threads) {
  const auto *schema = cudaq::qec::decoding::config::find_decoder_schema(name);

  auto in_schema = [&](const std::string &key) -> bool {
    if (!schema)
      return true; // no schema: accept all keys
    for (const auto &ps : schema->params)
      if (ps.key == key)
        return true;
    return false;
  };

  cudaqx::heterogeneous_map p;

  // ── detector_round (name-gated) ────────────────────────────────────────────
  if (name == "nv-fusion-decoder")
    p.insert("detector_round", dr);

  // ── Schema-gated ──────────────────────────────────────────────────────────
  // Zero means no leaf height was requested: omit the key entirely so the
  // decoder applies its own automatic schedule.
  if (block_leaf != 0 && in_schema("block_leaf_size"))
    p.insert("block_leaf_size", static_cast<std::size_t>(block_leaf));

  // User --param values: use the schema's param_kind to pick the correct
  // storage type.  This matters because on Linux x86-64 uint64_t
  // (unsigned long long) and std::size_t (unsigned long) are distinct types
  // in std::any, so a uint64 schema key stored as uint64_t would fail the
  // any_cast<std::size_t> inside the decoder and silently fall back to its
  // default value.  Decoders with no schema fall back to insert_typed.
  for (const auto &kv : extra) {
    const std::string &key = kv.first;
    // num_threads=0 asks the decoder to size itself from the machine, which it
    // does without knowing another instance exists: every instance of a row
    // would build a whole-machine pool.  Substituting this row's share keeps
    // the total near one thread per CPU.  Any explicit count is passed through
    // untouched, which is also the way to opt out.
    const std::string val =
        (auto_threads > 0 && key == "num_threads" && spells_zero(kv.second))
            ? std::to_string(auto_threads)
            : kv.second;
    if (!in_schema(key))
      continue;

    bool inserted = false;
    if (schema) {
      for (const auto &ps : schema->params) {
        if (ps.key != key)
          continue;
        namespace cfg = cudaq::qec::decoding::config;
        switch (ps.kind) {
        case cfg::param_kind::boolean:
          p.insert(key, val == "true");
          break;
        case cfg::param_kind::int32:
          p.insert(key, static_cast<int>(std::stoi(val)));
          break;
        case cfg::param_kind::uint64:
          p.insert(key, static_cast<std::size_t>(std::stoull(val)));
          break;
        case cfg::param_kind::f64:
          p.insert(key, std::stod(val));
          break;
        case cfg::param_kind::string:
          p.insert(key, val);
          break;
        default:
          insert_typed(p, key, val);
          break;
        }
        inserted = true;
        break;
      }
    }
    if (!inserted)
      insert_typed(p, key, val);
  }

  return p;
} // end - build_decoder_params()

// ── Timing
// ────────────────────────────────────────────────────────────────────

struct latency_stats {
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
};

double percentile(const std::vector<double> &sorted, double p) {
  if (sorted.empty())
    return 0.0;
  const double pos = (p / 100.0) * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(pos);
  const auto hi = std::min(lo + 1, sorted.size() - 1);
  return sorted[lo] * (1.0 - (pos - static_cast<double>(lo))) +
         sorted[hi] * (pos - static_cast<double>(lo));
}

latency_stats summarize(std::vector<double> v) {
  latency_stats s;
  if (v.empty())
    return s;
  std::sort(v.begin(), v.end());
  double sum = 0.0;
  for (auto x : v)
    sum += x;
  s.mean = sum / static_cast<double>(v.size());
  s.p50 = percentile(v, 50.0);
  s.p90 = percentile(v, 90.0);
  s.p99 = percentile(v, 99.0);
  s.min = v.front();
  s.max = v.back();
  return s;
}

double elapsed_us(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

// One point of the sweep: everything that identifies a row of the report.
struct sweep_point {
  int decoder_id = 0;
  std::string name;
  std::size_t distance = 0;
  std::size_t rounds = 0;
  double noise = 0.0;
  std::size_t instances = 1;
};

// Which decode path a point is measured on.
enum class bench_mode {
  batch,  ///< decode() on a whole syndrome at once
  stream, ///< enqueue_syndrome() round by round
};

struct measurement {
  int decoder_id = 0;
  std::string name;
  std::size_t distance = 0;
  std::size_t rounds = 0;
  double noise = 0.0;
  std::size_t instances = 1;
  // Timed shots per instance; total_shots is the sum over instances and is
  // what the logical error rate divides by.
  std::size_t shots = 0;
  std::size_t total_shots = 0;
  // Detector rounds in one shot, which is the number of rounds the streaming
  // path enqueues and the batch path decodes at once.  Stim emits one final
  // round from the data readout, so this is `rounds` + 1 rather than `rounds`.
  std::size_t rounds_per_shot = 0;
  latency_stats latency;
  latency_stats tail;
  // Detector rounds per second: the rate each instance of the point held over
  // its own timed window, summed.
  double throughput = 0.0;
  std::size_t logical_errors = 0;
};

uint64_t mask_from_result(const cudaq::qec::decoder_result &r) {
  uint64_t mask = 0;
  for (std::size_t i = 0; i < r.result.size() && i < 64; ++i)
    if (cudaq::qec::convert_soft_to_hard(r.result[i]))
      mask ^= uint64_t{1} << i;
  return mask;
}

uint64_t mask_from_corrections(const uint8_t *corr, std::size_t n) {
  uint64_t mask = 0;
  for (std::size_t i = 0; i < n && i < 64; ++i)
    if (corr[i])
      mask ^= uint64_t{1} << i;
  return mask;
}

// Shot an instance decodes on its i-th timed iteration.  Instances enter the
// shared pool at different points and wrap, so concurrent instances are never
// decoding the same syndrome in lockstep while still covering the same shots.
std::size_t timed_shot(const options &opts, std::size_t offset, std::size_t i) {
  return opts.warmup + (offset + i) % opts.shots;
}

void warmup_batch(cudaq::qec::decoder &dec, const benchmark_data &data,
                  const options &opts) {
  for (std::size_t s = 0; s < opts.warmup; ++s)
    (void)dec.decode(data.soft_syndromes[s]);
}

// Timed batch loop for one instance.  Only the decode() call is inside the
// stopwatch; the observable comparison that follows is bookkeeping.
void run_batch_instance(cudaq::qec::decoder &dec, const benchmark_data &data,
                        const options &opts, std::size_t offset,
                        instance_result &out) {
  out.shots = opts.shots;
  out.shot_us.reserve(opts.shots);
  for (std::size_t i = 0; i < opts.shots; ++i) {
    const std::size_t shot = timed_shot(opts, offset, i);
    const auto t0 = clock_type::now();
    auto decoded = dec.decode(data.soft_syndromes[shot]);
    out.shot_us.push_back(elapsed_us(t0, clock_type::now()));
    if (mask_from_result(decoded) != data.observable_masks[shot])
      ++out.logical_errors;
  }
} // end - run_batch_instance()

void spin_until(clock_type::time_point t) {
  while (clock_type::now() < t) {
  }
}

// One streaming shot: enqueue every round, then read the corrections out.
// Reports in-decoder work (sum of the enqueue calls, excluding any pacing
// spin) and the tail (final round plus corrections readout) separately, and
// returns the predicted observable mask.
uint64_t stream_shot(cudaq::qec::decoder &dec, const benchmark_data &data,
                     const options &opts,
                     const std::vector<std::size_t> &round_widths,
                     std::size_t shot, std::size_t num_obs, double *shot_us,
                     double *tail_us) {
  const std::vector<uint8_t> &syn = data.hard_syndromes[shot];
  double work = 0.0;
  std::size_t offset = 0;

  dec.reset_decoder();
  const auto shot_start = clock_type::now();
  for (std::size_t r = 0; r + 1 < round_widths.size(); ++r) {
    if (opts.round_interval.count() > 0)
      spin_until(shot_start + static_cast<int64_t>(r) * opts.round_interval);
    const auto rs = clock_type::now();
    dec.enqueue_syndrome(syn.data() + offset, round_widths[r]);
    work += elapsed_us(rs, clock_type::now());
    offset += round_widths[r];
  }
  if (opts.round_interval.count() > 0)
    spin_until(shot_start + static_cast<int64_t>(round_widths.size() - 1) *
                                opts.round_interval);
  const auto tail_start = clock_type::now();
  dec.enqueue_syndrome(syn.data() + offset, round_widths.back());
  const uint8_t *corr = dec.get_obs_corrections();
  const double tail = elapsed_us(tail_start, clock_type::now());

  if (shot_us)
    *shot_us = work + tail;
  if (tail_us)
    *tail_us = tail;
  return mask_from_corrections(corr, num_obs);
} // end - stream_shot()

void warmup_stream(cudaq::qec::decoder &dec, const benchmark_data &data,
                   const options &opts,
                   const std::vector<std::size_t> &round_widths) {
  const std::size_t num_obs = dec.get_num_observables();
  for (std::size_t s = 0; s < opts.warmup; ++s)
    (void)stream_shot(dec, data, opts, round_widths, s, num_obs, nullptr,
                      nullptr);
}

// Timed streaming loop for one instance.
void run_stream_instance(cudaq::qec::decoder &dec, const benchmark_data &data,
                         const options &opts,
                         const std::vector<std::size_t> &round_widths,
                         std::size_t offset, instance_result &out) {
  const std::size_t num_obs = dec.get_num_observables();
  double shot_us = 0.0;
  double tail_us = 0.0;

  out.shots = opts.shots;
  out.shot_us.reserve(opts.shots);
  out.tail_us.reserve(opts.shots);
  for (std::size_t i = 0; i < opts.shots; ++i) {
    const std::size_t shot = timed_shot(opts, offset, i);
    const uint64_t predicted = stream_shot(dec, data, opts, round_widths, shot,
                                           num_obs, &shot_us, &tail_us);
    out.shot_us.push_back(shot_us);
    out.tail_us.push_back(tail_us);
    if (predicted != data.observable_masks[shot])
      ++out.logical_errors;
  }
} // end - run_stream_instance()

// Fold every instance of a point into the single row the report prints.  The
// percentiles come from the pooled samples on purpose: a percentile taken from
// one instance says nothing about the instances contending with it.
measurement pool_measurement(const instance_pool_result &pool,
                             const options &opts, const sweep_point &point,
                             std::size_t rounds_per_shot) {
  measurement result;
  const auto totals = sum_instance_results(pool.instances);
  auto shot_us =
      concat_instance_samples(pool.instances, instance_sample_kind::shot);
  auto tail_us =
      concat_instance_samples(pool.instances, instance_sample_kind::tail);

  result.decoder_id = point.decoder_id;
  result.name = point.name;
  result.distance = point.distance;
  result.rounds = point.rounds;
  result.noise = point.noise;
  result.instances = point.instances;
  result.shots = opts.shots;
  result.total_shots = totals.shots;
  result.rounds_per_shot = rounds_per_shot;
  result.logical_errors = totals.logical_errors;
  result.latency = summarize(std::move(shot_us));
  // Batch mode records no tail samples; mirror the decode latency so a row
  // can be printed against either column.
  result.tail =
      tail_us.empty() ? result.latency : summarize(std::move(tail_us));
  // Rounds retired per second, each instance rated over its own timed window
  // and the rates then summed.  Rating them separately is what keeps an
  // instance that finishes early from being charged for the time it spent
  // waiting on the slowest one, which a single shared window would fold into
  // the denominator of every instance.  A window holds everything the shot
  // loop does, pacing spins included, so under --round_interval_us this
  // reports the paced round rate the decoders sustained rather than the rate
  // they are capable of.
  result.throughput = 0.0;
  for (const auto &instance : pool.instances) {
    if (instance.wall_us <= 0.0)
      continue;
    result.throughput += static_cast<double>(instance.shots * rounds_per_shot) *
                         1e6 / instance.wall_us;
  }
  return result;
} // end - pool_measurement()

// Run one sweep point across its decoder instances and return the pooled row.
//
// Every instance builds its own decoder inside its own thread rather than
// being handed one: decoder construction persistently pins the constructing
// thread to the decoder's CUDA device, and the library's rule is that the
// thread that builds a decoder is the thread that drives it.
measurement run_point(const options &opts, const benchmark_data &data,
                      const dem_matrices &matrices,
                      const cudaqx::heterogeneous_map &dec_params,
                      const std::vector<std::size_t> &round_widths,
                      bench_mode mode, const sweep_point &point) {
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders(point.instances);
  instance_work work;

  work.setup = [&](std::size_t instance) {
    std::optional<cudaq::qec::sparse_binary_matrix> D;
    if (mode == bench_mode::stream)
      D = identity_measurement_map(data.circuit.count_detectors());
    cudaq::qec::decoder_init inputs(
        matrices.H, cudaq::qec::sparse_binary_matrix(matrices.O),
        matrices.priors, std::move(D));
    auto dec = cudaq::qec::decoder::get(
        point.name, std::move(inputs),
        cudaq::qec::decode_result_type::observables, dec_params);
    // Stamp the sweep's id on the decoder so any [DecoderStats] line it logs
    // carries the same ID the table rows report.  Every instance of a point
    // shares that id, matching the one pooled row the point produces.
    dec->set_decoder_id(static_cast<uint32_t>(point.decoder_id));
    if (mode == bench_mode::stream) {
      warmup_stream(*dec, data, opts, round_widths);
    } else {
      warmup_batch(*dec, data, opts);
    }
    decoders[instance] = std::move(dec);
  }; // end - work.setup

  work.run = [&](std::size_t instance, instance_result &out) {
    const std::size_t offset =
        instance_shot_offset(instance, point.instances, opts.shots);
    if (mode == bench_mode::stream)
      run_stream_instance(*decoders[instance], data, opts, round_widths, offset,
                          out);
    else
      run_batch_instance(*decoders[instance], data, opts, offset, out);
  };

  const instance_pool_result pool =
      run_instances(point.instances, layout_for(opts, point.instances), work);
  const std::string failure = first_instance_error(pool.instances);
  if (!failure.empty())
    throw std::runtime_error(failure);
  return pool_measurement(pool, opts, point, round_widths.size());
} // end - run_point()

// ── Output
// ────────────────────────────────────────────────────────────────────

// Column widths for the flat results table.
constexpr int col_w_id = 4;
constexpr int col_w_decoder = 26;
constexpr int col_w_d = 4;
constexpr int col_w_rounds = 8;
constexpr int col_w_instances = 11;
constexpr int col_w_shots = 11; // timed shots per instance
constexpr int col_w_noise = 11;
// 11 so the widest latency heading ("decode p50", 10 characters) still leaves
// a separating space; anything narrower runs it into the noise column.
constexpr int col_w_lat = 11;
// 13 because a round rate is the shot rate times the rounds in a shot, so the
// integer here runs several digits longer than a shots/s figure would.
constexpr int col_w_tput = 13; // rounds/s summed over instances
constexpr int col_w_ler = 12;
// Divider width. The decoder name is printed with a two-space gutter, hence
// the + 2; deriving the total keeps it correct as columns come and go.
constexpr int table_width =
    col_w_id + col_w_decoder + 2 + col_w_d + col_w_rounds + col_w_instances +
    col_w_shots + col_w_noise + 3 * col_w_lat + col_w_tput + col_w_ler;

// The logical error rate divides by the shots of every instance, so a row must
// carry the total alongside the per-instance count.
double logical_error_rate(const measurement &row) {
  if (row.total_shots == 0)
    return 0.0;
  return static_cast<double>(row.logical_errors) /
         static_cast<double>(row.total_shots);
}

// CPU placement in one phrase, for the configuration header.
std::string describe_pinning(const core_layout &cores) {
  const std::size_t width = cores.width > 0 ? cores.width : 1;
  const std::size_t stride = effective_stride(cores);
  std::string text;

  if (cores.base < 0)
    return "none";
  text = "CPU " + std::to_string(cores.base) + "+";
  if (stride != 1)
    text += std::to_string(stride) + "*";
  text += "instance";
  if (width > 1)
    text += ", " + std::to_string(width) + " CPUs each";
  return text;
}

// Automatic placement differs per row, so report each instance count's blocks
// rather than a single layout the run never actually uses.
std::string describe_auto_pinning(const options &opts) {
  const auto cpus = allowed_cpus();
  std::string text = "auto over ";

  if (cpus.empty())
    return "auto (no CPU list available)";
  text += std::to_string(cpus.size()) + " CPUs (" +
          std::to_string(cpus.front()) + "-" + std::to_string(cpus.back()) +
          "):";
  for (const std::size_t n : opts.instances_list) {
    const core_layout layout = layout_for(opts, n);
    text += "  " + std::to_string(n) + " inst: ";
    text += layout.base < 0 ? std::string("unpinned")
                            : std::to_string(layout.width) + " CPUs each";
  }
  return text;
}

void print_table_header(const std::string &title, const std::string &lat_col) {
  std::cout << "\n" << title << "\n";
  // The round rate divides by a count no column carries, and one no reader
  // would guess from the rounds column, so spell the convention out.
  std::cout << "rounds/s: detector rounds retired per second, summed over "
               "instances; a shot carries rounds+1 of them\n";
  std::cout << std::right << std::setw(col_w_id) << "id" << std::left
            << std::setw(col_w_decoder + 2) << "  decoder" << std::right
            << std::setw(col_w_d) << "d" << std::setw(col_w_rounds) << "rounds"
            << std::setw(col_w_instances) << "instances"
            << std::setw(col_w_shots) << "shots/inst" << std::setw(col_w_noise)
            << "noise" << std::setw(col_w_lat) << (lat_col + " p50")
            << std::setw(col_w_lat) << "p90" << std::setw(col_w_lat) << "p99"
            << std::setw(col_w_tput) << "rounds/s" << std::setw(col_w_ler)
            << "LER"
            << "\n";
  std::cout << std::string(table_width, '-') << "\n";
}

void print_table_row(const measurement &row, bool use_tail) {
  const auto &lat = use_tail ? row.tail : row.latency;
  std::cout << std::right << std::fixed << std::setw(col_w_id) << row.decoder_id
            << "  " << std::left << std::setw(col_w_decoder) << row.name
            << std::right << std::setw(col_w_d) << row.distance
            << std::setw(col_w_rounds) << row.rounds
            << std::setw(col_w_instances) << row.instances
            << std::setw(col_w_shots) << row.shots << std::setw(col_w_noise)
            << std::scientific << std::setprecision(2) << row.noise
            << std::fixed << std::setprecision(2) << std::setw(col_w_lat)
            << lat.p50 << std::setw(col_w_lat) << lat.p90
            << std::setw(col_w_lat) << lat.p99 << std::setprecision(0)
            << std::setw(col_w_tput) << row.throughput << std::setprecision(4)
            << std::setw(col_w_ler) << logical_error_rate(row) << "\n";
}

void print_csv_row(const std::string &mode, const measurement &row) {
  std::cout << "csv," << mode << "," << row.decoder_id << "," << row.name << ","
            << row.distance << "," << row.rounds << "," << row.rounds_per_shot
            << "," << row.instances << "," << row.shots << ","
            << row.total_shots << "," << std::scientific << std::setprecision(2)
            << row.noise << "," << std::fixed << std::setprecision(3)
            << row.latency.p50 << "," << row.latency.p90 << ","
            << row.latency.p99 << "," << row.tail.p50 << "," << row.tail.p99
            << "," << std::setprecision(1) << row.throughput << ","
            << std::setprecision(4) << logical_error_rate(row) << "\n";
}

} // namespace

int main(int argc, char **argv) {
  options opts;
  int exit_code = 0;
  if (!parse_args(argc, argv, opts, exit_code))
    return exit_code;

  // Print configuration summary.
  std::cout << std::defaultfloat << "benchmark-qec-decoder"
            << "  shots=" << opts.shots << " (warmup " << opts.warmup << ")"
            << "  hardware_concurrency=" << std::thread::hardware_concurrency()
            << "\n"
            << "decoders:  ";
  for (std::size_t i = 0; i < opts.decoders.size(); ++i)
    std::cout << (i ? ", " : "") << "[" << i << "] " << opts.decoders[i];
  std::cout << "\n"
            << "round pacing="
            << (opts.round_interval.count() > 0
                    ? std::to_string(
                          static_cast<double>(opts.round_interval.count()) /
                          1e3) +
                          " us per round"
                    : std::string("none"))
            << "  output=observables\n"
            << "instances: ";
  for (std::size_t i = 0; i < opts.instances_list.size(); ++i)
    std::cout << (i ? ", " : "") << opts.instances_list[i];
  std::cout << "  pinning="
            << (opts.auto_pin ? describe_auto_pinning(opts)
                              : describe_pinning(opts.cores))
            << "\n";

  // Rewriting a value the user typed is only acceptable if the run says so.
  if (wants_machine_sized_threads(opts)) {
    std::cout << "num_threads=0 ->";
    for (const std::size_t n : opts.instances_list) {
      const std::size_t share = auto_thread_share(opts, n);
      std::cout << "  " << n << " inst: "
                << (share == 0 ? std::string("decoder default")
                               : std::to_string(share) + " threads");
    }
    std::cout << "\n";
  }

  // Oversubscription turns decode latency into scheduler noise, which is easy
  // to mistake for a decoder regression, so say so up front.
  const unsigned num_cpus = std::thread::hardware_concurrency();
  const std::size_t widest =
      *std::max_element(opts.instances_list.begin(), opts.instances_list.end());
  if (num_cpus > 0 && widest > num_cpus)
    std::cerr << "warning: " << widest << " decoder instances on " << num_cpus
              << " CPUs oversubscribes the machine; latencies will include "
                 "scheduling delay\n";
  // A stride under the width is legitimate for contention studies, but it is
  // also an easy typo, and it quietly undoes the isolation pinning is for.
  if (opts.cores.base >= 0 && effective_stride(opts.cores) < opts.cores.width)
    std::cerr << "warning: stride " << effective_stride(opts.cores)
              << " is narrower than width " << opts.cores.width
              << ", so instances share CPUs\n";

  // Accumulate all rows first so each mode's table is printed contiguously.
  std::vector<measurement> batch_rows, stream_rows;

  // Outer sweep: distances × rounds × noises.
  for (const std::size_t distance : opts.distances) {
    // Effective rounds list: use the distance itself when none specified.
    std::vector<std::size_t> effective_rounds;
    if (opts.rounds_list.empty())
      effective_rounds.push_back(distance);
    else
      effective_rounds = opts.rounds_list;

    for (const std::size_t rounds : effective_rounds) {
      // 0 = unset: let the decoder choose rather than imposing a schedule.
      const std::size_t block_leaf = opts.block_leaf_size;

      for (const double noise : opts.noises) {
        auto data = generate_data(opts, distance, rounds, noise);
        auto matrices = make_matrices(data.dem);
        const auto dr = detector_round_map(data.circuit);
        const auto round_widths = detectors_per_round(dr);

        for (int id = 0; id < static_cast<int>(opts.decoders.size()); ++id) {
          const auto &name = opts.decoders[static_cast<std::size_t>(id)];
          try {
            // Instance count innermost: consecutive rows then form the
            // scaling curve for one decoder at one configuration.
            for (const std::size_t instances : opts.instances_list) {
              // Rebuilt per row because a machine-sized thread request
              // resolves against this row's instance count.
              const cudaqx::heterogeneous_map dec_params =
                  build_decoder_params(name, dr, block_leaf, opts.extra_params,
                                       auto_thread_share(opts, instances));
              sweep_point point;
              point.decoder_id = id;
              point.name = name;
              point.distance = distance;
              point.rounds = rounds;
              point.noise = noise;
              point.instances = instances;

              if (opts.run_batch)
                batch_rows.push_back(run_point(opts, data, matrices, dec_params,
                                               round_widths, bench_mode::batch,
                                               point));
              if (opts.run_stream)
                stream_rows.push_back(run_point(opts, data, matrices,
                                                dec_params, round_widths,
                                                bench_mode::stream, point));
            } // end - for(instances)
          } catch (const std::exception &e) {
            std::cerr << "error: decoder [" << id << "] " << name
                      << " (d=" << distance << " r=" << rounds << " p=" << noise
                      << "): " << e.what() << "\n";
            return 1;
          }
        } // end - for(id)
      } // end - for(noise)
    } // end - for(rounds)
  } // end - for(distance)

  // Print batch table (all configs, then all stream configs). Latencies are
  // pooled over the instances of a row and rounds/s is summed over them.
  if (!batch_rows.empty()) {
    print_table_header("batch decode() -- latencies in microseconds, pooled "
                       "over instances",
                       "decode");
    for (const auto &row : batch_rows)
      print_table_row(row, /*use_tail=*/false);
  }
  if (!stream_rows.empty()) {
    print_table_header("streaming: last round + corrections -- latencies in "
                       "microseconds, pooled over instances",
                       "tail");
    for (const auto &row : stream_rows)
      print_table_row(row, /*use_tail=*/true);
  }

  // CSV output (interleaved batch then stream to keep mode grouping).
  if (opts.emit_csv) {
    std::cout << "csv,mode,id,decoder,d,rounds,det_rounds,instances,shots,"
                 "total_shots,noise,"
                 "p50,p90,p99,tail_p50,tail_p99,rounds_s,ler\n";
    for (const auto &row : batch_rows)
      print_csv_row("batch", row);
    for (const auto &row : stream_rows)
      print_csv_row("stream", row);
  }

  std::cout << std::defaultfloat;
  return 0;
}

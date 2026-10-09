/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
#include "cudaq/qec/extended_dem.h"
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
#include <tuple>
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

// The model form every decoder is built from.
enum class model_source {
  matrices, ///< H, O and priors of the whole shot
  chunks,   ///< per-round DEM chunks and the round count
};

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
  // Result form every decoder is built with. Streaming reads observable
  // corrections either way.
  cudaq::qec::decode_result_type output =
      cudaq::qec::decode_result_type::observables;
  model_source source = model_source::matrices;
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
      << "  --output F                 observables or errors (default\n"
      << "                             observables). errors times decode()\n"
      << "                             returning H columns, scored by O*e;\n"
      << "                             streaming still reads observables\n"
      << "  --source S                 matrices or chunks (default\n"
      << "                             matrices). chunks builds decoders from\n"
      << "                             per-round DEM chunks of the same "
         "model,\n"
      << "                             padded to one detector width per round\n"
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
    } else if (arg == "--output") {
      if (val == "observables") {
        opts.output = cudaq::qec::decode_result_type::observables;
      } else if (val == "errors") {
        opts.output = cudaq::qec::decode_result_type::errors;
      } else {
        std::cerr << "error: --output must be observables or errors\n";
        exit_code = 1;
        return false;
      }
    } else if (arg == "--source") {
      if (val == "matrices") {
        opts.source = model_source::matrices;
      } else if (val == "chunks") {
        opts.source = model_source::chunks;
      } else {
        std::cerr << "error: --source must be matrices or chunks\n";
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
  // Batch decode() takes detectors by contract, so both views are kept: the
  // folded detectors it decodes, and the raw measurement records the
  // streaming path enqueues for the decoder to fold itself.
  std::vector<std::vector<cudaq::qec::float_t>> soft_syndromes;
  std::vector<std::vector<uint8_t>> hard_syndromes;
  std::vector<std::vector<uint8_t>> hard_measurements;
  std::vector<uint64_t> observable_masks;
  // Detector d XORs measurement records measurement_map[d]; the column count
  // is circuit.count_measurements().
  std::vector<std::vector<std::uint32_t>> measurement_map;
  // Measurements each round contributes, in order. Not uniform: the final
  // data readout measures every data qubit, where a round measures ancillas.
  std::vector<std::size_t> round_widths;
};

// Defined below, beside the other detector-round helpers.
std::vector<int32_t> detector_round_map(const stim::Circuit &circuit);

// Resolve each DETECTOR's `rec[-k]` targets to absolute measurement indices,
// and the OBSERVABLE_INCLUDE records alongside them.
//
// Flattening expands REPEAT blocks, so a running measurement count over the
// flattened operations is enough: a record target's value() is its offset
// back from the count at that point.
struct circuit_records {
  std::vector<std::vector<std::uint32_t>> detector_to_measurements;
  std::vector<std::uint32_t> observable_measurements;
};

circuit_records records_of(const stim::Circuit &circuit) {
  circuit_records out;
  std::uint64_t seen = 0;
  out.detector_to_measurements.reserve(circuit.count_detectors());
  circuit.flattened().for_each_operation(
      [&](const stim::CircuitInstruction &op) {
        if (op.gate_type == stim::GateType::DETECTOR) {
          std::vector<std::uint32_t> rows;
          for (const auto target : op.targets)
            if (target.is_measurement_record_target())
              rows.push_back(static_cast<std::uint32_t>(
                  seen + static_cast<std::int64_t>(target.value())));
          out.detector_to_measurements.push_back(std::move(rows));
        } else if (op.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
          for (const auto target : op.targets)
            if (target.is_measurement_record_target())
              out.observable_measurements.push_back(static_cast<std::uint32_t>(
                  seen + static_cast<std::int64_t>(target.value())));
        } else {
          seen += op.count_measurement_results();
        }
      });
  return out;
} // end - records_of()

// Measurements per round, read off the measurement records each round's
// detectors touch: a round owns every record above what the rounds before it
// owned. The last entry is the data readout, which is wider than a round.
std::vector<std::size_t>
measurements_per_round(const circuit_records &records,
                       const std::vector<int32_t> &detector_rounds,
                       std::size_t total_measurements) {
  const auto rounds = static_cast<std::size_t>(*std::max_element(
                          detector_rounds.begin(), detector_rounds.end())) +
                      1;
  // Highest measurement record any detector of round r reads.
  std::vector<std::size_t> high(rounds, 0);
  for (std::size_t d = 0; d < detector_rounds.size(); ++d) {
    const auto r = static_cast<std::size_t>(detector_rounds[d]);
    for (const auto rec : records.detector_to_measurements[d])
      high[r] = std::max<std::size_t>(high[r], rec + 1);
  }
  for (std::size_t r = 1; r < rounds; ++r)
    high[r] = std::max(high[r], high[r - 1]);
  // Every remaining record belongs to the final readout.
  high.back() = std::max(high.back(), total_measurements);

  std::vector<std::size_t> widths(rounds, 0);
  std::size_t previous = 0;
  for (std::size_t r = 0; r < rounds; ++r) {
    widths[r] = high[r] - previous;
    previous = high[r];
  }
  return widths;
} // end - measurements_per_round()

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
  // Sample the measurement record, not detection events: the decoder is
  // handed what the hardware emits and folds it into detectors itself, which
  // is the work the decoding server's decoders do on every arriving round.
  const auto reference =
      stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(
          data.circuit);
  const auto measurements =
      stim::sample_batch_measurements<stim::MAX_BITWORD_WIDTH>(
          data.circuit, reference, total, rng, /*transposed=*/false);

  const auto records = records_of(data.circuit);
  data.measurement_map = records.detector_to_measurements;
  data.round_widths = measurements_per_round(
      records, detector_round_map(data.circuit),
      static_cast<std::size_t>(data.circuit.count_measurements()));

  const std::size_t nm =
      static_cast<std::size_t>(data.circuit.count_measurements());
  const std::size_t nd = data.measurement_map.size();
  data.soft_syndromes.resize(total);
  data.hard_syndromes.resize(total);
  data.hard_measurements.resize(total);
  data.observable_masks.assign(total, 0);
  for (std::size_t s = 0; s < total; ++s) {
    data.hard_measurements[s].assign(nm, 0);
    for (std::size_t m = 0; m < nm; ++m)
      if (measurements[m][s])
        data.hard_measurements[s][m] = 1;
    // Fold to detectors for the batch path, the same XOR the streaming
    // decoder performs through D.
    data.soft_syndromes[s].assign(nd, 0.0);
    data.hard_syndromes[s].assign(nd, 0);
    for (std::size_t d = 0; d < nd; ++d) {
      std::uint8_t parity = 0;
      for (const auto rec : data.measurement_map[d])
        parity ^= data.hard_measurements[s][rec];
      data.soft_syndromes[s][d] = parity ? 1.0 : 0.0;
      data.hard_syndromes[s][d] = parity;
    }
    // One logical observable for a memory experiment, read off the same
    // records the circuit's OBSERVABLE_INCLUDE names.
    std::uint8_t parity = 0;
    for (const auto rec : records.observable_measurements)
      parity ^= static_cast<std::uint8_t>(measurements[rec][s]);
    if (parity)
      data.observable_masks[s] ^= uint64_t{1};
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
  // Observables each H column flips, as a mask of the first 64. With chunks
  // set, the columns are the chunks', in the order they close to.
  std::vector<uint64_t> column_obs;
  // The same model as per-round chunks, for --source chunks.
  std::optional<cudaq::qec::dem_chunks_spec> chunks;
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
  m.column_obs.assign(ne, 0);
  std::vector<std::vector<index_type>> H_nested(ne);
  for (std::size_t col = 0; col < ne; ++col) {
    m.priors[col] = edges[col].p;
    for (auto det : edges[col].detectors)
      H_nested[col].push_back(static_cast<index_type>(det));
    for (auto o : edges[col].observables) {
      m.O.at({o, col}) = 1;
      if (o < 64)
        m.column_obs[col] |= uint64_t{1} << o;
    }
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

// Give every detector round the widest round's width, renumbering detector d
// of round r to r * width + d, in the matrices, the sampled syndromes and @p
// dr. Chunked models carry one detector width per round, and Stim's first and
// last rounds hold fewer detectors than the bulk; the added detectors flip
// with no fault, so they never fire.
void pad_rounds(dem_matrices &m, benchmark_data &data,
                std::vector<int32_t> &dr) {
  if (!std::is_sorted(dr.begin(), dr.end()))
    throw std::runtime_error("--source chunks needs detectors numbered round "
                             "by round");
  const std::size_t num_rounds = static_cast<std::size_t>(dr.back()) + 1;
  std::vector<std::size_t> first(num_rounds, dr.size());
  for (std::size_t det = dr.size(); det-- > 0;)
    first[static_cast<std::size_t>(dr[det])] = det;
  std::size_t width = 0;
  for (std::size_t r = 0; r < num_rounds; ++r)
    width = std::max(width, (r + 1 < num_rounds ? first[r + 1] : dr.size()) -
                                first[r]);
  std::vector<index_type> padded(dr.size());
  for (std::size_t det = 0; det < dr.size(); ++det) {
    const auto r = static_cast<std::size_t>(dr[det]);
    padded[det] = static_cast<index_type>(r * width + det - first[r]);
  }
  const std::size_t nd = num_rounds * width;

  auto columns = m.H.to_nested_csc();
  for (auto &column : columns)
    for (auto &det : column)
      det = padded[det];
  m.H = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      static_cast<index_type>(nd), static_cast<index_type>(columns.size()),
      columns);
  for (std::size_t s = 0; s < data.soft_syndromes.size(); ++s) {
    std::vector<cudaq::qec::float_t> soft(nd, 0.0);
    std::vector<uint8_t> hard(nd, 0);
    for (std::size_t det = 0; det < dr.size(); ++det) {
      soft[padded[det]] = data.soft_syndromes[s][det];
      hard[padded[det]] = data.hard_syndromes[s][det];
    }
    data.soft_syndromes[s] = std::move(soft);
    data.hard_syndromes[s] = std::move(hard);
  }
  // D follows the detectors into their padded positions. A padding detector
  // reads no measurement, so its row stays empty and it never fires.
  std::vector<std::vector<std::uint32_t>> remapped(nd);
  for (std::size_t det = 0; det < dr.size(); ++det)
    remapped[padded[det]] = std::move(data.measurement_map[det]);
  data.measurement_map = std::move(remapped);
  dr.resize(nd);
  for (std::size_t det = 0; det < nd; ++det)
    dr[det] = static_cast<int32_t>(det / width);
} // end - pad_rounds()

// Split the matrices into one DEM chunk per detector round and record them in
// m.chunks. A column belongs to the round of its lowest detector, and must
// reach no further than the next round. The rounds equal to the middle round,
// up to a shift, are the repeating bulk; every round before and after it is a
// phase of its own. Closing the chunks numbers detectors as @p dr does.
void add_chunks(dem_matrices &m, const std::vector<int32_t> &dr) {
  using namespace cudaq::qec;
  struct fault {
    double p = 0.0;
    std::vector<int64_t> rows, next_rows;
    uint64_t obs = 0;
    auto key() const { return std::tie(rows, next_rows, obs); }
  };
  if (!std::is_sorted(dr.begin(), dr.end()))
    throw std::runtime_error("--source chunks needs detectors numbered round "
                             "by round");
  const std::size_t num_rounds = static_cast<std::size_t>(dr.back()) + 1;
  std::vector<int64_t> first(num_rounds + 1, static_cast<int64_t>(dr.size()));
  for (std::size_t det = dr.size(); det-- > 0;)
    first[static_cast<std::size_t>(dr[det])] = static_cast<int64_t>(det);
  const auto width = [&](std::size_t r) {
    return r < num_rounds ? first[r + 1] - first[r] : int64_t{0};
  };

  std::vector<std::vector<fault>> rounds(num_rounds);
  const auto columns = m.H.to_nested_csc();
  for (std::size_t col = 0; col < columns.size(); ++col) {
    if (columns[col].empty())
      throw std::runtime_error("--source chunks needs every column to flip a "
                               "detector");
    std::vector<int64_t> dets(columns[col].begin(), columns[col].end());
    std::sort(dets.begin(), dets.end());
    const auto r = static_cast<std::size_t>(dr[dets.front()]);
    fault f;
    f.p = m.priors[col];
    f.obs = m.column_obs[col];
    for (const int64_t det : dets) {
      const auto at = static_cast<std::size_t>(dr[det]);
      if (at > r + 1)
        throw std::runtime_error("--source chunks needs every column within "
                                 "two consecutive rounds");
      (at == r ? f.rows : f.next_rows).push_back(det - first[at]);
    }
    rounds[r].push_back(std::move(f));
  }
  // Stim's decomposition repeats every two rounds: alternate rounds split a
  // mechanism into columns with the same detectors and observables. Merging
  // them leaves one model per round, up to rounding in the merged rates.
  for (auto &faults : rounds) {
    std::sort(faults.begin(), faults.end(),
              [](const fault &a, const fault &b) { return a.key() < b.key(); });
    std::vector<fault> merged;
    for (auto &f : faults) {
      if (!merged.empty() && merged.back().key() == f.key()) {
        double &p = merged.back().p;
        p = p * (1 - f.p) + f.p * (1 - p);
      } else {
        merged.push_back(std::move(f));
      }
    }
    faults = std::move(merged);
  }

  const auto same = [&](std::size_t a, std::size_t b) {
    if (width(a) != width(b) || width(a + 1) != width(b + 1) ||
        rounds[a].size() != rounds[b].size())
      return false;
    for (std::size_t i = 0; i < rounds[a].size(); ++i) {
      const fault &x = rounds[a][i], &y = rounds[b][i];
      if (x.key() != y.key() || std::abs(x.p - y.p) > 1e-9 * std::max(x.p, y.p))
        return false;
    }
    return true;
  };
  const std::size_t middle = num_rounds / 2;
  std::size_t bulk_first = middle, bulk_last = middle;
  while (bulk_first > 0 && same(bulk_first - 1, middle))
    --bulk_first;
  while (bulk_last + 1 < num_rounds && same(bulk_last + 1, middle))
    ++bulk_last;

  const std::size_t no = m.O.shape()[0];
  const auto chunk_of = [&](std::size_t r) {
    const auto &faults = rounds[r];
    const auto seam_rows = [&](int64_t rows, bool next) {
      std::vector<std::vector<int64_t>> by_row(static_cast<std::size_t>(rows));
      for (std::size_t i = 0; i < faults.size(); ++i)
        for (const int64_t row : next ? faults[i].next_rows : faults[i].rows)
          by_row[static_cast<std::size_t>(row)].push_back(
              static_cast<int64_t>(i));
      std::vector<int64_t> sparse;
      for (const auto &row : by_row) {
        sparse.insert(sparse.end(), row.begin(), row.end());
        sparse.push_back(-1);
      }
      return sparse;
    };
    dem_chunk_spec spec;
    spec.num_faults = faults.size();
    for (const auto &f : faults)
      spec.error_rates.push_back(f.p);
    spec.seam_specs.push_back(
        {seam_name::prev_round, {seam_rows(width(r), false), {}}});
    if (r + 1 < num_rounds)
      spec.seam_specs.push_back(
          {seam_name::next_round, {seam_rows(width(r + 1), true), {}}});
    for (std::size_t o = 0; o < no; ++o) {
      for (std::size_t i = 0; i < faults.size(); ++i)
        if (o < 64 && (faults[i].obs >> o & 1))
          spec.O_sparse.push_back(static_cast<int64_t>(i));
      spec.O_sparse.push_back(-1);
    }
    return spec;
  };

  dem_chunks_spec spec;
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  spec.num_rounds = num_rounds;
  std::vector<phase_id> chain;
  const auto add_phase = [&](const std::string &name, std::size_t r) {
    const phase_id id = phase_id::from_name(name);
    spec.phases.push_back({id, chunk_of(r)});
    chain.push_back(id);
  };
  for (std::size_t r = 0; r < bulk_first; ++r)
    add_phase("before" + std::to_string(r), r);
  add_phase("bulk", middle);
  for (std::size_t r = bulk_last + 1; r < num_rounds; ++r)
    add_phase("after" + std::to_string(r - bulk_last - 1), r);
  for (std::size_t i = 0; i + 1 < chain.size(); ++i)
    spec.connections.push_back({chain[i], chain[i + 1]});
  spec.connections.push_back({chain[bulk_first], chain[bulk_first]});

  m.column_obs.clear();
  for (const auto &faults : rounds)
    for (const auto &f : faults)
      m.column_obs.push_back(f.obs);
  m.chunks = std::move(spec);
} // end - add_chunks()

// The model's measurement-to-detector map: one row per detector, naming the
// measurement records it XORs. The decoder folds arriving rounds through this
// on the streaming path, which is why the benchmark supplies the real map
// rather than an identity standing in for pre-folded detectors.
cudaq::qec::sparse_binary_matrix
measurement_map_of(const benchmark_data &data) {
  return cudaq::qec::sparse_binary_matrix::from_nested_csr(
      static_cast<std::uint32_t>(data.measurement_map.size()),
      static_cast<std::uint32_t>(data.circuit.count_measurements()),
      data.measurement_map);
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
//                 needs temporal round information.  Withheld when @p dr is
//                 empty: chunks carry their rounds, and nv-fusion-decoder
//                 prefers the param over them.
//
//   rolling_error_output, rolling_error_margin -- nv-fusion-decoder
//                 construction knobs that the realtime YAML schema leaves
//                 out, since YAML always decodes to observables. Forwarded to
//                 nv-fusion-decoder by name.
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
  if (name == "nv-fusion-decoder" && !dr.empty())
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
    if (name == "nv-fusion-decoder" && key == "rolling_error_output") {
      p.insert(key, val == "true");
      continue;
    }
    if (name == "nv-fusion-decoder" && key == "rolling_error_margin") {
      p.insert(key, static_cast<uint64_t>(std::stoull(val)));
      continue;
    }
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

// Observables flipped by the H columns an error-output result selects.
uint64_t mask_from_errors(const cudaq::qec::decoder_result &r,
                          const std::vector<uint64_t> &column_obs) {
  uint64_t mask = 0;
  for (std::size_t i = 0; i < r.result.size() && i < column_obs.size(); ++i)
    if (cudaq::qec::convert_soft_to_hard(r.result[i]))
      mask ^= column_obs[i];
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
                        const dem_matrices &matrices, const options &opts,
                        std::size_t offset, instance_result &out) {
  const bool errors = opts.output == cudaq::qec::decode_result_type::errors;
  out.shots = opts.shots;
  out.shot_us.reserve(opts.shots);
  for (std::size_t i = 0; i < opts.shots; ++i) {
    const std::size_t shot = timed_shot(opts, offset, i);
    const auto t0 = clock_type::now();
    auto decoded = dec.decode(data.soft_syndromes[shot]);
    out.shot_us.push_back(elapsed_us(t0, clock_type::now()));
    const uint64_t predicted =
        errors ? mask_from_errors(decoded, matrices.column_obs)
               : mask_from_result(decoded);
    if (predicted != data.observable_masks[shot])
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
  const std::vector<uint8_t> &syn = data.hard_measurements[shot];
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
      D = measurement_map_of(data);
    auto inputs = [&] {
      if (!matrices.chunks)
        return cudaq::qec::decoder_init(
            matrices.H, cudaq::qec::sparse_binary_matrix(matrices.O),
            matrices.priors, std::move(D));
      return cudaq::qec::decoder_needs_model_matrices(
                 point.name, cudaq::qec::decoder_model_source::dem_chunks)
                 ? cudaq::qec::decoder_init::from_dem_chunks_closed(
                       *matrices.chunks, std::move(D))
                 : cudaq::qec::decoder_init::from_dem_chunks(*matrices.chunks,
                                                             std::move(D));
    }();
    auto dec = cudaq::qec::decoder::get(point.name, std::move(inputs),
                                        opts.output, dec_params);
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
      run_batch_instance(*decoders[instance], data, matrices, opts, offset,
                         out);
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
    col_w_shots + col_w_noise + 5 * col_w_lat + col_w_tput + col_w_ler;

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
            << "noise" << std::setw(col_w_lat) << (lat_col + " min")
            << std::setw(col_w_lat) << "p50" << std::setw(col_w_lat) << "p90"
            << std::setw(col_w_lat) << "p99" << std::setw(col_w_lat) << "max"
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
            << lat.min << std::setw(col_w_lat) << lat.p50
            << std::setw(col_w_lat) << lat.p90 << std::setw(col_w_lat)
            << lat.p99 << std::setw(col_w_lat) << lat.max
            << std::setprecision(0) << std::setw(col_w_tput) << row.throughput
            << std::setprecision(4) << std::setw(col_w_ler)
            << logical_error_rate(row) << "\n";
}

void print_csv_row(const std::string &mode, const measurement &row) {
  std::cout << "csv," << mode << "," << row.decoder_id << "," << row.name << ","
            << row.distance << "," << row.rounds << "," << row.rounds_per_shot
            << "," << row.instances << "," << row.shots << ","
            << row.total_shots << "," << std::scientific << std::setprecision(2)
            << row.noise << "," << std::fixed << std::setprecision(3)
            << row.latency.min << "," << row.latency.p50 << ","
            << row.latency.p90 << "," << row.latency.p99 << ","
            << row.latency.max << "," << row.tail.min << "," << row.tail.p50
            << "," << row.tail.p99 << "," << row.tail.max << ","
            << std::setprecision(1) << row.throughput << ","
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
  std::cout
      << "\n"
      << "round pacing="
      << (opts.round_interval.count() > 0
              ? std::to_string(
                    static_cast<double>(opts.round_interval.count()) / 1e3) +
                    " us per round"
              : std::string("none"))
      << "  output="
      << (opts.output == cudaq::qec::decode_result_type::errors ? "errors"
                                                                : "observables")
      << "  source="
      << (opts.source == model_source::chunks ? "chunks" : "matrices") << "\n"
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
  const std::size_t mode_count = static_cast<std::size_t>(opts.run_batch) +
                                 static_cast<std::size_t>(opts.run_stream);
  const std::size_t rounds_count =
      opts.rounds_list.empty() ? 1 : opts.rounds_list.size();
  const std::size_t total_points = opts.distances.size() * rounds_count *
                                   opts.noises.size() * opts.decoders.size() *
                                   opts.instances_list.size() * mode_count;
  std::size_t point_number = 0;

  // Keep stdout stable for tables and CSV consumers while making long sweeps
  // visibly advance on stderr. Each point includes construction and warmup, so
  // announce it before run_point() rather than only printing after completion.
  const auto run_and_record =
      [&](const benchmark_data &data, const dem_matrices &matrices,
          const cudaqx::heterogeneous_map &dec_params,
          const std::vector<std::size_t> &round_widths, bench_mode mode,
          const sweep_point &point, std::vector<measurement> &rows) {
        const char *mode_name = mode == bench_mode::batch ? "batch" : "stream";
        ++point_number;
        std::cerr << "[" << point_number << "/" << total_points << "] running "
                  << mode_name << " " << point.name << " d=" << point.distance
                  << " rounds=" << point.rounds << " noise=" << point.noise
                  << " instances=" << point.instances << " ..." << std::flush;
        const auto started = clock_type::now();
        rows.push_back(run_point(opts, data, matrices, dec_params, round_widths,
                                 mode, point));
        const double elapsed_seconds =
            std::chrono::duration<double>(clock_type::now() - started).count();
        std::cerr << " done (" << std::fixed << std::setprecision(2)
                  << elapsed_seconds << " s)\n"
                  << std::defaultfloat;
      }; // end - run_and_record()

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
        auto dr = detector_round_map(data.circuit);
        if (opts.source == model_source::chunks) {
          pad_rounds(matrices, data, dr);
          add_chunks(matrices, dr);
        }
        const auto &round_widths = data.round_widths;
        const std::vector<int32_t> param_dr =
            matrices.chunks ? std::vector<int32_t>{} : dr;

        for (int id = 0; id < static_cast<int>(opts.decoders.size()); ++id) {
          const auto &name = opts.decoders[static_cast<std::size_t>(id)];
          try {
            // Instance count innermost: consecutive rows then form the
            // scaling curve for one decoder at one configuration.
            for (const std::size_t instances : opts.instances_list) {
              // Rebuilt per row because a machine-sized thread request
              // resolves against this row's instance count.
              const cudaqx::heterogeneous_map dec_params = build_decoder_params(
                  name, param_dr, block_leaf, opts.extra_params,
                  auto_thread_share(opts, instances));
              sweep_point point;
              point.decoder_id = id;
              point.name = name;
              point.distance = distance;
              point.rounds = rounds;
              point.noise = noise;
              point.instances = instances;

              if (opts.run_batch)
                run_and_record(data, matrices, dec_params, round_widths,
                               bench_mode::batch, point, batch_rows);
              if (opts.run_stream)
                run_and_record(data, matrices, dec_params, round_widths,
                               bench_mode::stream, point, stream_rows);
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
                 "latency_min,p50,p90,p99,latency_max,tail_min,tail_p50,"
                 "tail_p99,tail_max,rounds_s,ler\n";
    for (const auto &row : batch_rows)
      print_csv_row("batch", row);
    for (const auto &row : stream_rows)
      print_csv_row("stream", row);
  }

  std::cout << std::defaultfloat;
  return 0;
}

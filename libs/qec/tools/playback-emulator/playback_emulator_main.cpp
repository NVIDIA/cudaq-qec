/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file playback_emulator_main.cpp
/// @brief CLI entry point for the playback emulator. Hand-parsed
/// `--flag=value` arguments, matching this repo's convention (see
/// tools/decoding-server/decoding_server.cpp) rather than a CLI-parsing
/// dependency.

#include "emulator.h"
#include "session.h"
#include "syndrome_source.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace cudaq::qec::playback;

namespace {

bool starts_with(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::string read_file(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void print_usage() {
  std::cout
      << "Usage: playback-emulator --schedule=<file> [options]\n"
         "\n"
         "Required:\n"
         "  --schedule=PATH     playback schedule text, one\n"
         "                      '<trigger> <op> [key=value...]' per line\n"
         "\n"
         "Options:\n"
         "  --config=PATH       multi_decoder_config YAML (decoders\n"
         "                      section); required for null/inproc,\n"
         "                      optional for udp (which can instead derive\n"
         "                      decoder_ids from --udp-endpoint=)\n"
         "  --tick=DURATION     wall-clock width of one tick (default: 1us)\n"
         "  --backend=NAME      null | inproc | udp (default: inproc)\n"
         "  --udp-endpoint=ID:HOST:PORT   repeatable; required for "
         "--backend=udp\n"
         "  --source=ID:PATH    static_source for source_id ID, one 0/1 bit\n"
         "                      string per round, one round per line in PATH\n"
         "  --stim-source=ID:key=value,...   stim_memory_source for source_id\n"
         "                      ID; repeatable. Keys: code, task, distance\n"
         "                      (required), rounds (default 3), seed (default\n"
         "                      1), and the noise probabilities\n"
         "                      before_measure_flip_probability,\n"
         "                      after_clifford_depolarization,\n"
         "                      before_round_data_depolarization,\n"
         "                      after_reset_flip_probability (default 0)\n"
         "  --out=PATH          write the run's CSV here (default: stdout)\n"
         "  --help\n";
}

std::uint64_t parse_duration_ns(const std::string &s) {
  std::size_t split = s.size();
  while (split > 0 && !std::isdigit(static_cast<unsigned char>(s[split - 1])))
    --split;
  const std::uint64_t value = std::stoull(s.substr(0, split));
  if (value == 0)
    throw std::runtime_error("duration '" + s + "' must be positive");
  const std::string unit = s.substr(split);
  std::uint64_t scale = 1;
  if (unit.empty() || unit == "ns")
    scale = 1;
  else if (unit == "us")
    scale = 1'000;
  else if (unit == "ms")
    scale = 1'000'000;
  else if (unit == "s")
    scale = 1'000'000'000;
  else
    throw std::runtime_error("unknown duration unit in '" + s + "'");
  std::uint64_t ns;
  if (__builtin_mul_overflow(value, scale, &ns))
    throw std::runtime_error("duration '" + s + "' overflows nanoseconds");
  return ns;
}

/// Everything one `--stim-source=ID:...` needs besides its source_id: the
/// seed (stim_memory_source's own constructor argument, not part of its
/// heterogeneous_map) and the rest of the key=value pairs, typed per key.
struct stim_source_spec {
  std::uint64_t seed = 1;
  cudaqx::heterogeneous_map params;
};

/// Parses `ID:key=value,key=value,...` (the same `ID:...` shape as
/// --source/--udp-endpoint). Keys are typed by name to match
/// stim_memory_source's params contract (see syndrome_source.cpp's
/// generate_circuit()): code/task are strings, distance/rounds are sizes,
/// the four noise knobs are doubles, seed is its own field.
std::pair<std::uint32_t, stim_source_spec>
parse_stim_source(const std::string &v) {
  const auto colon = v.find(':');
  if (colon == std::string::npos)
    throw std::runtime_error("--stim-source expects ID:key=value,...");
  const std::uint32_t id =
      static_cast<std::uint32_t>(std::stoul(v.substr(0, colon)));

  stim_source_spec spec;
  std::istringstream iss(v.substr(colon + 1));
  for (std::string pair; std::getline(iss, pair, ',');) {
    const auto eq = pair.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error("--stim-source: malformed key=value '" + pair +
                               "'");
    const std::string key = pair.substr(0, eq);
    const std::string val = pair.substr(eq + 1);
    if (key == "seed") {
      spec.seed = std::stoull(val);
    } else if (key == "code" || key == "task") {
      spec.params.insert(key, val);
    } else if (key == "distance" || key == "rounds") {
      spec.params.insert(key, static_cast<std::size_t>(std::stoul(val)));
    } else if (key == "before_measure_flip_probability" ||
               key == "after_clifford_depolarization" ||
               key == "before_round_data_depolarization" ||
               key == "after_reset_flip_probability") {
      spec.params.insert(key, std::stod(val));
    } else {
      throw std::runtime_error("--stim-source: unknown key '" + key + "'");
    }
  }
  return {id, std::move(spec)};
}

std::vector<std::vector<std::uint8_t>> read_rounds(const std::string &path) {
  std::vector<std::vector<std::uint8_t>> rounds;
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open source file: " + path);
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    std::vector<std::uint8_t> bits;
    bits.reserve(line.size());
    for (char c : line) {
      if (c != '0' && c != '1')
        throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                 ": malformed bit string '" + line +
                                 "': expected only '0'/'1'");
      bits.push_back(static_cast<std::uint8_t>(c - '0'));
    }
    rounds.push_back(std::move(bits));
  }
  return rounds;
}

} // namespace

int main(int argc, char **argv) {
  std::string config_path, schedule_path, backend_name = "inproc";
  std::string out_path;
  std::uint64_t tick_ns = 1000;
  std::unordered_map<std::uint64_t, std::string> udp_endpoints;
  std::unordered_map<std::uint32_t, std::string> source_files;
  std::unordered_map<std::uint32_t, stim_source_spec> stim_sources;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--help") {
        print_usage();
        return 0;
      } else if (starts_with(a, "--config=")) {
        config_path = a.substr(9);
      } else if (starts_with(a, "--schedule=")) {
        schedule_path = a.substr(11);
      } else if (starts_with(a, "--tick=")) {
        tick_ns = parse_duration_ns(a.substr(7));
      } else if (starts_with(a, "--backend=")) {
        backend_name = a.substr(10);
      } else if (starts_with(a, "--out=")) {
        out_path = a.substr(6);
      } else if (starts_with(a, "--udp-endpoint=")) {
        const std::string v = a.substr(15);
        const auto first_colon = v.find(':');
        if (first_colon == std::string::npos)
          throw std::runtime_error("--udp-endpoint expects ID:HOST:PORT");
        const std::uint64_t id = std::stoull(v.substr(0, first_colon));
        udp_endpoints[id] = v.substr(first_colon + 1);
      } else if (starts_with(a, "--source=")) {
        const std::string v = a.substr(9);
        const auto colon = v.find(':');
        if (colon == std::string::npos)
          throw std::runtime_error("--source expects ID:PATH");
        const std::uint32_t id =
            static_cast<std::uint32_t>(std::stoul(v.substr(0, colon)));
        source_files[id] = v.substr(colon + 1);
      } else if (starts_with(a, "--stim-source=")) {
        auto [id, spec] = parse_stim_source(a.substr(14));
        stim_sources[id] = std::move(spec);
      } else {
        std::cerr << "unrecognized argument: " << a << "\n";
        print_usage();
        return 1;
      }
    }

    if (schedule_path.empty()) {
      print_usage();
      return 1;
    }

    // --config's only job for `udp` is naming decoder_ids (for schedule
    // `session=` validation) -- the decoder itself lives on the server, not
    // here. `--udp-endpoint=ID:...` already names every id one by one, so
    // `udp` can skip --config entirely and derive decoder_ids from that.
    // `null`/`inproc` have no other source of ids (`inproc` also needs the
    // full config to build the decoder itself), so --config stays required
    // for those.
    cudaq::qec::decoding::config::multi_decoder_config config;
    std::vector<std::uint64_t> decoder_ids;
    if (!config_path.empty()) {
      config =
          cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
              read_file(config_path));
      for (const auto &d : config.decoders)
        decoder_ids.push_back(static_cast<std::uint64_t>(d.id));
    } else if (backend_name == "udp") {
      for (const auto &[id, endpoint] : udp_endpoints)
        decoder_ids.push_back(id);
      std::sort(decoder_ids.begin(), decoder_ids.end());
    } else {
      throw std::runtime_error(
          "--config is required unless --backend=udp (which can derive "
          "decoder_ids from --udp-endpoint= instead)");
    }

    std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
        owned_sessions;
    std::unordered_map<std::uint64_t, session *> router;

    if (backend_name == "null") {
      owned_sessions = make_null_sessions(decoder_ids);
    } else if (backend_name == "inproc") {
      owned_sessions = make_inproc_sessions(config);
    } else if (backend_name == "udp") {
      if (udp_endpoints.empty())
        throw std::runtime_error("--backend=udp requires at least one "
                                 "--udp-endpoint=ID:HOST:PORT");
      owned_sessions = make_udp_sessions(udp_endpoints);
    } else {
      throw std::runtime_error("unknown --backend='" + backend_name + "'");
    }
    route_sessions(owned_sessions, router);

    std::vector<std::unique_ptr<syndrome_source>> owned_sources;
    std::unordered_map<std::uint32_t, syndrome_source *> sources;
    for (const auto &[id, path] : source_files) {
      auto src = std::make_unique<static_source>(read_rounds(path));
      sources[id] = src.get();
      owned_sources.push_back(std::move(src));
    }
    for (const auto &[id, spec] : stim_sources) {
      if (sources.count(id))
        throw std::runtime_error("source_id " + std::to_string(id) +
                                 " given by both --source and --stim-source");
      auto src = std::make_unique<stim_memory_source>(spec.params, spec.seed);
      sources[id] = src.get();
      owned_sources.push_back(std::move(src));
    }

    auto sched = parse(read_file(schedule_path), decoder_ids, tick_ns);

    run_params params;
    auto p = plan(sched, router, sources, params);
    auto result = run(std::move(p));

    if (out_path.empty()) {
      write_csv(result, std::cout);
    } else {
      std::ofstream out(out_path);
      write_csv(result, out);
    }
  } catch (const std::exception &e) {
    std::cerr << "playback-emulator: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

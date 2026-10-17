/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file py_playback_emulator.cpp
/// @brief Python binding for the playback emulator. `parse()`/`plan()` and
/// their supporting machinery stay internal C++-only plumbing (see
/// emulator.h) -- the Python surface is just `run()`, returning a `run_result`
/// whose records are plain dicts keyed like the CLI tool's CSV columns.

#include "py_playback_emulator.h"

#include "cudaq.h"
#include "emulator.h"
#include "session.h"
#include "syndrome_source.h"
#include "type_casters.h"
#include "cudaq/qec/code.h"

#include <algorithm>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec::playback {

namespace {

/// `log[offset, offset + count)`, clamped so a record pointing past the log
/// slices empty.
template <typename T>
std::vector<T> slice(const std::vector<T> &log, std::size_t offset,
                     std::size_t count) {
  const auto first = std::min(offset, log.size());
  const auto last = std::min(first + count, log.size());
  return {log.begin() + first, log.begin() + last};
}

/// `log[offset, offset + count)` as a '0'/'1' string, as write_csv() writes it.
std::string bit_string(const std::vector<std::uint8_t> &log, std::size_t offset,
                       std::size_t count) {
  std::string out;
  for (auto bit : slice(log, offset, count))
    out.push_back(bit ? '1' : '0');
  return out;
}

/// One dict per record, keyed like write_csv()'s columns.
nb::list records_as_dicts(const run_result &r) {
  nb::list out;
  for (const auto &rec : r.records) {
    const auto first = rec.request_id_offset, count = rec.request_id_count;
    nb::dict d;
    d["event_index"] = rec.event_index;
    d["decoder_id"] = rec.decoder_id;
    d["op"] = to_string(rec.op);
    d["deadline_ns"] = rec.deadline_ns;
    d["call_ns"] = rec.call_ns;
    d["return_ns"] = rec.return_ns;
    d["status"] = rec.status;
    d["rounds_streamed"] = rec.rounds_streamed;
    d["read_completed"] = rec.read_completed;
    d["syndrome_bits"] =
        bit_string(r.syndrome_log, rec.syndrome_offset, rec.syndrome_count);
    d["correction_bits"] = bit_string(r.correction_log, rec.correction_offset,
                                      rec.correction_count);
    d["correction_mismatch"] = rec.correction_mismatch;
    d["request_ids"] = slice(r.request_id_log, first, count);
    d["dispatched"] = rec.dispatched;
    d["request_dispatch_ns"] = slice(r.request_dispatch_ns_log, first, count);
    d["request_return_ns"] = slice(r.request_return_ns_log, first, count);
    d["request_status"] = slice(r.request_status_log, first, count);
    out.append(std::move(d));
  }
  return out;
}

/// The Python `run_result`: the C++ result (for write_csv()) and its records,
/// converted to dicts once.
struct py_run_result {
  run_result result;
  nb::list records;
};

/// Maps a state-prep spec string to cudaq::qec::operation, a different
/// enum from the already-bound playback::operation of the same name.
cudaq::qec::operation state_prep_from_string(const std::string &name) {
  using cudaq::qec::operation;
  static const std::unordered_map<std::string, operation> kByName{
      {"x", operation::x},
      {"y", operation::y},
      {"z", operation::z},
      {"h", operation::h},
      {"s", operation::s},
      {"cx", operation::cx},
      {"cy", operation::cy},
      {"cz", operation::cz},
      {"stabilizer_round", operation::stabilizer_round},
      {"prep0", operation::prep0},
      {"prep1", operation::prep1},
      {"prepp", operation::prepp},
      {"prepm", operation::prepm},
  };
  auto it = kByName.find(name);
  if (it == kByName.end())
    throw std::invalid_argument("unknown state_prep: \"" + name + "\"");
  return it->second;
}

/// Builds one syndrome_source from a spec dict tagged by "type": "static",
/// "stim_memory", or "cudaq_memory" -- see run()'s docstring for each
/// type's keys.
std::unique_ptr<syndrome_source> make_source(const nb::dict &spec) {
  if (!spec.contains("type"))
    throw std::invalid_argument("source spec missing required \"type\" key");
  auto type = nb::cast<std::string>(spec["type"]);
  // An unknown key is an error, matching the CLI's --stim-source.
  static const std::unordered_map<std::string, std::vector<std::string>> kKeys{
      {"static", {"rounds"}},
      {"stim_memory",
       {"seed", "code", "task", "distance", "rounds",
        "before_measure_flip_probability", "after_clifford_depolarization",
        "before_round_data_depolarization", "after_reset_flip_probability"}},
      {"cudaq_memory", {"code", "state_prep", "max_rounds", "seed", "noise"}}};
  if (auto it = kKeys.find(type); it != kKeys.end())
    for (auto [k, v] : spec) {
      const auto key = nb::cast<std::string>(k);
      if (key != "type" && std::find(it->second.begin(), it->second.end(),
                                     key) == it->second.end())
        throw std::invalid_argument("unknown \"" + type + "\" source key \"" +
                                    key + "\"");
    }
  if (type == "static")
    return std::make_unique<static_source>(
        nb::cast<std::vector<std::vector<std::uint8_t>>>(spec["rounds"]));
  if (type == "stim_memory")
    // heterogeneous_map::get() ignores unrelated keys, so "type"/"seed"
    // riding along in the same dict is harmless.
    return std::make_unique<stim_memory_source>(
        cudaqx::hetMapFromKwargs(nb::cast<nb::kwargs>(spec)),
        spec.contains("seed") ? nb::cast<std::uint64_t>(spec["seed"]) : 1);
  if (type == "cudaq_memory") {
    cudaq::noise_model noise;
    if (spec.contains("noise"))
      noise = nb::cast<const cudaq::noise_model &>(spec["noise"]);
    return std::make_unique<cudaq_memory_source>(
        nb::cast<const code &>(spec["code"]),
        state_prep_from_string(nb::cast<std::string>(spec["state_prep"])),
        nb::cast<std::size_t>(spec["max_rounds"]), std::move(noise),
        nb::cast<std::uint64_t>(spec["seed"]));
  }
  throw std::invalid_argument("unknown source type: \"" + type + "\"");
}

/// cpu_roce_options from a spec dict ("device", "local_ip", "slots", ...).
cpu_roce_options make_cpu_roce_options(const std::optional<nb::dict> &spec) {
  if (!spec)
    throw std::invalid_argument(
        "run: cpu_roce_endpoints= requires cpu_roce_options= ({\"device\", "
        "\"local_ip\", ...})");
  cpu_roce_options opts;
  for (auto [k, v] : *spec) {
    const auto key = nb::cast<std::string>(k);
    if (key == "device")
      opts.device = nb::cast<std::string>(v);
    else if (key == "local_ip")
      opts.local_ip = nb::cast<std::string>(v);
    else if (key == "slots")
      opts.num_slots = nb::cast<std::uint32_t>(v);
    else if (key == "slot_size")
      opts.slot_size = nb::cast<std::uint32_t>(v);
    else if (key == "connect_timeout_ms")
      opts.connect_timeout_ms = nb::cast<std::uint32_t>(v);
    else
      throw std::invalid_argument("run: unknown cpu_roce_options key \"" + key +
                                  "\"");
  }
  return opts;
}

/// Parses, plans, and runs `schedule_text` in one call. Each of `decoders` /
/// `udp_endpoints` / `cpu_roce_endpoints` / `null_decoder_ids` names the
/// decoder_ids that backend serves; at least one must be given and no id may
/// appear in two. `sources` maps a schedule's source_id to a plain spec dict
/// (see `make_source`); a fresh syndrome_source is built from each spec for
/// this run alone.
py_run_result run_schedule(
    const std::string &schedule_text, std::uint64_t tick_ns,
    const std::unordered_map<std::uint32_t, nb::dict> &sources,
    const std::optional<cudaq::qec::decoding::config::multi_decoder_config>
        &decoders,
    const std::optional<std::unordered_map<std::uint64_t, std::string>>
        &udp_endpoints,
    std::uint32_t udp_timeout_ms,
    const std::optional<std::unordered_map<std::uint64_t, std::string>>
        &cpu_roce_endpoints,
    const std::optional<nb::dict> &cpu_roce_options_spec,
    std::uint32_t cpu_roce_timeout_ms,
    const std::optional<std::vector<std::uint64_t>> &null_decoder_ids,
    std::uint64_t lead_in_ns) {
  if (!decoders && !udp_endpoints && !cpu_roce_endpoints && !null_decoder_ids)
    throw std::invalid_argument(
        "run: specify at least one of decoders=, udp_endpoints=, "
        "cpu_roce_endpoints=, or null_decoder_ids= to select the session "
        "backend(s)");

  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
      owned_sessions;
  std::unordered_map<std::uint64_t, session *> router;

  if (decoders)
    adopt_sessions(make_inproc_sessions(*decoders), owned_sessions, router);
  if (udp_endpoints)
    adopt_sessions(make_udp_sessions(*udp_endpoints, udp_timeout_ms),
                   owned_sessions, router);
  if (cpu_roce_endpoints)
    adopt_sessions(
        make_cpu_roce_sessions(*cpu_roce_endpoints,
                               make_cpu_roce_options(cpu_roce_options_spec),
                               cpu_roce_timeout_ms),
        owned_sessions, router);
  if (null_decoder_ids)
    adopt_sessions(make_null_sessions(*null_decoder_ids), owned_sessions,
                   router);

  std::vector<std::uint64_t> known_decoder_ids;
  known_decoder_ids.reserve(router.size());
  for (auto &[id, _] : router)
    known_decoder_ids.push_back(id);

  std::unordered_map<std::uint32_t, std::unique_ptr<syndrome_source>>
      owned_sources;
  std::unordered_map<std::uint32_t, syndrome_source *> source_router;
  for (auto &[id, spec] : sources) {
    owned_sources[id] = make_source(spec);
    source_router[id] = owned_sources[id].get();
  }

  run_params params;
  params.lead_in_ns = lead_in_ns;

  auto sched = parse(schedule_text, known_decoder_ids, tick_ns);
  auto run_plan_ = plan(sched, router, source_router, params);

  // Release the GIL only for run() itself, not the dict-touching code around
  // it.
  run_result result;
  {
    nb::gil_scoped_release release;
    result = run(std::move(run_plan_));
  }
  nb::list records = records_as_dicts(result);
  return {std::move(result), std::move(records)};
}

} // namespace

void bindPlaybackEmulator(nb::module_ &mod) {
  auto m = mod.def_submodule("playback",
                             "Playback emulator: replay a pre-recorded RPC "
                             "schedule against decoders on a precise, "
                             "hardware-independent timing loop.");

  nb::class_<py_run_result>(m, "run_result")
      .def_ro(
          "records", &py_run_result::records,
          "One dict per schedule line, in schedule order, keyed like the "
          "CSV columns and holding the same values, except that per-request "
          "values are lists.")
      .def_prop_ro("warnings",
                   [](const py_run_result &r) { return r.result.warnings; })
      .def_prop_ro("t0_ns",
                   [](const py_run_result &r) { return r.result.t0_ns; })
      .def_prop_ro("tick_ns",
                   [](const py_run_result &r) { return r.result.tick_ns; })
      .def(
          "write_csv",
          [](const py_run_result &r) { return write_csv(r.result); },
          "Serialize this run's records to a CSV string.");

  m.def(
      "run", &run_schedule, nb::arg("schedule"), nb::arg("tick_ns") = 1000,
      nb::arg("sources") = nb::dict(), nb::arg("decoders") = nb::none(),
      nb::arg("udp_endpoints") = nb::none(), nb::arg("udp_timeout_ms") = 200,
      nb::arg("cpu_roce_endpoints") = nb::none(),
      nb::arg("cpu_roce_options") = nb::none(),
      nb::arg("cpu_roce_timeout_ms") = 200,
      nb::arg("null_decoder_ids") = nb::none(),
      nb::arg("lead_in_ns") = 20'000'000,
      "Parse, plan, and run a line-oriented playback schedule. `sources` "
      "maps a schedule's source_id -> a spec dict tagged by \"type\": "
      "\"static\" ({\"rounds\": [[...]]}), \"stim_memory\" ({\"seed\": N (1), "
      "\"code\", \"task\", \"distance\", \"rounds\" (optional), ...Stim "
      "noise knobs}), or \"cudaq_memory\" ({\"code\": a qec.Code, "
      "\"state_prep\": \"prep0\"|..., \"max_rounds\": N, \"seed\": N, "
      "\"noise\": a cudaq.NoiseModel (optional)}). "
      "Each of `decoders` (in-process decoders from a "
      "multi_decoder_config), `udp_endpoints` ({decoder_id: "
      "\"host:port\"}), `cpu_roce_endpoints` (same, plus `cpu_roce_options` "
      "= {\"device\", \"local_ip\", \"slots\" (8), \"slot_size\" (256), "
      "\"connect_timeout_ms\" (5000)}), and `null_decoder_ids` (discards "
      "everything) names the decoder_ids that backend serves; at least one "
      "must be given, and backends may be mixed as long as no decoder_id "
      "appears twice.");
}

} // namespace cudaq::qec::playback

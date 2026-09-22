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
/// emulator.h) -- the Python surface is just `run()`, returning the same
/// `run_result` the CLI tool gets.

#include "py_playback_emulator.h"

#include "cudaq.h"
#include "emulator.h"
#include "session.h"
#include "syndrome_source.h"
#include "type_casters.h"
#include "cudaq/qec/code.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec::playback {

namespace {

/// One event's [first, last) bounds into a log the same size as
/// request_id_log, clamped so a record pointing past the log slices empty.
std::pair<std::size_t, std::size_t> request_slice(const record &rec,
                                                  std::size_t log_size) {
  const auto first = std::min<std::size_t>(rec.request_id_offset, log_size);
  const auto last =
      std::min<std::size_t>(first + rec.request_id_count, log_size);
  return {first, last};
}

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
  if (type == "static")
    return std::make_unique<static_source>(
        nb::cast<std::vector<std::vector<std::uint8_t>>>(spec["rounds"]));
  if (type == "stim_memory")
    // heterogeneous_map::get() ignores unrelated keys, so "type"/"seed"
    // riding along in the same dict is harmless.
    return std::make_unique<stim_memory_source>(
        cudaqx::hetMapFromKwargs(nb::cast<nb::kwargs>(spec)),
        nb::cast<std::uint64_t>(spec["seed"]));
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

/// Parses, plans, and runs `schedule_text` in one call -- the session
/// backend is picked by which one of `decoders` / `udp_endpoints` /
/// `null_decoder_ids` is given (exactly one must be). `sources` maps a
/// schedule's source_id to a plain spec dict (see `make_source`); a fresh
/// syndrome_source is built from each spec for this run alone.
run_result run_schedule(
    const std::string &schedule_text, std::uint64_t tick_ns,
    const std::unordered_map<std::uint32_t, nb::dict> &sources,
    const std::optional<cudaq::qec::decoding::config::multi_decoder_config>
        &decoders,
    const std::optional<std::unordered_map<std::uint64_t, std::string>>
        &udp_endpoints,
    std::uint32_t udp_timeout_ms,
    const std::optional<std::vector<std::uint64_t>> &null_decoder_ids,
    std::uint64_t lead_in_ns) {
  if (int(decoders.has_value()) + int(udp_endpoints.has_value()) +
          int(null_decoder_ids.has_value()) !=
      1)
    throw std::invalid_argument(
        "run: specify exactly one of decoders=, udp_endpoints=, or "
        "null_decoder_ids= to select the session backend");

  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
      owned_sessions;
  std::unordered_map<std::uint64_t, session *> router;

  if (decoders) {
    owned_sessions = make_inproc_sessions(*decoders);
  } else if (udp_endpoints) {
    owned_sessions = make_udp_sessions(*udp_endpoints, udp_timeout_ms);
  } else {
    owned_sessions = make_null_sessions(*null_decoder_ids);
  }
  route_sessions(owned_sessions, router);

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

  // Release the GIL only for run() itself, not the dict-touching setup above.
  nb::gil_scoped_release release;
  return run(std::move(run_plan_));
}

} // namespace

void bindPlaybackEmulator(nb::module_ &mod) {
  auto m = mod.def_submodule("playback",
                             "Playback emulator: replay a pre-recorded RPC "
                             "schedule against decoders on a precise, "
                             "hardware-independent timing loop.");

  nb::enum_<operation>(m, "operation")
      .value("reset", operation::reset)
      .value("stream", operation::stream)
      .value("enqueue_data", operation::enqueue_data)
      .value("get_corrections", operation::get_corrections);

  nb::class_<record>(m, "record")
      .def_ro("event_index", &record::event_index)
      .def_ro("decoder_id", &record::decoder_id)
      .def_ro("op", &record::op)
      .def_ro("dispatched", &record::dispatched,
              "True once the dispatch loop actually reached this event; "
              "false if a hard error aborted the run first, in which case "
              "every other field is left default.")
      .def_ro("deadline_ns", &record::deadline_ns)
      .def_ro("call_ns", &record::call_ns)
      .def_ro("return_ns", &record::return_ns,
              "When this event's last reply/ack landed: the one request's "
              "reply for reset/get_corrections, or the max over a "
              "stream/enqueue_data's rounds.")
      .def_prop_ro(
          "status",
          [](const record &r) {
            // The two status spaces are disjoint by value
            // (RpcStatus 0..6, stream_terminate 100..103), so the
            // value picks the enum -- not the op, since a dry
            // source gives even an enqueue a SOURCE_EXHAUSTED.
            if (r.status == kNoStatus)
              return "NOT_DISPATCHED";
            return r.status >= 100
                       ? to_string(static_cast<stream_terminate>(r.status))
                       : to_string(static_cast<RpcStatus>(r.status));
          },
          "status as a human-readable string: a stream_terminate "
          "name for stream, an RpcStatus name for every other op, "
          "or NOT_DISPATCHED for an event an abort pre-empted.")
      .def_ro("rounds_streamed", &record::rounds_streamed)
      .def_ro("read_completed", &record::read_completed)
      .def_ro("syndrome_offset", &record::syndrome_offset)
      .def_ro("syndrome_count", &record::syndrome_count)
      .def_ro("correction_offset", &record::correction_offset)
      .def_ro("correction_count", &record::correction_count)
      .def_ro("correction_mismatch", &record::correction_mismatch)
      .def_ro("request_id_offset", &record::request_id_offset)
      .def_ro("request_id_count", &record::request_id_count,
              "How many RPCs this event sent: one per round for a stream, one "
              "for every other op, and zero if it sent nothing. Together with "
              "request_id_offset this slices run_result.request_id_log and "
              "the parallel request_dispatch_ns_log/request_return_ns_log/"
              "request_status_log; run_result.request_ids()/request_timings() "
              "do the slicing for you.");

  nb::class_<run_result>(m, "run_result")
      .def_ro("records", &run_result::records)
      .def_ro("syndrome_log", &run_result::syndrome_log)
      .def_ro("correction_log", &run_result::correction_log)
      .def_ro("request_id_log", &run_result::request_id_log)
      .def_ro("request_dispatch_ns_log", &run_result::request_dispatch_ns_log)
      .def_ro("request_return_ns_log", &run_result::request_return_ns_log)
      .def_ro("request_status_log", &run_result::request_status_log)
      .def(
          "request_ids",
          [](const run_result &r, std::size_t event_index) {
            const auto [first, last] = request_slice(r.records.at(event_index),
                                                     r.request_id_log.size());
            return std::vector<std::uint32_t>(r.request_id_log.begin() + first,
                                              r.request_id_log.begin() + last);
          },
          nb::arg("event_index"),
          "That event's slice of request_id_log: every request_id it put on "
          "the wire, in send order, for correlating against a server log.")
      .def(
          "request_timings",
          [](const run_result &r, std::size_t event_index) {
            const auto [first, last] = request_slice(r.records.at(event_index),
                                                     r.request_id_log.size());
            std::vector<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>>
                out;
            out.reserve(last - first);
            for (std::size_t i = first; i < last; ++i)
              out.emplace_back(r.request_id_log[i],
                               r.request_dispatch_ns_log[i],
                               r.request_return_ns_log[i]);
            return out;
          },
          nb::arg("event_index"),
          "That event's requests as (request_id, dispatch_ns, return_ns) "
          "tuples, in send order -- dispatch_ns is when the timing thread "
          "put it on the wire, return_ns is when its reply/ack landed (0 if "
          "never collected).")
      .def_ro("warnings", &run_result::warnings)
      .def_ro("t0_ns", &run_result::t0_ns)
      .def_ro("tick_ns", &run_result::tick_ns)
      .def(
          "write_csv", [](const run_result &r) { return write_csv(r); },
          "Serialize this run's records to a CSV string.");

  m.def("run", &run_schedule, nb::arg("schedule"), nb::arg("tick_ns"),
        nb::arg("sources"), nb::arg("decoders") = nb::none(),
        nb::arg("udp_endpoints") = nb::none(), nb::arg("udp_timeout_ms") = 200,
        nb::arg("null_decoder_ids") = nb::none(),
        nb::arg("lead_in_ns") = 20'000'000,
        "Parse, plan, and run a line-oriented playback schedule. `sources` "
        "maps a schedule's source_id -> a spec dict tagged by \"type\": "
        "\"static\" ({\"rounds\": [[...]]}), \"stim_memory\" ({\"seed\": N, "
        "\"code\", \"task\", \"distance\", \"rounds\" (optional), ...Stim "
        "noise knobs}), or \"cudaq_memory\" ({\"code\": a qec.Code, "
        "\"state_prep\": \"prep0\"|..., \"max_rounds\": N, \"seed\": N, "
        "\"noise\": a cudaq.NoiseModel (optional)}). "
        "Exactly one of `decoders` (in-process decoders from a "
        "multi_decoder_config), `udp_endpoints` ({decoder_id: "
        "\"host:port\"}), or `null_decoder_ids` (discards everything) selects "
        "the "
        "session backend.");
}

} // namespace cudaq::qec::playback

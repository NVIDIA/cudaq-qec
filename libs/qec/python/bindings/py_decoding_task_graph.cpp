/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "py_decoding_task_graph.h"
#include "type_casters.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/decoding_task_types.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec {

void bindDecodingTaskGraph(nb::module_ &mod) {

  // The round shape a syndrome mapping row names.  FINAL is how a caller says
  // the shot ended -- there is no separate flush verb.
  nb::enum_<cudaq::qec::round_type>(mod, "round_type")
      .value("BULK", cudaq::qec::round_type::BULK)
      .value("FINAL", cudaq::qec::round_type::FINAL);

  // -------------------------------------------------------------------------
  // I/O types
  // -------------------------------------------------------------------------

  nb::class_<measurement_results>(
      mod, "MeasurementResults",
      "Raw per-shot measurement record from the QPU.\n\n"
      "bits: list of 0/1 measurement outcomes (length = num_measurements).\n"
      "tag:  round tag identifying the shot.")
      .def(nb::init<>())
      .def_rw("bits", &measurement_results::bits)
      .def_rw("tag", &measurement_results::tag);

  nb::class_<detection_events>(
      mod, "DetectionEvents",
      "Aggregated detection events: s = D * m (mod 2).\n\n"
      "bits: float detector values (0.0 or 1.0), length = num_detectors.")
      .def(nb::init<>())
      .def_rw("bits", &detection_events::bits);

  nb::class_<error_pattern>(
      mod, "ErrorPattern",
      "Predicted error vector from a decoder.\n\n"
      "result:      soft error prediction per fault column.\n"
      "converged:   False if the decoder failed to converge.\n"
      "opt_results: plugin-specific diagnostics, or None.")
      .def(nb::init<>())
      .def_rw("converged", &error_pattern::converged)
      .def_rw("result", &error_pattern::result)
      .def_rw("opt_results", &error_pattern::opt_results);

  nb::class_<logical_outcome>(
      mod, "LogicalOutcome",
      "Final observable correction.\n\n"
      "bits:      one byte per observable (0 or 1).\n"
      "converged: False if the upstream decoder failed to converge.")
      .def(nb::init<>())
      .def_rw("bits", &logical_outcome::bits)
      .def_rw("converged", &logical_outcome::converged);

  nb::enum_<task_graph_streaming_status>(mod, "TaskGraphStreamingStatus")
      .value("PENDING", task_graph_streaming_status::pending)
      .value("AVAILABLE", task_graph_streaming_status::available)
      .value("COMPLETE", task_graph_streaming_status::complete)
      .value("FAILED", task_graph_streaming_status::failed);

  nb::class_<task_graph_output_progress>(mod, "TaskGraphOutputProgress")
      .def_ro("name", &task_graph_output_progress::name)
      .def_ro("value_offset", &task_graph_output_progress::value_offset)
      .def_ro("value_width", &task_graph_output_progress::value_width)
      .def_ro("generation", &task_graph_output_progress::generation)
      .def_ro("coverage_begin_round",
              &task_graph_output_progress::coverage_begin_round)
      .def_ro("coverage_end_round",
              &task_graph_output_progress::coverage_end_round)
      .def_ro("status", &task_graph_output_progress::status);

  nb::class_<task_graph_streaming_result>(mod, "TaskGraphStreamingResult")
      .def_ro("shot_epoch", &task_graph_streaming_result::shot_epoch)
      .def_ro("result_revision", &task_graph_streaming_result::result_revision)
      .def_ro("completed_operations",
              &task_graph_streaming_result::completed_operations)
      .def_ro("generation", &task_graph_streaming_result::generation)
      .def_ro("status", &task_graph_streaming_result::status)
      .def_ro("operation_completed",
              &task_graph_streaming_result::operation_completed)
      .def_ro("error", &task_graph_streaming_result::error)
      .def_ro("outputs", &task_graph_streaming_result::outputs)
      .def_ro("values", &task_graph_streaming_result::values);

  // -------------------------------------------------------------------------
  // Primitive task functions
  // -------------------------------------------------------------------------

  mod.def("apply_d_sparse", &apply_d_sparse, nb::arg("m"), nb::arg("d_sparse"),
          "Aggregate MeasurementResults → DetectionEvents via D_sparse XOR.");

  mod.def("apply_o_sparse", &apply_o_sparse, nb::arg("e"), nb::arg("o_sparse"),
          "Project ErrorPattern → LogicalOutcome via O_sparse.");

  mod.def("xor_logical", &xor_logical, nb::arg("a"), nb::arg("b"),
          "GF(2) XOR of two LogicalOutcomes (L_raw ⊕ L_flip → L_obs).");

  // -------------------------------------------------------------------------
  // TaskGraphNodeConfig
  // -------------------------------------------------------------------------

  // decoder_instance is deliberately not exposed: it is a unique_ptr to a
  // C++-constructed decoder, and `decoder_type` plus `params` names one from
  // the registry instead, which is what a config-driven caller has anyway.
  nb::class_<task_graph_node_config>(
      mod, "TaskGraphNodeConfig",
      "One node of a topology handed to DecodingTaskGraph.from_topology().\n\n"
      "id and kind name the node and its registered kind; the kind decides\n"
      "which of the rest are required. The model comes from exactly one of\n"
      "dem, the d_sparse/o_sparse/h_sparse triple, or stim_dem. The inner\n"
      "decoder is named by decoder_type and configured by params, while\n"
      "node_params configures the node itself (window_size, residual_from).")
      .def(nb::init<>())
      .def_rw("id", &task_graph_node_config::id)
      .def_rw("kind", &task_graph_node_config::kind)
      .def_rw("node_params", &task_graph_node_config::node_params)
      .def_rw("dem", &task_graph_node_config::dem)
      .def_rw("d_sparse", &task_graph_node_config::d_sparse)
      .def_rw("o_sparse", &task_graph_node_config::o_sparse)
      .def_rw("h_sparse", &task_graph_node_config::h_sparse)
      .def_rw("decoder_type", &task_graph_node_config::decoder_type)
      .def_rw("params", &task_graph_node_config::params)
      .def_rw("stim_dem", &task_graph_node_config::stim_dem)
      .def_prop_rw(
          "ensemble",
          [](task_graph_node_config &self) -> task_graph_ensemble_spec * {
            return self.ensemble ? &*self.ensemble : nullptr;
          },
          [](task_graph_node_config &self, task_graph_ensemble_spec &spec) {
            self.ensemble = std::move(spec);
          },
          nb::rv_policy::reference_internal,
          "When set, this entry is an ensemble pattern rather than a node: "
          "kind and the fields above are ignored. Assigning consumes the "
          "spec.");

  // Member node configs are consumed, as from_topology() consumes its nodes.
  nb::class_<task_graph_ensemble_spec>(
      mod, "TaskGraphEnsembleSpec",
      "Member subgraphs raced against each other, one answer kept; set on\n"
      "TaskGraphNodeConfig.ensemble in place of a kind. join_strategy names a\n"
      "selection strategy ('first', 'deadline') and join_params its\n"
      "parameters. add_member() consumes its node configs.")
      .def(nb::init<>())
      .def_rw("join_strategy", &task_graph_ensemble_spec::join_strategy)
      .def_rw("join_params", &task_graph_ensemble_spec::join_params)
      .def(
          "add_member",
          [](task_graph_ensemble_spec &self, nb::list nodes,
             std::vector<task_graph_edge> edges) {
            task_graph_ensemble_spec::member m;
            for (auto item : nodes)
              m.nodes.push_back(
                  std::move(nb::cast<task_graph_node_config &>(item)));
            m.edges = std::move(edges);
            self.members.push_back(std::move(m));
          },
          nb::arg("nodes"), nb::arg("edges"),
          "Append a member: its node configs (consumed) and the edges among "
          "them, naming the ensemble's input 'input' and the member's answer "
          "'output'.")
      .def_prop_ro("num_members", [](const task_graph_ensemble_spec &self) {
        return self.members.size();
      });

  nb::class_<task_graph_options>(
      mod, "TaskGraphOptions", "Construction-time task-graph execution policy.")
      .def(nb::init<>())
      .def_rw("async_streaming", &task_graph_options::async_streaming)
      .def_rw("async_queue_capacity", &task_graph_options::async_queue_capacity)
      .def_rw("async_coordinator_cpu",
              &task_graph_options::async_coordinator_cpu)
      .def_rw("async_spin_pause_iters",
              &task_graph_options::async_spin_pause_iters);

  // -------------------------------------------------------------------------
  // DecodingTaskGraph
  // -------------------------------------------------------------------------

  nb::class_<decoding_task_graph>(
      mod, "DecodingTaskGraph",
      "Composable decode pipeline backed by a general node/edge DAG.\n\n"
      "Factories bound here:\n"
      "  from_spec:     single decoder in a window-accumulation node.\n"
      "  from_topology: any topology of TaskGraphNodeConfig nodes and "
      "edges.\n"
      "  combine:       two independent decoders whose logical outcomes are "
      "XOR'd.\n\n"
      "A topology can also be described in configuration: set\n"
      "decoder_config.task_graph (see task_graph_spec) and let\n"
      "configure_decoders build it.\n\n"
      "One-shot graph execution:\n"
      "  run(m)         -> LogicalOutcome\n"
      "  run_all(m)     -> [LogicalOutcome]  # one per output\n\n"
      "Batch graph execution:\n"
      "  run_batch([m]) -> [LogicalOutcome]\n\n"
      "Streaming interface (per-round accumulation):\n"
      "  enqueue_syndrome(syndrome)       -> bool  # True when window fired\n"
      "  enqueue_syndrome(syndrome, type) -> bool  # fired, or tail committed\n"
      "  get_obs_corrections()            -> bytes # corrections buffer\n"
      "  clear_corrections()\n"
      "  reset_decoder()\n\n"
      "With TaskGraphOptions.async_streaming enabled, BULK enqueue calls\n"
      "return False after copying input; streaming_result() waits for "
      "accepted\n"
      "input and returns its completion state.\n"
      "Tagging the last round round_type.FINAL commits what a guarded window\n"
      "withheld; it is the only way to end a stream.")
      .def_static("from_spec", &decoding_task_graph::from_spec, nb::arg("spec"),
                  nb::arg("decoder_type"),
                  nb::arg("params") = cudaqx::heterogeneous_map{},
                  nb::arg("options") = task_graph_options{},
                  "Build from a DemChunksSpec and a named decoder plugin.\n\n"
                  "When spec.num_rounds is set, builds a fixed-T window.\n"
                  "When absent (streaming), builds a sliding window sized to\n"
                  "the minimum rounds required by the spec's phase structure.")
      .def_static(
          "from_topology",
          [](nb::list nodes, std::vector<task_graph_edge> edges,
             task_graph_options options) {
            // Moved out of, rather than copied: a node config owns a
            // unique_ptr, so the Python objects are consumed the way
            // combine()'s arguments are.
            std::vector<task_graph_node_config> cfgs;
            cfgs.reserve(nodes.size());
            for (auto item : nodes)
              cfgs.push_back(
                  std::move(nb::cast<task_graph_node_config &>(item)));
            return decoding_task_graph::from_topology(
                std::move(cfgs), std::move(edges), std::move(options));
          },
          nb::arg("nodes"), nb::arg("edges") = std::vector<task_graph_edge>{},
          nb::arg("options") = task_graph_options{},
          "Build a graph from TaskGraphNodeConfigs wired by task_graph_edges.\n"
          "The general builder: any registered kind, any number of outputs.\n"
          "Edges may be omitted for a single node, which is then wired from\n"
          "the raw shot to the sink.  The node configs are consumed.\n\n"
          "A graph with more than one output is read with run_all() and\n"
          "output_names(); run() and run_batch() refuse it.")
      .def_static(
          "combine", &decoding_task_graph::combine, nb::arg("left"),
          nb::arg("right"), nb::arg("options") = task_graph_options{},
          "Combine two single-node graphs: each branch decodes independently\n"
          "with its own D matrix, and their logical outcomes are XOR'd.\n"
          "Both input graphs are consumed.\n\n"
          "Branches with incompatible effective firing or commit schedules\n"
          "still form a valid one-shot and batch graph.  Such a graph reports\n"
          "supports_streaming() == False, and enqueue_syndrome() raises\n"
          "RuntimeError.")
      .def("run", &decoding_task_graph::run, nb::arg("m"),
           "Walk the node/edge DAG for one shot → LogicalOutcome.\n"
           "Does not touch the streaming accumulation buffer.\n"
           "Raises on a graph with more than one output; use run_all().")
      .def("run_all", &decoding_task_graph::run_all, nb::arg("m"),
           "Walk the DAG for one shot → one LogicalOutcome per output,\n"
           "in output_names() order.")
      .def("output_names", &decoding_task_graph::output_names,
           nb::rv_policy::reference_internal,
           "One name per graph output, indexing run_all() and naming the\n"
           "segments get_obs_corrections() concatenates.")
      .def("run_batch", &decoding_task_graph::run_batch, nb::arg("shots"),
           "run() for each MeasurementResults in shots.\n"
           "Single-output graphs only; loop run_all() for the rest.")
      .def(
          "enqueue_syndrome",
          [](decoding_task_graph &g, nb::bytes syndrome) {
            const auto *data =
                reinterpret_cast<const uint8_t *>(syndrome.c_str());
            const auto size = syndrome.size();
            nb::gil_scoped_release release;
            return g.enqueue_syndrome(data, size);
          },
          nb::arg("syndrome"),
          "Accumulate syndrome bytes into the window buffer.\n\n"
          "Synchronously, returns True when any window fired. With async\n"
          "streaming enabled, returns False after queueing BULK input; use\n"
          "streaming_result() to determine result readiness.\n"
          "An untagged round is a BULK round: it never ends the stream, so a\n"
          "guarded window's trailing rounds stay withheld.\n"
          "Raises RuntimeError when the graph does not support streaming.")
      .def(
          "enqueue_syndrome",
          [](decoding_task_graph &g, nb::bytes syndrome,
             cudaq::qec::round_type type) {
            const auto *data =
                reinterpret_cast<const uint8_t *>(syndrome.c_str());
            const auto size = syndrome.size();
            nb::gil_scoped_release release;
            return g.enqueue_syndrome(data, size, type);
          },
          nb::arg("syndrome"), nb::arg("type"),
          "Accumulate syndrome bytes, tagged with the round's shape.\n\n"
          "A FINAL round ends the shot, committing whatever a guarded window\n"
          "is still withholding.  Further rounds are dropped with a warning\n"
          "until clear_corrections() or reset_decoder().\n"
          "Raises ValueError for an empty FINAL round, which must carry the\n"
          "shot's final measurements, and RuntimeError when the graph does\n"
          "not support streaming.")
      .def(
          "get_obs_corrections",
          [](const decoding_task_graph &g) {
            std::vector<uint8_t> corrections;
            {
              nb::gil_scoped_release release;
              const uint8_t *data = g.get_obs_corrections();
              corrections.assign(data, data + g.get_num_observables());
            }
            return nb::bytes(reinterpret_cast<const char *>(corrections.data()),
                             corrections.size());
          },
          "Return the host-accessible corrections buffer.\n\n"
          "For an async-streaming graph, waits for records accepted before\n"
          "this call without closing or flushing the session.")
      .def(
          "streaming_result",
          [](const decoding_task_graph &g) {
            task_graph_streaming_result result;
            {
              nb::gil_scoped_release release;
              result = g.streaming_result();
            }
            return result;
          },
          "Snapshot of values and per-output streaming progress. With async "
          "streaming, waits for input accepted before this call. Status and "
          "generation distinguish unavailable outputs from genuine zeros.")
      .def("clear_corrections", &decoding_task_graph::clear_corrections,
           nb::call_guard<nb::gil_scoped_release>(),
           "Consume published corrections while preserving sliding-window "
           "overlap and publication generations.")
      .def("reset_decoder", &decoding_task_graph::reset_decoder,
           nb::call_guard<nb::gil_scoped_release>(),
           "Reset all per-shot state on every node.")
      .def("get_num_msyn_per_decode",
           &decoding_task_graph::get_num_msyn_per_decode,
           "Syndrome bytes consumed per full window.")
      .def("get_num_msyn_per_round",
           &decoding_task_graph::get_num_msyn_per_round,
           "Syndrome bytes in one round, the chunk size enqueue_syndrome "
           "expects.")
      .def("get_num_observables", &decoding_task_graph::get_num_observables,
           "Number of observable correction bytes.")
      .def("supports_streaming", &decoding_task_graph::supports_streaming,
           "Whether incremental streaming execution is supported.")
      .def("streaming_incompatibility_reason",
           &decoding_task_graph::streaming_incompatibility_reason,
           nb::rv_policy::reference_internal,
           "Empty when streaming is supported; otherwise the first "
           "build-time incompatibility.");
}

} // namespace cudaq::qec

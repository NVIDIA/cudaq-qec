/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Python bindings for extended_dem, dem_stitch/dem_close operations, and the
// DEM chunk-to-streaming-decoder utility functions.

#include "py_extended_dem.h"
#include "sparse_matrix_casters.h"
#include "type_casters.h"
#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/extended_dem.h"

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec {

void bindExtendedDem(nb::module_ &mod) {

  // -------------------------------------------------------------------------
  // seam_id / phase_id — compile-time FNV1a hash type
  // -------------------------------------------------------------------------
  nb::class_<seam_id>(mod, "SeamId",
                      "Identifier for a named seam or phase. Constructed from\n"
                      "a string name; two SeamIds are equal iff constructed\n"
                      "from the same string.")
      .def(nb::init<>())
      .def(nb::init<const char *>(), nb::arg("name"))
      .def_ro("value", &seam_id::value, "FNV1a-32 hash of the name string.")
      .def("__eq__", &seam_id::operator==)
      .def("__ne__", &seam_id::operator!=)
      .def("__lt__", &seam_id::operator<)
      .def("__hash__",
           [](seam_id s) { return static_cast<std::size_t>(s.value); })
      .def("__repr__",
           [](seam_id s) { return "SeamId(" + std::to_string(s.value) + ")"; });

  // Standard seam name constants
  nb::module_ seam_name_mod = mod.def_submodule(
      "seam_name", "Standard seam names for memory experiments.\n\n"
                   "  prev_round  — incoming syndrome seam\n"
                   "  next_round  — outgoing syndrome seam");
  seam_name_mod.attr("prev_round") = seam_name::prev_round;
  seam_name_mod.attr("next_round") = seam_name::next_round;

  nb::module_ phase_name_mod = mod.def_submodule(
      "phase_name", "Standard phase names for memory experiments:\n"
                    "  init, bulk, final");
  phase_name_mod.attr("init") = phase_name::dem_init;
  phase_name_mod.attr("bulk") = phase_name::dem_bulk;
  phase_name_mod.attr("final") = phase_name::dem_final;

  // -------------------------------------------------------------------------
  // extended_dem — mostly opaque; expose inspection helpers only
  // -------------------------------------------------------------------------
  // Bind the inner seam descriptor first
  nb::class_<extended_dem::seam>(mod, "ExtendedDemSeam",
                                 "Named seam descriptor: row range in H.")
      .def_ro("id", &extended_dem::seam::id, "SeamId identifying this seam.")
      .def_ro("row_begin", &extended_dem::seam::row_begin,
              "First row (inclusive) of this seam band in H.")
      .def_ro("row_end", &extended_dem::seam::row_end,
              "One past the last row of this seam band in H.")
      .def("num_rows", &extended_dem::seam::num_rows,
           "Number of rows in this seam band.");

  nb::class_<extended_dem>(
      mod, "ExtendedDem",
      "One chunk of a DEM with named seam boundaries.\n\n"
      "H holds all detector rows (seam bands + interior rows).\n"
      "O holds the observable-flip rows. error_rates has one entry\n"
      "per fault column.\n\n"
      "Users typically work through dem_chunks_from_spec() and\n"
      "dem_close_all() without inspecting fields directly.")
      .def(nb::init<>())
      .def("num_faults", &extended_dem::num_faults, "Number of fault columns.")
      .def("num_rows", &extended_dem::num_rows,
           "Total detector rows in H (seam + interior).")
      .def("num_observables", &extended_dem::num_observables,
           "Observable rows.")
      .def("num_interior_rows", &extended_dem::num_interior_rows,
           "Rows of H not covered by any seam.")
      .def("has_seam", &extended_dem::has_seam, nb::arg("id"),
           "True if a seam with the given SeamId exists.")
      .def("get_seam",
           nb::overload_cast<seam_id>(&extended_dem::get_seam, nb::const_),
           nb::arg("id"), "Return the seam descriptor for the given SeamId.")
      // The returned array owns its buffer, so the reference_internal policy
      // that properties default to does not apply.
      .def_prop_ro(
          "H",
          [](const extended_dem &self) {
            return sparse_binary_matrix_to_numpy(self.H);
          },
          nb::rv_policy::move,
          "All detector rows as a dense uint8 NumPy array [n_rows × n_faults].")
      .def_prop_ro(
          "O",
          [](const extended_dem &self) {
            return sparse_binary_matrix_to_numpy(self.O);
          },
          nb::rv_policy::move,
          "Observable rows as a dense uint8 NumPy array [k × n_faults].")
      .def_rw("error_rates", &extended_dem::error_rates,
              "Error rates, one per fault column.")
      .def_ro("seams", &extended_dem::seams, "List of named seam descriptors.")
      .def_rw("tags", &extended_dem::tags,
              "Tags per seam row (flat, in seams[] order).\n\n"
              "dem_stitch compares tags across a contracted boundary, but "
              "every builder here numbers them positionally as 0..width-1 per "
              "seam, so the comparison holds automatically once the widths "
              "match and adds no safety of its own. Assign your own "
              "identifiers here if you need the boundary check to verify that "
              "both sides name the same physical check.")
      .def("validate", &extended_dem::validate, nb::arg("context") = "dem",
           "Raise ValueError unless the chunk is internally consistent.");

  // -------------------------------------------------------------------------
  // extended_dem_from_css_matrices
  // -------------------------------------------------------------------------
  mod.def("extended_dem_from_css_matrices", &extended_dem_from_css_matrices,
          nb::arg("code"), nb::arg("noise"),
          "Build a one-round ExtendedDem from CSS generator matrices.\n\n"
          "H has seam bands for prev_round and next_round (identical for a\n"
          "one-round chunk). O holds the observable-flip rows.\n\n"
          "Args:\n"
          "    code:  CssCodes instance.\n"
          "    noise: CssNoise instance.\n"
          "Returns:\n"
          "    One-round ExtendedDem.");

  // -------------------------------------------------------------------------
  // dem_seam_spec / seam_spec_entry / dem_chunk_spec
  // -------------------------------------------------------------------------
  nb::class_<dem_seam_spec>(mod, "DemSeamSpec",
                            "Per-seam sparse spec for one named seam boundary.")
      .def(nb::init<>())
      .def_rw("H_sparse", &dem_seam_spec::H_sparse,
              "Syndrome rows, -1 terminated.")
      .def_rw("O_sparse", &dem_seam_spec::O_sparse,
              "Logical-frame rows, -1 terminated (non-memory).");

  nb::class_<seam_spec_entry>(mod, "SeamSpecEntry",
                              "A (SeamId, DemSeamSpec) pair.")
      .def(nb::init<>())
      .def_rw("id", &seam_spec_entry::id, "SeamId of this seam.")
      .def_rw("spec", &seam_spec_entry::spec, "DemSeamSpec for this seam.");

  nb::class_<dem_chunk_spec>(
      mod, "DemChunkSpec",
      "One DEM phase/chunk as flat index lists.\n\n"
      "Two mutually exclusive forms: H_sparse at chunk level (memory\n"
      "experiments, all seams share the same H rows), or seam_specs with\n"
      "per-seam H_sparse/O_sparse for non-memory circuits.\n"
      "O_sparse at chunk level → ExtendedDem.O (decoder observable).")
      .def(nb::init<>())
      .def_rw("num_faults", &dem_chunk_spec::num_faults,
              "Fault columns; all matrices have this width.")
      .def_rw("H_sparse", &dem_chunk_spec::H_sparse,
              "Shared H rows for all seams, -1 terminated (memory shorthand).")
      .def_rw("seam_specs", &dem_chunk_spec::seam_specs,
              "Per-seam specs as a list of SeamSpecEntry (non-memory form).")
      .def_rw("O_sparse", &dem_chunk_spec::O_sparse,
              "Observable rows (-1 terminated).")
      .def_rw("error_rates", &dem_chunk_spec::error_rates,
              "One prior per fault column.")
      .def("is_empty", &dem_chunk_spec::is_empty,
           "True iff no field has been set.")
      .def("expand", &dem_chunk_spec::expand, nb::arg("ids"),
           "Expand H_sparse shorthand into seam_specs for the given ids.")
      .def("validate", &dem_chunk_spec::validate, nb::arg("context"),
           "Raise ValueError on the first inconsistency.");

  mod.def(
      "dem_chunk_from_spec", &dem_chunk_from_spec, nb::arg("spec"),
      nb::arg("seam_names") = std::vector<seam_id>{},
      nb::arg("context") = "dem_chunk",
      "Build one ExtendedDem from a DemChunkSpec.\n\n"
      "Args:\n"
      "    spec:       DemChunkSpec to materialize.\n"
      "    seam_names: SeamIds for shorthand expansion (ignored if\n"
      "                seam_specs is already populated).\n"
      "    context:    Prefix for any error message.\n"
      "Returns:\n"
      "    ExtendedDem whose seam rows are tagged positionally, 0..width-1\n"
      "    within each seam. See ExtendedDem.tags: positional tags make\n"
      "    the seam-tag check in dem_stitch equivalent to its width\n"
      "    check.");

  // -------------------------------------------------------------------------
  // seam_connection / phase_connection / phase_spec_entry / dem_chunks_spec
  // -------------------------------------------------------------------------
  nb::class_<seam_connection>(
      mod, "SeamConnection",
      "Which seam of one phase contracts against which seam of the next.")
      .def(nb::init<>())
      .def_rw("from_seam", &seam_connection::from_seam,
              "Seam contracting forward out of a phase.")
      .def_rw("to_seam", &seam_connection::to_seam,
              "Seam contracting backward into the next phase.");

  nb::class_<phase_connection>(mod, "PhaseConnection",
                               "Directed edge in the phase graph.")
      .def(nb::init<>())
      .def_rw("from_phase", &phase_connection::from_phase)
      .def_rw("to_phase", &phase_connection::to_phase)
      .def("is_self", &phase_connection::is_self,
           "True when from_phase == to_phase (marks the repeating phase).");

  nb::class_<phase_spec_entry>(mod, "PhaseSpecEntry",
                               "A (PhaseId, DemChunkSpec) pair.")
      .def(nb::init<>())
      .def_rw("id", &phase_spec_entry::id, "PhaseId of this phase.")
      .def_rw("spec", &phase_spec_entry::spec, "DemChunkSpec for this phase.");

  nb::class_<dem_chunks_spec>(
      mod, "DemChunksSpec",
      "Multi-phase DEM decomposition with arbitrary named phases.\n\n"
      "Standard memory experiment (linear chain init → bulk×N → final):\n"
      "  connections = [{init,bulk}, {bulk,bulk}, {bulk,final}]\n"
      "  seam        = {next_round, prev_round}\n"
      "  num_rounds  = T   (None for streaming — round count unknown at config "
      "time)\n\n"
      "dem_chunks_from_spec() calls phase_sequence() to materialise the full "
      "chunk\n"
      "list. num_rounds is required for that call when a self-loop is "
      "present;\n"
      "for streaming leave it as None and process bulk chunks as they arrive.")
      .def(nb::init<>())
      .def_rw("phases", &dem_chunks_spec::phases, "List of PhaseSpecEntry.")
      .def_rw("connections", &dem_chunks_spec::connections,
              "List of PhaseConnection (phase graph edges).")
      .def_rw("seam", &dem_chunks_spec::seam,
              "SeamConnection used between all adjacent phases.")
      .def_rw("num_rounds", &dem_chunks_spec::num_rounds,
              "Total rounds to materialise (None for streaming).\n"
              "Required by phase_sequence() when a self-loop is present.")
      .def("is_empty", &dem_chunks_spec::is_empty,
           "True when no phases or connections have been set.")
      .def("has_repeating_phase", &dem_chunks_spec::has_repeating_phase,
           "True when any connection is a self-loop.")
      .def("repeating_phase", &dem_chunks_spec::repeating_phase,
           "Return the PhaseId of the self-connected phase.")
      .def("phase_sequence", &dem_chunks_spec::phase_sequence,
           "Expand connections into an ordered list of PhaseIds.\n"
           "Requires num_rounds when a self-loop is present.")
      .def("get_phase", &dem_chunks_spec::get_phase, nb::arg("id"),
           "Return the DemChunkSpec for the phase with the given PhaseId.\n"
           "Raises ValueError if no such phase exists.")
      .def("validate", &dem_chunks_spec::validate,
           "Raise ValueError if the spec is inconsistent.");

  mod.def("dem_chunks_from_spec", &dem_chunks_from_spec, nb::arg("spec"),
          "Expand a DemChunksSpec into a sequence of ExtendedDem chunks.\n\n"
          "Args:\n"
          "    spec: DemChunksSpec describing the phases and connections.\n"
          "Returns:\n"
          "    List of ExtendedDem in phase_sequence() order.");

  // -------------------------------------------------------------------------
  // prior_combine_mode
  // -------------------------------------------------------------------------
  nb::enum_<prior_combine_mode>(mod, "PriorCombineMode",
                                "Strategy for dem_merge_duplicate_columns().")
      .value("or_combine", prior_combine_mode::or_combine,
             "p = 1/2*(1-prod(1-2p_i)): GF(2)/XOR merge. Use for physical "
             "fault mechanisms.")
      .value("sum_combine", prior_combine_mode::sum_combine,
             "p = min(1, sum(p_i)): linear approximation for small priors.");

  // -------------------------------------------------------------------------
  // dem_merge_duplicate_columns / are/assert_dem_columns_unique
  // -------------------------------------------------------------------------
  mod.def("dem_merge_duplicate_columns", &dem_merge_duplicate_columns,
          nb::arg("dem"), nb::arg("mode") = prior_combine_mode::or_combine,
          "Merge fault columns with identical row support.\n\n"
          "Args:\n"
          "    dem:  ExtendedDem to process.\n"
          "    mode: PriorCombineMode.\n"
          "Returns:\n"
          "    New ExtendedDem with unique-support columns.");

  mod.def("are_dem_columns_unique", &are_dem_columns_unique, nb::arg("dem"),
          "Return True iff every fault column has a unique row-support set.");

  mod.def("assert_dem_columns_unique", &assert_dem_columns_unique,
          nb::arg("dem"),
          "Raise ValueError if any two columns share row support.");

  mod.def("dem_stitch_merged", &dem_stitch_merged, nb::arg("dem_chunks"),
          nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          nb::arg("mode") = prior_combine_mode::or_combine,
          "Stitch chunks then merge duplicate columns.\n\n"
          "Args:\n"
          "    dem_chunks: Non-empty list of ExtendedDem.\n"
          "    from_seam:  SeamId contracting forward (default: next_round).\n"
          "    to_seam:    SeamId contracting backward (default: prev_round).\n"
          "    mode:       PriorCombineMode for the merge step.");

  // -------------------------------------------------------------------------
  // dem_stitch / dem_stitch_all
  // -------------------------------------------------------------------------
  mod.def("dem_stitch", &dem_stitch, nb::arg("a"), nb::arg("b"),
          nb::arg("from_seam"), nb::arg("to_seam"),
          "Stitch two adjacent DEM chunks.\n\n"
          "Contracts a.seams[from_seam] with b.seams[to_seam].\n\n"
          "Args:\n"
          "    a:          Left ExtendedDem.\n"
          "    b:          Right ExtendedDem.\n"
          "    from_seam:  SeamId of A's outgoing seam.\n"
          "    to_seam:    SeamId of B's incoming seam.\n"
          "Returns:\n"
          "    Stitched ExtendedDem.");

  mod.def("dem_stitch_all", &dem_stitch_all, nb::arg("dem_chunks"),
          nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          "Stitch a list of adjacent DEM chunks left-to-right.\n\n"
          "Args:\n"
          "    dem_chunks: Non-empty list of ExtendedDem.\n"
          "    from_seam:  SeamId contracting forward (default: next_round).\n"
          "    to_seam:    SeamId contracting backward (default: prev_round).\n"
          "Returns:\n"
          "    Fully-stitched ExtendedDem.");

  // -------------------------------------------------------------------------
  // dem_close / dem_close_all
  // -------------------------------------------------------------------------
  mod.def(
      "dem_close", &dem_close, nb::arg("dem"),
      nb::arg("to_seam") = seam_name::prev_round,
      nb::arg("from_seam") = seam_name::next_round,
      "Collapse an ExtendedDem into a flat detector_error_model.\n\n"
      "to_seam rows appear first in the output (detector[0] = syndrome[0]\n"
      "vs. zero initial state). The from_seam rows are dropped. A chunk\n"
      "carrying any other non-empty seam is rejected.\n\n"
      "Args:\n"
      "    dem:       ExtendedDem to close.\n"
      "    to_seam:   SeamId whose rows appear first (default: prev_round).\n"
      "    from_seam: SeamId whose rows are dropped (default: next_round).");

  mod.def("dem_close_all", &dem_close_all, nb::arg("dem_chunks"),
          nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          "Build a flat detector_error_model from T chunks in O(T) time.\n\n"
          "Args:\n"
          "    dem_chunks: Non-empty list of ExtendedDem.\n"
          "    from_seam:  SeamId contracting forward (default: next_round).\n"
          "    to_seam:    SeamId contracting backward (default: prev_round).\n"
          "Returns:\n"
          "    detector_error_model ready for any decoder.");

  // -------------------------------------------------------------------------
  // Streaming decoder utilities
  // -------------------------------------------------------------------------
  mod.def("dem_chunk_rounds", &dem_chunk_rounds, nb::arg("dem_chunk"),
          nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          "Return how many measurement rounds a DEM chunk spans.");

  mod.def("dem_chunks_to_rounds", &dem_chunks_to_rounds, nb::arg("dem_chunks"),
          nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          "Return the total rounds a list of DEM chunks describes.");

  mod.def("dem_chunks_to_detector_round", &dem_chunks_to_detector_round,
          nb::arg("dem_chunks"), nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          "Return the round index for each detector in dem_close_all(chunks).");

  mod.def("dem_chunks_to_d_sparse", &dem_chunks_to_d_sparse,
          nb::arg("dem_chunks"), nb::arg("from_seam") = seam_name::next_round,
          nb::arg("to_seam") = seam_name::prev_round,
          "Return the D_sparse measurement-to-detector map (memory exp only).\n"
          "d_sparse[det_id] lists raw measurement bit positions that\n"
          "XOR-combine to fire that detector.");

  mod.def("dem_chunks_to_o_sparse", &dem_chunks_to_o_sparse,
          nb::arg("dem_chunks"),
          "Return the O_sparse observable-flip map from T DEM chunks.\n"
          "o_sparse[obs_id] lists global fault column indices that flip\n"
          "observable obs_id.");

  // sparse_binary_matrix has no Python class, so hand back a scipy CSC
  // matrix rather than an unconvertible C++ object.
  mod.def(
      "dem_chunks_to_pcm",
      [](const std::vector<extended_dem> &dem_chunks, seam_id from_seam,
         seam_id to_seam) {
        return sparse_binary_matrix_to_scipy_csc(
            dem_chunks_to_pcm(dem_chunks, from_seam, to_seam));
      },
      nb::arg("dem_chunks"), nb::arg("from_seam") = seam_name::next_round,
      nb::arg("to_seam") = seam_name::prev_round,
      "Build a canonicalized parity-check matrix from T DEM chunks.\n\n"
      "Returns a scipy.sparse.csc_matrix, so scipy must be installed.");
}

} // namespace cudaq::qec

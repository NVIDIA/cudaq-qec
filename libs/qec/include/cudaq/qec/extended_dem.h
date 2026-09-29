/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One chunk of a detector error model (extended_dem) with named seam
// boundaries, plus the operations that compose chunks into a flat
// detector_error_model.
//
// Build:
//   extended_dem_from_css_matrices()        — one-round chunk from CSS matrices
//   dem_chunk_from_spec(spec, seam_names)   — from a DemChunkSpec
//   dem_chunks_from_spec(spec)              — expand a full DemChunksSpec
//
// Compose:
//   dem_stitch(a, b, from, to)     — contract two adjacent chunks
//   dem_stitch_all(chunks, ...)    — left-fold stitch over a list
//   dem_stitch_merged(chunks, ...) — stitch then merge duplicate columns
//   dem_close(dem, to_seam)        — collapse a single chunk to a flat DEM
//   dem_close_all(chunks, ...)     — O(T) close without quadratic stitching
//
// Merge:
//   dem_merge_duplicate_columns    — collapse identical fault columns
//   are_dem_columns_unique         — check for duplicates
//   assert_dem_columns_unique      — throw if duplicates found
//
// Streaming helpers:
//   dem_chunk_rounds               — how many measurement rounds a chunk spans
//   dem_chunks_to_rounds           — total rounds across a list
//   dem_chunks_to_detector_round   — per-detector round index
//   dem_chunks_to_o_sparse         — observable-flip map
//   dem_chunks_to_pcm              — parity-check matrix
//
// The matching D_sparse map is memory-circuit-specific, so it lives in
// dem_chunks_memory.h (dem_chunks_to_d_sparse) rather than here.
//
// Each chunk holds a single H matrix (all detector rows) plus a list of named
// seam descriptors that identify which row bands participate in stitching.
// Interior rows are the complement — rows of H not covered by any seam.
//
// Standard seam names for memory experiments (defined in seam_name::):
//   prev_round  — incoming syndrome boundary
//   next_round  — outgoing syndrome boundary
//
// Seam and phase names are auto-derived from the name string via a compile-time
// FNV1a hash so the user never assigns an ID manually. A name registry
// (seam_id::register_name / seam_id::name) is available for diagnostics.

#pragma once

#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::qec {

// ---------------------------------------------------------------------------
// seam_id / phase_id — compile-time hash type
// ---------------------------------------------------------------------------

/// @brief Compile-time FNV1a-32 hash of a name string.
constexpr uint32_t fnv1a_32(const char *s, uint32_t h = 2166136261u) noexcept {
  return *s ? fnv1a_32(s + 1, (h ^ static_cast<uint8_t>(*s)) * 16777619u) : h;
}

/// @brief Lightweight identifier for a named seam or phase.
///
/// The value is the FNV1a-32 hash of the name string, computed at compile
/// time. Two seam_ids are equal iff they were constructed from the same name.
/// The same string always produces the same ID, so no registry is required
/// for identity comparison. A name registry is available via register_name()
/// and name() for human-readable diagnostics and error messages.
struct seam_id {
  uint32_t value = 0;

  constexpr seam_id() noexcept = default;
  constexpr explicit seam_id(const char *name) noexcept
      : value(fnv1a_32(name)) {}

  constexpr bool operator==(seam_id o) const noexcept {
    return value == o.value;
  }
  constexpr bool operator!=(seam_id o) const noexcept {
    return value != o.value;
  }
  constexpr bool operator<(seam_id o) const noexcept { return value < o.value; }

  /// Register a display name for diagnostics. Same name is a no-op; a
  /// different name for the same id throws (FNV1a-32 collision).
  static void register_name(seam_id id, std::string_view name);

  /// Return the registered name, or "seam:<hex>" if not registered.
  std::string name() const;
};

/// @brief Same type as seam_id; separate alias for readability at call sites
/// that refer to phase identities rather than seam identities.
using phase_id = seam_id;

// ---------------------------------------------------------------------------
// Standard names
// ---------------------------------------------------------------------------

namespace seam_name {
/// Incoming syndrome boundary.
constexpr seam_id prev_round{"prev_round"};
/// Outgoing syndrome boundary.
constexpr seam_id next_round{"next_round"};
///@cond INTERNAL
static_assert(prev_round != next_round,
              "seam_id hash collision between prev_round and next_round");
///@endcond
} // namespace seam_name

namespace phase_name {
/// First round; string value "init". C++ identifier prefixed to avoid clash
/// with context-sensitive keywords.
constexpr phase_id dem_init{"init"};
/// Repeated middle round; string value "bulk".
constexpr phase_id dem_bulk{"bulk"};
/// Last round; string value "final".
constexpr phase_id dem_final{"final"};
///@cond INTERNAL
static_assert(dem_init != dem_bulk && dem_bulk != dem_final,
              "phase_id hash collision in standard phase names");
///@endcond
} // namespace phase_name

// ---------------------------------------------------------------------------
// extended_dem
// ---------------------------------------------------------------------------

/// @brief One DEM chunk with named seam boundaries.
///
/// H holds all detector rows (seam bands + interior rows). Each named seam is
/// a contiguous row band within H, identified by a seam_id and
/// [row_begin, row_end). Interior rows are H rows not covered by any seam.
/// O holds the observable-flip rows (decoder output).
///
/// For a one-round CSS chunk built by extended_dem_from_css_matrices():
///   H rows [0, d):   prev_round seam
///   H rows [d, 2d):  next_round seam
///   H rows [2d, ..): interior (empty for a single round)
struct extended_dem {
  sparse_binary_matrix H; ///< All detector rows: seam bands + interior
  sparse_binary_matrix O; ///< [k x n_faults] observable-flip rows
  // Future (non-memory): sparse_binary_matrix L; // logical-frame rows
  // Future (heralded):   sparse_binary_matrix V; // virtual detector rows
  std::vector<double> error_rates; ///< one per fault column

  /// Named seam descriptor: which rows in H form this seam.
  struct seam {
    seam_id id;
    uint32_t row_begin = 0;
    uint32_t row_end = 0;
    uint32_t num_rows() const { return row_end - row_begin; }
  };
  std::vector<seam> seams; ///< named seam row bands

  /// One tag per seam row, stored flat in seams[] order.
  ///
  /// A tag is meant to name the physical check behind a seam row, and
  /// dem_stitch() compares tags across a contracted boundary. Be aware that
  /// both in-tree producers -- extended_dem_from_css_matrices() and
  /// dem_chunk_from_spec() -- number tags positionally as 0..width-1 within
  /// each seam. For chunks built either way the two sides therefore hold
  /// identical sequences whenever their widths agree, so the tag comparison
  /// reduces to the width comparison made just before it and rules out
  /// nothing extra. Notably, two chunks from unrelated codes that happen to
  /// share a seam width will stitch without complaint.
  ///
  /// The comparison only does independent work if a caller overwrites this
  /// field with identifiers of its own; neither the spec form nor its YAML
  /// schema carries tags, so that has to be done on the built chunk.
  std::vector<uint64_t> tags;

  // Accessors

  /// Number of fault columns (H.num_cols()).
  uint32_t num_faults() const;

  /// Total detector rows in H (seam rows + interior rows).
  uint32_t num_rows() const;

  /// Observable rows (O.num_rows()).
  uint32_t num_observables() const;

  /// Rows of H not covered by any seam band.
  uint32_t num_interior_rows() const;

  /// True if a seam with the given id exists.
  bool has_seam(seam_id id) const;

  /// Return the seam descriptor with the given id.
  /// @throws std::out_of_range if no such seam exists.
  const seam &get_seam(seam_id id) const;
  seam &get_seam(seam_id id);

  /// Append a new named seam. Throws if id is already present.
  seam &add_seam(seam_id id, uint32_t row_begin, uint32_t row_end);

  /// @brief Throw std::invalid_argument unless this chunk is internally
  /// consistent.
  ///
  /// Checks:
  ///   - H.num_cols() == error_rates.size()
  ///   - O.num_cols() == error_rates.size() (or O is empty)
  ///   - Each seam's row_begin <= row_end and row_end <= H.num_rows()
  ///   - No two non-empty seam row bands overlap
  ///   - tags.size() == sum of seam row counts
  ///
  /// O is exempt from the column check only when it is default-constructed
  /// (0 x 0), which is the "no observables" convention.
  ///
  /// @param context Prefix for the error message.
  void validate(const char *context) const;
};

// ---------------------------------------------------------------------------
// Build a one-round extended_dem from CSS matrices
// ---------------------------------------------------------------------------

/// @brief Build a one-round extended_dem from CSS matrices and noise.
///
/// Emits one fault column per active fault. The H matrix has:
///   rows [0, d):  prev_round seam (same rows as next_round for a one-round
///   chunk) rows [d, 2d): next_round seam rows [2d, ..): interior (empty for
///   a one-round chunk)
///
/// O holds the observable-flip rows (lz/lx columns). Tags are sequential.
///
/// @param code  CSS code matrices.
/// @param noise Depolarizing noise rates.
/// @return      One-round extended_dem.
/// @throws std::invalid_argument on dimension or per-qubit size mismatch.
extended_dem extended_dem_from_css_matrices(const css_code_matrices &code,
                                            const css_noise_params &noise);

// ---------------------------------------------------------------------------
// Declarative chunk specs (the form carried in decoder configuration YAML)
// ---------------------------------------------------------------------------

/// @brief Per-seam sparse spec: rows for one named seam boundary.
///
/// H_sparse and O_sparse use the -1-terminated row encoding:
/// [0, 1, 5, -1, 1, 2, 6, -1] describes two rows.
struct dem_seam_spec {
  std::vector<std::int64_t> H_sparse; ///< -1-terminated syndrome rows
  std::vector<std::int64_t> O_sparse; ///< -1-terminated logical-frame rows
  bool operator==(const dem_seam_spec &) const = default;
};

struct seam_spec_entry {
  seam_id id;
  dem_seam_spec spec;
  bool operator==(const seam_spec_entry &) const = default;
};

/// @brief One DEM phase/chunk as flat index lists.
///
/// Two mutually exclusive forms:
///
///   Shorthand: H_sparse at chunk level. The expander assigns this single H
///   to every seam named in the connection (all seams share the same H rows).
///   Valid for CSS memory experiments where prev_round.H == next_round.H.
///
///   Full per-seam: seam_specs populated directly with per-seam H_sparse and
///   O_sparse. Required when seams carry different H rows (non-memory gadgets).
///
/// O_sparse at chunk level maps to extended_dem::O (the decoder observable).
struct dem_chunk_spec {
  uint64_t num_faults = 0;

  /// Shorthand: shared H for all seams. Mutually exclusive with seam_specs.
  std::vector<std::int64_t> H_sparse;

  /// Full per-seam form. Mutually exclusive with H_sparse.
  std::vector<seam_spec_entry> seam_specs;

  /// Chunk-level observable rows → extended_dem::O.
  std::vector<std::int64_t> O_sparse;

  std::vector<double> error_rates;

  /// True when nothing has been set.
  bool is_empty() const;
  bool operator==(const dem_chunk_spec &) const = default;

  /// @brief Expand H_sparse shorthand into seam_specs using the provided IDs.
  ///
  /// No-op when seam_specs is already populated. The shorthand H_sparse is
  /// assigned to every seam in ids with an empty O_sparse.
  ///
  /// @param ids Seam IDs to create entries for (e.g. {next_round, prev_round})
  void expand(const std::vector<seam_id> &ids);

  /// @brief Check internal consistency.
  ///
  /// Validates num_faults, error_rates, mutual exclusion of H_sparse /
  /// seam_specs, and all H_sparse / O_sparse index lists.
  ///
  /// @param context Prefix for error messages.
  /// @throws std::invalid_argument on the first violation.
  void validate(const std::string &context) const;
};

/// @brief Build an extended_dem from a dem_chunk_spec.
///
/// Calls spec.expand(seam_names) to resolve the shorthand form if needed,
/// then materializes H (all seam bands stacked, interior empty) and O.
/// Tags are numbered positionally within each seam, 0..width-1, which leaves
/// dem_stitch()'s tag check equivalent to its width check; see
/// extended_dem::tags.
///
/// @param spec       Chunk to materialize.
/// @param seam_names Seam IDs for expand(); ignored when seam_specs is set.
/// @param context    Prefix for error messages.
/// @return           The chunk as an extended_dem.
/// @throws std::invalid_argument if the spec is inconsistent.
extended_dem dem_chunk_from_spec(const dem_chunk_spec &spec,
                                 const std::vector<seam_id> &seam_names = {},
                                 const std::string &context = "dem_chunk");

// ---------------------------------------------------------------------------
// Phase / connection specs for multi-phase decompositions
// ---------------------------------------------------------------------------

/// @brief Which seam of one phase contracts against which seam of the next.
struct seam_connection {
  seam_id from_seam; ///< seam contracting forward out of a phase
  seam_id to_seam;   ///< seam contracting backward into the next phase
  bool operator==(const seam_connection &) const = default;
};

/// @brief Directed edge in the phase graph.
///
/// A self-connection (from_phase == to_phase) marks the repeating phase in a
/// linear chain.
struct phase_connection {
  phase_id from_phase;
  phase_id to_phase;
  bool is_self() const { return from_phase == to_phase; }
  bool operator==(const phase_connection &) const = default;
};

struct phase_spec_entry {
  phase_id id;
  dem_chunk_spec spec;
  bool operator==(const phase_spec_entry &) const = default;
};

/// @brief Multi-phase DEM decomposition with arbitrary named phases.
///
/// Standard memory experiment (linear chain A → B×N → C):
///   connections = [{init,bulk}, {bulk,bulk}, {bulk,final}]
///   seam        = {next_round, prev_round}
///   num_rounds  = T   (omit for streaming — num_rounds unknown at config time)
///
/// dem_chunks_from_spec() calls phase_sequence() to materialise the full
/// ordered chunk list. num_rounds is only needed for that call:
/// phase_sequence() throws when a self-loop is present and num_rounds is
/// absent. For a linear chain (no self-loop) phase_sequence() derives the
/// length from the connections.
struct dem_chunks_spec {
  std::vector<phase_spec_entry> phases;
  std::vector<phase_connection> connections;
  seam_connection seam;
  std::optional<uint64_t> num_rounds;

  /// True when no phases or connections have been set.
  bool is_empty() const;
  bool operator==(const dem_chunks_spec &) const = default;

  /// True when any connection is a self-loop.
  bool has_repeating_phase() const;

  /// Return the phase_id of the self-connected (repeating) phase.
  /// @throws std::invalid_argument if there is no or more than one self-loop.
  phase_id repeating_phase() const;

  /// @brief Expand connections + num_rounds into an ordered phase_id list.
  ///
  /// With a self-loop (repeating phase) and num_rounds = T:
  ///   result = [A, B, B, ...(T-2 times)..., B, C]
  /// Without a self-loop, the sequence length is derived from the connections
  /// and num_rounds is not consulted.
  ///
  /// @throws std::invalid_argument if a self-loop is present and num_rounds
  ///         is absent, or if the connection graph is not a valid linear
  ///         chain: a phase with more than one non-self outgoing edge, or a
  ///         cycle among the non-self edges, is rejected.
  std::vector<phase_id> phase_sequence() const;

  /// Return the spec for the phase with the given id.
  /// @throws std::invalid_argument if no such phase exists.
  const dem_chunk_spec &get_phase(phase_id id) const;

  /// @brief Validate phase specs, connection graph, and seam widths.
  /// @throws std::invalid_argument on the first violation.
  void validate() const;
};

/// @brief Expand a dem_chunks_spec into a sequence of extended_dem chunks.
///
/// Calls phase_sequence() to determine the ordered phase list, then calls
/// dem_chunk_from_spec() for each phase with the seam names from spec.seam.
///
/// @param spec Validated phase/connection spec.
/// @return     Sequence of chunks ready to stitch or close.
/// @throws std::invalid_argument if the spec is inconsistent.
std::vector<extended_dem> dem_chunks_from_spec(const dem_chunks_spec &spec);

// ---------------------------------------------------------------------------
// Stitch operations
// ---------------------------------------------------------------------------

/// @brief Stitch two adjacent DEM chunks: contract a.seams[from] with
/// b.seams[to].
///
/// The contracted seam rows become new interior rows in the result. Row
/// ordering in the result H:
///   a.seams[to_seam] rows          (only A fault columns)
///   a interior rows                (only A fault columns)
///   contracted seam rows           (both A and B fault columns)
///   b interior rows                (only B fault columns)
///   b.seams[from_seam] rows        (only B fault columns)
///
/// Only from_seam and to_seam may carry rows. A chunk holding any other
/// non-empty named seam is rejected, because such a seam has no counterpart
/// on either side and no defined position in the result.
///
/// @param a         Left DEM chunk.
/// @param b         Right DEM chunk.
/// @param from_seam Seam in A contracting forward.
/// @param to_seam   Seam in B contracting backward.
/// @return          Stitched extended_dem.
/// @throws std::invalid_argument on seam width, tag, or observable mismatch,
///         or if either chunk carries an additional non-empty seam.
extended_dem dem_stitch(const extended_dem &a, const extended_dem &b,
                        seam_id from_seam, seam_id to_seam);

/// @brief Stitch a span of adjacent DEM chunks left-to-right.
///
/// Applies the same from_seam/to_seam connection to every adjacent pair.
/// Defaults to the standard memory seam names so existing call sites need
/// no changes.
extended_dem dem_stitch_all(const std::vector<extended_dem> &chunks,
                            seam_id from_seam = seam_name::next_round,
                            seam_id to_seam = seam_name::prev_round);

// ---------------------------------------------------------------------------
// Duplicate fault columns
// ---------------------------------------------------------------------------

/// @brief Prior-combining strategy for dem_merge_duplicate_columns().
enum class prior_combine_mode { or_combine, sum_combine };

/// @brief Merge fault columns with identical row support into single columns.
///
/// Row support is compared across all blocks: H rows in row order, then O
/// rows. Columns are compared over GF(2). The merged prior is computed under
/// the chosen rule.
///
/// @param dem   Input extended_dem (not modified).
/// @param mode  How to combine priors of merged columns.
/// @return      New extended_dem with unique-support columns.
extended_dem dem_merge_duplicate_columns(
    const extended_dem &dem,
    prior_combine_mode mode = prior_combine_mode::or_combine);

/// @brief Return true iff every fault column has a unique row-support set.
bool are_dem_columns_unique(const extended_dem &dem);

/// @brief Throw std::invalid_argument if any two columns share row support.
void assert_dem_columns_unique(const extended_dem &dem);

/// @brief Stitch DEM chunks left-to-right then merge duplicate columns.
extended_dem
dem_stitch_merged(const std::vector<extended_dem> &chunks,
                  seam_id from_seam = seam_name::next_round,
                  seam_id to_seam = seam_name::prev_round,
                  prior_combine_mode mode = prior_combine_mode::or_combine);

// ---------------------------------------------------------------------------
// Close operations
// ---------------------------------------------------------------------------

/// @brief Collapse an extended_dem into a flat detector_error_model.
///
/// Emits rows of H in order:
///   - seams[to_seam] rows first (detector[0] = syndrome[0] vs. zero state)
///   - interior rows next
///   - O as observables_flips_matrix
///
/// seams[from_seam] rows are dropped: the final round has no next round to
/// compare against. A chunk carrying a non-empty seam that is neither
/// to_seam nor from_seam is rejected instead, since dropping it would
/// silently discard detector rows. Both seam names default to the
/// memory-experiment common case.
///
/// @param dem       Chunk to close.
/// @param to_seam   Seam whose rows appear first in the output detectors.
/// @param from_seam Seam whose rows are intentionally dropped.
/// @return          detector_error_model ready for any decoder.
/// @throws std::invalid_argument if dem carries an additional non-empty seam.
detector_error_model dem_close(const extended_dem &dem,
                               seam_id to_seam = seam_name::prev_round,
                               seam_id from_seam = seam_name::next_round);

/// @brief Build a flat detector_error_model from T DEM chunks in O(T) time.
///
/// Equivalent to dem_close(dem_stitch_all(chunks, from, to), to) but avoids
/// the O(T²) left-fold accumulation. Detector rows are always emitted in the
/// same order as a stitched result.
///
/// As with dem_stitch, a chunk carrying a non-empty seam other than from_seam
/// or to_seam is rejected rather than having those rows silently dropped.
detector_error_model dem_close_all(const std::vector<extended_dem> &chunks,
                                   seam_id from_seam = seam_name::next_round,
                                   seam_id to_seam = seam_name::prev_round);

// ---------------------------------------------------------------------------
// Streaming decoder integration utilities
// ---------------------------------------------------------------------------

/// @brief How many rounds one DEM chunk spans.
uint32_t dem_chunk_rounds(const extended_dem &chunk,
                          seam_id from_seam = seam_name::next_round,
                          seam_id to_seam = seam_name::prev_round);

/// @brief Total rounds a sequence of DEM chunks describes.
std::size_t dem_chunks_to_rounds(const std::vector<extended_dem> &chunks,
                                 seam_id from_seam = seam_name::next_round,
                                 seam_id to_seam = seam_name::prev_round);

/// @brief Extract the detector→round mapping from a sequence of DEM chunks.
std::vector<std::int32_t>
dem_chunks_to_detector_round(const std::vector<extended_dem> &chunks,
                             seam_id from_seam = seam_name::next_round,
                             seam_id to_seam = seam_name::prev_round);

/// @brief Extract the O_sparse observable-flip map from T DEM chunks.
///
/// Keyed on fault columns; seam names are not needed.
std::vector<std::vector<uint32_t>>
dem_chunks_to_o_sparse(const std::vector<extended_dem> &chunks);

/// @brief Build a canonicalized CSC parity-check matrix from chunks.
sparse_binary_matrix
dem_chunks_to_pcm(const std::vector<extended_dem> &chunks,
                  seam_id from_seam = seam_name::next_round,
                  seam_id to_seam = seam_name::prev_round);

} // namespace cudaq::qec

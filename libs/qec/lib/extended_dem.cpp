/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Implements extended_dem using a single H matrix with named seam row bands.
//
// Each fault column of a one-round chunk has identical rows in both the
// prev_round and next_round seam bands, because syndrome[r] participates in
// both seam detectors: detector[r] = syndrome[r] XOR syndrome[r-1] (left)
// and detector[r+1] = syndrome[r+1] XOR syndrome[r] (right).
//
// Stitching contracts a.seams[from].H rows against b.seams[to].H rows onto
// the same output row indices, so each fault in A and B independently
// contributes to the seam-detector row.
//
// dem_close() writes seams[to_seam] rows first, then interior rows, then O.

#include "cudaq/qec/extended_dem.h"
#include "dem_construction_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudaq::qec {

using namespace detail;

// ---------------------------------------------------------------------------
// seam_id name registry
// ---------------------------------------------------------------------------

static std::unordered_map<uint32_t, std::string> &seam_name_registry() {
  static std::unordered_map<uint32_t, std::string> reg;
  return reg;
}

// Display name for diagnostics; throws on FNV1a-32 name collision.
void seam_id::register_name(seam_id id, std::string_view name) {
  auto &reg = seam_name_registry();
  char hex[16] = {};
  auto [it, inserted] = reg.emplace(id.value, name);

  if (inserted)
    return;
  if (it->second == name)
    return;

  std::snprintf(hex, sizeof(hex), "%08x", id.value);
  throw std::invalid_argument(
      std::string("seam_id::register_name: FNV1a-32 collision at 0x") + hex +
      " between '" + it->second + "' and '" + std::string(name) + "'");
} // end - register_name()

std::string seam_id::name() const {
  const auto &reg = seam_name_registry();
  auto it = reg.find(value);
  if (it != reg.end())
    return it->second;
  char buf[16];
  std::snprintf(buf, sizeof(buf), "seam:%08x", value);
  return buf;
}

static const bool seam_names_registered = [] {
  seam_id::register_name(seam_name::prev_round, "prev_round");
  seam_id::register_name(seam_name::next_round, "next_round");
  seam_id::register_name(phase_name::dem_init, "init");
  seam_id::register_name(phase_name::dem_bulk, "bulk");
  seam_id::register_name(phase_name::dem_final, "final");
  return true;
}();

// col_list: nested CSC representation (one vector<index_type> per column)
using col_list = std::vector<std::vector<sparse_binary_matrix::index_type>>;

// ---------------------------------------------------------------------------
// extended_dem accessors
// ---------------------------------------------------------------------------

uint32_t extended_dem::num_faults() const {
  return static_cast<uint32_t>(error_rates.size());
}

uint32_t extended_dem::num_rows() const { return H.num_rows(); }

uint32_t extended_dem::num_observables() const { return O.num_rows(); }

uint32_t extended_dem::num_interior_rows() const {
  uint32_t seam_total = 0;
  for (const auto &s : seams)
    seam_total += s.num_rows();
  const uint32_t h_rows = H.num_rows();
  return h_rows > seam_total ? h_rows - seam_total : 0;
}

bool extended_dem::has_seam(seam_id id) const {
  for (const auto &s : seams)
    if (s.id == id)
      return true;
  return false;
}

const extended_dem::seam &extended_dem::get_seam(seam_id id) const {
  for (const auto &s : seams)
    if (s.id == id)
      return s;
  throw std::out_of_range("extended_dem::get_seam: no seam with id " +
                          id.name());
}

extended_dem::seam &extended_dem::get_seam(seam_id id) {
  for (auto &s : seams)
    if (s.id == id)
      return s;
  throw std::out_of_range("extended_dem::get_seam: no seam with id " +
                          id.name());
}

extended_dem::seam &extended_dem::add_seam(seam_id id, uint32_t row_begin,
                                           uint32_t row_end) {
  if (has_seam(id))
    throw std::invalid_argument("extended_dem::add_seam: duplicate seam " +
                                id.name());
  seams.push_back({id, row_begin, row_end});
  return seams.back();
}

void extended_dem::validate(const char *context) const {
  const uint32_t n = num_faults();
  auto chk = [&](const sparse_binary_matrix &m, const char *label) {
    if (m.num_cols() != n)
      throw std::invalid_argument(std::string(context) + ": " + label +
                                  " has " + std::to_string(m.num_cols()) +
                                  " cols but chunk has " + std::to_string(n) +
                                  " faults");
  };
  chk(H, "H");
  // A default-constructed O (0 x 0) is the "no observables" convention. Any
  // other O has to be column-aligned with H, because the merge and o_sparse
  // helpers read it one fault column at a time.
  if (O.num_rows() > 0 || O.num_cols() > 0)
    chk(O, "O");
  const uint32_t n_rows = H.num_rows();
  for (const auto &s : seams) {
    if (s.row_end > n_rows)
      throw std::invalid_argument(std::string(context) + ": seam row_end " +
                                  std::to_string(s.row_end) +
                                  " > H.num_rows() " + std::to_string(n_rows));
    if (s.row_begin > s.row_end)
      throw std::invalid_argument(std::string(context) +
                                  ": seam has row_begin > row_end");
  }
  // Overlapping bands would make num_interior_rows(), which subtracts the
  // summed seam widths, disagree with the row-membership test that actually
  // materializes interior rows, so row counts and row contents would diverge.
  std::vector<std::pair<uint32_t, uint32_t>> bands;
  bands.reserve(seams.size());
  for (const auto &s : seams)
    if (s.num_rows() > 0)
      bands.emplace_back(s.row_begin, s.row_end);
  std::sort(bands.begin(), bands.end());
  for (std::size_t i = 1; i < bands.size(); ++i)
    if (bands[i].first < bands[i - 1].second)
      throw std::invalid_argument(std::string(context) + ": seam row bands [" +
                                  std::to_string(bands[i - 1].first) + "," +
                                  std::to_string(bands[i - 1].second) +
                                  ") and [" + std::to_string(bands[i].first) +
                                  "," + std::to_string(bands[i].second) +
                                  ") overlap");
  uint32_t tag_count = 0;
  for (const auto &s : seams)
    tag_count += s.num_rows();
  if (tags.size() != tag_count)
    throw std::invalid_argument(
        std::string(context) + ": tags.size()=" + std::to_string(tags.size()) +
        " != sum of seam rows " + std::to_string(tag_count));
} // end - extended_dem::validate()

// ---------------------------------------------------------------------------
// Internal matrix helpers
// ---------------------------------------------------------------------------

static sparse_binary_matrix hcat(const sparse_binary_matrix &a,
                                 const sparse_binary_matrix &b) {
  auto cols = a.to_nested_csc();
  const auto b_cols = b.to_nested_csc();
  cols.insert(cols.end(), b_cols.begin(), b_cols.end());
  const auto n_total = static_cast<std::size_t>(a.num_cols()) + b.num_cols();
  if (n_total > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("hcat: combined column count exceeds uint32_t");
  return sparse_binary_matrix::from_nested_csc(
      a.num_rows(), static_cast<uint32_t>(n_total), cols);
}

static sparse_binary_matrix vstack(const sparse_binary_matrix &a,
                                   const sparse_binary_matrix &b) {
  if (b.num_cols() != a.num_cols())
    throw std::invalid_argument("vstack: column count mismatch (" +
                                std::to_string(a.num_cols()) + " vs " +
                                std::to_string(b.num_cols()) + ")");
  const uint32_t n_cols = a.num_cols();
  const uint32_t a_rows = a.num_rows();
  const auto total_rows = static_cast<std::size_t>(a_rows) + b.num_rows();
  if (total_rows > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("vstack: combined row count exceeds uint32_t");
  auto a_csc = a.to_nested_csc();
  const auto b_csc = b.to_nested_csc();
  col_list combined(n_cols);
  for (uint32_t c = 0; c < n_cols; ++c) {
    combined[c] = std::move(a_csc[c]);
    for (auto r : b_csc[c])
      combined[c].push_back(a_rows + r);
  }
  return sparse_binary_matrix::from_nested_csc(
      static_cast<uint32_t>(total_rows), n_cols, combined);
}

static sparse_binary_matrix pad_right(const sparse_binary_matrix &m,
                                      uint32_t n_total) {
  auto cols = m.to_nested_csc();
  cols.resize(n_total);
  return sparse_binary_matrix::from_nested_csc(m.num_rows(), n_total, cols);
}

static sparse_binary_matrix pad_left(const sparse_binary_matrix &m,
                                     uint32_t n_left, uint32_t n_total) {
  const auto src = m.to_nested_csc();
  col_list cols(n_total);
  for (uint32_t c = 0; c < m.num_cols(); ++c)
    cols[n_left + c] = src[c];
  return sparse_binary_matrix::from_nested_csc(m.num_rows(), n_total, cols);
}

static sparse_binary_matrix zero_matrix(uint32_t nrows, uint32_t ncols) {
  return sparse_binary_matrix::from_nested_csc(nrows, ncols, col_list(ncols));
}

static void write_sparse_block(cudaqx::tensor<uint8_t> &dst,
                               const sparse_binary_matrix &src,
                               std::size_t row_offset, std::size_t col_offset) {
  const auto cols = src.to_nested_csc();
  const uint32_t nc = src.num_cols();
  for (uint32_t c = 0; c < nc; ++c)
    for (auto r : cols[c])
      dst.at({row_offset + r, col_offset + c}) ^= 1;
}

// Extract rows [row_begin, row_end) from H as a new sparse matrix.
static sparse_binary_matrix extract_row_band(const sparse_binary_matrix &H,
                                             uint32_t row_begin,
                                             uint32_t row_end) {
  const uint32_t n_rows = row_end - row_begin;
  const uint32_t n_cols = H.num_cols();
  if (n_rows == 0)
    return zero_matrix(0, n_cols);
  const auto csr = H.to_nested_csr();
  col_list selected(n_rows);
  for (uint32_t r = 0; r < n_rows; ++r)
    selected[r] = csr[row_begin + r];
  return sparse_binary_matrix::from_nested_csr(n_rows, n_cols, selected);
}

// Extract all H rows not covered by any seam band.
static sparse_binary_matrix
extract_interior_rows(const sparse_binary_matrix &H,
                      const std::vector<extended_dem::seam> &seams) {
  const uint32_t n_rows = H.num_rows();
  const uint32_t n_cols = H.num_cols();
  const auto csr = H.to_nested_csr();
  col_list interior;
  for (uint32_t r = 0; r < n_rows; ++r) {
    bool in_seam = false;
    for (const auto &s : seams)
      if (r >= s.row_begin && r < s.row_end) {
        in_seam = true;
        break;
      }
    if (!in_seam)
      interior.push_back(csr[r]);
  }
  return sparse_binary_matrix::from_nested_csr(
      static_cast<uint32_t>(interior.size()), n_cols, interior);
}

// Tag offset for a given seam in the flat tags vector.
static uint32_t seam_tag_offset(const extended_dem &dem, seam_id id) {
  uint32_t offset = 0;
  for (const auto &s : dem.seams) {
    if (s.id == id)
      return offset;
    offset += s.num_rows();
  }
  throw std::out_of_range("seam_tag_offset: seam " + id.name() + " not found");
}

// Validate that the outgoing seam of 'a' and the incoming seam of 'b' have
// the same width and identical per-row tags. An absent seam is treated as
// width 0; the check is skipped only when both sides are absent/empty.
//
// The tag loop below is a hook rather than an active check: both in-tree
// producers tag seam rows positionally, so once the widths match the two tag
// sequences are both 0..width-1 and always agree. It bites only for tags a
// caller assigned itself. See extended_dem::tags.
static void check_seam_boundary(const extended_dem &a, const extended_dem &b,
                                seam_id from_seam, seam_id to_seam,
                                const std::string &context) {
  const uint32_t from_w =
      a.has_seam(from_seam) ? a.get_seam(from_seam).num_rows() : 0u;
  const uint32_t to_w =
      b.has_seam(to_seam) ? b.get_seam(to_seam).num_rows() : 0u;
  if (from_w == 0 && to_w == 0)
    return;
  if (from_w != to_w)
    throw std::invalid_argument(context + ": seam width mismatch (" +
                                std::to_string(from_w) + " vs " +
                                std::to_string(to_w) + ")");
  const uint32_t off_a = seam_tag_offset(a, from_seam);
  const uint32_t off_b = seam_tag_offset(b, to_seam);
  for (uint32_t k = 0; k < from_w; ++k)
    if (a.tags[off_a + k] != b.tags[off_b + k])
      throw std::invalid_argument(context + ": tag mismatch at seam row " +
                                  std::to_string(k));
}

// Composition only has defined semantics for the two seams named in the
// connection: their rows either pair up across a boundary or become the
// leading detector band. A third named seam has no counterpart on either
// side, and the row bands it owns are excluded from the interior, so it
// would be dropped from the result without this check.
static void reject_extra_seams(const extended_dem &dem, seam_id from_seam,
                               seam_id to_seam, const std::string &context) {
  for (const auto &s : dem.seams) {
    if (s.id == from_seam || s.id == to_seam || s.num_rows() == 0)
      continue;
    throw std::invalid_argument(
        context + ": seam " + s.id.name() + " carries " +
        std::to_string(s.num_rows()) + " rows but is neither from_seam (" +
        from_seam.name() + ") nor to_seam (" + to_seam.name() +
        "); composing chunks that carry additional named seams is not "
        "supported");
  }
}

// ---------------------------------------------------------------------------
// sparse_from_terminated_rows and validate_index_list
// ---------------------------------------------------------------------------

static sparse_binary_matrix
sparse_from_terminated_rows(const std::vector<std::int64_t> &rows,
                            uint64_t num_faults) {
  std::vector<std::vector<sparse_binary_matrix::index_type>> nested;
  std::vector<sparse_binary_matrix::index_type> current;
  const std::size_t n_rows =
      static_cast<std::size_t>(std::count(rows.begin(), rows.end(), -1));
  nested.reserve(n_rows);
  for (auto value : rows) {
    if (value == -1) {
      nested.push_back(current);
      current.clear();
      continue;
    }
    current.push_back(static_cast<sparse_binary_matrix::index_type>(value));
  }
  if (!current.empty())
    throw std::invalid_argument(
        "sparse_from_terminated_rows: input did not end with -1 terminator");
  return sparse_binary_matrix::from_nested_csr(
             static_cast<sparse_binary_matrix::index_type>(nested.size()),
             static_cast<sparse_binary_matrix::index_type>(num_faults), nested)
      .to_csc();
}

static void validate_index_list(const std::vector<std::int64_t> &rows,
                                uint64_t num_faults, const std::string &context,
                                const std::string &field) {
  if (rows.empty())
    return;
  if (rows.back() != -1)
    throw std::invalid_argument(context + "." + field +
                                " must end with a -1 row terminator");
  std::unordered_set<std::int64_t> row_cols;
  for (auto value : rows) {
    if (value == -1) {
      row_cols.clear();
      continue;
    }
    if (value < 0 || static_cast<uint64_t>(value) >= num_faults)
      throw std::invalid_argument(context + "." + field + " index " +
                                  std::to_string(value) + " out of range");
    if (!row_cols.insert(value).second)
      throw std::invalid_argument(context + "." + field +
                                  " duplicate column index " +
                                  std::to_string(value) + " in a row");
  }
}

// ---------------------------------------------------------------------------
// extended_dem_from_css_matrices
// ---------------------------------------------------------------------------

extended_dem extended_dem_from_css_matrices(const css_code_matrices &code,
                                            const css_noise_params &noise) {
  validate_noise_rates(noise);
  extended_dem result;
  const std::size_t n = resolve_num_qubits(code);
  if (n == 0) {
    auto check_zero_rows = [](const sparse_binary_matrix &m, const char *name) {
      if (m.num_rows() > 0)
        throw std::invalid_argument(
            std::string(name) + " has " + std::to_string(m.num_rows()) +
            " rows but zero columns; all matrices must be empty when "
            "num_qubits cannot be determined");
    };
    check_zero_rows(code.hz, "hz");
    check_zero_rows(code.hx, "hx");
    check_zero_rows(code.lz, "lz");
    check_zero_rows(code.lx, "lx");
    if (!noise.px_per_qubit.empty() || !noise.py_per_qubit.empty() ||
        !noise.pz_per_qubit.empty())
      throw std::invalid_argument(
          "per-qubit noise vectors are non-empty but all code matrices have "
          "zero columns");
    return result;
  }

  check_num_cols(code.hz, n, "hz");
  check_num_cols(code.hx, n, "hx");
  check_num_cols(code.lz, n, "lz");
  check_num_cols(code.lx, n, "lx");
  check_per_qubit_size(noise.px_per_qubit, n, "px_per_qubit");
  check_per_qubit_size(noise.py_per_qubit, n, "py_per_qubit");
  check_per_qubit_size(noise.pz_per_qubit, n, "pz_per_qubit");

  const auto hz_csc = padded_nested_csc(code.hz, n);
  const auto hx_csc = padded_nested_csc(code.hx, n);
  const auto lz_csc = padded_nested_csc(code.lz, n);
  const auto lx_csc = padded_nested_csc(code.lx, n);

  const uint32_t nz = static_cast<uint32_t>(code.hz.num_rows());
  const uint32_t nx = static_cast<uint32_t>(code.hx.num_rows());
  const uint32_t kz = static_cast<uint32_t>(code.lz.num_rows());
  const uint32_t kx = static_cast<uint32_t>(code.lx.num_rows());

  if (static_cast<uint64_t>(nz) + nx > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "extended_dem_from_css_matrices: hz+hx rows exceed uint32_t");
  if (static_cast<uint64_t>(kz) + kx > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "extended_dem_from_css_matrices: lz+lx rows exceed uint32_t");

  const uint32_t d = nz + nx;
  const uint32_t k = kz + kx;

  check_rate_vector_size(noise.pm_per_check, static_cast<std::size_t>(d),
                         "pm_per_check", "n_checks");

  const auto x_qubits = active_qubits(noise.px, noise.px_per_qubit, n);
  const auto z_qubits = active_qubits(noise.pz, noise.pz_per_qubit, n);
  const auto y_qubits = active_qubits(noise.py, noise.py_per_qubit, n);
  const auto m_checks =
      active_checks(noise.pm, noise.pm_per_check, static_cast<std::size_t>(d));

  const std::size_t n_faults_sz =
      x_qubits.size() + z_qubits.size() + y_qubits.size() + m_checks.size();
  if (n_faults_sz > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "extended_dem_from_css_matrices: active fault count exceeds uint32_t");
  const uint32_t n_faults = static_cast<uint32_t>(n_faults_sz);
  if (n_faults == 0)
    return result;

  col_list syn_cols(n_faults), obs_cols(n_faults);
  std::vector<double> priors;
  priors.reserve(n_faults);

  uint32_t col = 0;
  for (const std::size_t q : x_qubits) {
    for (auto r : hz_csc[q])
      syn_cols[col].push_back(r);
    for (auto r : lz_csc[q])
      obs_cols[col].push_back(r);
    priors.push_back(qubit_rate(noise.px, noise.px_per_qubit, q));
    ++col;
  }
  for (const std::size_t q : z_qubits) {
    for (auto r : hx_csc[q])
      syn_cols[col].push_back(nz + r);
    for (auto r : lx_csc[q])
      obs_cols[col].push_back(kz + r);
    priors.push_back(qubit_rate(noise.pz, noise.pz_per_qubit, q));
    ++col;
  }
  for (const std::size_t q : y_qubits) {
    for (auto r : hz_csc[q])
      syn_cols[col].push_back(r);
    for (auto r : hx_csc[q])
      syn_cols[col].push_back(nz + r);
    for (auto r : lz_csc[q])
      obs_cols[col].push_back(r);
    for (auto r : lx_csc[q])
      obs_cols[col].push_back(kz + r);
    priors.push_back(qubit_rate(noise.py, noise.py_per_qubit, q));
    ++col;
  }
  for (const std::size_t ck : m_checks) {
    syn_cols[col].push_back(static_cast<sparse_binary_matrix::index_type>(ck));
    priors.push_back(check_rate(noise.pm, noise.pm_per_check, ck));
    ++col;
  }

  // Build syndrome matrix once; H = vstack(syn, syn) for two seam bands.
  auto syn_mat = sparse_binary_matrix::from_nested_csc(d, n_faults, syn_cols);
  result.H = vstack(syn_mat, syn_mat);
  result.O = sparse_binary_matrix::from_nested_csc(k, n_faults, obs_cols);
  result.error_rates = std::move(priors);

  // Positional tags, identical for both seams: row i of either seam is check
  // i of the same round-to-round syndrome matrix. Being positional, they carry
  // no identity beyond the seam width -- see extended_dem::tags.
  std::vector<uint64_t> half_tags(d);
  std::iota(half_tags.begin(), half_tags.end(), uint64_t{0});
  result.tags = half_tags;
  result.tags.insert(result.tags.end(), half_tags.begin(), half_tags.end());

  result.add_seam(seam_name::prev_round, 0, d);
  result.add_seam(seam_name::next_round, d, 2 * d);
  return result;
} // end - extended_dem_from_css_matrices()

// ---------------------------------------------------------------------------
// dem_chunk_spec methods
// ---------------------------------------------------------------------------

bool dem_chunk_spec::is_empty() const {
  return num_faults == 0 && H_sparse.empty() && seam_specs.empty() &&
         O_sparse.empty() && error_rates.empty();
}

void dem_chunk_spec::expand(const std::vector<seam_id> &ids) {
  if (!seam_specs.empty())
    return;
  if (H_sparse.empty())
    return;
  for (const seam_id id : ids)
    seam_specs.push_back({id, {H_sparse, {}}});
  H_sparse.clear();
}

void dem_chunk_spec::validate(const std::string &context) const {
  if (num_faults == 0)
    throw std::invalid_argument(context + ".num_faults must be positive");
  if (num_faults > std::numeric_limits<sparse_binary_matrix::index_type>::max())
    throw std::invalid_argument(context + ".num_faults exceeds uint32_t max");
  if (!H_sparse.empty() && !seam_specs.empty())
    throw std::invalid_argument(
        context + ": H_sparse and seam_specs are mutually exclusive");
  if (error_rates.size() != num_faults)
    throw std::invalid_argument(
        context + ".error_rates has " + std::to_string(error_rates.size()) +
        " entries but num_faults=" + std::to_string(num_faults));
  for (std::size_t i = 0; i < error_rates.size(); ++i)
    if (!(error_rates[i] >= 0.0 && error_rates[i] <= 1.0))
      throw std::invalid_argument(context + ".error_rates[" +
                                  std::to_string(i) + "] out of [0,1]");
  validate_index_list(H_sparse, num_faults, context, "H_sparse");
  validate_index_list(O_sparse, num_faults, context, "O_sparse");
  for (const auto &e : seam_specs) {
    validate_index_list(e.spec.H_sparse, num_faults, context,
                        "seam[" + e.id.name() + "].H_sparse");
    validate_index_list(e.spec.O_sparse, num_faults, context,
                        "seam[" + e.id.name() + "].O_sparse");
  }
}

// ---------------------------------------------------------------------------
// dem_chunk_from_spec
// ---------------------------------------------------------------------------

extended_dem dem_chunk_from_spec(const dem_chunk_spec &spec,
                                 const std::vector<seam_id> &seam_names,
                                 const std::string &context) {
  dem_chunk_spec working = spec;
  working.validate(context);
  working.expand(seam_names);

  for (const auto &e : working.seam_specs)
    if (!e.spec.O_sparse.empty())
      throw std::invalid_argument(
          context + ": per-seam O_sparse in seam '" + e.id.name() +
          "' is not yet supported; use chunk-level O_sparse instead");

  extended_dem result;
  result.error_rates = working.error_rates;

  // Build H by vstacking seam bands in seam_specs order.
  uint32_t row_cursor = 0;
  for (const auto &e : working.seam_specs) {
    auto seam_H =
        sparse_from_terminated_rows(e.spec.H_sparse, working.num_faults);
    const uint32_t seam_rows = seam_H.num_rows();
    if (result.H.num_rows() == 0 && result.H.num_cols() == 0) {
      if (seam_rows > 0)
        result.H = seam_H;
      else
        result.H = zero_matrix(0, static_cast<uint32_t>(working.num_faults));
    } else if (seam_rows > 0) {
      result.H = vstack(result.H, seam_H);
    }
    result.add_seam(e.id, row_cursor, row_cursor + seam_rows);
    for (uint32_t k = 0; k < seam_rows; ++k)
      result.tags.push_back(static_cast<uint64_t>(k));
    row_cursor += seam_rows;
  }
  if (result.H.num_cols() == 0)
    result.H = zero_matrix(0, static_cast<uint32_t>(working.num_faults));

  result.O = sparse_from_terminated_rows(working.O_sparse, working.num_faults);
  return result;
} // end - dem_chunk_from_spec()

// ---------------------------------------------------------------------------
// dem_chunks_spec methods
// ---------------------------------------------------------------------------

bool dem_chunks_spec::is_empty() const {
  return phases.empty() && connections.empty();
}

bool dem_chunks_spec::has_repeating_phase() const {
  for (const auto &c : connections)
    if (c.is_self())
      return true;
  return false;
}

phase_id dem_chunks_spec::repeating_phase() const {
  phase_id result;
  bool found = false;
  for (const auto &c : connections) {
    if (c.is_self()) {
      if (found)
        throw std::invalid_argument(
            "dem_chunks_spec: more than one self-connected phase");
      result = c.from_phase;
      found = true;
    }
  }
  if (!found)
    throw std::invalid_argument(
        "dem_chunks_spec: no self-connected (repeating) phase");
  return result;
}

std::vector<phase_id> dem_chunks_spec::phase_sequence() const {
  if (phases.empty())
    throw std::invalid_argument(
        "dem_chunks_spec::phase_sequence: no phases defined");
  if (connections.empty())
    throw std::invalid_argument(
        "dem_chunks_spec::phase_sequence: no connections defined");

  // Collect non-self edges and identify the repeating phase.
  std::vector<std::pair<phase_id, phase_id>> chain;
  phase_id rep_phase;
  bool has_rep = false;
  for (const auto &c : connections) {
    if (c.is_self()) {
      if (has_rep)
        throw std::invalid_argument(
            "dem_chunks_spec::phase_sequence: multiple self-loops");
      rep_phase = c.from_phase;
      has_rep = true;
    } else {
      chain.push_back({c.from_phase, c.to_phase});
    }
  }

  // num_rounds is required only when a self-loop (repeating phase) is present.
  if (has_rep && !num_rounds.has_value())
    throw std::invalid_argument(
        "dem_chunks_spec::phase_sequence: num_rounds required when a "
        "self-loop is present");
  if (has_rep && *num_rounds < 2)
    throw std::invalid_argument(
        "dem_chunks_spec::phase_sequence: num_rounds must be >= 2");

  if (chain.empty()) {
    std::vector<phase_id> seq(*num_rounds, rep_phase);
    return seq;
  }

  // Find entry: in chain.from_phase but not in any chain.to_phase.
  std::set<uint32_t> to_phases_set;
  for (const auto &[f, t] : chain)
    to_phases_set.insert(t.value);

  phase_id entry = chain[0].first;
  for (const auto &[f, t] : chain)
    if (to_phases_set.find(f.value) == to_phases_set.end()) {
      entry = f;
      break;
    }

  // For a linear chain without a self-loop, sequence length = chain edges + 1.
  // With a self-loop, num_rounds controls how many repeating copies are
  // inserted.
  const std::size_t non_repeating = chain.size() + 1;
  const std::size_t rounds = has_rep ? *num_rounds : non_repeating;

  if (has_rep && rounds < non_repeating)
    throw std::invalid_argument("dem_chunks_spec::phase_sequence: num_rounds " +
                                std::to_string(rounds) + " too small for " +
                                std::to_string(non_repeating) +
                                " non-repeating phases");
  const std::size_t rep_count = has_rep ? rounds - non_repeating : 0;

  // Build a successor map so that connection order does not matter.
  std::map<uint32_t, phase_id> successor;
  for (const auto &[f, t] : chain) {
    if (!successor.emplace(f.value, t).second)
      throw std::invalid_argument("dem_chunks_spec::phase_sequence: phase " +
                                  f.name() +
                                  " has multiple non-self outgoing edges");
  }

  std::vector<phase_id> seq;
  seq.reserve(rounds);
  // Each phase is visited at most once; the repeating phase is expanded in
  // place rather than by walking its self-loop. Without this guard a cyclic
  // connection graph (A->B, B->A) never reaches a phase without a successor
  // and the walk grows the sequence until memory runs out.
  std::set<uint32_t> visited;
  phase_id current = entry;
  while (true) {
    if (!visited.insert(current.value).second)
      throw std::invalid_argument(
          "dem_chunks_spec::phase_sequence: connection graph is cyclic; "
          "phase " +
          current.name() + " is reachable twice");
    seq.push_back(current);
    if (has_rep && current == rep_phase) {
      for (std::size_t r = 0; r < rep_count; ++r)
        seq.push_back(rep_phase);
    }
    auto it = successor.find(current.value);
    if (it == successor.end())
      break;
    current = it->second;
  }

  if (seq.size() != rounds)
    throw std::invalid_argument("dem_chunks_spec::phase_sequence: computed " +
                                std::to_string(seq.size()) +
                                " phases but expected " +
                                std::to_string(rounds));
  return seq;
} // end - dem_chunks_spec::phase_sequence()

const dem_chunk_spec &dem_chunks_spec::get_phase(phase_id id) const {
  for (const auto &e : phases)
    if (e.id == id)
      return e.spec;
  throw std::invalid_argument("dem_chunks_spec::get_phase: phase " + id.name() +
                              " not found");
}

void dem_chunks_spec::validate() const {
  if (is_empty())
    throw std::invalid_argument("dem_chunks_spec: no phases or connections");
  if (connections.empty())
    throw std::invalid_argument(
        "dem_chunks_spec: connections must not be empty");
  if (phases.empty())
    throw std::invalid_argument("dem_chunks_spec: phases must not be empty");
  if (num_rounds.has_value() && *num_rounds < 2)
    throw std::invalid_argument("dem_chunks_spec: num_rounds must be >= 2");
  for (const auto &e : phases)
    e.spec.validate("dem_chunks." + e.id.name());
}

// ---------------------------------------------------------------------------
// dem_chunks_from_spec
// ---------------------------------------------------------------------------

std::vector<extended_dem> dem_chunks_from_spec(const dem_chunks_spec &spec) {
  spec.validate();
  const auto seq = spec.phase_sequence();
  std::vector<extended_dem> result;
  result.reserve(seq.size());
  for (std::size_t i = 0; i < seq.size(); ++i) {
    const auto &phase_spec = spec.get_phase(seq[i]);
    std::vector<seam_id> ids;
    // Every phase gets the incoming seam (to_seam / prev_round). For the first
    // phase this becomes the initial-state detector band: dem_close_all emits
    // those rows first, compared against the zero initial syndrome.
    ids.push_back(spec.seam.to_seam);
    if (i + 1 < seq.size())
      ids.push_back(spec.seam.from_seam);
    result.push_back(dem_chunk_from_spec(
        phase_spec, ids, "dem_chunks." + std::to_string(seq[i].value)));
  }
  return result;
} // end - dem_chunks_from_spec()

// ---------------------------------------------------------------------------
// dem_stitch
// ---------------------------------------------------------------------------

extended_dem dem_stitch(const extended_dem &a, const extended_dem &b,
                        seam_id from_seam, seam_id to_seam) {
  a.validate("dem_stitch: left");
  b.validate("dem_stitch: right");
  if (a.num_observables() != b.num_observables())
    throw std::invalid_argument("dem_stitch: observable count mismatch (" +
                                std::to_string(a.num_observables()) + " vs " +
                                std::to_string(b.num_observables()) + ")");

  if (!a.has_seam(from_seam))
    throw std::invalid_argument("dem_stitch: left chunk has no seam with id " +
                                std::to_string(from_seam.value) +
                                " (from_seam); cannot contract");
  if (!b.has_seam(to_seam))
    throw std::invalid_argument("dem_stitch: right chunk has no seam with id " +
                                std::to_string(to_seam.value) +
                                " (to_seam); cannot contract");

  reject_extra_seams(a, from_seam, to_seam, "dem_stitch: left");
  reject_extra_seams(b, from_seam, to_seam, "dem_stitch: right");

  check_seam_boundary(a, b, from_seam, to_seam, "dem_stitch");
  const auto &a_from = a.get_seam(from_seam);
  const auto &b_to = b.get_seam(to_seam);
  const uint32_t seam_d = a_from.num_rows();

  const auto n_A_sz = static_cast<std::size_t>(a.num_faults());
  const auto n_total_sz = n_A_sz + b.num_faults();
  if (n_total_sz > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error(
        "dem_stitch: combined fault count exceeds uint32_t");
  const uint32_t n_A = static_cast<uint32_t>(n_A_sz);
  const uint32_t n_total = static_cast<uint32_t>(n_total_sz);

  // Extract the five row bands (leading/trailing seams may be absent for
  // asymmetric phase chunks like init/final — handle gracefully)
  auto a_to_rows = a.has_seam(to_seam)
                       ? extract_row_band(a.H, a.get_seam(to_seam).row_begin,
                                          a.get_seam(to_seam).row_end)
                       : zero_matrix(0, n_A);
  auto a_int_rows = extract_interior_rows(a.H, a.seams);
  auto a_from_rows = extract_row_band(a.H, a_from.row_begin, a_from.row_end);
  auto b_to_rows = extract_row_band(b.H, b_to.row_begin, b_to.row_end);
  auto b_int_rows = extract_interior_rows(b.H, b.seams);
  auto b_from_rows =
      b.has_seam(from_seam)
          ? extract_row_band(b.H, b.get_seam(from_seam).row_begin,
                             b.get_seam(from_seam).row_end)
          : zero_matrix(0, static_cast<uint32_t>(b.num_faults()));

  // Build result H
  // Row layout: a_to | a_interior | contracted_seam | b_interior | b_from
  auto contracted = hcat(a_from_rows, b_to_rows);
  auto top = pad_right(a_to_rows, n_total);
  auto top_int = pad_right(a_int_rows, n_total);
  // contracted already has n_total cols
  auto bot_int = pad_left(b_int_rows, n_A, n_total);
  auto bot = pad_left(b_from_rows, n_A, n_total);

  // Vstack non-empty parts
  sparse_binary_matrix result_H = zero_matrix(0, n_total);
  auto push = [&](const sparse_binary_matrix &m) {
    if (m.num_rows() > 0)
      result_H = result_H.num_rows() > 0 ? vstack(result_H, m) : m;
  };
  push(top);
  push(top_int);
  push(contracted);
  push(bot_int);
  push(bot);

  if (result_H.num_cols() == 0)
    result_H = zero_matrix(0, n_total);

  extended_dem r;
  r.H = result_H;
  r.O = hcat(a.O, b.O);
  r.error_rates.reserve(n_total);
  r.error_rates.insert(r.error_rates.end(), a.error_rates.begin(),
                       a.error_rates.end());
  r.error_rates.insert(r.error_rates.end(), b.error_rates.begin(),
                       b.error_rates.end());

  // Result seam descriptors (seams may be absent for asymmetric phase chunks)
  const uint32_t a_to_n = a_to_rows.num_rows();
  const uint32_t a_int_n = a.num_interior_rows();
  const uint32_t b_int_n = b.num_interior_rows();
  const uint32_t b_from_n = b_from_rows.num_rows();
  r.add_seam(to_seam, 0, a_to_n);
  const uint32_t b_from_start = a_to_n + a_int_n + seam_d + b_int_n;
  r.add_seam(from_seam, b_from_start, b_from_start + b_from_n);

  // Tags: from A's to_seam (if present), then from B's from_seam (if present)
  if (a.has_seam(to_seam)) {
    const uint32_t a_to_off = seam_tag_offset(a, to_seam);
    for (uint32_t k = 0; k < a_to_n; ++k)
      r.tags.push_back(a.tags[a_to_off + k]);
  }
  if (b.has_seam(from_seam)) {
    const uint32_t b_from_off = seam_tag_offset(b, from_seam);
    for (uint32_t k = 0; k < b_from_n; ++k)
      r.tags.push_back(b.tags[b_from_off + k]);
  }

  return r;
} // end - dem_stitch()

// ---------------------------------------------------------------------------
// dem_stitch_all
// ---------------------------------------------------------------------------

namespace {

std::size_t seam_rows_or_zero(const extended_dem &chunk, seam_id id) {
  return chunk.has_seam(id) ? chunk.get_seam(id).num_rows() : 0u;
}

col_list nested_csc(const sparse_binary_matrix &m) {
  return m.layout() == sparse_binary_matrix_layout::csc
             ? m.to_nested_csc()
             : m.to_csc().to_nested_csc();
}

// Row and column geometry of a whole chunk chain, computed up front so the
// chain can be filled in one pass. Row order is the one a left fold of
// dem_stitch() produces:
//
//   chunk 0 to_seam | chunk 0 interior | seam(0,1) | chunk 1 interior | ...
//   ... | chunk T-1 interior | chunk T-1 from_seam
//
// Dropping the trailing band leaves exactly dem_close_all()'s detector rows,
// which is what lets both entry points below share this plan.
struct chain_layout {
  std::vector<std::size_t> col_off;      ///< T+1 cumulative fault columns
  std::vector<std::size_t> interior_row; ///< T interior band starts
  std::vector<std::size_t> seam_row;     ///< T-1 contracted seam band starts
  std::size_t lead = 0;                  ///< rows of chunk 0's to_seam
  std::size_t trail = 0;                 ///< rows of chunk T-1's from_seam
  std::size_t trail_row = 0;             ///< first row of the trailing band
  std::size_t num_faults = 0;
};

chain_layout plan_chain(const std::vector<extended_dem> &chunks,
                        seam_id from_seam, seam_id to_seam) {
  const std::size_t T = chunks.size();
  chain_layout plan;
  plan.col_off.assign(T + 1, 0);
  plan.interior_row.assign(T, 0);
  plan.seam_row.assign(T - 1, 0);
  plan.lead = seam_rows_or_zero(chunks.front(), to_seam);
  plan.trail = seam_rows_or_zero(chunks.back(), from_seam);

  std::size_t rows = plan.lead;
  for (std::size_t i = 0; i < T; ++i) {
    plan.col_off[i + 1] = plan.col_off[i] + chunks[i].num_faults();
    plan.interior_row[i] = rows;
    rows += chunks[i].num_interior_rows();
    if (i + 1 < T) {
      plan.seam_row[i] = rows;
      rows += seam_rows_or_zero(chunks[i], from_seam);
    }
  }
  plan.trail_row = rows;
  plan.num_faults = plan.col_off.back();
  return plan;
}

// Where each of chunk i's own H rows lands in the assembled chain. A chunk's
// to_seam shares its rows with the predecessor's from_seam, which is what
// contracting the boundary means; chunk 0's to_seam has no predecessor and
// heads the result instead.
std::vector<uint32_t> chain_row_map(const extended_dem &chunk, std::size_t i,
                                    std::size_t T, const chain_layout &plan,
                                    seam_id from_seam, seam_id to_seam) {
  const std::size_t to_base = (i == 0) ? 0 : plan.seam_row[i - 1];
  const std::size_t from_base = (i + 1 < T) ? plan.seam_row[i] : plan.trail_row;

  const bool has_to = chunk.has_seam(to_seam);
  const bool has_from = chunk.has_seam(from_seam);
  const uint32_t to_begin = has_to ? chunk.get_seam(to_seam).row_begin : 0u;
  const uint32_t to_end = has_to ? chunk.get_seam(to_seam).row_end : 0u;
  const uint32_t from_begin =
      has_from ? chunk.get_seam(from_seam).row_begin : 0u;
  const uint32_t from_end = has_from ? chunk.get_seam(from_seam).row_end : 0u;

  // Rows outside both bands are interior. reject_extra_seams() has already
  // ruled out a third band carrying rows, so this agrees with
  // extract_interior_rows(), which excludes every seam.
  std::vector<uint32_t> row_map(chunk.H.num_rows(), 0);
  std::size_t interior_seen = 0;
  for (uint32_t r = 0; r < chunk.H.num_rows(); ++r) {
    if (has_to && r >= to_begin && r < to_end)
      row_map[r] = static_cast<uint32_t>(to_base + (r - to_begin));
    else if (has_from && r >= from_begin && r < from_end)
      row_map[r] = static_cast<uint32_t>(from_base + (r - from_begin));
    else
      row_map[r] =
          static_cast<uint32_t>(plan.interior_row[i] + interior_seen++);
  }
  return row_map;
}

// Fault columns are block-disjoint across chunks, so each chunk's columns are
// copied once with their rows remapped. That is linear in total non-zeros,
// where a left fold of dem_stitch() re-copies the whole accumulator per step
// and so costs O(T^2).
//
// row_limit drops rows at or past it, which is how the closed forms discard
// the trailing from_seam band without a second pass.
col_list chain_h_columns(const std::vector<extended_dem> &chunks,
                         const chain_layout &plan, seam_id from_seam,
                         seam_id to_seam, std::size_t row_limit) {
  const std::size_t T = chunks.size();
  col_list cols(plan.num_faults);
  for (std::size_t i = 0; i < T; ++i) {
    const auto row_map =
        chain_row_map(chunks[i], i, T, plan, from_seam, to_seam);
    const auto local = nested_csc(chunks[i].H);
    for (std::size_t j = 0; j < local.size(); ++j) {
      auto &dst = cols[plan.col_off[i] + j];
      dst.reserve(local[j].size());
      for (auto r : local[j])
        if (row_map[r] < row_limit)
          dst.push_back(row_map[r]);
    }
  }
  return cols;
}

col_list chain_o_columns(const std::vector<extended_dem> &chunks,
                         const chain_layout &plan) {
  col_list cols(plan.num_faults);
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    // A chunk with no observables may leave O default-constructed (0 x 0).
    if (chunks[i].O.num_cols() == 0)
      continue;
    const auto local = nested_csc(chunks[i].O);
    for (std::size_t j = 0; j < local.size(); ++j)
      cols[plan.col_off[i] + j] = local[j];
  }
  return cols;
}

// The checks a left fold performed through its repeated dem_stitch() calls.
// Validating the inputs is equivalent to validating each accumulator, since
// every accumulator is built from them.
void validate_chain(const std::vector<extended_dem> &chunks, seam_id from_seam,
                    seam_id to_seam, const std::string &fn) {
  const std::size_t T = chunks.size();
  for (std::size_t i = 0; i < T; ++i) {
    chunks[i].validate((fn + ": chunk " + std::to_string(i)).c_str());
    reject_extra_seams(chunks[i], from_seam, to_seam,
                       fn + ": chunk " + std::to_string(i));
  }
  for (std::size_t i = 1; i < T; ++i)
    if (chunks[i].num_observables() != chunks[0].num_observables())
      throw std::invalid_argument(fn + ": observable count mismatch at chunk " +
                                  std::to_string(i));
  for (std::size_t i = 0; i + 1 < T; ++i) {
    if (!chunks[i].has_seam(from_seam))
      throw std::invalid_argument(fn + ": chunk " + std::to_string(i) +
                                  " has no seam with id " + from_seam.name() +
                                  " (from_seam); cannot contract");
    if (!chunks[i + 1].has_seam(to_seam))
      throw std::invalid_argument(fn + ": chunk " + std::to_string(i + 1) +
                                  " has no seam with id " + to_seam.name() +
                                  " (to_seam); cannot contract");
    check_seam_boundary(chunks[i], chunks[i + 1], from_seam, to_seam,
                        fn + ": boundary " + std::to_string(i));
  }
}

} // namespace

extended_dem dem_stitch_all(const std::vector<extended_dem> &chunks,
                            seam_id from_seam, seam_id to_seam) {
  if (chunks.empty())
    throw std::invalid_argument("dem_stitch_all: dem_chunks must be non-empty");
  if (chunks.size() == 1) {
    chunks[0].validate("dem_stitch_all: chunk 0");
    return chunks[0];
  }
  validate_chain(chunks, from_seam, to_seam, "dem_stitch_all");

  const std::size_t T = chunks.size();
  const chain_layout plan = plan_chain(chunks, from_seam, to_seam);
  const std::size_t n_rows = plan.trail_row + plan.trail;
  const auto n_faults = static_cast<uint32_t>(plan.num_faults);

  extended_dem result;
  // Remapped rows arrive out of order within a column, so canonicalize.
  result.H = sparse_binary_matrix::from_nested_csc(
                 static_cast<uint32_t>(n_rows), n_faults,
                 chain_h_columns(chunks, plan, from_seam, to_seam, n_rows))
                 .canonicalize();
  result.O = sparse_binary_matrix::from_nested_csc(
      chunks[0].num_observables(), n_faults, chain_o_columns(chunks, plan));
  result.error_rates.reserve(plan.num_faults);
  for (const auto &chunk : chunks)
    result.error_rates.insert(result.error_rates.end(),
                              chunk.error_rates.begin(),
                              chunk.error_rates.end());

  result.add_seam(to_seam, 0, static_cast<uint32_t>(plan.lead));
  result.add_seam(from_seam, static_cast<uint32_t>(plan.trail_row),
                  static_cast<uint32_t>(plan.trail_row + plan.trail));

  // Tags come from the two surviving seams, as with a pairwise stitch.
  if (chunks[0].has_seam(to_seam)) {
    const uint32_t off = seam_tag_offset(chunks[0], to_seam);
    for (std::size_t k = 0; k < plan.lead; ++k)
      result.tags.push_back(chunks[0].tags[off + k]);
  }
  if (chunks[T - 1].has_seam(from_seam)) {
    const uint32_t off = seam_tag_offset(chunks[T - 1], from_seam);
    for (std::size_t k = 0; k < plan.trail; ++k)
      result.tags.push_back(chunks[T - 1].tags[off + k]);
  }
  return result;
} // end - dem_stitch_all()

// ---------------------------------------------------------------------------
// dem_close
// ---------------------------------------------------------------------------

detector_error_model dem_close(const extended_dem &dem, seam_id to_seam,
                               seam_id from_seam) {
  dem.validate("dem_close");
  reject_extra_seams(dem, from_seam, to_seam, "dem_close");
  const uint32_t n_faults = dem.num_faults();
  const uint32_t n_obs = dem.num_observables();
  const uint32_t n_lead =
      dem.has_seam(to_seam) ? dem.get_seam(to_seam).num_rows() : 0;
  const uint32_t n_interior = dem.num_interior_rows();
  const uint32_t n_det = n_lead + n_interior;

  detector_error_model result;
  result.detector_error_matrix = cudaqx::tensor<uint8_t>({n_det, n_faults});
  result.observables_flips_matrix = cudaqx::tensor<uint8_t>({n_obs, n_faults});
  result.error_rates = dem.error_rates;

  if (n_lead > 0) {
    const auto &ls = dem.get_seam(to_seam);
    auto lead_H = extract_row_band(dem.H, ls.row_begin, ls.row_end);
    write_sparse_block(result.detector_error_matrix, lead_H, 0, 0);
  }
  auto interior = extract_interior_rows(dem.H, dem.seams);
  write_sparse_block(result.detector_error_matrix, interior, n_lead, 0);
  write_sparse_block(result.observables_flips_matrix, dem.O, 0, 0);
  return result;
} // end - dem_close()

// ---------------------------------------------------------------------------
// dem_close_all (O(T) forward pass)
// ---------------------------------------------------------------------------

detector_error_model dem_close_all(const std::vector<extended_dem> &chunks,
                                   seam_id from_seam, seam_id to_seam) {
  if (chunks.empty())
    throw std::invalid_argument("dem_close_all: dem_chunks must be non-empty");
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    chunks[i].validate(("dem_close_all: chunk " + std::to_string(i)).c_str());
    reject_extra_seams(chunks[i], from_seam, to_seam,
                       "dem_close_all: chunk " + std::to_string(i));
  }

  const uint32_t k = chunks[0].num_observables();
  for (std::size_t i = 1; i < chunks.size(); ++i)
    if (chunks[i].num_observables() != k)
      throw std::invalid_argument(
          "dem_close_all: observable count mismatch at chunk " +
          std::to_string(i));

  // Validate adjacent seam widths and tags using the same helper as dem_stitch.
  const std::size_t T = chunks.size();
  for (std::size_t i = 0; i + 1 < T; ++i)
    check_seam_boundary(chunks[i], chunks[i + 1], from_seam, to_seam,
                        "dem_close_all: boundary " + std::to_string(i));

  // Cumulative fault column offsets
  std::vector<std::size_t> col_off(T + 1, 0);
  for (std::size_t i = 0; i < T; ++i)
    col_off[i + 1] = col_off[i] + chunks[i].num_faults();
  const std::size_t n_faults = col_off.back();

  const std::size_t lead =
      chunks[0].has_seam(to_seam) ? chunks[0].get_seam(to_seam).num_rows() : 0;
  std::size_t total_interior = 0;
  for (const auto &c : chunks)
    total_interior += c.num_interior_rows();
  std::size_t total_seam = 0;
  for (std::size_t i = 0; i + 1 < T; ++i)
    if (chunks[i].has_seam(from_seam))
      total_seam += chunks[i].get_seam(from_seam).num_rows();
  const std::size_t n_det = lead + total_interior + total_seam;

  detector_error_model result;
  result.detector_error_matrix = cudaqx::tensor<uint8_t>({n_det, n_faults});
  result.observables_flips_matrix =
      cudaqx::tensor<uint8_t>({static_cast<std::size_t>(k), n_faults});
  result.error_rates.reserve(n_faults);

  // Write leading seam of chunk 0
  if (lead > 0 && chunks[0].has_seam(to_seam)) {
    const auto &s = chunks[0].get_seam(to_seam);
    auto lead_H = extract_row_band(chunks[0].H, s.row_begin, s.row_end);
    write_sparse_block(result.detector_error_matrix, lead_H, 0, col_off[0]);
  }

  std::size_t row_cursor = lead;
  for (std::size_t i = 0; i < T; ++i) {
    auto interior = extract_interior_rows(chunks[i].H, chunks[i].seams);
    if (interior.num_rows() > 0) {
      write_sparse_block(result.detector_error_matrix, interior, row_cursor,
                         col_off[i]);
      row_cursor += interior.num_rows();
    }
    if (i + 1 < T && chunks[i].has_seam(from_seam) &&
        chunks[i + 1].has_seam(to_seam)) {
      const auto &fs = chunks[i].get_seam(from_seam);
      const auto &ts = chunks[i + 1].get_seam(to_seam);
      auto from_H = extract_row_band(chunks[i].H, fs.row_begin, fs.row_end);
      auto to_H = extract_row_band(chunks[i + 1].H, ts.row_begin, ts.row_end);
      write_sparse_block(result.detector_error_matrix, from_H, row_cursor,
                         col_off[i]);
      write_sparse_block(result.detector_error_matrix, to_H, row_cursor,
                         col_off[i + 1]);
      row_cursor += fs.num_rows();
    }
    write_sparse_block(result.observables_flips_matrix, chunks[i].O, 0,
                       col_off[i]);
    result.error_rates.insert(result.error_rates.end(),
                              chunks[i].error_rates.begin(),
                              chunks[i].error_rates.end());
  } // end - for(i)

  return result;
} // end - dem_close_all()

// ---------------------------------------------------------------------------
// Streaming utilities
// ---------------------------------------------------------------------------

static uint32_t chunk_seam_width(const extended_dem &chunk, seam_id from_seam,
                                 seam_id to_seam) {
  if (chunk.has_seam(to_seam) && chunk.get_seam(to_seam).num_rows() > 0)
    return chunk.get_seam(to_seam).num_rows();
  if (chunk.has_seam(from_seam) && chunk.get_seam(from_seam).num_rows() > 0)
    return chunk.get_seam(from_seam).num_rows();
  return 0;
}

static void validate_dem_chunk_sequence(const std::vector<extended_dem> &chunks,
                                        seam_id from_seam, seam_id to_seam,
                                        const char *fn) {
  if (chunks.empty())
    throw std::invalid_argument(std::string(fn) +
                                ": dem_chunks must be non-empty");
  const uint32_t d = chunk_seam_width(chunks[0], from_seam, to_seam);
  if (d == 0)
    throw std::invalid_argument(std::string(fn) + ": chunk 0 has no seam rows");
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    if (chunks[i].num_interior_rows() % d != 0)
      throw std::invalid_argument(
          std::string(fn) + ": chunk " + std::to_string(i) +
          " interior rows not a whole number of rounds of " +
          std::to_string(d));
    for (const auto &s : chunks[i].seams)
      if (s.num_rows() != 0 && s.num_rows() != d)
        throw std::invalid_argument(std::string(fn) + ": chunk " +
                                    std::to_string(i) + " seam width " +
                                    std::to_string(s.num_rows()) +
                                    " != sequence width " + std::to_string(d));
  }
}

uint32_t dem_chunk_rounds(const extended_dem &chunk, seam_id from_seam,
                          seam_id to_seam) {
  const uint32_t d = chunk_seam_width(chunk, from_seam, to_seam);
  if (d == 0)
    throw std::invalid_argument(
        "dem_chunk_rounds: chunk has no seam rows on either side");
  if (chunk.num_interior_rows() % d != 0)
    throw std::invalid_argument(
        "dem_chunk_rounds: interior rows not a whole number of rounds");
  // A non-zero leading seam (to_seam with rows) counts as one round.
  // A zero-row seam is a structural placeholder and doesn't contribute.
  const bool has_leading =
      chunk.has_seam(to_seam) && chunk.get_seam(to_seam).num_rows() > 0;
  const uint32_t leading = has_leading ? 1u : 0u;
  return leading + chunk.num_interior_rows() / d;
}

std::size_t dem_chunks_to_rounds(const std::vector<extended_dem> &chunks,
                                 seam_id from_seam, seam_id to_seam) {
  validate_dem_chunk_sequence(chunks, from_seam, to_seam,
                              "dem_chunks_to_rounds");
  std::size_t rounds = 0;
  for (const auto &c : chunks)
    rounds += dem_chunk_rounds(c, from_seam, to_seam);
  return rounds;
}

std::vector<std::int32_t>
dem_chunks_to_detector_round(const std::vector<extended_dem> &chunks,
                             seam_id from_seam, seam_id to_seam) {
  validate_dem_chunk_sequence(chunks, from_seam, to_seam,
                              "dem_chunks_to_detector_round");
  const uint32_t d = chunk_seam_width(chunks[0], from_seam, to_seam);
  std::vector<std::int32_t> result;
  std::int32_t round = 0;
  result.reserve(chunks.size() * static_cast<std::size_t>(d));
  for (const auto &c : chunks) {
    const uint32_t rounds = dem_chunk_rounds(c, from_seam, to_seam);
    for (uint32_t r = 0; r < rounds; ++r, ++round)
      result.insert(result.end(), d, round);
  }
  return result;
}

std::vector<std::vector<uint32_t>>
dem_chunks_to_o_sparse(const std::vector<extended_dem> &chunks) {
  if (chunks.empty())
    throw std::invalid_argument(
        "dem_chunks_to_o_sparse: dem_chunks must be non-empty");
  for (std::size_t i = 0; i < chunks.size(); ++i)
    chunks[i].validate(
        ("dem_chunks_to_o_sparse: chunk " + std::to_string(i)).c_str());

  const uint32_t k_obs = chunks[0].num_observables();
  for (std::size_t i = 1; i < chunks.size(); ++i)
    if (chunks[i].num_observables() != k_obs)
      throw std::invalid_argument(
          "dem_chunks_to_o_sparse: observable count mismatch at chunk " +
          std::to_string(i));

  std::vector<std::vector<uint32_t>> o_sparse(k_obs);
  std::size_t col_off = 0;
  for (const auto &c : chunks) {
    const uint32_t nc = c.num_faults();
    if (nc > 0 && col_off + nc - 1 > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error(
          "dem_chunks_to_o_sparse: global fault column exceeds uint32_t");
    // A chunk with no observables may carry a default-constructed O whose
    // column pointers are empty, so it cannot be indexed per fault column.
    if (c.O.num_cols() == 0) {
      col_off += nc;
      continue;
    }
    sparse_binary_matrix obs_converted;
    const sparse_binary_matrix &obs_csc =
        c.O.layout() == sparse_binary_matrix_layout::csc
            ? c.O
            : (obs_converted = c.O.to_csc(), obs_converted);
    const auto &obs_ptr = obs_csc.ptr();
    const auto &obs_rows_v = obs_csc.indices();
    for (uint32_t fc = 0; fc < nc; ++fc)
      for (auto p = obs_ptr[fc]; p != obs_ptr[fc + 1]; ++p)
        o_sparse[obs_rows_v[p]].push_back(static_cast<uint32_t>(col_off + fc));
    col_off += nc;
  }
  return o_sparse;
}

sparse_binary_matrix dem_chunks_to_pcm(const std::vector<extended_dem> &chunks,
                                       seam_id from_seam, seam_id to_seam) {
  if (chunks.empty())
    throw std::invalid_argument(
        "dem_chunks_to_pcm: dem_chunks must be non-empty");
  validate_chain(chunks, from_seam, to_seam, "dem_chunks_to_pcm");

  // The same forward pass dem_stitch_all() makes, stopping before the trailing
  // from_seam band that closing drops. Building CSC directly keeps peak memory
  // proportional to the non-zeros; routing through dem_close_all()'s dense
  // detector_error_matrix would make it the detector x fault product.
  const chain_layout plan = plan_chain(chunks, from_seam, to_seam);
  return sparse_binary_matrix::from_nested_csc(
             static_cast<uint32_t>(plan.trail_row),
             static_cast<uint32_t>(plan.num_faults),
             chain_h_columns(chunks, plan, from_seam, to_seam, plan.trail_row))
      .canonicalize();
}

// ---------------------------------------------------------------------------
// dem_merge_duplicate_columns
// ---------------------------------------------------------------------------

namespace {

col_list canonical_columns(const sparse_binary_matrix &mat) {
  if (mat.layout() == sparse_binary_matrix_layout::csc)
    return mat.canonicalize().to_nested_csc();
  return mat.to_csc().canonicalize().to_nested_csc();
}

// Flat per-column support across all blocks: H rows in row order, then O rows.
struct dem_column_supports {
  col_list H_cols, O_cols;
  std::vector<uint32_t> flat;
  std::vector<std::size_t> off;
  uint32_t obs_base;

  explicit dem_column_supports(const extended_dem &dem)
      : H_cols(canonical_columns(dem.H)), O_cols(canonical_columns(dem.O)),
        obs_base(dem.num_rows()) {
    const std::size_t n = dem.num_faults();
    // A chunk with no observables may carry a default-constructed O with no
    // columns at all; treat it as n empty columns so it can be indexed
    // alongside H.
    if (O_cols.empty())
      O_cols.assign(n, {});
    std::size_t total = 0;
    for (std::size_t j = 0; j < n; ++j)
      total += H_cols[j].size() + O_cols[j].size();
    flat.reserve(total);
    off.reserve(n + 1);
    off.push_back(0);
    for (std::size_t j = 0; j < n; ++j) {
      for (auto r : H_cols[j])
        flat.push_back(static_cast<uint32_t>(r));
      for (auto r : O_cols[j])
        flat.push_back(obs_base + static_cast<uint32_t>(r));
      off.push_back(flat.size());
    }
  }

  std::size_t num_columns() const { return off.size() - 1; }
  std::size_t size(std::size_t j) const { return off[j + 1] - off[j]; }
  const uint32_t *begin_col(std::size_t j) const {
    return flat.data() + off[j];
  }
};

} // namespace

extended_dem dem_merge_duplicate_columns(const extended_dem &dem,
                                         prior_combine_mode mode) {
  dem.validate("dem_merge_duplicate_columns");
  const std::size_t n = dem.num_faults();
  if (n == 0)
    return dem;

  const dem_column_supports cols(dem);

  struct column_group {
    std::size_t representative;
    double accumulator;
    std::size_t members;
  };
  std::map<std::vector<uint32_t>, column_group> sup_map;

  for (std::size_t j = 0; j < n; ++j) {
    std::vector<uint32_t> sup(cols.begin_col(j),
                              cols.begin_col(j) + cols.size(j));
    const double p = dem.error_rates[j];
    auto it = sup_map.find(sup);
    if (it == sup_map.end()) {
      const double init =
          (mode == prior_combine_mode::or_combine) ? (1.0 - 2.0 * p) : p;
      sup_map.emplace(std::move(sup), column_group{j, init, 1});
    } else {
      if (mode == prior_combine_mode::or_combine)
        it->second.accumulator *= (1.0 - 2.0 * p);
      else
        it->second.accumulator += p;
      ++it->second.members;
    }
  }

  const auto new_n = static_cast<uint32_t>(sup_map.size());
  col_list new_H(new_n), new_O(new_n);
  std::vector<double> new_rates;
  new_rates.reserve(new_n);

  std::size_t k = 0;
  for (const auto &[sup, group] : sup_map) {
    const std::size_t rep = group.representative;
    new_H[k] = cols.H_cols[rep];
    new_O[k] = cols.O_cols[rep];
    const double fp = group.members == 1
                          ? dem.error_rates[rep]
                          : (mode == prior_combine_mode::or_combine
                                 ? 0.5 * (1.0 - group.accumulator)
                                 : std::min(1.0, group.accumulator));
    new_rates.push_back(fp);
    ++k;
  }

  extended_dem result;
  result.H =
      sparse_binary_matrix::from_nested_csc(dem.num_rows(), new_n, new_H);
  result.O = sparse_binary_matrix::from_nested_csc(dem.num_observables(), new_n,
                                                   new_O);
  result.error_rates = std::move(new_rates);
  result.seams = dem.seams;
  result.tags = dem.tags;
  return result;
} // end - dem_merge_duplicate_columns()

bool are_dem_columns_unique(const extended_dem &dem) {
  dem.validate("are_dem_columns_unique");
  const std::size_t n = dem.num_faults();
  const dem_column_supports cols(dem);
  std::map<std::vector<uint32_t>, std::size_t> seen;
  for (std::size_t j = 0; j < n; ++j) {
    std::vector<uint32_t> sup(cols.begin_col(j),
                              cols.begin_col(j) + cols.size(j));
    if (!seen.emplace(std::move(sup), j).second)
      return false;
  }
  return true;
} // end - are_dem_columns_unique()

void assert_dem_columns_unique(const extended_dem &dem) {
  dem.validate("assert_dem_columns_unique");
  if (!are_dem_columns_unique(dem))
    throw std::invalid_argument(
        "extended_dem has duplicate fault columns; "
        "call dem_merge_duplicate_columns() to merge them.");
} // end - assert_dem_columns_unique()

extended_dem dem_stitch_merged(const std::vector<extended_dem> &chunks,
                               seam_id from_seam, seam_id to_seam,
                               prior_combine_mode mode) {
  return dem_merge_duplicate_columns(dem_stitch_all(chunks, from_seam, to_seam),
                                     mode);
} // end - dem_stitch_merged()

} // namespace cudaq::qec

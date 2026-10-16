/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/codes/surface_code.h"
#include <algorithm>
#include <cctype>
#include <iomanip>

using cudaq::qec::operation;

namespace cudaq::qec::surface_code {
namespace {

std::string normalize_orientation(std::string value) {
  auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), is_not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(),
              value.end());
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

surface_role role_for_parity(sc_orientation orientation, bool odd_parity) {
  // The first orientation character controls the bulk checkerboard. The H/V
  // character controls boundary placement only, so XV/XH share one bulk
  // assignment and ZV/ZH share the other.
  const bool x_on_even_parity =
      orientation == sc_orientation::XV || orientation == sc_orientation::XH;
  const bool z_on_even_parity =
      orientation == sc_orientation::ZV || orientation == sc_orientation::ZH;
  if (!x_on_even_parity && !z_on_even_parity)
    throw std::runtime_error("Unhandled surface-code orientation.");
  return (x_on_even_parity != odd_parity) ? surface_role::amx
                                          : surface_role::amz;
}

} // namespace

sc_orientation sc_orientation_from_str(const std::string &s) {
  const auto orientation = normalize_orientation(s);
  if (orientation == "XV" || orientation == "O1")
    return sc_orientation::XV;
  if (orientation == "XH" || orientation == "O2")
    return sc_orientation::XH;
  if (orientation == "ZV" || orientation == "O3")
    return sc_orientation::ZV;
  if (orientation == "ZH" || orientation == "O4")
    return sc_orientation::ZH;
  throw std::runtime_error("Invalid surface-code orientation '" + s +
                           "'. Expected XV(O1), XH(O2), ZV(O3), or ZH(O4).");
}

vec2d::vec2d(int row_in, int col_in) : row(row_in), col(col_in) {}

vec2d operator+(const vec2d &lhs, const vec2d &rhs) {
  return {lhs.row + rhs.row, lhs.col + rhs.col};
}

vec2d operator-(const vec2d &lhs, const vec2d &rhs) {
  return {lhs.row - rhs.row, lhs.col - rhs.col};
}

bool operator==(const vec2d &lhs, const vec2d &rhs) {
  return lhs.row == rhs.row && lhs.col == rhs.col;
}

// impose 2d ordering for reproducibility
bool operator<(const vec2d &lhs, const vec2d &rhs) {
  // sort by row component.
  // if row is tied, sort by col.
  if (lhs.row != rhs.row) {
    return lhs.row < rhs.row;
  }
  return lhs.col < rhs.col;
}

void stabilizer_grid::generate_grid_roles() {
  // init grid to all empty
  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      surface_role role = surface_role::empty;
      roles[idx] = role;
    }
  }

  // set alternating x/z interior
  for (size_t row = 1; row < grid_length - 1; ++row) {
    for (size_t col = 1; col < grid_length - 1; ++col) {
      size_t idx = row * grid_length + col;
      const bool is_odd_parity = (row + col) % 2;
      roles[idx] = role_for_parity(orientation_, is_odd_parity);
    }
  }

  const bool horizontal_boundaries_use_even_parity =
      orientation_ == sc_orientation::XH || orientation_ == sc_orientation::ZH;

  // Boundary sites alternate around the perimeter. The top/bottom and
  // left/right edges therefore occupy complementary parities. Since
  // is_odd_parity is false for even parity, the top/bottom predicate is
  // intentionally the inverse of horizontal_boundaries_use_even_parity.
  // The left/right predicate selects the opposite edge parity class: when
  // top/bottom uses even parity, == selects odd sites, and vice versa.

  // set top/bottom boundaries for weight 2 stabs
  for (size_t row = 0; row < grid_length; row += grid_length - 1) {
    for (size_t col = 1; col < grid_length - 1; ++col) {
      size_t idx = row * grid_length + col;
      const bool is_odd_parity = (row + col) % 2;
      if (is_odd_parity != horizontal_boundaries_use_even_parity)
        roles[idx] = role_for_parity(orientation_, is_odd_parity);
    }
  }

  // set left/right boundaries for weight 2 stabs
  for (size_t row = 1; row < grid_length - 1; ++row) {
    for (size_t col = 0; col < grid_length; col += grid_length - 1) {
      size_t idx = row * grid_length + col;
      const bool is_odd_parity = (row + col) % 2;
      if (is_odd_parity == horizontal_boundaries_use_even_parity)
        roles[idx] = role_for_parity(orientation_, is_odd_parity);
    }
  }
}

void stabilizer_grid::generate_grid_indices() {
  size_t z_count = 0;
  size_t x_count = 0;
  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case surface_role::amz:
        z_stab_coords.push_back(vec2d(row, col));
        z_stab_indices[vec2d(row, col)] = z_count;
        z_count++;
        break;
      case surface_role::amx:
        x_stab_coords.push_back(vec2d(row, col));
        x_stab_indices[vec2d(row, col)] = x_count;
        x_count++;
        break;
      case surface_role::empty:
        // nothing
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
  }

  // Data qubit grid
  // This grid is a on a different coordinate system than the stabilizer grid
  // Can think of this grid as offset from the stabilizer grid by a half unit
  // right and down. data_row = stabilizer_row + 0.5 data_col = stabilizer_col +
  // 0.5
  for (size_t row = 0; row < distance; ++row) {
    for (size_t col = 0; col < distance; ++col) {
      size_t idx = row * distance + col;
      data_coords.push_back(vec2d(row, col));
      data_indices[vec2d(row, col)] = idx;
    }
  }
}

void stabilizer_grid::generate_stabilizers() {
  for (size_t i = 0; i < x_stab_coords.size(); ++i) {
    std::vector<size_t> current_stab;
    for (int row_offset = -1; row_offset < 1; ++row_offset) {
      for (int col_offset = -1; col_offset < 1; ++col_offset) {
        int row = x_stab_coords[i].row + row_offset;
        int col = x_stab_coords[i].col + col_offset;
        vec2d trial_coord(row, col);
        if (data_indices.find(trial_coord) != data_indices.end()) {
          current_stab.push_back(data_indices[trial_coord]);
        }
      }
    }
    std::sort(current_stab.begin(), current_stab.end());
    x_stabilizers.push_back(current_stab);
  }

  for (size_t i = 0; i < z_stab_coords.size(); ++i) {
    std::vector<size_t> current_stab;
    for (int row_offset = -1; row_offset < 1; ++row_offset) {
      for (int col_offset = -1; col_offset < 1; ++col_offset) {
        int row = z_stab_coords[i].row + row_offset;
        int col = z_stab_coords[i].col + col_offset;
        vec2d trial_coord(row, col);
        if (data_indices.find(trial_coord) != data_indices.end()) {
          current_stab.push_back(data_indices[trial_coord]);
        }
      }
    }
    std::sort(current_stab.begin(), current_stab.end());
    z_stabilizers.push_back(current_stab);
  }
}

stabilizer_grid::stabilizer_grid() {}

stabilizer_grid::stabilizer_grid(uint32_t distance, sc_orientation orientation)
    : distance(distance), grid_length(distance + 1), orientation_(orientation),
      roles(grid_length * grid_length) {
  // generate a 2d grid of roles
  generate_grid_roles();
  // now use grid to set coord
  generate_grid_indices();
  // now use coords to set the stabilizers
  generate_stabilizers();
}

sc_orientation stabilizer_grid::get_orientation() const { return orientation_; }

void stabilizer_grid::print_stabilizer_coords() const {
  int width = std::to_string(grid_length).length();

  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case amz:
        std::cout << "Z(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case amx:
        std::cout << "X(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case empty:
        std::cout << std::setw(2 * width + 6) << " ";
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_stabilizer_indices() const {
  auto orig_flags = std::cout.flags();
  int width = std::to_string(z_stab_indices.size()).length() + 2;
  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case amz:
        std::cout << "Z" << std::left << std::setw(width)
                  << z_stab_indices.at(vec2d(row, col));
        break;
      case amx:
        std::cout << "X" << std::left << std::setw(width)
                  << x_stab_indices.at(vec2d(row, col));
        break;
      case empty:
        std::cout << std::setw(width + 1) << " ";
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout.flags(orig_flags);
}

void stabilizer_grid::print_stabilizer_maps() const {
  std::cout << x_stab_coords.size() << " mx ancilla qubits:\n";
  for (size_t i = 0; i < x_stab_coords.size(); ++i) {
    std::cout << "amx[" << i << "] @ (" << x_stab_coords[i].row << ", "
              << x_stab_coords[i].col << ")\n";
  }
  std::cout << z_stab_coords.size() << " mz ancilla qubits:\n";
  for (size_t i = 0; i < z_stab_coords.size(); ++i) {
    std::cout << "amz[" << i << "] @ (" << z_stab_coords[i].row << ", "
              << z_stab_coords[i].col << ")\n";
  }
  std::cout << "\n";

  std::cout << x_stab_coords.size() << " mx ancilla qubits:\n";
  for (const auto &[k, v] : x_stab_indices) {
    std::cout << "@(" << k.row << "," << k.col << "): amx[" << v << "]\n";
  }
  std::cout << z_stab_coords.size() << " mz ancilla qubits:\n";
  for (const auto &[k, v] : z_stab_indices) {
    std::cout << "@(" << k.row << "," << k.col << "): amz[" << v << "]\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_data_grid() const {
  auto orig_flags = std::cout.flags();
  int width = std::to_string(distance).length() + 2;

  for (size_t row = 0; row < distance; ++row) {
    for (size_t col = 0; col < distance; ++col) {
      size_t idx = row * distance + col;
      std::cout << "d" << std::left << std::setw(width) << idx;
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout.flags(orig_flags);
}

void stabilizer_grid::print_stabilizer_grid() const {
  int width = std::to_string(grid_length).length();

  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case amz:
        std::cout << "Z(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case amx:
        std::cout << "X(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case empty:
        std::cout << "e(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_stabilizers() const {
  for (size_t s_i = 0; s_i < x_stabilizers.size(); ++s_i) {
    std::cout << "s[" << s_i << "]: ";
    for (size_t op_i = 0; op_i < x_stabilizers[s_i].size(); ++op_i) {
      std::cout << "X" << x_stabilizers[s_i][op_i] << " ";
    }
    std::cout << "\n";
  }

  size_t offset = x_stabilizers.size();
  for (size_t s_i = 0; s_i < z_stabilizers.size(); ++s_i) {
    std::cout << "s[" << s_i + offset << "]: ";
    for (size_t op_i = 0; op_i < z_stabilizers[s_i].size(); ++op_i) {
      std::cout << "Z" << z_stabilizers[s_i][op_i] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

std::vector<cudaq::spin_op_term>
stabilizer_grid::get_spin_op_stabilizers() const {
  std::vector<cudaq::spin_op_term> spin_op_stabs;
  for (size_t s_i = 0; s_i < x_stabilizers.size(); ++s_i) {
    std::string stab(data_coords.size(), 'I');
    for (auto elem : x_stabilizers[s_i]) {
      stab[elem] = 'X';
    }
    spin_op_stabs.emplace_back(cudaq::spin_op::from_word(stab));
  }
  for (size_t s_i = 0; s_i < z_stabilizers.size(); ++s_i) {
    std::string stab(data_coords.size(), 'I');
    for (auto elem : z_stabilizers[s_i]) {
      stab[elem] = 'Z';
    }
    spin_op_stabs.emplace_back(cudaq::spin_op::from_word(stab));
  }
  return spin_op_stabs;
}

namespace {

/// Corner offsets from a stabilizer grid coordinate to its four data qubits,
/// listed in CNOT execution order (timesteps 1..4). Row-major ("Z"-shaped)
/// visits NW, NE, SW, SE so the last two CNOTs touch a horizontally adjacent
/// data pair; column-major ("N"-shaped) visits NW, SW, NE, SE so the last two
/// touch a vertically adjacent pair. Assigning one shape to the X plaquettes
/// and the transposed shape to the Z plaquettes is the standard zigzag
/// schedule pair for the rotated surface code (Tomita & Svore,
/// https://arxiv.org/abs/1404.3747; hook errors per Dennis et al.,
/// https://arxiv.org/abs/quant-ph/0110143): it steers hook errors against
/// the logical operators and stays conflict-free if the two timestep
/// sequences are ever interleaved.
constexpr int row_major_order[4][2] = {{-1, -1}, {-1, 0}, {0, -1}, {0, 0}};
constexpr int col_major_order[4][2] = {{-1, -1}, {0, -1}, {-1, 0}, {0, 0}};

cudaqx::tensor<uint8_t>
build_cnot_schedule(const std::vector<vec2d> &stab_coords,
                    const std::map<vec2d, size_t> &data_indices,
                    std::size_t num_data, bool use_col_major_order) {
  const auto &order = use_col_major_order ? col_major_order : row_major_order;
  std::vector<std::vector<uint8_t>> rows;
  rows.reserve(stab_coords.size());
  for (const auto &sc : stab_coords) {
    std::vector<uint8_t> row(num_data, 0);
    for (int step = 0; step < 4; ++step) {
      vec2d dq(sc.row + order[step][0], sc.col + order[step][1]);
      auto it = data_indices.find(dq);
      if (it != data_indices.end())
        row[it->second] = step + 1;
    }
    rows.push_back(std::move(row));
  }

  // Match the row order of to_parity_matrix(): within one stabilizer type the
  // sort comparator orders rows by the index of the first supported data
  // qubit. That order is total only because no two same-type surface-code
  // stabilizers share a first-support index — enforce it rather than assume
  // it, since a violation would silently pair schedule rows with the wrong
  // ancilla/parity rows.
  std::vector<std::pair<std::size_t, std::size_t>>
      keyed; // (first support, row)
  keyed.reserve(rows.size());
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto it = std::find_if(rows[r].begin(), rows[r].end(),
                           [](uint8_t v) { return v != 0; });
    keyed.emplace_back(it - rows[r].begin(), r);
  }
  std::sort(keyed.begin(), keyed.end());
  for (std::size_t r = 1; r < keyed.size(); ++r)
    if (keyed[r].first == keyed[r - 1].first)
      throw std::runtime_error(
          "surface-code CNOT schedule: two same-type stabilizers share their "
          "first supported data qubit, so the schedule rows cannot be aligned "
          "with the parity-matrix rows.");

  cudaqx::tensor<uint8_t> t({rows.size(), num_data});
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (std::size_t c = 0; c < num_data; ++c)
      t.at({r, c}) = rows[keyed[r].second][c];
  return t;
}

/// Flatten a schedule matrix into (stabilizer, data) index pairs ordered by
/// timestep within each stabilizer — the replay format used by kernels that
/// take an explicit CNOT pair list.
std::vector<std::size_t>
schedule_to_pairs(const cudaqx::tensor<uint8_t> &sched) {
  std::vector<std::size_t> pairs;
  const auto num_stabs = sched.shape()[0];
  const auto num_data = sched.shape()[1];
  std::vector<std::pair<uint8_t, std::size_t>> row; // (step, data)
  for (std::size_t s = 0; s < num_stabs; ++s) {
    row.clear();
    for (std::size_t d = 0; d < num_data; ++d)
      if (sched.at({s, d}) != 0)
        row.emplace_back(sched.at({s, d}), d);
    std::sort(row.begin(), row.end());
    for (const auto &[step, d] : row) {
      pairs.push_back(s);
      pairs.push_back(d);
    }
  }
  return pairs;
}

} // namespace

cudaqx::tensor<uint8_t> stabilizer_grid::get_cnot_schedule_x() const {
  // A fault on the ancilla mid-round propagates onto the data qubits of the
  // CNOTs that have not executed yet ("hook error"): an X⊗X pair on the last
  // two data qubits in the schedule. If the X logical runs along the top row
  // (horizontal), that pair must be vertical (column-major order) so hook
  // chains run against the logical instead of along it; otherwise the pair
  // must be horizontal (row-major order).
  const bool x_logical_on_top_row =
      orientation_ == sc_orientation::XV || orientation_ == sc_orientation::ZH;
  return build_cnot_schedule(x_stab_coords, data_indices, distance * distance,
                             /*use_col_major_order=*/x_logical_on_top_row);
}

cudaqx::tensor<uint8_t> stabilizer_grid::get_cnot_schedule_z() const {
  // Same reasoning as get_cnot_schedule_x() for Z⊗Z hook pairs versus the Z
  // logical, which runs perpendicular to the X logical: the Z plaquettes use
  // the transposed order. The two orders also form a conflict-free pair if
  // the X and Z timesteps are ever interleaved.
  const bool x_logical_on_top_row =
      orientation_ == sc_orientation::XV || orientation_ == sc_orientation::ZH;
  return build_cnot_schedule(z_stab_coords, data_indices, distance * distance,
                             /*use_col_major_order=*/!x_logical_on_top_row);
}

std::vector<std::size_t> stabilizer_grid::get_cnot_schedule_pairs_x() const {
  return schedule_to_pairs(get_cnot_schedule_x());
}

std::vector<std::size_t> stabilizer_grid::get_cnot_schedule_pairs_z() const {
  return schedule_to_pairs(get_cnot_schedule_z());
}

std::vector<cudaq::spin_op_term>
stabilizer_grid::get_spin_op_observables() const {
  std::vector<cudaq::spin_op_term> spin_op_obs;

  // The valid logical pair depends on orientation. For XV and ZH, the X logical
  // runs along the top row of data qubits and the Z logical along the left
  // column. For XH and ZV the boundary types are swapped, so the assignment is
  // exchanged: X logical along the left column, Z logical along the top row.
  // Picking the wrong pair yields operators that do not commute with the
  // stabilizers.
  const bool x_logical_on_top_row =
      orientation_ == sc_orientation::XV || orientation_ == sc_orientation::ZH;

  // Support that runs along the top row of data qubits.
  std::string top_row_obs(data_coords.size(), 'I');
  // Support that runs along the left column of data qubits.
  std::string left_col_obs(data_coords.size(), 'I');

  if (x_logical_on_top_row) {
    // X obs runs along top row of data qubits.
    for (size_t i = 0; i < distance; ++i)
      top_row_obs[i] = 'X';
    // Z obs runs along left col of data qubits.
    for (size_t i = 0; i < data_coords.size(); i += distance)
      left_col_obs[i] = 'Z';
    spin_op_obs.emplace_back(cudaq::spin_op::from_word(top_row_obs));
    spin_op_obs.emplace_back(cudaq::spin_op::from_word(left_col_obs));
  } else {
    // X obs runs along left col of data qubits.
    for (size_t i = 0; i < data_coords.size(); i += distance)
      left_col_obs[i] = 'X';
    // Z obs runs along top row of data qubits.
    for (size_t i = 0; i < distance; ++i)
      top_row_obs[i] = 'Z';
    spin_op_obs.emplace_back(cudaq::spin_op::from_word(left_col_obs));
    spin_op_obs.emplace_back(cudaq::spin_op::from_word(top_row_obs));
  }

  return spin_op_obs;
}

surface_code::surface_code(const heterogeneous_map &options) : code() {
  if (!options.contains("distance"))
    throw std::runtime_error(
        "[surface_code] distance not provided. distance must be provided via "
        "qec::get_code(..., options) options map.");
  distance = options.get<std::size_t>("distance");
  grid = stabilizer_grid(
      distance,
      sc_orientation_from_str(options.get<std::string>("orientation", "ZH")));

  m_stabilizers = grid.get_spin_op_stabilizers();
  m_pauli_observables = grid.get_spin_op_observables();

  // Sort now to avoid repeated sorts later.
  sortStabilizerOps(m_stabilizers);
  sortStabilizerOps(m_pauli_observables);

  operation_encodings.insert(std::make_pair(operation::x, x));
  operation_encodings.insert(std::make_pair(operation::z, z));
  operation_encodings.insert(std::make_pair(operation::cx, cx));
  operation_encodings.insert(std::make_pair(operation::cz, cz));
  operation_encodings.insert(
      std::make_pair(operation::stabilizer_round, stabilizer));
  operation_encodings.insert(std::make_pair(operation::prep0, prep0));
  operation_encodings.insert(std::make_pair(operation::prep1, prep1));
  operation_encodings.insert(std::make_pair(operation::prepp, prepp));
  operation_encodings.insert(std::make_pair(operation::prepm, prepm));
}

std::size_t surface_code::get_num_data_qubits() const {
  return distance * distance;
}

std::size_t surface_code::get_num_ancilla_qubits() const {
  return distance * distance - 1;
}

std::size_t surface_code::get_num_ancilla_x_qubits() const {
  return (distance * distance - 1) / 2;
}

std::size_t surface_code::get_num_ancilla_z_qubits() const {
  return (distance * distance - 1) / 2;
}

std::size_t surface_code::get_num_x_stabilizers() const {
  return (distance * distance - 1) / 2;
}

std::size_t surface_code::get_num_z_stabilizers() const {
  return (distance * distance - 1) / 2;
}

cudaqx::tensor<uint8_t> surface_code::get_stabilizer_schedule_x() const {
  return grid.get_cnot_schedule_x();
}

cudaqx::tensor<uint8_t> surface_code::get_stabilizer_schedule_z() const {
  return grid.get_cnot_schedule_z();
}

/// @brief Register the surace_code type
CUDAQ_EXT_PT_REGISTER_TYPE(surface_code)

} // namespace cudaq::qec::surface_code

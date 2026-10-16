/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qec/code.h"
#include "cudaq/qec/patch.h"
#include <string>

using namespace cudaqx;

namespace cudaq::qec::surface_code {

/// @brief enumerates the role of a grid site in the surface codes stabilizer
/// grid
enum surface_role { amx, amz, empty };

/// @brief Surface-code orientation: controls which Pauli type (X or Z) occupies
/// the bulk interior and which boundaries are X-type vs Z-type.
///
/// Geometry names follow the NVIDIA/Ising-Decoding naming convention
/// (https://github.com/NVIDIA/Ising-Decoding): Ising's `code_rotation`
/// convention uses first character = bulk syndrome type, second character =
/// rotated_type. By geometry: XV and ZH place X-type stabilizers on the
/// left/right boundaries (Z on top/bottom); XH and ZV place X-type on
/// top/bottom (Z on left/right). XV vs ZH (and XH vs ZV) differ in the bulk
/// X/Z checkerboard.
///
/// Observable convention: get_spin_op_observables() returns the valid logical
/// pair for the orientation. XV/ZH use X obs along the top row and Z obs along
/// the left column; XH/ZV swap them (X along the left column, Z along the top
/// row). See get_spin_op_observables() for details.
enum class sc_orientation { XV, XH, ZV, ZH };

/// @brief Convert a surface-code orientation string to a sc_orientation enum
/// (XV, XH, ZV, ZH, or Ising's O1-O4 aliases, where O1/O2/O3/O4 map to
/// XV/XH/ZV/ZH).
sc_orientation sc_orientation_from_str(const std::string &s);

/// @brief describes the 2d coordinate on the stabilizer grid
struct vec2d {
  int row;
  int col;

  vec2d(int row_in, int col_in);
};

vec2d operator+(const vec2d &lhs, const vec2d &rhs);
vec2d operator-(const vec2d &lhs, const vec2d &rhs);
bool operator==(const vec2d &lhs, const vec2d &rhs);
bool operator<(const vec2d &lhs, const vec2d &rhs);

// clang-format off
/// @brief Generates and keeps track of the 2d grid of stabilizers in the
/// rotated surface code.
/// Following same layout convention as in: https://arxiv.org/abs/2311.10687
/// Grid layout is arranged from left to right, top to bottom (row major storage)
/// grid_length = 4 example:
/// ```
/// (0,0)   (0,1)   (0,2)   (0,3)
/// (1,0)   (1,1)   (1,2)   (1,3)
/// (2,0)   (2,1)   (2,2)   (2,3)
/// (3,0)   (3,1)   (3,2)   (3,3)
/// ```
// clang-format on

///
/// Each entry on the grid can be an X stabilizer, Z stabilizer,
/// or empty, as is needed on the edges.
/// The diagrams below show the default ZH orientation. Other orientations
/// assign X/Z roles according to sc_orientation.
/// The grid length of 4 corresponds to a distance 3 surface code, which results
/// in:
/// ```
/// e(0,0)  e(0,1)  Z(0,2)  e(0,3)
/// X(1,0)  Z(1,1)  X(1,2)  e(1,3)
/// e(2,0)  X(2,1)  Z(2,2)  X(2,3)
/// e(3,0)  Z(3,1)  e(3,2)  e(3,3)
/// ```
///
/// This is seen through the `print_stabilizer_grid()` member function.
/// To get rid of the empty sites, the `print_stabilizer_coords()` function is
/// used:
/// ```
///                 Z(0,2)
/// X(1,0)  Z(1,1)  X(1,2)
///         X(2,1)  Z(2,2)  X(2,3)
///         Z(3,1)
/// ```
///
/// and to get the familiar visualization of the distance three surface code,
/// the `print_stabilizer_indices` results in:
/// ```
///         Z0
/// X0  Z1  X1
///     X2  Z2  X3
///     Z3
/// ```
///
/// The data qubits are located at the four corners of each of the weight-4
/// stabilizers. They are also organized with index increasing from left to
/// right, top to bottom:
/// ```
/// d0  d1  d2
/// d3  d4  d5
/// d6  d7  d8
/// ```
class stabilizer_grid {
private:
  /// @brief Generates this->roles
  void generate_grid_roles();
  /// @brief Generates {x,z}_stab_coords and indices
  void generate_grid_indices();
  /// @brief Generates {x,z}_stabilizers
  void generate_stabilizers();

public:
  /// @brief The distance of the code
  /// determines the number of data qubits per dimension
  uint32_t distance = 0;

  /// @brief length of the stabilizer grid
  /// for distance = d data qubits,
  /// the stabilizer grid has length d+1
  uint32_t grid_length = 0;

private:
  sc_orientation orientation_ = sc_orientation::ZH;

public:
  /// @brief flattened vector of the stabilizer grid sites roles'
  /// grid idx -> role
  /// stored in row major order
  std::vector<surface_role> roles;

  /// @brief x stab index -> 2d coord
  std::vector<vec2d> x_stab_coords;

  /// @brief z stab index -> 2d coord
  std::vector<vec2d> z_stab_coords;

  /// @brief 2d coord -> z stab index
  std::map<vec2d, size_t> x_stab_indices;

  /// @brief 2d coord -> z stab index
  std::map<vec2d, size_t> z_stab_indices;

  /// @brief data index -> 2d coord
  /// data qubits are in an offset 2D coord system from stabilizers
  std::vector<vec2d> data_coords;

  /// @brief 2d coord -> data index
  std::map<vec2d, size_t> data_indices;

  /// @brief Each element is an X stabilizer specified by the data qubits it has
  /// support on
  /// In surface code, can have weight 2 or weight 4 stabs
  /// So {x,z}_stabilizer[i].size() == 2 || 4
  std::vector<std::vector<size_t>> x_stabilizers;

  /// @brief Each element is an Z stabilizer specified by the data qubits it has
  /// support on
  std::vector<std::vector<size_t>> z_stabilizers;

  /// @brief Construct the grid from the code's distance
  stabilizer_grid(uint32_t distance,
                  sc_orientation orientation = sc_orientation::ZH);
  /// @brief Empty constructor
  stabilizer_grid();

  /// @brief Get the orientation used to construct this stabilizer grid.
  sc_orientation get_orientation() const;

  /// @brief Print a 2d grid of stabilizer roles
  void print_stabilizer_grid() const;

  /// @brief Print a 2d grid of stabilizer coords
  void print_stabilizer_coords() const;

  /// @brief Print a 2d grid of stabilizer indices
  void print_stabilizer_indices() const;

  /// @brief Print a 2d grid of data qubit indices
  void print_data_grid() const;

  /// @brief Print the coord <--> indices maps
  void print_stabilizer_maps() const;

  /// @brief Print the stabilizers in sparse pauli format
  void print_stabilizers() const;

  /// @brief Get the stabilizers as a vector of cudaq::spin_op_terms
  std::vector<cudaq::spin_op_term> get_spin_op_stabilizers() const;

  /// @brief Get the CNOT schedule matrix for the X stabilizers.
  ///
  /// @return Tensor with the same shape and support pattern as the X block of
  /// the parity check matrix (num_x_stabilizers x distance^2). Entry 0 means
  /// the ancilla does not touch that data qubit; entry k in [1, 4] means the
  /// ancilla-data CNOT executes at timestep k of the stabilizer round.
  ///
  /// @note The per-plaquette CNOT order is chosen per the grid's orientation
  /// so that mid-round ancilla faults ("hook errors"), which propagate onto
  /// the data qubits of the remaining CNOTs, land perpendicular to the
  /// same-type logical operator instead of along it, following the standard
  /// zigzag schedule of https://arxiv.org/abs/1404.3747. Rows are ordered to
  /// match the rows of %to_parity_matrix() / %code::get_parity_x().
  cudaqx::tensor<uint8_t> get_cnot_schedule_x() const;

  /// @brief Get the CNOT schedule matrix for the Z stabilizers.
  ///
  /// @return Tensor with the same shape and support pattern as the Z block of
  /// the parity check matrix (num_z_stabilizers x distance^2). Entry 0 means
  /// the ancilla does not touch that data qubit; entry k in [1, 4] means the
  /// data-ancilla CNOT executes at timestep k of the stabilizer round.
  ///
  /// @note See get_cnot_schedule_x() for the hook-error rationale and row
  /// ordering.
  cudaqx::tensor<uint8_t> get_cnot_schedule_z() const;

  /// @brief Get the X-stabilizer CNOT schedule as a flat list of
  /// (stabilizer index, data index) pairs, ordered by timestep within each
  /// stabilizer — the replay format for kernels that take an explicit CNOT
  /// pair list. Stabilizer indices match the rows of get_cnot_schedule_x().
  std::vector<std::size_t> get_cnot_schedule_pairs_x() const;

  /// @brief Z-stabilizer counterpart of get_cnot_schedule_pairs_x().
  std::vector<std::size_t> get_cnot_schedule_pairs_z() const;

  /// @brief Get the observables as a vector of cudaq::spin_op_terms
  ///
  /// @return The X logical observable first, followed by the Z logical
  /// observable.
  ///
  /// @note Returns the correct logical pair for the grid's orientation. For XV
  /// and ZH the X observable runs along the top row of data qubits and the Z
  /// observable along the left column; for XH and ZV the assignment is swapped
  /// (X along the left column, Z along the top row) to match their boundary
  /// types. The returned X/Z observables commute with the stabilizers and
  /// anticommute with each other for every orientation.
  std::vector<cudaq::spin_op_term> get_spin_op_observables() const;
};

/// \pure_device_kernel
///
/// @brief Apply X gate to a surface_code patch
/// @param p The patch to apply the X gate to
__qpu__ void x(patch p);

/// \pure_device_kernel
///
/// @brief Apply Z gate to a surface_code patch
/// @param p The patch to apply the Z gate to
__qpu__ void z(patch p);

/// \pure_device_kernel
///
/// @brief Apply controlled-X gate between two surface_code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cx(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Apply controlled-Z gate between two surface_code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cz(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |0⟩ state
/// @param p The patch to prepare
__qpu__ void prep0(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |1⟩ state
/// @param p The patch to prepare
__qpu__ void prep1(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |+⟩ state
/// @param p The patch to prepare
__qpu__ void prepp(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |-⟩ state
/// @param p The patch to prepare
__qpu__ void prepm(patch p);

/// \pure_device_kernel
///
/// @brief Perform stabilizer measurements on a surface_code patch
/// @param p The patch to measure
/// @param x_stabilizer_schedule Flattened X-stabilizer CNOT schedule matrix
/// (see stabilizer_grid::get_cnot_schedule_x): entry 0 = no support, entry
/// k >= 1 = CNOT at timestep k
/// @param z_stabilizer_schedule Flattened Z-stabilizer CNOT schedule matrix
/// (see stabilizer_grid::get_cnot_schedule_z)
/// @return Vector of measurement results
__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch p, const std::vector<std::size_t> &x_stabilizer_schedule,
           const std::vector<std::size_t> &z_stabilizer_schedule);

/// @brief surface_code implementation
class surface_code : public cudaq::qec::code {
protected:
  /// @brief The code distance parameter
  std::size_t distance;

  /// @brief Get the number of data qubits in the surface_code
  /// @return Number of data qubits (distance^2 for surface_code)
  std::size_t get_num_data_qubits() const override;

  /// @brief Get the number of total ancilla qubits in the surface_code
  /// @return Number of data qubits (distance^2 - 1 for surface_code)
  std::size_t get_num_ancilla_qubits() const override;

  /// @brief Get the number of X ancilla qubits in the surface_code
  /// @return Number of data qubits ((distance^2 - 1)/2 for surface_code)
  std::size_t get_num_ancilla_x_qubits() const override;

  /// @brief Get the number of Z ancilla qubits in the surface_code
  /// @return Number of data qubits ((distance^2 - 1)/2 for surface_code)
  std::size_t get_num_ancilla_z_qubits() const override;

  /// @brief Get number of X stabilizer that can be measured
  /// @return Number of X-type stabilizers
  std::size_t get_num_x_stabilizers() const override;

  /// @brief Get number of Z stabilizer that can be measured
  /// @return Number of Z-type stabilizers
  std::size_t get_num_z_stabilizers() const override;

public:
  /// @brief Constructor for the surface_code
  surface_code(const heterogeneous_map &);
  // Grid constructor would be useful

  /// @brief Get the hook-error-aware X-stabilizer CNOT schedule matrix.
  /// See stabilizer_grid::get_cnot_schedule_x().
  cudaqx::tensor<uint8_t> get_stabilizer_schedule_x() const override;

  /// @brief Get the hook-error-aware Z-stabilizer CNOT schedule matrix.
  /// See stabilizer_grid::get_cnot_schedule_z().
  cudaqx::tensor<uint8_t> get_stabilizer_schedule_z() const override;

  /// @brief Extension creator function for the surface_code
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      surface_code, static std::unique_ptr<cudaq::qec::code> create(
                        const cudaqx::heterogeneous_map &options) {
        return std::make_unique<surface_code>(options);
      })

  /// @brief Grid to keep track of topological arrangement of qubits.
  stabilizer_grid grid;
};

} // namespace cudaq::qec::surface_code

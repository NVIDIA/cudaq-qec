# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import cudaq
import cudaq_qec as qec


def test_basic_construction_d3():
    g = qec.stabilizer_grid(3)
    assert g.distance == 3
    assert g.grid_length == 4

    assert len(g.x_stab_coords) == 4
    assert len(g.z_stab_coords) == 4
    assert len(g.x_stabilizers) == 4
    assert len(g.z_stabilizers) == 4

    assert len(g.data_coords) == 9


def test_orientation_construction_d3():
    g = qec.stabilizer_grid(3, qec.sc_orientation.XH)
    assert g.orientation == qec.sc_orientation.XH
    assert len(g.x_stabilizers) == 4
    assert len(g.z_stabilizers) == 4

    alias_grid = qec.stabilizer_grid(3, "O3")
    assert alias_grid.orientation == qec.sc_orientation.ZV
    assert len(alias_grid.x_stabilizers) == 4
    assert len(alias_grid.z_stabilizers) == 4

    canonical_grid = qec.stabilizer_grid(3, " ZH ")
    assert canonical_grid.orientation == qec.sc_orientation.ZH
    assert len(canonical_grid.x_stabilizers) == 4
    assert len(canonical_grid.z_stabilizers) == 4


def test_get_code_orientation_option():
    code = qec.get_code("surface_code", distance=3, orientation="O3")
    assert isinstance(code, qec.Code)

    np.testing.assert_array_equal(
        code.get_observables_x(),
        np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(
        code.get_observables_z(),
        np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8))


def test_roles_layout_d3():
    g = qec.stabilizer_grid(3)
    roles = g.roles
    assert len(roles) == g.grid_length * g.grid_length

    def role_at(r, c):
        return roles[r * g.grid_length + c]

    # e(0,0)  e(0,1)  Z(0,2)  e(0,3)
    assert qec.role_to_str(role_at(0, 0)) == "e"
    assert qec.role_to_str(role_at(0, 1)) == "e"
    assert qec.role_to_str(role_at(0, 2)) == "Z"
    assert qec.role_to_str(role_at(0, 3)) == "e"

    # X(1,0)  Z(1,1)  X(1,2)  e(1,3)
    assert qec.role_to_str(role_at(1, 0)) == "X"
    assert qec.role_to_str(role_at(1, 1)) == "Z"
    assert qec.role_to_str(role_at(1, 2)) == "X"
    assert qec.role_to_str(role_at(1, 3)) == "e"

    # e(2,0)  X(2,1)  Z(2,2)  X(2,3)
    assert qec.role_to_str(role_at(2, 0)) == "e"
    assert qec.role_to_str(role_at(2, 1)) == "X"
    assert qec.role_to_str(role_at(2, 2)) == "Z"
    assert qec.role_to_str(role_at(2, 3)) == "X"

    # e(3,0)  Z(3,1)  e(3,2)  e(3,3)
    assert qec.role_to_str(role_at(3, 0)) == "e"
    assert qec.role_to_str(role_at(3, 1)) == "Z"
    assert qec.role_to_str(role_at(3, 2)) == "e"
    assert qec.role_to_str(role_at(3, 3)) == "e"


def test_index_maps_and_stabilizers_d3():
    g = qec.stabilizer_grid(3)

    xi = g.x_stab_indices
    zi = g.z_stab_indices
    di = g.data_indices

    assert (1, 0) in xi
    assert (0, 2) in zi
    assert (3, 1) in zi
    assert (2, 3) in xi

    for r in range(3):
        for c in range(3):
            assert di[(r, c)] == r * 3 + c

    # X weigth-2 stabilizer
    x_idx = xi[(1, 0)]
    supp = g.x_stabilizers[x_idx]
    assert supp == sorted([0, 3])

    # Z weigth-2 stabilizer
    z_idx = zi[(0, 2)]
    supp = g.z_stabilizers[z_idx]
    assert supp == sorted([1, 2])


def test_text_formatters_d3():
    g = qec.stabilizer_grid(3)

    txt_grid = g.format_stabilizer_grid()
    txt_coords = g.format_stabilizer_coords()
    txt_idx = g.format_stabilizer_indices()
    txt_data = g.format_data_grid()
    txt_stabs = g.format_stabilizers()

    assert "Z(" in txt_grid and "X(" in txt_grid
    assert "Z(" in txt_coords and "X(" in txt_coords
    assert "Z0" in txt_idx and "X0" in txt_idx
    assert "d0" in txt_data and "d8" in txt_data
    assert "X" in txt_stabs and "Z" in txt_stabs


def test_count_d3():
    g = qec.stabilizer_grid(3)
    assert g.distance == 3
    assert g.grid_length == 4

    stabs = g.get_spin_op_stabilizers()
    obs = g.get_spin_op_observables()

    assert isinstance(stabs, list)
    assert isinstance(obs, list)
    assert len(stabs) == 8
    assert len(obs) == 2

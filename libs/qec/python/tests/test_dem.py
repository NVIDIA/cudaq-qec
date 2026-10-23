# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec

import subprocess


def is_nvidia_gpu_available():
    """Check if NVIDIA GPU is available using nvidia-smi command."""
    try:
        result = subprocess.run(["nvidia-smi"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode == 0 and "GPU" in result.stdout:
            return True
    except FileNotFoundError:
        # The nvidia-smi command is not found, indicating no NVIDIA GPU drivers
        return False
    return False


# Helper function to convert a binary matrix to a convenient string
def mat_to_str(mat):
    s = ''
    if mat.ndim == 1:
        for col in mat:
            if col == 0:
                s += '.'
            elif col < 0:
                s += '-'
            elif col >= 10:
                s += '*'
            else:
                s += str(int(col))
        s += '\n'
        return s
    else:
        for row in mat:
            for col in row:
                if col == 0:
                    s += '.'
                elif col < 0:
                    s += '-'
                elif col >= 10:
                    s += '*'
                else:
                    s += str(int(col))
            s += '\n'
    return s


# Use the fixture to set the target
@pytest.fixture(scope="module", autouse=True)
def set_target():
    cudaq.set_target("stim")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize(
    "dem_fn",
    [
        qec.dem_from_memory_circuit,
        qec.x_dem_from_memory_circuit,
        qec.z_dem_from_memory_circuit,
    ],
)
def test_dem_from_memory_circuit_requires_noise_model(dem_fn):
    # DEM generation needs noise mechanisms, so omitted/None noise must fail
    # before the binding dereferences an empty optional noise model.
    code = qec.get_code('steane')
    statePrep = qec.operation.prep0

    with pytest.raises(RuntimeError, match="requires a noise model"):
        dem_fn(code, statePrep, 1)

    with pytest.raises(RuntimeError, match="requires a noise model"):
        dem_fn(code, statePrep, 1, None)


def test_dem_from_memory_circuit():
    code = qec.get_code('steane')
    p = 0.01
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 2

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    # Mechanisms that share a detector syndrome but flip a different observable
    # are kept as distinct columns (they are not merged), so the same detector
    # column pattern can appear more than once with a distinct observable row.
    expected_detector_error_matrix = """
11111.....1...............................
..1.1111...111............................
...11..111..1.1111........................
..........111.1.1.1111.....1..............
.............1.11..1.1111...111...........
.................1..11..111..1.1111.......
...........................111.1.1.1111...
..............................1.11..1.111.
..................................1..11.11
"""
    # Uncomment the following line to get a string representation of the DEM
    # that you can compare to expected_detector_error_matrix.
    # print(mat_to_str(dem.detector_error_matrix), end='')
    assert '\n' + mat_to_str(
        dem.detector_error_matrix) == expected_detector_error_matrix

    # Uncomment the following line to get a string representation of the error
    # rates that you can compare to expected_error_rates.
    # print(np.round(dem.error_rates, 4))

    # The following error rates were captured from the above print statement and
    # are considered "truth" data now.
    expected_error_rates = [
        0.0132, 0.0053, 0.0236, 0.0158, 0.021, 0.0236, 0.008, 0.0236, 0.0053,
        0.0132, 0.0106, 0.0053, 0.0053, 0.0106, 0.0053, 0.0053, 0.0053, 0.0106,
        0.0132, 0.0158, 0.0158, 0.0184, 0.0106, 0.008, 0.0158, 0.0053, 0.0132,
        0.0106, 0.0053, 0.0053, 0.0106, 0.0053, 0.0053, 0.0053, 0.0106, 0.0027,
        0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027
    ]
    assert np.allclose(dem.error_rates, expected_error_rates, atol=1e-4)

    expected_observables_flips_matrix = (
        '.1....11.1.....1.......11.1.....1......111\n')
    # Uncomment the following line to get a string representation of the
    # observables flips matrix that you can compare to
    # expected_observables_flips_matrix.
    assert mat_to_str(
        dem.observables_flips_matrix) == expected_observables_flips_matrix


def test_x_dem_from_memory_circuit():
    code = qec.get_code('steane')
    p = 0.01
    noise = cudaq.NoiseModel()
    # X stabilizers detect Z errors, but we need X errors for X basis prep
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prepp  # X basis preparation
    nRounds = 2

    dem = qec.x_dem_from_memory_circuit(code, statePrep, nRounds, noise)

    # X DEM should have different structure from Z DEM
    # Verify basic properties
    assert dem.detector_error_matrix.shape[0] > 0
    assert dem.detector_error_matrix.shape[1] > 0
    assert len(dem.error_rates) == dem.detector_error_matrix.shape[1]
    assert dem.observables_flips_matrix.shape[0] > 0
    assert dem.observables_flips_matrix.shape[
        1] == dem.detector_error_matrix.shape[1]

    # Error rates should be positive
    assert all(rate >= 0 for rate in dem.error_rates)
    # At least some non-zero rates
    assert any(rate > 0 for rate in dem.error_rates)


def test_decoding_from_dem_from_memory_circuit():
    cudaq.set_random_seed(13)
    code = qec.get_code('steane')
    Lz = code.get_observables_z()
    p = 0.01
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 6
    nShots = 200

    dem = qec.dem_from_memory_circuit(code, statePrep, nRounds, noise)
    syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots,
                                                nRounds, noise)
    syndromes = np.asarray(syndromes, dtype=np.uint8)
    logical_measurements = ((Lz @ data.transpose()) % 2).flatten().astype(
        np.uint8)

    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(syndromes)
    error_predictions = np.asarray(dr.result, dtype=np.uint8)
    dr_converged = np.asarray(dr.converged, dtype=np.uint8)

    data_predictions = (dem.observables_flips_matrix @ error_predictions.T) % 2
    print(f'data_predictions.shape: {data_predictions.shape}')
    print(f'dr_converged        : {mat_to_str(dr_converged)}', end='')
    print(f'data_predictions    : {mat_to_str(data_predictions)}', end='')
    print(f'logical_measurements: {mat_to_str(logical_measurements)}', end='')

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    print(f'nLogicalErrorsWithoutDecoding : {nLogicalErrorsWithoutDecoding}')
    print(f'nLogicalErrorsWithDecoding    : {nLogicalErrorsWithDecoding}')
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


# TODO: Enable tensor network decoder once that goes into main.
@pytest.mark.parametrize(
    "decoder_name,error_rate",
    [
        ("single_error_lut", 0.003),
        ("nv-qldpc-decoder", 0.003),
        # ("tensor_network_decoder", 0.003),
    ])
def test_decoding_from_surface_code_dem_from_memory_circuit(
        decoder_name, error_rate):
    # If this machine only has a CPU, then skip the nv-qldpc-decoder test.
    if decoder_name == "nv-qldpc-decoder" and not is_nvidia_gpu_available():
        pytest.skip("nv-qldpc-decoder requires a GPU")

    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    Lz = code.get_observables_z()
    p = error_rate
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 5
    nShots = 2000

    # Sample the memory circuit with errors
    syndromes, data = qec.z_sample_memory_circuit(code, statePrep, nShots,
                                                  nRounds, noise)

    logical_measurements = (Lz @ data.transpose()) % 2
    # only one logical qubit, so do not need the second axis
    logical_measurements = logical_measurements.flatten()

    # Sum syndromes across the second dimension
    num_syn_triggered_per_shot = np.sum(syndromes, axis=1)

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    try:
        if decoder_name == "tensor_network_decoder":
            # Print the shape of the matrices
            decoder = qec.get_decoder('tensor_network_decoder',
                                      dem.detector_error_matrix,
                                      logical_obs=dem.observables_flips_matrix,
                                      noise_model=dem.error_rates,
                                      contractor_name="cutensornet",
                                      dtype="float32",
                                      device="cuda:0")
        elif decoder_name == "nv-qldpc-decoder":
            osd_method = 3  # Combination Sweep
            decoder = qec.get_decoder('nv-qldpc-decoder',
                                      dem.detector_error_matrix,
                                      max_iterations=50,
                                      error_rate_vec=np.array(dem.error_rates),
                                      use_sparsity=True,
                                      use_osd=osd_method > 0,
                                      osd_order=1000,
                                      osd_method=osd_method)
        elif decoder_name == "single_error_lut":
            decoder = qec.get_decoder('single_error_lut',
                                      dem.detector_error_matrix)
        else:
            raise ValueError(f'Invalid decoder name: {decoder_name}')
    except Exception as e:
        print(
            f'Error getting decoder: {e}, probably not available in this build')
        pytest.skip(f'{decoder_name} not available in this build')
    print(f'decoder_name: {decoder_name}')

    dr = decoder.decode_batch(syndromes)
    error_predictions = np.asarray(dr.result, dtype=np.uint8)
    dr_converged = np.asarray(dr.converged, dtype=np.uint8)
    if decoder_name == "tensor_network_decoder":
        # Tensor network decoder returns the observable flips, not the error predictions.
        data_predictions = np.round(dr.result).astype(np.uint8).T
    else:
        data_predictions = (
            dem.observables_flips_matrix @ error_predictions.T) % 2

    print(f'data_predictions.shape: {data_predictions.shape}')
    if nShots < 200:
        print(f'num_syn_per_shot    : {mat_to_str(num_syn_triggered_per_shot)}',
              end='')
        print(f'dr_converged        : {mat_to_str(dr_converged)}', end='')
        print(f'data_predictions    : {mat_to_str(data_predictions)}', end='')
        print(f'logical_measurements: {mat_to_str(logical_measurements)}',
              end='')

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    print(f'nLogicalErrorsWithoutDecoding : {nLogicalErrorsWithoutDecoding}')
    print(f'nLogicalErrorsWithDecoding    : {nLogicalErrorsWithDecoding}')
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


def test_pymatching_observable_model_does_not_select_output_basis():
    H = np.eye(2, dtype=np.uint8)
    O = np.array([[1, 1]], dtype=np.uint8)
    rates = np.array([0.1, 0.1])
    syndrome = np.array([1, 0], dtype=np.uint8)

    error_decoder = qec.get_decoder('pymatching', H, O=O, error_rate_vec=rates)
    error_result = error_decoder.decode(syndrome)
    np.testing.assert_array_equal(error_result.result, [1.0, 0.0])

    observable_decoder = qec.get_decoder('pymatching',
                                         H,
                                         O=O,
                                         output='observables',
                                         error_rate_vec=rates)
    observable_result = observable_decoder.decode(syndrome)
    np.testing.assert_array_equal(observable_result.result, [1.0])

    parallel_H = np.array([[1, 1]], dtype=np.uint8)
    parallel_O = np.array([[1, 0]], dtype=np.uint8)
    parallel_rates = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        qec.get_decoder('pymatching',
                        parallel_H,
                        O=parallel_O,
                        error_rate_vec=parallel_rates)
    with pytest.raises(ValueError):
        qec.get_decoder('pymatching',
                        parallel_H,
                        O=parallel_O,
                        output='errors',
                        error_rate_vec=parallel_rates)
    qec.get_decoder('pymatching',
                    parallel_H,
                    O=parallel_O,
                    output='observables',
                    error_rate_vec=parallel_rates)


def test_pymatching_decode_to_observable_surface_code_dem():
    """Test explicit PyMatching observable output on a surface-code DEM."""
    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    Lz = code.get_observables_z()
    p = 0.003
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 5
    nShots = 2000

    syndromes, data = qec.z_sample_memory_circuit(code, statePrep, nShots,
                                                  nRounds, noise)

    logical_measurements = (Lz @ data.transpose()) % 2
    logical_measurements = logical_measurements.flatten()

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)

    decoder = qec.get_decoder(
        'pymatching',
        dem.detector_error_matrix,
        O=dem.observables_flips_matrix,
        output='observables',
        error_rate_vec=np.array(dem.error_rates),
    )

    dr = decoder.decode_batch(syndromes)
    # Constructed for observable output, so each row is observable flips
    # (length num_observables), not error predictions.
    obs_per_shot = np.asarray(dr.result, dtype=np.float64)
    data_predictions = np.round(obs_per_shot).astype(np.uint8).T

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


def test_pcm_extend_to_n_rounds():
    # This test independently compares the functionality of dem_from_memory_circuit
    # (of two different numbers of rounds) to pcm_extend_to_n_rounds.
    statePrep = qec.operation.prep0
    nRounds = 5
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.003), 1)
    code = qec.get_code('surface_code', distance=5)
    dem05 = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    nSyndromesPerRound = code.get_num_z_stabilizers()
    dem15 = qec.z_dem_from_memory_circuit(code, statePrep, 3 * nRounds, noise)
    H05 = dem05.detector_error_matrix
    H15 = dem15.detector_error_matrix
    H15_new, column_list = qec.pcm_extend_to_n_rounds(H05, nSyndromesPerRound,
                                                      3 * nRounds + 1)
    # Check if H15 == H15_new, one column at a time.
    for c in range(H15.shape[1]):
        if not np.allclose(H15[:, c], H15_new[:, c]):
            print(f'Column {c} is not equal')
            # Use join to print the columns as a string without spaces
            print(f'H15    [:, c] : {"".join(map(str, H15[:, c]))}')
            print(f'H15_new[:, c] : {"".join(map(str, H15_new[:, c]))}')
            assert False
    assert len(column_list) == H15_new.shape[1]

    # Check the extended error rates.
    H15_new_error_rates = np.array(dem05.error_rates)[column_list]
    assert np.allclose(H15_new_error_rates, dem15.error_rates, atol=1e-6)


def test_dem_from_memory_circuit_boundary_row_format():
    """Test the data layout produced by dem_from_memory_circuit
    Syndromes are formatted as:
        [initial boundary: numFixed rows]
        [(numRounds - 1) interior round-to-round transitions, each
         numSyndromesPerRound rows, laid out as [Z rows][X rows]]
        [final boundary: numFixed rows]
    where `numFixed` is the number of stabilizers matching the prep/readout
    basis (Z stabilizers for prep0/prep1), and both boundary groups contain
    only that basis 
    """
    code = qec.get_code('steane')
    numXStabs = code.get_num_x_stabilizers()
    numZStabs = code.get_num_z_stabilizers()
    assert numXStabs == numZStabs

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    statePrep = qec.operation.prep0  # Z-basis prep => Z is the "fixed" basis.

    for nRounds in [1, 2, 3, 4, 6]:
        full_dem = qec.dem_from_memory_circuit(code, statePrep, nRounds, noise)
        z_dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
        x_dem = qec.x_dem_from_memory_circuit(code, statePrep, nRounds, noise)

        expected_z_rows = (nRounds + 1) * numZStabs
        expected_x_rows = max(nRounds - 1, 0) * numXStabs

        assert z_dem.detector_error_matrix.shape[0] == expected_z_rows, (
            f"nRounds={nRounds}: z_dem has "
            f"{z_dem.detector_error_matrix.shape[0]} rows, expected "
            f"{expected_z_rows}")
        assert x_dem.detector_error_matrix.shape[0] == expected_x_rows, (
            f"nRounds={nRounds}: x_dem has "
            f"{x_dem.detector_error_matrix.shape[0]} rows, expected "
            f"{expected_x_rows}")

        # Z (the fixed/boundary basis) and X (the other basis) partition the
        # full DEM's rows disjointly and exhaustively.
        assert (z_dem.detector_error_matrix.shape[0] +
                x_dem.detector_error_matrix.shape[0] ==
                full_dem.detector_error_matrix.shape[0])

    # With only one round, there are zero interior rounds, so an X-only DEM under a Z-prep
    # experiment has literally no rows or columns at all.
    x_dem_one_round = qec.x_dem_from_memory_circuit(code, statePrep, 1, noise)
    assert x_dem_one_round.detector_error_matrix.shape == (0, 0)

    # ...and, symmetrically, an X-prep experiment flips which basis is
    # "fixed": now it is the Z-only DEM that is empty for a single round.
    xPrep = qec.operation.prepp
    z_dem_one_round = qec.z_dem_from_memory_circuit(code, xPrep, 1, noise)
    assert z_dem_one_round.detector_error_matrix.shape == (0, 0)


# ---------------------------------------------------------------------------
# Supplementary regression tests retained on top of PR #610.
# #610 already fixed canonicalize_for_rounds (observable-aware merging,
# zero-syndrome handling, order-independent composition) and added C++ unit
# tests for it. These tests add what #610 does not have: an external Stim
# oracle cross-check of the synthesized DEM and a Stim-DEM decode smoke test
# through the PyMatching plugin. Stim is optional, so the imports live inside
# the tests via pytest.importorskip and platforms without Stim skip cleanly.
# ---------------------------------------------------------------------------


def _build_steane_z_memory_stim_circuit(stim_mod, p, n_rounds):
    # Mirrors Steane prep0 and stabilizer schedules in steane_device.cpp.
    # Qubits: data 0..6, ancx 7..9, ancz 10..12.
    stab_supports = [
        [0, 1, 2, 3],
        [1, 2, 4, 5],
        [2, 3, 5, 6],
    ]
    prep0_cx_pairs = [(0, 1), (4, 5), (6, 3), (6, 5), (4, 2), (0, 3), (4, 1),
                      (3, 2)]

    c = stim_mod.Circuit()
    c.append("R", list(range(13)))
    c.append("H", [0, 4, 6])
    for ctrl, tgt in prep0_cx_pairs:
        c.append("CX", [ctrl, tgt])
        c.append("DEPOLARIZE2", [ctrl, tgt], p)

    for r in range(n_rounds):
        c.append("H", [7, 8, 9])
        for xi, support in enumerate(stab_supports):
            for di in support:
                c.append("CX", [7 + xi, di])
                c.append("DEPOLARIZE2", [7 + xi, di], p)
        c.append("H", [7, 8, 9])

        for zi, support in enumerate(stab_supports):
            for di in support:
                c.append("CX", [di, 10 + zi])
                c.append("DEPOLARIZE2", [di, 10 + zi], p)

        # cudaq-qec measures ancz first, then ancx.
        c.append("M", [10, 11, 12, 7, 8, 9])
        c.append("R", [7, 8, 9, 10, 11, 12])

        # Z-DEM keeps ancz detectors only; later rounds compare to prior ancz.
        if r == 0:
            for zi in range(3):
                c.append("DETECTOR", [stim_mod.target_rec(-6 + zi)])
        else:
            for zi in range(3):
                c.append("DETECTOR", [
                    stim_mod.target_rec(-6 + zi),
                    stim_mod.target_rec(-12 + zi),
                ])

    c.append("M", [0, 1, 2, 3, 4, 5, 6])

    # Final boundary: reconstruct each Z stabilizer from the transversal data
    # measurement and compare it to the last ancz round, catching errors that
    # happen between the last syndrome round and the final readout. Data
    # qubit di lands at rec(-7 + di); the last ancz round (measured as
    # [10, 11, 12, 7, 8, 9] before this M) lands at rec(-13 + zi).
    for zi, support in enumerate(stab_supports):
        c.append("DETECTOR", [stim_mod.target_rec(-7 + di) for di in support] +
                 [stim_mod.target_rec(-13 + zi)])

    # Z_L = Z_4 Z_5 Z_6.
    c.append("OBSERVABLE_INCLUDE",
             [stim_mod.target_rec(i) for i in (-3, -2, -1)], 0)
    return c


def _independent_merge(p, q):
    # P(A xor B) for independent events.
    return p + q - 2.0 * p * q


def _stim_dem_to_multiset(stim_dem):
    # Match cudaq-qec canonicalize_for_rounds by dropping no-syndrome errors.
    bucket = {}
    for inst in stim_dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        det_set = []
        obs_set = []
        for tgt in inst.targets_copy():
            if tgt.is_relative_detector_id():
                det_set.append(tgt.val)
            elif tgt.is_logical_observable_id():
                obs_set.append(tgt.val)
        if not det_set:
            continue
        key = (tuple(sorted(det_set)), tuple(sorted(obs_set)))
        if key in bucket:
            bucket[key] = _independent_merge(bucket[key], prob)
        else:
            bucket[key] = prob
    return bucket


def _cudaq_dem_to_multiset(dem):
    # Same keying as _stim_dem_to_multiset: (detectors, observables) -> prob.
    H = np.asarray(dem.detector_error_matrix)
    O = np.asarray(dem.observables_flips_matrix)
    rates = np.asarray(dem.error_rates)
    bucket = {}
    for c in range(H.shape[1]):
        key = (tuple(np.flatnonzero(H[:, c]).tolist()),
               tuple(np.flatnonzero(O[:, c]).tolist()))
        prob = float(rates[c])
        if key in bucket:
            bucket[key] = _independent_merge(bucket[key], prob)
        else:
            bucket[key] = prob
    return bucket


def _stim_dem_to_arrays(stim_dem):
    # Split decomposed Stim errors into graphlike H/O columns for PyMatching.
    n_dets = stim_dem.num_detectors
    n_obs = stim_dem.num_observables
    h_cols, o_cols, rates = [], [], []
    for inst in stim_dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        components = [[]]
        for tgt in inst.targets_copy():
            if tgt.is_separator():
                components.append([])
            else:
                components[-1].append(tgt)
        for comp in components:
            h_col = np.zeros(n_dets, dtype=np.uint8)
            o_col = np.zeros(n_obs, dtype=np.uint8)
            for tgt in comp:
                if tgt.is_relative_detector_id():
                    h_col[tgt.val] = 1
                elif tgt.is_logical_observable_id():
                    o_col[tgt.val] = 1
            # Matching cannot use a purely-logical component with no syndrome.
            if h_col.sum() == 0:
                continue
            h_cols.append(h_col)
            o_cols.append(o_col)
            rates.append(prob)
    H = np.stack(h_cols, axis=1) if h_cols else np.zeros(
        (n_dets, 0), dtype=np.uint8)
    O = np.stack(o_cols, axis=1) if o_cols else np.zeros(
        (n_obs, 0), dtype=np.uint8)
    return H, O, np.asarray(rates, dtype=np.float64)


def test_z_dem_from_memory_circuit_against_stim_oracle():
    # Cross-check cudaq-qec's Steane Z-DEM against an independent Stim DEM built
    # from an equivalent circuit. Both sides drop no-syndrome errors so the
    # comparison matches dem_from_memory_circuit's remove_zero_syndrome_errors.
    stim_mod = pytest.importorskip(
        "stim",
        reason=
        "stim not installed; skipping Stim oracle cross-check for Steane Z-DEM")

    code = qec.get_code('steane')
    p = 0.01
    n_rounds = 2
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    cudaq_dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0,
                                              n_rounds, noise)

    stim_circuit = _build_steane_z_memory_stim_circuit(stim_mod, p, n_rounds)
    stim_dem = stim_circuit.detector_error_model(decompose_errors=False)

    # ASSERT: detector/observable counts must agree before comparing columns,
    # otherwise the signature keys below are not comparable.
    assert cudaq_dem.detector_error_matrix.shape[0] == stim_dem.num_detectors, (
        f"num_detectors mismatch: cudaq="
        f"{cudaq_dem.detector_error_matrix.shape[0]}, "
        f"stim={stim_dem.num_detectors}")
    assert cudaq_dem.observables_flips_matrix.shape[
        0] == stim_dem.num_observables, (
            f"num_observables mismatch: cudaq="
            f"{cudaq_dem.observables_flips_matrix.shape[0]}, "
            f"stim={stim_dem.num_observables}")

    cudaq_terms = _cudaq_dem_to_multiset(cudaq_dem)
    stim_terms = _stim_dem_to_multiset(stim_dem)

    # ASSERT: identical (detector, observable) signature sets. This is the core
    # regression guard: if canonicalization ever again merged same-syndrome
    # different-observable mechanisms, cudaq would be missing observable keys.
    cudaq_keys = set(cudaq_terms)
    stim_keys = set(stim_terms)
    assert cudaq_keys == stim_keys, (
        f"DEM key sets differ. cudaq-only keys: "
        f"{sorted(cudaq_keys - stim_keys)}; "
        f"stim-only keys: {sorted(stim_keys - cudaq_keys)}")

    # ASSERT: per-signature probabilities agree; the tolerance absorbs benign
    # differences in how the two tools group equivalent Pauli outcomes.
    for k in cudaq_keys:
        assert np.isclose(
            cudaq_terms[k], stim_terms[k], atol=1e-4,
            rtol=1e-3), (f"probability mismatch at {k}: "
                         f"cudaq={cudaq_terms[k]}, stim={stim_terms[k]}")


def test_pymatching_decodes_stim_surface_code_dem():
    # End-to-end smoke test: feed a Stim-generated rotated surface-code DEM
    # (which contains parallel matching edges) through cudaq-qec's PyMatching
    # plugin and confirm decoding reduces the logical error count.
    stim_mod = pytest.importorskip(
        "stim",
        reason="stim not installed; skipping Stim-based PyMatching decode test")

    distance = 5
    n_rounds = 5
    p = 0.003
    n_shots = 2000

    circuit = stim_mod.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=n_rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    # Matching plugin expects graphlike columns (1 or 2 detectors).
    stim_dem = circuit.detector_error_model(decompose_errors=True)

    H, O, rates = _stim_dem_to_arrays(stim_dem)

    # Sample syndromes and true observable flips from the same Stim circuit.
    sampler = circuit.compile_detector_sampler(seed=13)
    syndromes_bool, obs_bool = sampler.sample(shots=n_shots,
                                              separate_observables=True)
    syndromes = syndromes_bool.astype(np.uint8)
    logical_measurements = obs_bool.astype(np.uint8).flatten()

    # Surface-code DEMs have parallel edges; let PyMatching merge them.
    try:
        decoder = qec.get_decoder(
            'pymatching',
            H,
            O=O,
            output='observables',
            error_rate_vec=rates,
            merge_strategy='independent',
        )
    except Exception as e:
        pytest.skip(f'pymatching decoder unavailable in this build: {e}')

    dr = decoder.decode_batch(syndromes)
    # Constructed for observable output, so these are observable flips.
    obs_per_shot = np.asarray(dr.result, dtype=np.float64)
    data_predictions = np.round(obs_per_shot).astype(np.uint8).flatten()

    # ASSERT: decoding the observable corrects more logical flips than leaving
    # the raw measurement uncorrected, i.e. the matching decode is useful.
    n_errors_without_decoding = int(np.sum(logical_measurements))
    n_errors_with_decoding = int(np.sum(data_predictions ^
                                        logical_measurements))
    assert n_errors_with_decoding < n_errors_without_decoding, (
        f"PyMatching did not reduce logical errors: "
        f"with_decoding={n_errors_with_decoding}, "
        f"without_decoding={n_errors_without_decoding}")


@pytest.mark.parametrize("statePrep,sample_fn,dem_fn", [
    (qec.operation.prep0, qec.sample_memory_circuit,
     qec.dem_from_memory_circuit),
    (qec.operation.prep0, qec.z_sample_memory_circuit,
     qec.z_dem_from_memory_circuit),
    (qec.operation.prepp, qec.x_sample_memory_circuit,
     qec.x_dem_from_memory_circuit),
])
def test_sample_memory_circuit_aligned_with_dem(statePrep, sample_fn, dem_fn):
    """x_sample_memory_circuit/z_sample_memory_circuit must line up
    column-for-column, in the same detector order, with the rows produced by
    x_dem_from_memory_circuit/z_dem_from_memory_circuit for both a Z-basis
    (prep0/z) and an X-basis (prepp/x) memory experiment.
    """
    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    is_z_prep = statePrep in (qec.operation.prep0, qec.operation.prep1)
    L = code.get_observables_z() if is_z_prep else code.get_observables_x()
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.003), 1)
    nRounds = 5
    nShots = 2000

    dem = dem_fn(code, statePrep, nRounds, noise)
    syndromes, data = sample_fn(code, statePrep, nShots, nRounds, noise)

    # Shape alignment: one syndrome column per DEM detector row.
    assert dem.detector_error_matrix.shape[0] == syndromes.shape[1], (
        f"statePrep={statePrep}, {dem_fn.__name__} has "
        f"{dem.detector_error_matrix.shape[0]} detector rows but "
        f"{sample_fn.__name__} produced {syndromes.shape[1]} syndrome "
        f"columns")

    logical_measurements = ((L @ data.transpose()) % 2).flatten()

    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(syndromes)
    error_predictions = np.asarray(dr.result, dtype=np.uint8)
    data_predictions = (dem.observables_flips_matrix @ error_predictions.T) % 2

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    print(nLogicalErrorsWithoutDecoding, nLogicalErrorsWithDecoding)
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding, (
        f"statePrep={statePrep}: decoding {sample_fn.__name__} syndromes "
        f"with {dem_fn.__name__}'s DEM did not reduce the logical error "
        f"rate ({nLogicalErrorsWithDecoding} vs "
        f"{nLogicalErrorsWithoutDecoding}), suggesting misaligned detector "
        f"order")


def test_decoder_context_from_memory_circuit_requires_noise_model():
    # Like the *_dem_from_memory_circuit bindings, the decoder_context binding
    # dereferences an optional noise model, so noise=None/omitted must fail.
    code = qec.get_code('steane')
    statePrep = qec.operation.prep0

    with pytest.raises(RuntimeError, match="requires a noise model"):
        qec.decoder_context_from_memory_circuit(code, statePrep, 1)

    with pytest.raises(RuntimeError, match="requires a noise model"):
        qec.decoder_context_from_memory_circuit(code, statePrep, 1, None)


def test_decoder_context_and_components():
    # decoder_context_from_memory_circuit returns a decoder_context;
    # canonicalization is deferred until a component method is called.
    # full_component() matches dem_from_memory_circuit;
    # x_component() / z_component() reproduce x_/z_dem_from_memory_circuit.
    code = qec.get_code('steane')
    prep = qec.operation.prep0
    nRounds = 3
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.05), 1)

    ctx = qec.decoder_context_from_memory_circuit(code, prep, nRounds, noise)
    dem = qec.dem_from_memory_circuit(code, prep, nRounds, noise)
    z = qec.z_dem_from_memory_circuit(code, prep, nRounds, noise)
    x = qec.x_dem_from_memory_circuit(code, prep, nRounds, noise)

    # full_component() matches the plain entry point.
    fc_dem, fc_m2d, fc_m2o = ctx.full_component()
    assert mat_to_str(fc_dem.detector_error_matrix) == mat_to_str(
        dem.detector_error_matrix)

    # m2d/m2o maps are present and aligned to the DEM (Python binding surface).
    assert len(fc_m2d) == fc_dem.detector_error_matrix.shape[0]
    assert len(fc_m2o) == fc_dem.observables_flips_matrix.shape[0]
    assert ctx.num_measurements > 0
    assert all(0 <= m < ctx.num_measurements for row in fc_m2d for m in row)

    # x_component() / z_component() reproduce the per-type DEMs and partition
    # the detectors
    zc_dem, zc_m2d, zc_m2o = ctx.z_component()
    xc_dem, xc_m2d, xc_m2o = ctx.x_component()
    assert mat_to_str(zc_dem.detector_error_matrix) == mat_to_str(
        z.detector_error_matrix)
    assert mat_to_str(xc_dem.detector_error_matrix) == mat_to_str(
        x.detector_error_matrix)
    assert (zc_dem.detector_error_matrix.shape[0] +
            xc_dem.detector_error_matrix.shape[0] ==
            fc_dem.detector_error_matrix.shape[0])
    assert len(zc_m2d) == zc_dem.detector_error_matrix.shape[0]


def test_decoder_context_d_sparse_layout():
    # qec.d_sparse(m2d) flattens a measurement-to-detector map into the
    # -1-terminated sparse vector a realtime decoder expects.
    code = qec.get_code('steane')
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.05), 1)
    ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, 3,
                                                  noise)
    fc_dem, fc_m2d, fc_m2o = ctx.full_component()

    d_sparse = qec.d_sparse(fc_m2d)
    # One -1 terminator per detector row.
    assert d_sparse.count(-1) == len(fc_m2d)
    # Reconstruct the m2d rows from the flattened form and compare.
    rebuilt, row = [], []
    for v in d_sparse:
        if v == -1:
            rebuilt.append(row)
            row = []
        else:
            row.append(v)
    assert rebuilt == [list(r) for r in fc_m2d]


def test_d_sparse_does_not_require_explicit_measurement_width():
    assert qec.d_sparse([[2], [], [0, 4]]) == [2, -1, -1, 0, 4, -1]


def test_decoder_context_single_type_code_empty_component():
    # The repetition code has only Z stabilizers, so its X context has no
    # detectors. Verify the empty context is self-consistent (no detectors,
    # empty maps), and the Z context keeps every detector.
    code = qec.get_code('repetition', distance=3)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.05), 1)
    ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, 3,
                                                  noise)

    xc_dem, xc_m2d, xc_m2o = ctx.x_component()
    assert xc_dem.num_detectors() == 0
    assert len(xc_m2d) == 0
    assert qec.d_sparse(xc_m2d) == []  # empty m2d -> empty D_sparse

    zc_dem, zc_m2d, zc_m2o = ctx.z_component()
    fc_dem, fc_m2d, fc_m2o = ctx.full_component()
    assert zc_dem.num_detectors() == fc_dem.num_detectors()
    assert len(zc_m2d) == zc_dem.num_detectors()


# ---------------------------------------------------------------------------
# Tests for dem_stitch / dem_stitch_all / dem_close / dem_close_all and the
# dem_chunks_to_* utility functions exposed via py_extended_dem.cpp.
# All tests use a rep-3 single-round chunk (d=2 seam rows, 3 faults).
# ---------------------------------------------------------------------------


def _rep3_dem_chunk(px=0.01):
    code = qec.css_matrices_from_code(qec.get_code("repetition", distance=3))
    noise = qec.CssNoise()
    noise.px_per_qubit = [px, px, px]
    return qec.extended_dem_from_css_matrices(code, noise)


def test_dem_stitch_interior_grows():
    # After one stitch the contracted seam rows become interior (d=2).
    c = _rep3_dem_chunk()
    ab = qec.dem_stitch(c, c, qec.seam_name.next_round,
                        qec.seam_name.prev_round)
    assert ab.num_interior_rows() == 2
    assert ab.get_seam(qec.seam_name.prev_round).num_rows() == 2
    assert ab.num_faults() == 6  # 2 rounds × 3 qubits


def test_dem_stitch_error_rates_concatenated():
    px = 0.01
    c = _rep3_dem_chunk(px)
    ab = qec.dem_stitch(c, c, qec.seam_name.next_round,
                        qec.seam_name.prev_round)
    assert len(ab.error_rates) == 6
    assert all(abs(p - px) < 1e-12 for p in ab.error_rates)


def test_dem_stitch_all_empty_raises():
    with pytest.raises(Exception):
        qec.dem_stitch_all([])


def test_dem_stitch_all_single_chunk_identity():
    c = _rep3_dem_chunk()
    result = qec.dem_stitch_all([c])
    assert result.num_interior_rows() == c.num_interior_rows()
    assert result.num_faults() == c.num_faults()


def test_dem_stitch_all_three_chunks_interior():
    c = _rep3_dem_chunk()
    abc = qec.dem_stitch_all([c, c, c])
    # 2 stitches × 2 contracted seam rows each = 4 interior rows
    assert abc.num_interior_rows() == 4
    assert abc.num_faults() == 9  # 3 rounds × 3 qubits


def test_dem_close_single_chunk_shape():
    # A single rep-3 chunk has d=2 prev_round seam rows and 3 fault columns.
    # dem_close emits the prev_round seam rows as detectors and drops next_round.
    px = 0.01
    chunk = _rep3_dem_chunk(px)
    closed = qec.dem_close(chunk)
    d = chunk.get_seam(qec.seam_name.prev_round).num_rows()
    assert closed.num_detectors() == d
    assert closed.num_error_mechanisms() == chunk.num_faults()
    assert np.allclose(closed.error_rates, chunk.error_rates)


def test_dem_close_all_matches_close_stitch_all():
    # dem_close_all must produce the same DEM as dem_close(dem_stitch_all(...)).
    T = 4
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    via_stitch = qec.dem_close(qec.dem_stitch_all(chunks))
    via_close_all = qec.dem_close_all(chunks)
    assert via_close_all.num_detectors() == via_stitch.num_detectors()
    assert via_close_all.num_error_mechanisms() == \
        via_stitch.num_error_mechanisms()
    assert np.array_equal(np.asarray(via_close_all.detector_error_matrix),
                          np.asarray(via_stitch.detector_error_matrix))
    assert np.array_equal(np.asarray(via_close_all.observables_flips_matrix),
                          np.asarray(via_stitch.observables_flips_matrix))
    assert np.allclose(via_close_all.error_rates, via_stitch.error_rates)


def test_dem_close_all_empty_raises():
    with pytest.raises(Exception):
        qec.dem_close_all([])


def test_dem_chunk_rounds_single():
    c = _rep3_dem_chunk()
    assert qec.dem_chunk_rounds(c) == 1


def test_dem_chunk_rounds_stitched():
    c = _rep3_dem_chunk()
    ab = qec.dem_stitch(c, c, qec.seam_name.next_round,
                        qec.seam_name.prev_round)
    assert qec.dem_chunk_rounds(ab) == 2


def test_dem_chunks_to_rounds():
    T = 5
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    assert qec.dem_chunks_to_rounds(chunks) == T


def test_dem_chunks_to_detector_round_length_and_values():
    # T chunks of d=2 seam rows → T*d detector entries; entry i belongs to
    # round i//d.
    T = 3
    d = 2
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    rounds = qec.dem_chunks_to_detector_round(chunks)
    assert len(rounds) == T * d
    for i, r in enumerate(rounds):
        assert r == i // d, f"detector {i}: expected round {i // d}, got {r}"


def test_dem_chunks_to_d_sparse_length():
    T = 3
    d = 2
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    d_sparse = qec.dem_chunks_to_d_sparse(chunks)
    # One entry per detector
    assert len(d_sparse) == T * d


def test_dem_chunks_to_o_sparse_length():
    T = 3
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    o_sparse = qec.dem_chunks_to_o_sparse(chunks)
    # One entry per observable (rep-3 has 1)
    assert len(o_sparse) == chunks[0].num_observables()


# ---------------------------------------------------------------------------
# Declarative phase specs
# ---------------------------------------------------------------------------

REP5_CHECKS = 4


def _make_phase_entry(name_str, num_faults, H_sparse, O_sparse, error_rates):
    s = qec.DemChunkSpec()
    s.num_faults = num_faults
    s.H_sparse = list(H_sparse)
    s.O_sparse = list(O_sparse)
    s.error_rates = list(error_rates)
    e = qec.PhaseSpecEntry()
    e.id = qec.SeamId(name_str)
    e.spec = s
    return e


def _make_connection(from_str, to_str):
    c = qec.PhaseConnection()
    c.from_phase = qec.SeamId(from_str)
    c.to_phase = qec.SeamId(to_str)
    return c


def _rep5_spec(num_rounds=5):
    """A d=5 repetition code as init / bulk / final phases.

    Fault columns 0..4 are data qubits and 5..8 measurement errors; the final
    phase measures destructively so it has only the five data columns.
    Spec-built chunks carry only seam rows (no interior rows).
    """
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _make_phase_entry("init", 9,
                          [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1],
                          [0, -1], [0.02] * 9),
        _make_phase_entry("bulk", 9,
                          [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1],
                          [0, -1], [0.02] * 9),
        _make_phase_entry("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1],
                          [0, -1], [0.02] * 5),
    ]
    spec.connections = [
        _make_connection("init", "bulk"),
        _make_connection("bulk", "bulk"),
        _make_connection("bulk", "final"),
    ]
    spec.seam = seam
    spec.num_rounds = num_rounds
    return spec


def test_dem_chunks_spec_validates():
    spec = _rep5_spec()
    spec.validate()
    assert spec.has_repeating_phase()
    assert not spec.is_empty()
    assert qec.DemChunksSpec().is_empty()


def test_dem_chunk_spec_round_trips_through_attributes():
    spec = _rep5_spec()
    assert spec.get_phase(qec.SeamId("init")).num_faults == 9
    assert spec.get_phase(qec.SeamId("final")).num_faults == 5


def test_dem_chunks_from_spec_expands_to_round_count():
    for rounds in range(3, 8):
        spec = _rep5_spec(rounds)
        chunks = qec.dem_chunks_from_spec(spec)
        assert len(chunks) == rounds
        # init has a prev_round seam (initial-state detector band).
        assert chunks[0].has_seam(qec.seam_name.prev_round)
        assert chunks[0].get_seam(qec.seam_name.prev_round).num_rows() > 0
        # final has no outgoing seam.
        assert not chunks[-1].has_seam(qec.seam_name.next_round)
        assert qec.dem_chunks_to_rounds(chunks) == rounds


def test_dem_chunks_from_spec_closes_to_one_detector_band_per_round():
    # init's prev_round seam gives the initial-state detector band (round 0 vs
    # zero), and each inter-chunk boundary gives one more band: rounds total.
    for rounds in range(3, 8):
        spec = _rep5_spec(rounds)
        chunks = qec.dem_chunks_from_spec(spec)
        dem = qec.dem_close_all(chunks)
        assert dem.detector_error_matrix.shape[0] == rounds * REP5_CHECKS
        assert len(qec.dem_chunks_to_d_sparse(chunks)) == rounds * REP5_CHECKS


def test_dem_chunk_from_spec_matches_the_phase_it_describes():
    spec = _rep5_spec()
    init_spec = spec.get_phase(qec.SeamId("init"))
    init = qec.dem_chunk_from_spec(init_spec, [qec.seam_name.next_round])
    assert init.num_faults() == 9
    assert not init.has_seam(qec.seam_name.prev_round)
    assert init.has_seam(qec.seam_name.next_round)
    assert init.get_seam(qec.seam_name.next_round).num_rows() == REP5_CHECKS
    # Called directly with only next_round: no prev_round seam → 0 rounds.
    # (dem_chunks_from_spec always adds prev_round; this test calls
    # dem_chunk_from_spec directly with an explicit seam list.)
    assert qec.dem_chunk_rounds(init) == 0


def test_dem_chunks_spec_rejects_mismatched_error_rate_count():
    spec = _rep5_spec()
    for entry in spec.phases:
        if entry.id == qec.SeamId("init"):
            entry.spec.error_rates = [0.02] * 8  # wrong count
    with pytest.raises(Exception):
        spec.validate()


def test_dem_chunks_from_spec_requires_num_rounds_for_repeating_phase():
    spec = _rep5_spec()
    spec.num_rounds = None
    with pytest.raises(Exception):
        qec.dem_chunks_from_spec(spec)


def test_dem_chunks_from_spec_linear_chain_needs_no_num_rounds():
    # A two-phase linear chain (no self-loop) works without num_rounds.
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _make_phase_entry("init", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1],
                          [0, -1], [0.02] * 5),
        _make_phase_entry("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1],
                          [0, -1], [0.02] * 5),
    ]
    spec.connections = [_make_connection("init", "final")]
    spec.seam = seam
    # num_rounds absent — valid for a linear chain with no self-loop
    chunks = qec.dem_chunks_from_spec(spec)
    assert len(chunks) == 2


def test_dem_chunks_from_spec_rejects_a_cyclic_connection_graph():
    # A -> B and B -> A has no terminal phase; walking it used to grow the
    # phase sequence until the process ran out of memory.
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _make_phase_entry("A", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1],
                          [0, -1], [0.02] * 5),
        _make_phase_entry("B", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1],
                          [0, -1], [0.02] * 5),
    ]
    spec.connections = [_make_connection("A", "B"), _make_connection("B", "A")]
    spec.seam = seam

    with pytest.raises(Exception):
        spec.phase_sequence()
    with pytest.raises(Exception):
        qec.dem_chunks_from_spec(spec)


# ---------------------------------------------------------------------------
# Matrix interop: sparse_binary_matrix has no Python class, so the bindings
# accept/return NumPy and scipy matrices on its behalf.
# ---------------------------------------------------------------------------


def test_extended_dem_matrices_are_numpy_arrays():
    c = _rep3_dem_chunk()
    H, O = c.H, c.O

    assert isinstance(H, np.ndarray)
    assert isinstance(O, np.ndarray)
    assert H.dtype == np.uint8
    assert H.shape == (c.num_rows(), c.num_faults())
    assert O.shape == (c.num_observables(), c.num_faults())

    # Each access returns an independent copy, so mutating one cannot corrupt
    # the chunk it came from.
    H[0, 0] ^= 1
    assert not np.array_equal(H, c.H)


def test_css_codes_matrices_accept_numpy_arrays():
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    lz = np.array([[1, 0, 0]], dtype=np.uint8)

    codes = qec.CssCodes()
    codes.hz = hz
    codes.lz = lz

    assert np.array_equal(codes.hz, hz)
    assert np.array_equal(codes.lz, lz)

    # A hand-built CssCodes has to drive DEM construction, not just read back.
    noise = qec.CssNoise()
    noise.px_per_qubit = [0.01, 0.01, 0.01]
    chunk = qec.extended_dem_from_css_matrices(codes, noise)
    assert chunk.num_faults() == 3
    assert chunk.num_observables() == 1


def test_css_codes_matrices_accept_scipy_sparse():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)

    codes = qec.CssCodes()
    codes.hz = scipy_sparse.csr_matrix(hz)
    assert np.array_equal(codes.hz, hz)


def test_css_codes_matrices_accept_non_uint8_dtype():
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    codes = qec.CssCodes()
    codes.hz = hz
    assert np.array_equal(codes.hz, hz.astype(np.uint8))


def test_dem_chunks_to_pcm_returns_a_scipy_matrix():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    c = _rep3_dem_chunk()
    pcm = qec.dem_chunks_to_pcm([c, c])

    assert scipy_sparse.issparse(pcm)
    assert scipy_sparse.isspmatrix_csc(pcm)
    # Same detector geometry as the closed DEM it is built from.
    dem = qec.dem_close_all([c, c])
    assert pcm.shape == dem.detector_error_matrix.shape
    assert np.array_equal(pcm.toarray().astype(np.uint8),
                          dem.detector_error_matrix)


if __name__ == "__main__":
    pytest.main()

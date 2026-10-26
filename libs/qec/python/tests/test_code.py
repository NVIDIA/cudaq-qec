# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec
from cudaq_qec import patch
from cudaq_qec.plugins.codes import example


def test_get_code():
    steane = qec.get_code("steane")
    assert isinstance(steane, qec.Code)


def test_get_available_codes():
    codes = qec.get_available_codes()
    assert isinstance(codes, list)
    assert "steane" in codes


def test_code_parity_matrices():
    steane = qec.get_code("steane")

    parity = steane.get_parity()
    assert isinstance(parity, np.ndarray)
    assert parity.shape == (6, 14)

    parity_x = steane.get_parity_x()
    assert isinstance(parity, np.ndarray)
    assert parity_x.shape == (3, 7)

    parity_z = steane.get_parity_z()
    assert isinstance(parity, np.ndarray)
    assert parity_z.shape == (3, 7)


def test_code_stabilizer_schedule_matrices():
    # Codes without a schedule override return the plain parity matrices.
    steane = qec.get_code("steane")
    assert np.array_equal(steane.get_stabilizer_schedule_x(),
                          steane.get_parity_x())
    assert np.array_equal(steane.get_stabilizer_schedule_z(),
                          steane.get_parity_z())

    # The surface code overrides the schedule: same support pattern as the
    # parity matrices, entries are CNOT timesteps in [1, 4].
    surface = qec.get_code("surface_code", distance=3)
    for schedule, parity in [
        (surface.get_stabilizer_schedule_x(), surface.get_parity_x()),
        (surface.get_stabilizer_schedule_z(), surface.get_parity_z())
    ]:
        assert isinstance(schedule, np.ndarray)
        assert schedule.shape == parity.shape
        assert np.array_equal(schedule != 0, parity != 0)
        assert schedule.max() <= 4


def test_repetition_empty_x_matrices_preserve_rank():
    repetition = qec.get_code("repetition", distance=3)

    # Repetition is Z-only; empty X-side results must still be matrices with
    # one column per data qubit so Python callers can inspect their shape.
    parity_x = repetition.get_parity_x()
    assert isinstance(parity_x, np.ndarray)
    assert parity_x.dtype == np.uint8
    assert parity_x.shape == (0, 3)

    observables_x = repetition.get_observables_x()
    assert isinstance(observables_x, np.ndarray)
    assert observables_x.dtype == np.uint8
    assert observables_x.shape == (0, 3)

    parity_z = repetition.get_parity_z()
    assert parity_z.shape == (2, 3)

    observables_z = repetition.get_observables_z()
    assert observables_z.shape == (1, 3)


def test_code_stabilizers():
    steane = qec.get_code("steane")
    stabilizers = steane.get_stabilizers()
    assert isinstance(stabilizers, list)
    assert len(stabilizers) == 6
    assert all(isinstance(stab, cudaq.Operator) for stab in stabilizers)
    stabStrings = [term.get_pauli_word() for term in stabilizers]
    expected = [
        "ZZZZIII", "XXXXIII", "IXXIXXI", "IIXXIXX", "IZZIZZI", "IIZZIZZ"
    ]
    assert set(expected) == set(stabStrings)


def test_sample_memory_circuit():
    steane = qec.get_code("steane")
    nShots, nRounds = 10, 4
    # Shape: (nShots, k) where k = 2*numAncz + (nRounds-1)*numCols
    # For Steane: numAncz=3, numCols=6, nRounds=4 → k = 2*3 + 3*6 = 24
    nDetectors = 24

    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=nShots,
                                                       numRounds=nRounds)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (nShots, nDetectors)
    print(syndromes)

    syndromes_with_op, dataResults = qec.sample_memory_circuit(
        steane, qec.operation.prep1, nShots, nRounds)
    assert isinstance(syndromes_with_op, np.ndarray)
    print(syndromes_with_op)
    assert syndromes_with_op.shape == (nShots, nDetectors)


def test_custom_steane_code():
    ops = ["ZZZZIII", "XXXXIII", "IXXIXXI", "IIXXIXX", "IZZIZZI", "IIZZIZZ"]
    custom_steane = qec.get_code("steane", stabilizers=ops)
    assert isinstance(custom_steane, qec.Code)

    parity = custom_steane.get_parity()
    assert parity.shape == (6, 14)

    expected_parity = np.array([
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1
    ])
    print(parity)
    np.testing.assert_array_equal(parity, expected_parity.reshape(6, 14))


def test_noisy_simulation():
    cudaq.set_target('stim')

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel('x',
                                qec.TwoQubitDepolarization(.1),
                                num_controls=1)
    steane = qec.get_code("steane")
    nShots, nRounds = 10, 4
    nDetectors = 24
    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=nShots,
                                                       numRounds=nRounds,
                                                       noise=noise)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (nShots, nDetectors)
    print(syndromes)
    assert np.any(syndromes)
    cudaq.reset_target()


def test_python_code():
    steane = qec.get_code("py-steane-example")
    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=10,
                                                       numRounds=4)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (10, 24)
    print(syndromes)
    assert not np.any(syndromes)


# A stabilizer_round kernel that honors the schedule entries: the CNOT for a
# nonzero entry k executes at timestep k (the example plugin's kernel instead
# tests `== 1`, i.e. plain parity support).
@cudaq.kernel
def _scheduled_stabilizer(logicalQubit: patch, x_schedule: list[int],
                          z_schedule: list[int]) -> list[cudaq.measure_handle]:
    h(logicalQubit.ancx)
    for step in range(1, 5):
        for xi in range(len(logicalQubit.ancx)):
            for di in range(len(logicalQubit.data)):
                if x_schedule[xi * len(logicalQubit.data) + di] == step:
                    x.ctrl(logicalQubit.ancx[xi], logicalQubit.data[di])
    h(logicalQubit.ancx)
    for step in range(1, 5):
        for zi in range(len(logicalQubit.ancz)):
            for di in range(len(logicalQubit.data)):
                if z_schedule[zi * len(logicalQubit.data) + di] == step:
                    x.ctrl(logicalQubit.data[di], logicalQubit.ancz[zi])

    results = mz([*logicalQubit.ancz, *logicalQubit.ancx])

    reset(logicalQubit.ancx)
    reset(logicalQubit.ancz)
    return results


def _make_scheduled_steane(name, schedule_x, schedule_z):
    """Register a Python Steane code that provides stabilizer schedules."""

    @qec.code(name)
    class ScheduledSteane:

        def __init__(self, **kwargs):
            qec.Code.__init__(self, **kwargs)
            self.stabilizers = [
                cudaq.SpinOperator.from_word(w) for w in [
                    "XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI",
                    "IIZZIZZ"
                ]
            ]
            self.pauli_observables = [
                cudaq.SpinOperator.from_word(p) for p in ["IIIIXXX", "IIIIZZZ"]
            ]
            self.operation_encodings = {
                qec.operation.prep0: example.prep0,
                qec.operation.stabilizer_round: _scheduled_stabilizer
            }

        def get_num_data_qubits(self):
            return 7

        def get_num_ancilla_x_qubits(self):
            return 3

        def get_num_ancilla_z_qubits(self):
            return 3

        def get_num_ancilla_qubits(self):
            return 6

        def get_num_x_stabilizers(self):
            return 3

        def get_num_z_stabilizers(self):
            return 3

        def get_stabilizer_schedule_x(self):
            return schedule_x

        def get_stabilizer_schedule_z(self):
            return schedule_z

    return qec.get_code(name)


def test_python_code_stabilizer_schedule():
    # A Python code may provide schedule matrices: same support as the parity
    # matrices, with entries giving the interaction timestep.
    steane = qec.get_code("py-steane-example")
    parity_x = steane.get_parity_x()
    parity_z = steane.get_parity_z()

    # Without the optional methods, the schedule defaults to the parity.
    assert np.array_equal(steane.get_stabilizer_schedule_x(), parity_x)
    assert np.array_equal(steane.get_stabilizer_schedule_z(), parity_z)

    # Number the interactions within each stabilizer to form a schedule.
    schedule_x = (np.cumsum(parity_x, axis=1) * parity_x).astype(np.uint8)
    schedule_z = (np.cumsum(parity_z, axis=1) * parity_z).astype(np.uint8)
    code = _make_scheduled_steane("py-steane-scheduled", schedule_x, schedule_z)
    assert np.array_equal(code.get_stabilizer_schedule_x(), schedule_x)
    assert np.array_equal(code.get_stabilizer_schedule_z(), schedule_z)

    # The schedule must flow through to the sampled memory circuit.
    syndromes, _ = qec.sample_memory_circuit(code, numShots=10, numRounds=4)
    assert syndromes.shape == (10, 24)
    assert not np.any(syndromes)


def test_python_code_stabilizer_schedule_invalid():
    steane = qec.get_code("py-steane-example")
    parity_x = steane.get_parity_x()
    parity_z = steane.get_parity_z()

    # Wrong shape.
    bad = _make_scheduled_steane("py-steane-bad-schedule-shape",
                                 parity_x[:, :-1], parity_z)
    with pytest.raises(RuntimeError, match="shape"):
        bad.get_stabilizer_schedule_x()

    # Support pattern differs from the parity matrix.
    flipped = parity_x.copy()
    flipped[0, 0] ^= 1
    bad = _make_scheduled_steane("py-steane-bad-schedule-support", flipped,
                                 parity_z)
    with pytest.raises(RuntimeError, match="support"):
        bad.get_stabilizer_schedule_x()


# stabilizer_round kernels with invalid (non list[cudaq.measure_handle])
# return annotations. The bodies are minimal; only the annotation matters.
@cudaq.kernel
def _stab_list_bool(q: patch, xs: list[int], zs: list[int]) -> list[bool]:
    return mz([*q.ancx, *q.ancz])


@cudaq.kernel
def _stab_bool(q: patch, xs: list[int], zs: list[int]) -> bool:
    return mz(q.ancx[0])


@cudaq.kernel
def _stab_int(q: patch, xs: list[int], zs: list[int]) -> int:
    return cudaq.to_integer(cudaq.to_bools(mz([*q.ancx, *q.ancz])))


@cudaq.kernel
def _stab_void(q: patch, xs: list[int], zs: list[int]):
    mz([*q.ancx, *q.ancz])


@cudaq.kernel
def _stab_none(q: patch, xs: list[int], zs: list[int]) -> None:
    mz([*q.ancx, *q.ancz])


@pytest.mark.parametrize("kernel,name", [
    (_stab_list_bool, 'py-bad-stab-list-bool'),
    (_stab_bool, 'py-bad-stab-bool'),
    (_stab_int, 'py-bad-stab-int'),
    (_stab_void, 'py-bad-stab-void'),
    (_stab_none, 'py-bad-stab-none'),
])
def test_stabilizer_round_invalid_annotation(kernel, name):
    # A stabilizer_round must return list[cudaq.measure_handle]; any other
    # annotation drops the measurement handles, so registration must reject it.
    @qec.code(name)
    class BadCode:

        def __init__(self, **kwargs):
            qec.Code.__init__(self, **kwargs)
            self.stabilizers = [
                cudaq.SpinOperator.from_word(w) for w in [
                    "XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI",
                    "IIZZIZZ"
                ]
            ]
            self.pauli_observables = [
                cudaq.SpinOperator.from_word(p) for p in ["IIIIXXX", "IIIIZZZ"]
            ]
            self.operation_encodings = {qec.operation.stabilizer_round: kernel}

        def get_num_data_qubits(self):
            return 7

        def get_num_ancilla_x_qubits(self):
            return 3

        def get_num_ancilla_z_qubits(self):
            return 3

        def get_num_ancilla_qubits(self):
            return 6

        def get_num_x_stabilizers(self):
            return 3

        def get_num_z_stabilizers(self):
            return 3

    with pytest.raises(RuntimeError, match="list\\[cudaq.measure_handle\\]"):
        qec.get_code(name)


def test_invalid_code():
    with pytest.raises(RuntimeError):
        qec.get_code("invalid_code_name")


def test_invalid_operation():
    steane = qec.get_code("steane")
    with pytest.raises(TypeError):
        qec.sample_memory_circuit(steane, "invalid_op", 10, 4)


def test_generate_random_bit_flips():
    # Test case 1: error_prob = 0
    nBits = 10
    error_prob = 0

    data = qec.generate_random_bit_flips(nBits, error_prob)
    print(f"data shape: {data.shape}")

    assert len(data.shape) == 1
    assert data.shape[0] == 10
    assert np.all(data == 0)


def test_steane_code_capacity():
    # Test case 1: error_prob = 0
    steane = qec.get_code("steane")
    Hz = steane.get_parity_z()
    n_shots = 10
    error_prob = 0

    syndromes, data = qec.sample_code_capacity(Hz, n_shots, error_prob)

    assert len(Hz.shape) == 2
    assert Hz.shape[0] == 3
    assert Hz.shape[1] == 7
    assert syndromes.shape[0] == n_shots
    assert syndromes.shape[1] == Hz.shape[0]
    assert data.shape[0] == n_shots
    assert data.shape[1] == Hz.shape[1]

    # Error prob = 0 should be all zeros
    assert np.all(data == 0)
    assert np.all(syndromes == 0)

    # Test case 2: error_prob = 0.15
    error_prob = 0.15
    seed = 1337

    syndromes, data = qec.sample_code_capacity(Hz,
                                               n_shots,
                                               error_prob,
                                               seed=seed)

    assert len(Hz.shape) == 2
    assert Hz.shape[0] == 3
    assert Hz.shape[1] == 7
    assert syndromes.shape[0] == n_shots
    assert syndromes.shape[1] == Hz.shape[0]
    assert data.shape[0] == n_shots
    assert data.shape[1] == Hz.shape[1]

    # Known seeded data for error_prob = 0.15
    seeded_data = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]])

    checked_syndromes = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0],
                                  [0, 1, 1], [0, 0, 0]])

    assert np.array_equal(data, seeded_data)
    assert np.array_equal(syndromes, checked_syndromes)

    # Test case 3: error_prob = 0.25
    error_prob = 0.25
    seed = 1337

    syndromes, data = qec.sample_code_capacity(Hz,
                                               n_shots,
                                               error_prob,
                                               seed=seed)

    assert len(Hz.shape) == 2
    assert Hz.shape[0] == 3
    assert Hz.shape[1] == 7
    assert syndromes.shape[0] == n_shots
    assert syndromes.shape[1] == Hz.shape[0]
    assert data.shape[0] == n_shots
    assert data.shape[1] == Hz.shape[1]

    # Known seeded data for error_prob = 0.25
    seeded_data = np.array([[0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]])

    checked_syndromes = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0],
                                  [0, 1, 1], [0, 0, 0]])

    assert np.array_equal(data, seeded_data)
    assert np.array_equal(syndromes, checked_syndromes)


def test_het_map_from_kwargs_bool():
    steane = qec.get_code("steane", bool_true=True, bool_false=False)
    assert isinstance(steane, qec.Code)


def test_version():
    assert "CUDA-Q QEC" in qec.__version__


def test_python_steane_code_direct_construction():
    # Construct the example plugin class directly and check the Steane layout
    # (qubit counts, stabilizer/observable words, and operation encodings).
    code = example.MySteaneCodeImpl()
    assert code.get_num_data_qubits() == 7
    assert code.get_num_ancilla_x_qubits() == 3
    assert code.get_num_ancilla_z_qubits() == 3
    assert code.get_num_ancilla_qubits() == 6
    assert code.get_num_x_stabilizers() == 3
    assert code.get_num_z_stabilizers() == 3
    stab_words = [term.get_pauli_word() for term in code.stabilizers]
    assert stab_words == [
        "XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI", "IIZZIZZ"
    ]
    obs_words = [term.get_pauli_word() for term in code.pauli_observables]
    assert obs_words == ["IIIIXXX", "IIIIZZZ"]
    encodings = code.operation_encodings
    assert encodings[qec.operation.prep0] is example.prep0
    assert encodings[qec.operation.stabilizer_round] is example.stabilizer


def test_python_steane_kernel_bodies(monkeypatch):
    # Run the Python bodies retained on the kernel callables: prep0 applies
    # three Hadamards and eight CNOTs; stabilizer uses both CNOT orientations,
    # measures Z then X ancillas, and resets both ancilla registers.
    class _Reg:

        def __init__(self, ids):
            self._ids = list(ids)

        def __len__(self):
            return len(self._ids)

        def __getitem__(self, i):
            return self._ids[i]

        def __iter__(self):
            return iter(self._ids)

    class _Patch:

        def __init__(self):
            self.data = _Reg(range(7))
            self.ancx = _Reg(range(10, 13))
            self.ancz = _Reg(range(20, 23))

    hadamards = []
    cnots = []
    measures = []
    resets = []

    def fake_h(*qubits):
        hadamards.append(qubits)

    class _X:

        def ctrl(self, control, target):
            cnots.append((control, target))

    def fake_mz(qubits):
        measures.append(list(qubits))
        return "mz-sentinel"

    def fake_reset(qubits):
        resets.append(qubits)

    for fn in (example.prep0.kernelFunction, example.stabilizer.kernelFunction):
        monkeypatch.setitem(fn.__globals__, "h", fake_h)
        monkeypatch.setitem(fn.__globals__, "x", _X())
        monkeypatch.setitem(fn.__globals__, "mz", fake_mz)
        monkeypatch.setitem(fn.__globals__, "reset", fake_reset)

    logical = _Patch()
    example.prep0.kernelFunction(logical)
    assert hadamards == [(0, 4, 6)]
    assert cnots == [(0, 1), (4, 5), (6, 3), (6, 5), (4, 2), (0, 3), (4, 1),
                     (3, 2)]

    hadamards.clear()
    cnots.clear()
    n_data = 7
    x_stab = [0] * (3 * n_data)
    z_stab = [0] * (3 * n_data)
    x_stab[0 * n_data + 0] = 1
    x_stab[1 * n_data + 2] = 1
    x_stab[2 * n_data + 6] = 1
    z_stab[0 * n_data + 1] = 1
    z_stab[1 * n_data + 3] = 1
    z_stab[2 * n_data + 5] = 1
    result = example.stabilizer.kernelFunction(logical, x_stab, z_stab)
    assert hadamards == [(logical.ancx,), (logical.ancx,)]
    assert cnots == [(10, 0), (11, 2), (12, 6), (1, 20), (3, 21), (5, 22)]
    assert measures == [[20, 21, 22, 10, 11, 12]]
    assert resets == [logical.ancx, logical.ancz]
    assert result == "mz-sentinel"


if __name__ == "__main__":
    pytest.main()

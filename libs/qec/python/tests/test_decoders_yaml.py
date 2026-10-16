# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq_qec as qec


def is_nv_qldpc_decoder_available():
    """
    Helper function to check if the NV-QLDPC decoder is available.
    """
    try:
        H_list = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 1]]

        H_np = np.array(H_list, dtype=np.uint8)
        nv_dec_gpu_and_cpu = qec.get_decoder("nv-qldpc-decoder", H_np)
        return True
    except Exception as e:
        return False


def check_decoder_yaml_roundtrip(multi_config):
    """
    Helper function to test that a decoder configuration can be serialized to
    and from YAML.
    """
    # Serialize to YAML
    config_str = multi_config.to_yaml_str(200)

    # Deserialize from YAML
    multi_config_from_yaml = qec.multi_decoder_config.from_yaml_str(config_str)

    # And now serialize the deserialized configuration back to YAML, just for
    # good measure.
    round_trip_config_str = multi_config_from_yaml.to_yaml_str(200)

    # Validate
    match_strings = round_trip_config_str == config_str
    match_configs = multi_config_from_yaml == multi_config

    assert match_strings, "YAML strings don't match after round trip"
    assert match_configs, "Configs don't match after round trip"


def check_decoder_creation(multi_config):
    """
    Helper function to create and finalize a decoder configuration.
    """
    status = qec.configure_decoders(multi_config)
    assert status == 0, f"configure_decoders returned non-zero status: {status}"
    qec.finalize_decoders()


def create_test_empty_decoder_config(decoder_id):
    """
    Helper function to create a sample, skeleton test decoder configuration for
    a single error LUT decoder.
    """
    config = qec.decoder_config()
    config.id = decoder_id
    config.type = "single_error_lut"
    config.block_size = 20
    config.syndrome_size = 10

    # Create sparse H matrix representation from a zero matrix
    H = np.zeros((config.syndrome_size, config.block_size), dtype=np.uint8)
    config.H_sparse = qec.pcm_to_sparse_vec(H)

    # Create sparse O matrix representation from a zero matrix
    O = np.zeros((2, config.block_size), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)

    # Generate timelike sparse detector matrix
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)
    return config


def create_test_decoder_config_nv_qldpc(decoder_id):
    """
    Helper function to create a sample, skeleton test decoder configuration for
    the NV-QLDPC decoder.
    """
    config = create_test_empty_decoder_config(decoder_id)
    config.type = "nv-qldpc-decoder"
    config.error_rate_vec = [0.1] * config.block_size

    # Create NV-QLDPC decoder configuration (a parameter dict; keys are
    # governed by the decoder's registered schema)
    config.decoder_custom_args = {
        "use_sparsity": True,
        "max_iterations": 50,
        "use_osd": True,
        "osd_order": 60,
        "osd_method": 3,
        "n_threads": 128,
        "bp_batch_size": 1,
        "osd_batch_size": 16,
        "iter_per_check": 2,
        "clip_value": 10.0,
        "bp_method": 3,
        "scale_factor": 1.0,
        "proc_float": "fp64",
        # Relay-BP configuration
        "gamma0": 0.0,
        "gamma_dist": [0.1, 0.2],
        "srelay_config": {
            "pre_iter": 5,
            "num_sets": 10,
            "stopping_criterion": "NConv",
            "stop_nconv": 10,
        },
        # explicit_gammas must have num_sets rows (10 in this case)
        "explicit_gammas": [[0.1] * config.block_size for _ in range(10)],
        "bp_seed": 42,
        "composition": 1,
    }

    return config


def test_single_decoder():
    """
    Test YAML serialization/deserialization and creation of a single NV-QLDPC decoder.
    """
    if not is_nv_qldpc_decoder_available():
        pytest.skip("NV-QLDPC decoder is not available, skipping test")
    multi_config = qec.multi_decoder_config()
    config = create_test_decoder_config_nv_qldpc(0)
    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_multi_decoder():
    """
    Test YAML serialization/deserialization and creation of multiple NV-QLDPC decoders.
    """
    if not is_nv_qldpc_decoder_available():
        pytest.skip("NV-QLDPC decoder is not available, skipping test")
    multi_config = qec.multi_decoder_config()
    config1 = create_test_decoder_config_nv_qldpc(0)
    config2 = create_test_decoder_config_nv_qldpc(1)
    multi_config.decoders = [config1, config2]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_multi_lut_decoder():
    """
    Test YAML serialization/deserialization and creation of a multi-error LUT decoder.
    """
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "multi_error_lut"

    config.decoder_custom_args = {"lut_error_depth": 2}

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_single_lut_decoder():
    """
    Test YAML serialization/deserialization and creation of a single-error LUT decoder.
    """
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "single_error_lut"

    config.decoder_custom_args = {}

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_sliding_window_decoder():
    """
    Test YAML serialization/deserialization and creation of a sliding window decoder.
    """
    n_rounds = 4
    n_errs_per_round = 30
    n_syndromes_per_round = 10
    n_cols = n_rounds * n_errs_per_round
    n_rows = n_rounds * n_syndromes_per_round
    weight = 3

    # Generate random PCM
    pcm = qec.generate_random_pcm(n_rounds=n_rounds,
                                  n_errs_per_round=n_errs_per_round,
                                  n_syndromes_per_round=n_syndromes_per_round,
                                  weight=weight,
                                  seed=13)
    pcm = qec.sort_pcm_columns(pcm, n_syndromes_per_round)

    # Top-level decoder config
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "sliding_window"
    config.block_size = n_cols
    config.syndrome_size = n_rows

    # Convert PCM to sparse representation
    config.H_sparse = qec.pcm_to_sparse_vec(pcm)

    # Create sparse O matrix (2 x n_cols zero matrix)
    O = np.zeros((2, n_cols), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)

    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)

    # Sliding window config. inner_decoder_params is validated against the
    # schema registered under inner_decoder_name.
    config.decoder_custom_args = {
        "window_size": 1,
        "step_size": 1,
        "num_syndromes_per_round": n_syndromes_per_round,
        "straddle_start_round": False,
        "straddle_end_round": True,
        "inner_decoder_name": "multi_error_lut",
        "inner_decoder_params": {
            "lut_error_depth": 2
        },
    }
    config.error_rate_vec = [0.1] * config.block_size

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_sliding_window_boundary_syndromes_roundtrip():
    """
    Test that a sliding_window's num_boundary_syndromes parameter survives a
    YAML round trip. This is serialization-only (the boundary-layout decoding
    behavior is exercised by the direct-decoder tests in test_sliding_window).
    """
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "sliding_window"
    config.block_size = 6
    config.syndrome_size = 4

    H = np.zeros((config.syndrome_size, config.block_size), dtype=np.uint8)
    config.H_sparse = qec.pcm_to_sparse_vec(H)
    O = np.zeros((1, config.block_size), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)

    config.decoder_custom_args = {
        "window_size": 1,
        "step_size": 1,
        "num_syndromes_per_round": 2,
        "num_boundary_syndromes": 1,
        "inner_decoder_name": "single_error_lut",
    }
    config.error_rate_vec = [0.1] * config.block_size

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)


# ---------------------------------------------------------------------------
# Chunk-form configurations
# ---------------------------------------------------------------------------

REP5_CHECKS = 4

BULK_H_SPARSE = "[ 0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1 ]"
BULK_RATES = "[ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 ]"


def chunk_form_yaml(num_rounds):
    """A d=5 repetition code written as phases rather than flat matrices.

    num_rounds lives inside dem_chunks, alongside the phases and the
    connection graph it expands. Pass None to omit it (the streaming form).
    """
    rounds = "" if num_rounds is None else f"  num_rounds: {num_rounds}\n"
    return f"""
id: 0
type: single_error_lut
dem_chunks:
  seam:
    from: next_round
    to: prev_round
  connections:
    - {{from: init, to: bulk}}
    - {{from: bulk, to: bulk}}
    - {{from: bulk, to: final}}
{rounds}  phases:
    - name: init
      spec:
        num_faults: 9
        H_sparse: {BULK_H_SPARSE}
        O_sparse: [ 0, -1 ]
        error_rates: {BULK_RATES}
    - name: bulk
      spec:
        num_faults: 9
        H_sparse: {BULK_H_SPARSE}
        O_sparse: [ 0, -1 ]
        error_rates: {BULK_RATES}
    - name: final
      spec:
        num_faults: 5
        H_sparse: [ 0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1 ]
        O_sparse: [ 0, -1 ]
        error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02 ]
"""


def expected_syndrome_size(num_rounds):
    """init's prev_round seam gives the initial-state detector band, and each
    inter-chunk boundary gives one more: num_rounds bands total."""
    return num_rounds * REP5_CHECKS


def expected_block_size(num_rounds):
    """init plus (num_rounds - 2) bulk copies at 9 faults each, and a
    destructively measured final phase with only its 5 data columns."""
    return 9 * (num_rounds - 1) + 5


def test_chunk_form_parses_with_no_flat_matrices():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))

    assert config.dem_chunks is not None
    assert config.dem_chunks.num_rounds == 5
    assert len(config.dem_chunks.phases) == 3
    assert len(config.dem_chunks.connections) == 3
    assert config.dem_chunks.has_repeating_phase()
    assert config.dem_chunks.get_phase(qec.SeamId("init")).num_faults == 9
    assert config.dem_chunks.get_phase(qec.SeamId("final")).num_faults == 5
    # The flat fields are derived, so they stay unset until expansion.
    assert config.block_size == 0
    assert config.syndrome_size == 0
    assert len(config.H_sparse) == 0


def test_expand_dem_chunks_fills_the_flat_fields():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    closed = qec.qecrt.config.expand_dem_chunks(config)

    assert closed is not None
    assert config.syndrome_size == expected_syndrome_size(5)
    assert config.block_size == expected_block_size(5)
    assert len(closed.error_rates) == config.block_size
    # One -1 terminator per row, so H and D are as tall as the DEM.
    assert config.H_sparse.count(-1) == config.syndrome_size
    assert config.D_sparse.count(-1) == config.syndrome_size


def test_both_forms_warns_and_flat_form_wins(capfd):
    """A leftover H_sparse disables the entire chunk spec. expand_dem_chunks()
    produces that same state, so it has to keep parsing, but it must say so.

    The warning comes from the C++ logger writing to the process stderr, hence
    capfd rather than capsys.
    """
    yaml = chunk_form_yaml(5) + """block_size: 2
syndrome_size: 2
H_sparse: [ 0, -1, 1, -1 ]
O_sparse: [ 0, -1 ]
D_sparse: [ 0, -1, 1, -1 ]
"""
    config = qec.decoder_config.from_yaml_str(yaml)
    captured = capfd.readouterr()

    assert "both H_sparse and dem_chunks" in captured.err
    # The flat matrix is what a decoder would be built from, and dem_chunks
    # survives so the document still round-trips.
    assert config.syndrome_size == 2
    assert list(config.H_sparse) == [0, -1, 1, -1]
    assert config.dem_chunks is not None


def test_expand_dem_chunks_is_a_no_op_on_a_flat_config():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(3))
    qec.qecrt.config.expand_dem_chunks(config)
    flat_H = list(config.H_sparse)

    # Already flat now, so a second call must leave it alone.
    assert qec.qecrt.config.expand_dem_chunks(config) is None
    assert list(config.H_sparse) == flat_H


def test_same_chunks_serve_different_round_counts():
    sizes = {}
    # 3 is the shortest sequence these phases expand to (init, bulk, final).
    for rounds in (3, 4, 5, 8):
        config = qec.decoder_config.from_yaml_str(chunk_form_yaml(rounds))
        qec.qecrt.config.expand_dem_chunks(config)
        sizes[rounds] = (config.syndrome_size, config.block_size)

    for rounds, (syndrome_size, block_size) in sizes.items():
        assert syndrome_size == expected_syndrome_size(rounds)
        assert block_size == expected_block_size(rounds)


def test_chunk_form_builds_a_working_decoder():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [config]

    assert qec.qecrt.config.configure_decoders(multi_config) == 0
    qec.qecrt.config.finalize_decoders()


def test_num_rounds_is_optional_until_expansion():
    # Streaming configs leave num_rounds out: the round count is unknown when
    # the file is written. Expansion is what needs it, because the repeating
    # phase has no other way to know how many copies to emit.
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(None))
    assert config.dem_chunks.num_rounds is None

    with pytest.raises(Exception):
        qec.qecrt.config.expand_dem_chunks(config)


def test_num_rounds_is_rejected_outside_dem_chunks():
    yaml_str = """
id: 0
type: single_error_lut
num_rounds: 5
block_size: 3
syndrome_size: 2
H_sparse: [ 0, -1, 1, -1 ]
O_sparse: [ 0, -1 ]
D_sparse: [ 0, -1, 0, 1, -1 ]
"""
    with pytest.raises(Exception):
        qec.decoder_config.from_yaml_str(yaml_str)


def test_num_rounds_below_two_is_rejected():
    # num_rounds counts init and final as well as the bulk copies between
    # them, so 2 is the smallest value the parser accepts.
    for rounds in (0, 1):
        with pytest.raises(Exception):
            qec.decoder_config.from_yaml_str(chunk_form_yaml(rounds))

    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(2))
    assert config.dem_chunks.num_rounds == 2
    # These particular phases need three rounds to lay out, so 2 parses but
    # cannot expand. The parser only enforces the floor common to every spec.
    with pytest.raises(Exception):
        qec.qecrt.config.expand_dem_chunks(config)


def test_incomplete_flat_config_is_rejected():
    # dem_chunks made the flat fields optional in the mapping; a flat config
    # still has to name all of them, or it parses into an empty DEM that only
    # fails when a decoder is built from it.
    complete = {
        "block_size": "block_size: 3",
        "syndrome_size": "syndrome_size: 2",
        "H_sparse": "H_sparse: [ 0, -1, 1, -1 ]",
        "O_sparse": "O_sparse: [ 0, -1 ]",
        "D_sparse": "D_sparse: [ 0, -1, 0, 1, -1 ]",
    }
    header = "id: 0\ntype: single_error_lut\n"
    assert qec.decoder_config.from_yaml_str(header +
                                            "\n".join(complete.values()))

    for omitted in complete:
        yaml_str = header + "\n".join(
            line for key, line in complete.items() if key != omitted)
        with pytest.raises(Exception):
            qec.decoder_config.from_yaml_str(yaml_str)


def test_config_describing_no_dem_at_all_is_rejected():
    with pytest.raises(Exception):
        qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\n")


def test_expand_dem_chunks_is_exported_at_top_level():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    assert qec.expand_dem_chunks(config) is not None
    assert config.block_size == expected_block_size(5)


def test_derived_fields_are_rejected_in_chunk_form():
    with_block_size = chunk_form_yaml(5).replace(
        "type: single_error_lut",
        f"type: single_error_lut\nblock_size: {expected_block_size(5)}")
    with pytest.raises(Exception):
        qec.decoder_config.from_yaml_str(with_block_size)


def _phase_entry(name, num_faults, h_sparse):
    entry = qec.PhaseSpecEntry()
    entry.id = qec.SeamId(name)
    entry.spec.num_faults = num_faults
    entry.spec.H_sparse = h_sparse
    entry.spec.O_sparse = [0, -1]
    entry.spec.error_rates = [0.02] * num_faults
    return entry


def _connection(from_name, to_name):
    connection = qec.PhaseConnection()
    connection.from_phase = qec.SeamId(from_name)
    connection.to_phase = qec.SeamId(to_name)
    return connection


def test_chunk_form_can_be_built_programmatically():
    config = qec.decoder_config()
    config.id = 0
    config.type = "single_error_lut"

    bulk_h = [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1]
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _phase_entry("init", 9, bulk_h),
        _phase_entry("bulk", 9, bulk_h),
        _phase_entry("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]),
    ]
    spec.connections = [
        _connection("init", "bulk"),
        _connection("bulk", "bulk"),
        _connection("bulk", "final"),
    ]
    spec.seam = seam
    spec.num_rounds = 4
    config.dem_chunks = spec

    qec.qecrt.config.expand_dem_chunks(config)
    assert config.syndrome_size == expected_syndrome_size(4)
    assert config.block_size == expected_block_size(4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

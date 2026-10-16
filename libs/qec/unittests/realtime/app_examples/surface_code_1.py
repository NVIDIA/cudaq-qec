# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# For full test script: surface_code-1-test.sh
#
# Python counterpart to surface_code-1.cpp.
# See its header comment for the overall flow description.
#
# Limitations vs the C++ version:
#   - --save_syndrome / --load_syndrome: _set_syndrome_capture_callback is not
#     bound to Python; syndrome capture requires the C++ binary.
#   - The QPU stabilizer-round and state-prep kernels are hardcoded below
#     because code.get_operation() is not available for C++-registered codes.

import os
import sys
import argparse
from typing import List, Optional

# Force stim as the default simulator for emulation
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec

sys.tracebacklimit = 999

# ── decoder configuration helpers ────────────────────────────────────────────


def decoder_args(decoder_type: str, params: list = None) -> dict:
    """Return the custom-args dict for a named decoder, with any
    --param key=value overrides applied on top."""
    if decoder_type == "nv-qldpc-decoder":
        args = {
            "use_sparsity": True,
            "max_iterations": 50,
            "bp_method": 3,  # min-sum + dmem (required for relay)
            "composition": 1,  # sequential relay
            "gamma0": 0.0,
            "clip_value": 200.0,
            "repeatable": True,
            "srelay_config": {
                "pre_iter": 5,
                "num_sets": 10,
                "stopping_criterion": "All",
                "stop_nconv": 1,
            },
            "gamma_dist": [0.1, 0.2],
        }
    elif decoder_type == "pymatching":
        args = {"merge_strategy": "smallest_weight"}
    elif decoder_type == "multi_error_lut":
        args = {"lut_error_depth": 2}
    else:
        raise RuntimeError(f"Unknown decoder type: {decoder_type}")
    if not params:
        return args
    schema = {
        p["key"]: p["kind"] for p in qec.decoder_param_schema(decoder_type)
    }
    for kv in params:
        if "=" not in kv:
            raise RuntimeError(f"--param requires key=value format: {kv}")
        key, val = kv.split("=", 1)
        if key not in schema:
            raise RuntimeError(
                f"Unknown parameter '{key}' for decoder '{decoder_type}'")
        kind = schema[key]
        try:
            if kind == "int32":
                args[key] = int(val)
            elif kind in ("float64", "f64"):
                args[key] = float(val)
            elif kind in ("float64_vec", "f64_vec"):
                args[key] = [float(x) for x in val.split(",")]
            elif kind == "boolean":
                if val not in ("true", "false", "1", "0"):
                    raise ValueError
                args[key] = val in ("true", "1")
            elif kind == "string":
                args[key] = val
            else:
                raise RuntimeError(f"--param: unsupported type for '{key}'")
        except (ValueError, RuntimeError) as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"--param '{key}': invalid value '{val}'") from e
    return args


def build_multi_decoder_config(dem, m2d, num_syndromes_per_round: int,
                               num_boundary_syndromes: int,
                               opts) -> qec.multi_decoder_config:
    d_sparse_vec = qec.d_sparse(m2d)
    error_rates = list(dem.error_rates)
    decoder_list = []
    for i in range(opts.num_logical):
        dc = qec.decoder_config()
        dc.id = i
        dc.block_size = dem.num_error_mechanisms()
        dc.syndrome_size = dem.num_detectors()
        dc.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
        dc.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
        dc.D_sparse = d_sparse_vec
        dc.error_rate_vec = error_rates

        if opts.decoder_type == "sliding_window":
            dc.type = "sliding_window"
            dc.decoder_custom_args = {
                "window_size":
                    opts.sw_window_size,
                "step_size":
                    opts.sw_step_size,
                "num_syndromes_per_round":
                    num_syndromes_per_round,
                "num_boundary_syndromes":
                    num_boundary_syndromes,
                "straddle_start_round":
                    False,
                "straddle_end_round":
                    True,
                "inner_decoder_name":
                    opts.sw_inner_decoder,
                "inner_decoder_params":
                    decoder_args(opts.sw_inner_decoder, opts.decoder_params),
            }
        else:
            dc.type = opts.decoder_type
            dc.decoder_custom_args = decoder_args(opts.decoder_type,
                                                  opts.decoder_params)

        decoder_list.append(dc)
    multi_cfg = qec.multi_decoder_config()
    multi_cfg.decoders = decoder_list
    return multi_cfg


@cudaq.kernel
def prep_0_data(data: cudaq.qview) -> None:
    reset(data)


@cudaq.kernel
def stabilizer_round(data: cudaq.qview, anc: cudaq.qview, num_ancz: int,
                     num_ancx: int, x_schedule: List[int],
                     z_schedule: List[int]) -> List[cudaq.measure_handle]:
    """anc = [ancz[0..num_ancz-1], ancx[0..num_ancx-1]]."""
    for i in range(len(anc)):
        reset(anc[i])

    num_data = len(data)

    for i in range(num_ancx):
        h(anc[num_ancz + i])

    for step in range(1, 5):
        # X stabilizers: ancx[xi] is control, data[di] is target
        for xi in range(num_ancx):
            for di in range(num_data):
                if x_schedule[xi * num_data + di] == step:
                    cx(anc[num_ancz + xi], data[di])
        # Z stabilizers: data[di] is control, ancz[zi] is target
        for zi in range(num_ancz):
            for di in range(num_data):
                if z_schedule[zi * num_data + di] == step:
                    cx(data[di], anc[zi])

    for i in range(num_ancx):
        h(anc[num_ancz + i])

    return mz(anc)


@cudaq.kernel
def memory_circuit(data: cudaq.qview, anc: cudaq.qview, num_ancz: int,
                   num_ancx: int, num_rounds: int, x_schedule: List[int],
                   z_schedule: List[int],
                   decoder_id: int) -> List[cudaq.measure_handle]:
    """Per-logical memory circuit: prep, lock-in, rounds, data readout."""
    prep_0_data(data)

    syndrome = stabilizer_round(data, anc, num_ancz, num_ancx, x_schedule,
                                z_schedule)
    qec.enqueue_syndromes(decoder_id, syndrome, 0)

    for _r in range(num_rounds - 1):
        s = stabilizer_round(data, anc, num_ancz, num_ancx, x_schedule,
                             z_schedule)
        qec.enqueue_syndromes(decoder_id, s, 0)

    data_meas = mz(data)
    qec.enqueue_syndromes(decoder_id, data_meas, 0)
    return data_meas


@cudaq.kernel
def demo_circuit_qpu(num_data: int, num_ancx: int, num_ancz: int,
                     num_rounds: int, num_logical: int, x_schedule: List[int],
                     z_schedule: List[int], obs_matrix_flat: List[int],
                     num_obs: int) -> int:
    """Shot kernel: run num_logical independent memory experiments.

    Return layout: low 32 bits = per-logical corrected observables (bit i),
    high 32 bits = total corrections applied this shot.
    """
    data = cudaq.qvector(num_logical * num_data)
    anc_stride = num_ancz + num_ancx
    anc = cudaq.qvector(num_logical * anc_stride)

    for i in range(num_logical):
        qec.reset_decoder(i)

    num_corrections = 0
    ret = 0
    for i in range(num_logical):
        data_meas = memory_circuit(data[i * num_data:(i + 1) * num_data],
                                   anc[i * anc_stride:(i + 1) * anc_stride],
                                   num_ancz, num_ancx, num_rounds, x_schedule,
                                   z_schedule, i)

        obs = False
        for q in range(num_data):
            if obs_matrix_flat[q] != 0 and bool(data_meas[q]):
                obs = not obs

        correction = qec.get_corrections(i, num_obs, False)
        if correction[0]:
            obs = not obs
            num_corrections += 1

        if obs:
            ret = ret | (1 << i)

    ret = ret | (num_corrections << 32)
    return ret


# ── host-side functions ───────────────────────────────────────────────────────


def setup_decoders(code, state_prep_op, opts, noise) -> bool:
    """Configure the decoders. Returns False if --save_dem wrote the config
    (no shots should be run); True if decoders are ready for shots."""
    if opts.load_dem:
        try:
            with open(opts.dem_filename) as f:
                cfg = qec.multi_decoder_config.from_yaml_str(f.read())
        except OSError:
            raise RuntimeError(
                f"Could not open dem config file: {opts.dem_filename}")

        if cfg.decoders:
            is_z_prep = state_prep_op in (qec.operation.prep0,
                                          qec.operation.prep1)
            num_ancx = code.get_num_ancilla_x_qubits()
            num_ancz = code.get_num_ancilla_z_qubits()
            num_boundary = num_ancz if is_z_prep else num_ancx
            expected = (2 * num_boundary + (opts.num_rounds - 1) *
                        (num_ancx + num_ancz))
            if cfg.decoders[0].syndrome_size != expected:
                raise RuntimeError(
                    f"Loaded DEM syndrome_size ({cfg.decoders[0].syndrome_size})"
                    f" does not match current geometry ({expected} = "
                    f"2*num_boundary + (num_rounds-1)*num_stabilizers); "
                    f"check --distance and --num_rounds")

            # also cross-check the raw-measurement span:
            # the largest measurement index in D_sparse, plus one.
            if len(cfg.decoders[0].D_sparse) == 0:
                raise RuntimeError("Loaded DEM has empty D_sparse")
            loaded_measurements = max(cfg.decoders[0].D_sparse) + 1
            expected_measurements = (opts.num_rounds * (num_ancx + num_ancz) +
                                     code.get_num_data_qubits())
            if loaded_measurements != expected_measurements:
                raise RuntimeError(
                    f"Loaded DEM measurement span ({loaded_measurements}) does "
                    f"not match current geometry ({expected_measurements} = "
                    f"num_rounds*num_stabilizers + num_data); "
                    f"check --distance and --num_rounds")

        qec.configure_decoders(cfg)
        print(f"Loaded decoder config from {opts.dem_filename} "
              f"({len(cfg.decoders)} decoders)")
        return True

    if opts.p_cnot == 0.0:
        raise RuntimeError(
            "Cannot build a DEM with p_cnot = 0.0 (no noise means no error mechanisms)"
        )

    leaf_decoder = (opts.sw_inner_decoder if opts.decoder_type
                    == "sliding_window" else opts.decoder_type)
    decompose_errors = (leaf_decoder == "pymatching")
    ctx = qec.decoder_context_from_memory_circuit(code, state_prep_op,
                                                  opts.num_rounds, noise,
                                                  decompose_errors)
    dem, m2d, _ = ctx.full_component()
    print(f"DEM: {dem.num_detectors()} detectors x "
          f"{dem.num_error_mechanisms()} error mechanisms")

    is_z_prep = state_prep_op in (qec.operation.prep0, qec.operation.prep1)
    num_ancx = code.get_num_ancilla_x_qubits()
    num_ancz = code.get_num_ancilla_z_qubits()
    num_syndromes_per_round = num_ancx + num_ancz
    num_boundary = num_ancz if is_z_prep else num_ancx

    cfg = build_multi_decoder_config(dem, m2d, num_syndromes_per_round,
                                     num_boundary, opts)

    if opts.save_dem:
        with open(opts.dem_filename, 'w') as f:
            f.write(cfg.to_yaml_str(200))
        print(f"Saved decoder config to {opts.dem_filename}")
        return False

    qec.configure_decoders(cfg)
    return True


def run_shots(num_shots: int, noise, obs_flat: list, num_obs: int, opts,
              x_schedule: list, z_schedule: list, num_data: int, num_ancx: int,
              num_ancz: int) -> list:
    is_remote = cudaq.get_target().name not in ("stim", "default")
    kwargs = {} if is_remote else {"noise_model": noise}
    return cudaq.run(demo_circuit_qpu,
                     num_data,
                     num_ancx,
                     num_ancz,
                     opts.num_rounds,
                     opts.num_logical,
                     x_schedule,
                     z_schedule,
                     obs_flat,
                     num_obs,
                     shots_count=num_shots,
                     **kwargs)


def report_results(run_result: list, num_logical: int) -> dict:
    print(f"Result size: {len(run_result)}")
    num_non_zero = 0
    num_corrections = 0
    for shot in run_result:
        num_corrections += shot >> 32
        for j in range(num_logical):
            if (shot >> j) & 1:
                num_non_zero += 1
    print(f"Number of non-zero values measured : {num_non_zero}")
    print(f"Number of corrections decoder found: {num_corrections}")
    return {"num_non_zero": num_non_zero, "num_corrections": num_corrections}


def demo_circuit_host(code, opts) -> Optional[dict]:
    state_prep_op = qec.operation.prep0

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x",
                                qec.TwoQubitDepolarization(opts.p_cnot),
                                num_controls=1)

    num_data = code.get_num_data_qubits()
    num_ancx = code.get_num_ancilla_x_qubits()
    num_ancz = code.get_num_ancilla_z_qubits()

    x_schedule = code.get_stabilizer_schedule_x().flatten().tolist()
    z_schedule = code.get_stabilizer_schedule_z().flatten().tolist()

    obs_matrix = code.get_observables_z()  # prep0 preserves logical Z
    num_obs = obs_matrix.shape[0]
    obs_flat = obs_matrix.flatten().tolist()

    if not setup_decoders(code, state_prep_op, opts, noise):
        return None  # --save_dem: config written, no shots

    # DEM characterization above always used the default stim target.
    # Now switch to the run target if the user asked for quantinuum.
    if opts.target == "quantinuum":
        extra = {}
        if opts.project_id:
            extra["project"] = opts.project_id
        if not opts.machine_name.endswith("SC"):
            if opts.max_cost > 0:
                extra["max_cost"] = opts.max_cost
            if opts.max_qubits > 0:
                extra["max_qubits"] = opts.max_qubits
        cudaq.set_target("quantinuum",
                         emulate=opts.emulate,
                         machine=opts.machine_name,
                         extra_payload_provider="decoder",
                         **extra)

    if opts.seed >= 0:
        cudaq.set_random_seed(opts.seed)

    run_result = run_shots(opts.num_shots, noise, obs_flat, num_obs, opts,
                           x_schedule, z_schedule, num_data, num_ancx, num_ancz)
    return report_results(run_result, opts.num_logical)


# ── argument parsing + validation ─────────────────────────────────────────────

_LEAF_DECODERS = ("multi_error_lut", "nv-qldpc-decoder", "pymatching")
_ALL_DECODERS = _LEAF_DECODERS + ("sliding_window",)


def run(argv: List[str]) -> Optional[dict]:
    """Parse argv, run the experiment, and return the result dict
    {"num_non_zero", "num_corrections"} (None for --save_dem, which only writes
    the config). Raises RuntimeError on invalid arguments or setup failures."""
    parser = argparse.ArgumentParser(description="Surface code demo 1")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--num_shots", type=int, default=10)
    parser.add_argument("--p_cnot",
                        type=float,
                        default=0.001,
                        help="Two-qubit depolarizing rate on CNOT gates")
    parser.add_argument("--num_logical", type=int, default=1)
    parser.add_argument("--num_rounds",
                        type=int,
                        default=-1,
                        help="Defaults to distance")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Simulator seed; negative leaves it unseeded")
    parser.add_argument("--decoder_type",
                        type=str,
                        default="multi_error_lut",
                        choices=list(_ALL_DECODERS))
    parser.add_argument(
        "--sw_window_size",
        type=int,
        default=-1,
        help="Sliding window size in rounds; defaults to distance")
    parser.add_argument("--sw_step_size",
                        type=int,
                        default=1,
                        help="Sliding window step size")
    parser.add_argument("--sw_inner_decoder",
                        type=str,
                        default=None,
                        help="Inner decoder for sliding_window")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        dest="decoder_params",
        metavar="KEY=VALUE",
        help="Override a decoder parameter (repeatable). For sliding_window "
        "the override targets the inner decoder. Example: --param lut_error_depth=1"
    )
    parser.add_argument("--save_dem",
                        type=str,
                        default=None,
                        help="Characterize DEM, write config YAML, and exit")
    parser.add_argument("--load_dem",
                        type=str,
                        default=None,
                        help="Load decoder config from YAML instead of "
                        "characterizing in-process")
    parser.add_argument("--target",
                        type=str,
                        default="stim",
                        help="Execution target (default: stim)")
    parser.add_argument("--machine_name",
                        type=str,
                        default="",
                        help="Machine name for Quantinuum target")
    parser.add_argument("--emulate",
                        action="store_true",
                        help="Use emulation when running on Quantinuum target")
    parser.add_argument("--project_id",
                        type=str,
                        default="",
                        help="Project ID for non-emulate Quantinuum runs")
    parser.add_argument("--max_cost",
                        type=int,
                        default=-1,
                        help="Max cost for Quantinuum target.")
    parser.add_argument("--max_qubits",
                        type=int,
                        default=-1,
                        help="Max qubits for Quantinuum target.")

    opts = parser.parse_args(argv)

    # Expand sentinel defaults
    if opts.num_rounds == -1:
        opts.num_rounds = opts.distance
    if opts.sw_window_size == -1:
        opts.sw_window_size = opts.distance

    sw_inner_decoder_set = opts.sw_inner_decoder is not None
    if not sw_inner_decoder_set:
        opts.sw_inner_decoder = "multi_error_lut"

    if sw_inner_decoder_set and opts.decoder_type in _LEAF_DECODERS:
        raise RuntimeError("--sw_inner_decoder is only valid with "
                           "--decoder_type sliding_window")

    # Attach dem_filename so host functions can access it uniformly
    opts.save_dem_flag = opts.save_dem is not None
    opts.load_dem_flag = opts.load_dem is not None
    opts.dem_filename = opts.save_dem or opts.load_dem or ""
    # Reuse the names the host functions expect
    opts.save_dem = opts.save_dem_flag
    opts.load_dem = opts.load_dem_flag

    if opts.save_dem and opts.load_dem:
        raise RuntimeError("Cannot use both --save_dem and --load_dem together")

    if opts.decoder_type == "sliding_window":
        if opts.sw_inner_decoder not in _LEAF_DECODERS:
            raise RuntimeError(
                f"--sw_inner_decoder must be one of {_LEAF_DECODERS}")
        if opts.sw_step_size < 1:
            raise RuntimeError(
                f"sw_step_size ({opts.sw_step_size}) must be >= 1")
        if opts.sw_window_size > opts.num_rounds:
            raise RuntimeError(
                f"sw_window_size ({opts.sw_window_size}) must be "
                f"<= num_rounds ({opts.num_rounds})")
        if opts.sw_step_size > opts.sw_window_size:
            raise RuntimeError(f"sw_step_size ({opts.sw_step_size}) must be "
                               f"<= sw_window_size ({opts.sw_window_size})")

    if opts.num_rounds < opts.distance:
        raise RuntimeError(f"num_rounds ({opts.num_rounds}) must be at least "
                           f"equal to distance ({opts.distance})")

    if opts.num_logical > 32:
        raise RuntimeError("num_logical > 32 is not supported.")

    if opts.target == "quantinuum":
        if opts.machine_name == "":
            if not opts.emulate:
                raise RuntimeError(
                    "machine_name must be set when target is quantinuum")
            opts.machine_name = "Helios-LocalE"
        if not opts.emulate and not opts.project_id:
            raise RuntimeError(
                "project_id must be set for non-emulate quantinuum runs")

    print(f"Running with p_cnot = {opts.p_cnot}, distance = {opts.distance}, "
          f"num_shots = {opts.num_shots}, num_rounds = {opts.num_rounds}")

    code = qec.get_code("surface_code", distance=opts.distance)
    return demo_circuit_host(code, opts)


def main(argv: List[str]) -> int:
    """CLI wrapper around run(): returns 0 on success and 1 on error."""
    try:
        run(argv)
    except Exception as e:
        print(f"Error: {e}")
        qec.finalize_decoders()
        return 1
    qec.finalize_decoders()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

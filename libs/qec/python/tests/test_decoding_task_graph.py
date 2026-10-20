# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import threading

import pytest
import cudaq_qec as qec

# ---------------------------------------------------------------------------
# Helpers — same rep-5 spec used in test_dem.py
# ---------------------------------------------------------------------------


def _make_phase_entry(name_str, num_faults, H_sparse, O_sparse, error_rates):
    entry = qec.PhaseSpecEntry()
    entry.id = qec.SeamId(name_str)
    spec = qec.DemChunkSpec()
    spec.num_faults = num_faults
    spec.H_sparse = H_sparse
    spec.O_sparse = O_sparse
    spec.error_rates = error_rates
    entry.spec = spec
    return entry


def _make_connection(from_str, to_str):
    c = qec.PhaseConnection()
    c.from_phase = qec.SeamId(from_str)
    c.to_phase = qec.SeamId(to_str)
    return c


def _rep5_spec(num_rounds=5):
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


def _streaming_spec():
    spec = _rep5_spec(5)
    spec.num_rounds = None
    return spec


# ---------------------------------------------------------------------------
# from_spec — construction and basic properties
# ---------------------------------------------------------------------------


def test_from_spec_builds_graph():
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    assert graph.get_num_msyn_per_decode() > 0
    assert graph.get_num_observables() > 0


def test_from_spec_streaming_builds_graph():
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    assert graph.get_num_msyn_per_decode() > 0
    assert graph.get_num_observables() > 0


# ---------------------------------------------------------------------------
# One-shot execution — run / run_all; run_batch handles multiple records
# ---------------------------------------------------------------------------


def test_run_zero_syndromes_gives_zero_corrections():
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    m = qec.MeasurementResults()
    m.bits = [0] * n
    lo = graph.run(m)
    assert lo.converged
    assert all(b == 0 for b in lo.bits)


def _window_node(id_):
    node = qec.TaskGraphNodeConfig()
    node.id = id_
    node.kind = "single_window"
    node.dem = _rep5_spec(5)
    node.decoder_type = "single_error_lut"
    return node


def _tg_edge(from_node, to_node, from_output="output", to_input="input"):
    edge = qec.task_graph_edge()
    edge.from_node = from_node
    edge.from_output = from_output
    edge.to_node = to_node
    edge.to_input = to_input
    return edge


def test_from_topology_builds_a_multi_output_graph():
    graph = qec.DecodingTaskGraph.from_topology(
        [_window_node("a"), _window_node("b")],
        [
            _tg_edge("measurements", "a"),
            _tg_edge("measurements", "b"),
            _tg_edge("a", "output", to_input="a_out"),
            _tg_edge("b", "output", to_input="b_out"),
        ],
    )

    assert list(graph.output_names()) == ["a_out", "b_out"]

    m = qec.MeasurementResults()
    m.bits = [0] * graph.get_num_msyn_per_decode()

    outcomes = graph.run_all(m)
    assert len(outcomes) == 2
    for lo in outcomes:
        assert all(b == 0 for b in lo.bits)

    # One outcome cannot answer for two outputs, and the message says which
    # call was made.
    with pytest.raises(RuntimeError, match="run"):
        graph.run(m)


def test_from_topology_races_an_ensemble_of_windows():
    ens = qec.TaskGraphEnsembleSpec()
    ens.join_strategy = "deadline"
    ens.join_params = {"deadline_us": 200000}
    # One member answers through `output`; the other pushes its own
    # candidates and takes the token.
    ens.add_member([_window_node("a")],
                   [_tg_edge("input", "a"),
                    _tg_edge("a", "output")])
    b = _window_node("b")
    b.node_params = {"push_candidates": True, "cancellable": True}
    ens.add_member([b], [_tg_edge("input", "b")])
    assert ens.num_members == 2

    node = qec.TaskGraphNodeConfig()
    node.id = "ens"
    node.ensemble = ens
    assert node.ensemble.join_strategy == "deadline"

    graph = qec.DecodingTaskGraph.from_topology([node])
    # Fed raw measurements, the ensemble is one-shot only: size the shot from
    # a lone window over the same DEM.
    probe = qec.DecodingTaskGraph.from_topology([_window_node("probe")])
    m = qec.MeasurementResults()
    m.bits = [0] * probe.get_num_msyn_per_decode()
    assert all(b == 0 for b in graph.run(m).bits)


def test_streaming_result_tracks_independent_outputs_consume_and_reset():
    fast = _window_node("fast")
    slow = _window_node("slow")
    slow.dem = _rep5_spec(10)
    graph = qec.DecodingTaskGraph.from_topology(
        [fast, slow],
        [
            _tg_edge("measurements", "fast"),
            _tg_edge("measurements", "slow"),
            _tg_edge("fast", "output", to_input="fast"),
            _tg_edge("slow", "output", to_input="slow"),
        ],
    )
    half_window = bytes(5 * graph.get_num_msyn_per_round())

    assert graph.enqueue_syndrome(half_window)
    partial = graph.streaming_result()
    assert partial.outputs[0].generation == 1
    assert partial.outputs[0].status == qec.TaskGraphStreamingStatus.AVAILABLE
    assert partial.outputs[1].generation == 0
    assert partial.outputs[1].status == qec.TaskGraphStreamingStatus.PENDING
    assert partial.generation == 0

    assert graph.enqueue_syndrome(half_window)
    # Python returns a snapshot, not a live alias into the graph-owned result.
    assert partial.outputs[1].generation == 0
    assert partial.generation == 0
    available = graph.streaming_result()
    assert available.generation == 1
    epoch = available.shot_epoch
    revision = available.result_revision

    graph.clear_corrections()
    consumed = graph.streaming_result()
    assert consumed.shot_epoch == epoch
    assert consumed.result_revision == revision + 1
    assert consumed.generation == 1
    assert consumed.status == qec.TaskGraphStreamingStatus.PENDING

    graph.reset_decoder()
    reset = graph.streaming_result()
    assert reset.shot_epoch == epoch + 1
    assert reset.result_revision == 0
    assert reset.generation == 0
    assert all(output.generation == 0 for output in reset.outputs)


def test_run_all_and_output_names_agree_with_run():
    # The single-output case: run_all() returns the one outcome run() does.
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    m = qec.MeasurementResults()
    m.bits = [0] * n

    names = graph.output_names()
    assert len(names) == 1
    outcomes = graph.run_all(m)
    assert len(outcomes) == len(names)
    assert list(outcomes[0].bits) == list(graph.run(m).bits)


def test_run_batch_matches_sequential():
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    shots = []
    for _ in range(4):
        m = qec.MeasurementResults()
        m.bits = [0] * n
        shots.append(m)
    results = graph.run_batch(shots)
    assert len(results) == 4
    for lo in results:
        assert all(b == 0 for b in lo.bits)


# ---------------------------------------------------------------------------
# Streaming interface — enqueue_syndrome / corrections
# ---------------------------------------------------------------------------


def test_enqueue_syndrome_fires_and_returns_zero_corrections():
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    fired = graph.enqueue_syndrome(bytes(n))
    assert fired
    corr = graph.get_obs_corrections()
    assert len(corr) == graph.get_num_observables()
    assert all(b == 0 for b in corr)
    result = graph.streaming_result()
    assert result.operation_completed
    assert result.result_revision == 1
    assert result.status == qec.TaskGraphStreamingStatus.AVAILABLE
    assert result.outputs[0].generation == 1
    assert result.outputs[0].status == qec.TaskGraphStreamingStatus.AVAILABLE
    assert list(result.values) == [0] * graph.get_num_observables()


def test_async_streaming_uses_existing_api_with_large_input():
    options = qec.TaskGraphOptions()
    options.async_streaming = True
    options.async_queue_capacity = 2
    options.async_spin_pause_iters = 17
    assert options.async_coordinator_cpu == -1
    assert options.async_spin_pause_iters == 17
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut",
                                            options=options)
    capacity = graph.get_num_msyn_per_decode()
    assert capacity > graph.get_num_msyn_per_round()

    # BULK reports acceptance asynchronously; completion arrives in snapshots.
    assert graph.enqueue_syndrome(bytes(capacity)) is False
    result = graph.streaming_result()

    assert result.generation == 1
    assert result.operation_completed
    assert bytes(graph.get_obs_corrections()) == bytes(result.values)
    graph.clear_corrections()


def _guarded_graph(async_streaming=False):
    node = qec.TaskGraphNodeConfig()
    node.id = "guarded"
    node.kind = "sliding_window"
    node.dem = _rep5_spec(5)
    node.decoder_type = "single_error_lut"
    node.node_params = {
        "window_size": 5,
        "step_size": 3,
        "start_guard": 1,
        "end_guard": 1,
    }
    options = qec.TaskGraphOptions()
    options.async_streaming = async_streaming
    options.async_queue_capacity = 1
    return qec.DecodingTaskGraph.from_topology([node], options=options)


def test_async_clear_mid_shot_matches_guarded_synchronous_graph():
    synchronous = _guarded_graph()
    asynchronous = _guarded_graph(async_streaming=True)
    window = synchronous.get_num_msyn_per_decode()
    per_round = synchronous.get_num_msyn_per_round()
    first = bytearray(window)
    first[window - per_round] = 1

    assert synchronous.enqueue_syndrome(bytes(first))
    assert asynchronous.enqueue_syndrome(bytes(first)) is False
    synchronous.clear_corrections()
    asynchronous.clear_corrections()
    sync_consumed = synchronous.streaming_result()
    async_consumed = asynchronous.streaming_result()
    assert async_consumed.generation == sync_consumed.generation
    assert bytes(async_consumed.values) == bytes(sync_consumed.values)
    assert [o.status for o in async_consumed.outputs
           ] == [o.status for o in sync_consumed.outputs]

    stride = bytes(3 * per_round)
    assert synchronous.enqueue_syndrome(stride)
    assert asynchronous.enqueue_syndrome(stride) is False
    last_round = bytes(per_round)
    assert synchronous.enqueue_syndrome(last_round, qec.round_type.FINAL)
    assert asynchronous.enqueue_syndrome(last_round, qec.round_type.FINAL)
    assert bytes(asynchronous.streaming_result().values) == bytes(
        synchronous.streaming_result().values)


def test_async_combine_uses_existing_streaming_api():
    synchronous = qec.DecodingTaskGraph.combine(
        qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut"),
        qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut"),
    )
    options = qec.TaskGraphOptions()
    options.async_streaming = True
    options.async_queue_capacity = 2
    asynchronous = qec.DecodingTaskGraph.combine(
        qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut"),
        qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut"),
        options=options,
    )
    syndrome = bytearray(synchronous.get_num_msyn_per_decode())
    syndrome[0] = 1

    assert synchronous.enqueue_syndrome(bytes(syndrome), qec.round_type.FINAL)
    assert asynchronous.enqueue_syndrome(bytes(syndrome), qec.round_type.FINAL)
    result = asynchronous.streaming_result()
    assert result.status == qec.TaskGraphStreamingStatus.COMPLETE
    assert result.operation_completed
    assert bytes(result.values) == bytes(synchronous.get_obs_corrections())


def test_async_streaming_rejects_overlapping_python_producers():
    options = qec.TaskGraphOptions()
    options.async_streaming = True
    options.async_queue_capacity = 1
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5),
                                            "single_error_lut",
                                            options=options)
    assert graph.enqueue_syndrome(bytes(
        graph.get_num_msyn_per_round())) is False
    graph.streaming_result()

    num_threads = 8
    iterations = 50
    rendezvous = threading.Barrier(num_threads)
    errors = []

    def produce():
        # Capacity one creates public-call backpressure while the coordinator
        # consumes the previous record. Since the binding releases the GIL,
        # competing Python threads must then hit the producer-call guard.
        syndrome = bytes(64 * graph.get_num_msyn_per_round())
        for _ in range(iterations):
            rendezvous.wait()
            try:
                graph.enqueue_syndrome(syndrome)
            except RuntimeError as error:
                errors.append(str(error))

    producers = [threading.Thread(target=produce) for _ in range(num_threads)]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join(timeout=10)
        assert not producer.is_alive()

    # Whether producers overlap depends on thread scheduling, so a run without
    # overlap is not a failure. The C++ test
    # AsyncStreamingRejectsOverlappingProducers covers the guard itself.
    assert all("concurrent producers" in error for error in errors)
    graph.streaming_result()
    if not errors:
        pytest.skip("producers never overlapped on this runner")


def test_empty_final_round_is_rejected():
    # FINAL carries the shot's final measurements.
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    with pytest.raises(ValueError):
        graph.enqueue_syndrome(b"", qec.round_type.FINAL)


def test_clear_corrections_zeros_buffer():
    graph = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    graph.enqueue_syndrome(bytes(n))
    graph.clear_corrections()
    assert all(b == 0 for b in graph.get_obs_corrections())


def test_reset_decoder_clears_streaming_state():
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    graph.enqueue_syndrome(bytes(n))
    graph.reset_decoder()
    # After reset, re-filling the full window must fire again cleanly.
    assert graph.enqueue_syndrome(bytes(n))
    assert all(b == 0 for b in graph.get_obs_corrections())


def test_a_final_round_is_a_noop_on_an_unguarded_window():
    # _streaming_spec() builds an unguarded window, which withholds nothing, so
    # FINAL must not manufacture a second commit of the last window. The
    # guarded tail is pinned in C++ (AFinalRoundCommitsTheTailOnAFullWindow),
    # where guards are reachable.
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    assert graph.enqueue_syndrome(bytes(n), qec.round_type.FINAL)
    before = bytes(graph.get_obs_corrections())

    # The shot is closed: further rounds are dropped with a warning, not an
    # exception, and the corrections do not move.
    assert graph.enqueue_syndrome(bytes(n), qec.round_type.BULK) is False
    assert bytes(graph.get_obs_corrections()) == before

    # clear_corrections() is the between-shots call, so it re-opens.  Feed
    # rounds rather than a whole shot: clear_corrections zeroes the
    # accumulator, not the window buffer, so the window still holds carry-over.
    graph.clear_corrections()
    per_round = graph.get_num_msyn_per_round()
    fired = [
        graph.enqueue_syndrome(bytes(per_round)) for _ in range(n // per_round)
    ]
    assert any(fired), "the graph stayed closed after clear_corrections()"


def test_round_type_is_exposed():
    # FINAL is the only public way to end a stream now, so the enum has to be
    # reachable from Python.
    assert qec.round_type.BULK != qec.round_type.FINAL


def test_streaming_partial_fill_does_not_fire():
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    n = graph.get_num_msyn_per_decode()
    if n > 1:
        fired = graph.enqueue_syndrome(bytes(n - 1))
        assert not fired


def test_per_round_size_divides_the_window():
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    per_round = graph.get_num_msyn_per_round()
    capacity = graph.get_num_msyn_per_decode()
    assert per_round > 0
    assert capacity > per_round
    assert capacity % per_round == 0


def test_streaming_fires_every_round_once_window_is_full():
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    per_round = graph.get_num_msyn_per_round()
    rounds_per_window = graph.get_num_msyn_per_decode() // per_round
    for _ in range(rounds_per_window - 1):
        assert not graph.enqueue_syndrome(bytes(per_round))
    # Full window, then one fire per round because the stride is one round.
    assert graph.enqueue_syndrome(bytes(per_round))
    assert graph.enqueue_syndrome(bytes(per_round))


def test_oversized_enqueue_is_consumed_rather_than_dropped():
    graph = qec.DecodingTaskGraph.from_spec(_streaming_spec(),
                                            "single_error_lut")
    per_round = graph.get_num_msyn_per_round()
    capacity = graph.get_num_msyn_per_decode()
    assert capacity // per_round > 1

    # The graph meters a chunk to each node's remaining room, so more than a
    # window's worth in one call drives every fire it covers and carries the
    # remainder forward. get_num_msyn_per_decode() is a size to prefer, not a
    # limit; a caller sending whole rounds sees no difference.
    assert not graph.enqueue_syndrome(bytes(per_round))
    assert graph.enqueue_syndrome(bytes(capacity + 1))


# ---------------------------------------------------------------------------
# combine — two-decoder graph
# ---------------------------------------------------------------------------


def test_combine_zero_syndromes_zero_corrections_one_shot():
    left = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    right = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    combined = qec.DecodingTaskGraph.combine(left, right)
    assert combined is not None

    n = combined.get_num_msyn_per_decode()
    m = qec.MeasurementResults()
    m.bits = [0] * n
    lo = combined.run(m)
    assert lo.converged
    assert all(b == 0 for b in lo.bits)


def test_combine_streaming_xors_corrections():
    left = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    right = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    combined = qec.DecodingTaskGraph.combine(left, right)

    n = combined.get_num_msyn_per_decode()
    assert combined.enqueue_syndrome(bytes(n))
    corr = combined.get_obs_corrections()
    assert len(corr) == combined.get_num_observables()
    # XOR of two identical zero-syndrome results is zero.
    assert all(b == 0 for b in corr)


def test_combine_with_unequal_windows_is_batch_only():
    left = qec.DecodingTaskGraph.from_spec(_rep5_spec(5), "single_error_lut")
    right = qec.DecodingTaskGraph.from_spec(_rep5_spec(7), "single_error_lut")
    combined = qec.DecodingTaskGraph.combine(left, right)

    assert not combined.supports_streaming()
    assert "window_size" in combined.streaming_incompatibility_reason()

    m = qec.MeasurementResults()
    m.bits = [0] * combined.get_num_msyn_per_decode()
    assert combined.run(m).converged
    with pytest.raises(RuntimeError, match="window_size"):
        combined.enqueue_syndrome(bytes(combined.get_num_msyn_per_round()))

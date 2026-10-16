# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq_qec as qec


def create_test_matrix():
    np.random.seed(42)
    return np.random.randint(0, 2, (10, 20)).astype(np.uint8)


def create_test_syndrome():
    np.random.seed(42)
    return np.random.random(10).tolist()


H = create_test_matrix()


def test_decoder_initialization():
    decoder = qec.get_decoder('example_byod', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_initialization_with_fortran_order():
    # Fortran-order (column-major) arrays are now handled via stride-aware
    # scanning and should work correctly.
    H_f = np.eye(10, 20, dtype=np.uint8, order='F')
    H_c = np.ascontiguousarray(H_f)
    decoder_f = qec.get_decoder('single_error_lut_example', H_f)
    decoder_c = qec.get_decoder('single_error_lut_example', H_c)
    syndrome = np.zeros(H_f.shape[0], dtype=np.uint8)
    r_f = decoder_f.decode(syndrome)
    r_c = decoder_c.decode(syndrome)
    assert r_f.converged == r_c.converged
    assert list(r_f.result) == list(r_c.result)


def test_decoder_api():
    # Test decode_batch
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode_batch(
        [create_test_syndrome(), create_test_syndrome()])
    assert isinstance(result, qec.BatchDecoderResult)
    assert len(result) == 2
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert hasattr(result, 'opt_results')
    assert isinstance(result.converged, np.ndarray)
    assert result.converged.shape == (2,)
    assert isinstance(result.result, np.ndarray)
    assert result.result.shape == (2, 10)
    assert len(result.opt_results) == 2
    assert all(opt is None for opt in result.opt_results)
    first = result[0]
    assert isinstance(first, qec.DecoderResult)
    assert first.converged == result.converged[0]
    np.testing.assert_array_equal(first.result, result.result[0])
    assert first.opt_results is None
    last = result[-1]
    np.testing.assert_array_equal(last.result, result.result[-1])
    sliced = result[:1]
    assert isinstance(sliced, qec.BatchDecoderResult)
    assert sliced.result.shape == (1, 10)
    assert sliced.converged.shape == (1,)
    assert len(sliced.opt_results) == 1
    iterated = list(result)
    assert len(iterated) == 2
    assert all(isinstance(r, qec.DecoderResult) for r in iterated)
    np.testing.assert_array_equal(iterated[1].result, result.result[1])

    # Empty batch: shape (0, 0); per-shot width is undefined without input.
    empty_result = decoder.decode_batch([])
    assert isinstance(empty_result, qec.BatchDecoderResult)
    assert empty_result.result.shape == (0, 0)
    assert empty_result.converged.shape == (0,)
    assert len(empty_result) == 0
    assert len(empty_result.opt_results) == 0

    # Test decode_async
    decoder = qec.get_decoder('example_byod', H)
    result_async = decoder.decode_async(create_test_syndrome())
    assert hasattr(result_async, 'get')
    assert hasattr(result_async, 'ready')

    result = result_async.get()
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)
    assert result.result.shape == (10,)


def test_decode_async_temporary_keeps_decoder_alive():
    # The future must retain this temporary decoder through virtual decode.
    async_result = qec.get_decoder('example_byod',
                                   H).decode_async(create_test_syndrome())
    result = async_result.get()

    # ASSERT: successful output proves the temporary survived until completion.
    assert result.converged and result.result.shape == (10,)


def test_decoder_result_structure():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    # Test basic structure
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert hasattr(result, 'opt_results')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)
    assert result.result.shape == (10,)

    # Test opt_results functionality
    assert result.opt_results is None  # Default should be None

    # Test that opt_results is preserved in async decode
    async_result = decoder.decode_async(create_test_syndrome())
    result = async_result.get()
    assert hasattr(result, 'opt_results')
    assert result.opt_results is None


def test_batch_decoder_result_constructor():
    result = np.zeros((2, 3), dtype=np.float64)
    converged = np.array([True, False], dtype=np.bool_)
    batch_result = qec.BatchDecoderResult(result, converged)

    assert isinstance(batch_result.result, np.ndarray)
    assert batch_result.result.shape == (2, 3)
    assert batch_result.converged.tolist() == [True, False]
    assert batch_result.opt_results == [None, None]
    assert batch_result[np.int64(0)].converged is True
    assert batch_result[::2].result.shape == (1, 3)
    assert np.shares_memory(batch_result[::2].result, batch_result.result)

    empty_result = qec.BatchDecoderResult(np.empty((0, 3), dtype=np.float64),
                                          np.array([], dtype=np.bool_))
    assert len(empty_result) == 0
    assert empty_result.result.shape == (0, 3)
    assert empty_result.converged.shape == (0,)
    assert list(empty_result) == []
    with pytest.raises(IndexError):
        empty_result[0]

    # Cross-array invariants we still enforce.
    with pytest.raises(RuntimeError, match="row count must match"):
        qec.BatchDecoderResult(result,
                               np.array([True, False, True], dtype=np.bool_))

    with pytest.raises(RuntimeError, match="opt_results length must match"):
        qec.BatchDecoderResult(result, converged, [None])

    # Rank is enforced by nanobind, surfaced as TypeError.
    with pytest.raises(TypeError):
        qec.BatchDecoderResult(np.zeros(3, dtype=np.float64), converged)

    # dtype and contiguity are coerced silently, not rejected.
    int_result = qec.BatchDecoderResult(np.zeros((2, 3), dtype=np.int32),
                                        converged)
    assert int_result.result.dtype == batch_result.result.dtype
    assert int_result.result.flags.c_contiguous

    f_order = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
    assert not f_order.flags.c_contiguous
    f_result = qec.BatchDecoderResult(f_order, converged)
    assert f_result.result.flags.c_contiguous


def test_python_decoder_batch_preserves_opt_results():

    @qec.decoder("python_opt_results_byod")
    class PythonOptResultsDecoder:

        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H

        def decode(self, syndrome):
            res = qec.DecoderResult()
            res.converged = True
            res.result = np.arange(self.H.shape[1], dtype=np.float64)
            res.opt_results = {
                "syndrome_weight": int(np.count_nonzero(syndrome)),
                "tag": "python"
            }
            return res

    decoder = qec.get_decoder("python_opt_results_byod", H)
    batch_result = decoder.decode_batch(
        [np.zeros(H.shape[0]), np.ones(H.shape[0])])

    assert isinstance(batch_result, qec.BatchDecoderResult)
    assert batch_result.result.shape == (2, H.shape[1])
    assert batch_result.converged.tolist() == [True, True]
    assert batch_result.opt_results[0]["syndrome_weight"] == 0
    assert batch_result.opt_results[1]["syndrome_weight"] == H.shape[0]
    assert batch_result[1].opt_results["tag"] == "python"


def test_batch_opt_results_defaults_to_none():
    # Decoders that produce no batch-level results must report None rather
    # than an empty dict, so callers can distinguish "not supported" from
    # "supported but empty".
    decoder = qec.get_decoder('example_byod', H)
    batch_result = decoder.decode_batch(
        [create_test_syndrome(), create_test_syndrome()])
    assert hasattr(batch_result, 'batch_opt_results')
    assert batch_result.batch_opt_results is None

    # Same for a decoder whose per-shot opt_results are populated: the two
    # channels are independent.
    empty_result = decoder.decode_batch([])
    assert empty_result.batch_opt_results is None


def test_batch_opt_results_constructor_and_slicing():
    result = np.zeros((3, 4), dtype=np.float64)
    converged = np.ones(3, dtype=bool)

    # Omitted -> None, keeping the pre-existing 2- and 3-argument forms valid.
    assert qec.BatchDecoderResult(result, converged).batch_opt_results is None
    assert qec.BatchDecoderResult(result, converged,
                                  None).batch_opt_results is None

    payload = {"counts": np.array([1, 2, 3], dtype=np.int32), "width": 12}
    batch_result = qec.BatchDecoderResult(result, converged, None, payload)
    assert batch_result.batch_opt_results["width"] == 12
    np.testing.assert_array_equal(batch_result.batch_opt_results["counts"],
                                  [1, 2, 3])

    # Batch-level arrays are indexed by position in the full batch, so a slice
    # drops them instead of carrying through values that no longer line up
    # with the sliced rows.
    assert batch_result[:2].batch_opt_results is None

    # Integer indexing yields a DecoderResult, which has no batch-level
    # channel at all.
    assert not hasattr(batch_result[0], 'batch_opt_results')


def test_python_decoder_can_return_batch_opt_results():

    @qec.decoder("python_batch_opt_results_byod")
    class PythonBatchOptResultsDecoder:

        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H

        def decode(self, syndrome):
            res = qec.DecoderResult()
            res.converged = True
            res.result = np.zeros(self.H.shape[1], dtype=np.float64)
            return res

        def decode_batch(self, syndromes):
            n = len(syndromes)
            return qec.BatchDecoderResult(
                np.zeros((n, self.H.shape[1]), dtype=np.float64),
                np.ones(n, dtype=bool),
                None,
                {"weights": np.arange(n, dtype=np.float32)},
            )

    decoder = qec.get_decoder("python_batch_opt_results_byod", H)
    batch_result = decoder.decode_batch(
        [np.zeros(H.shape[0]), np.ones(H.shape[0])])
    weights = batch_result.batch_opt_results["weights"]
    assert weights.dtype == np.float32
    np.testing.assert_array_equal(weights, [0.0, 1.0])


def test_python_decoder_batch_override_must_return_batch_decoder_result():

    @qec.decoder("python_bad_batch_byod")
    class PythonBadBatchDecoder:

        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H

        def decode(self, syndrome):
            res = qec.DecoderResult()
            res.converged = True
            res.result = np.zeros(self.H.shape[1], dtype=np.float64)
            return res

        def decode_batch(self, syndromes):
            # Pre-batch return shape; the decorator should reject this.
            return [self.decode(s) for s in syndromes]

    decoder = qec.get_decoder("python_bad_batch_byod", H)
    with pytest.raises(TypeError, match="must return a BatchDecoderResult"):
        decoder.decode_batch([np.zeros(H.shape[0]), np.ones(H.shape[0])])


def test_decoder_plugin_initialization():
    decoder = qec.get_decoder('single_error_lut_example', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_double_vec():
    vec = np.array([1, 2, 3], dtype=np.float64)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_float_vec():
    vec = np.array([1, 2, 3], dtype=np.float32)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_uint8_vec():
    vec = np.array([1, 2, 3], dtype=np.uint8)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_int32_vec():
    vec = np.array([1, 2, 3], dtype=np.int32)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_int16_vec():
    vec = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert "Unsupported array data type" in repr(e)


def test_decoder_kwargs_accept_lists_and_2d_arrays():
    decoder = qec.get_decoder(
        'single_error_lut_example',
        H,
        flat_values=[0.1, 0.2, 0.3],
        nested_values=[[0.1, 0.2], [0.3, 0.4]],
        matrix_float32=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        matrix_float64=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        matrix_int32=np.array([[1, 2], [3, 4]], dtype=np.int32),
        matrix_uint8=np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_result_structure():
    decoder = qec.get_decoder('single_error_lut_example', H)
    result = decoder.decode(create_test_syndrome())

    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)


def test_single_error_lut_example_uses_canonical_threshold():
    H = np.array([[0, 1]], dtype=np.uint8)
    decoder = qec.get_decoder('single_error_lut_example', H)
    result = decoder.decode([0.5])

    # For this H, syndrome "1" maps to a correction on qubit 1.
    assert result.converged is True
    assert np.array_equal(result.result, [0.0, 1.0])


def test_decoder_result_values():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    assert result.converged is True
    assert isinstance(result.result, np.ndarray)
    assert np.all((0 <= result.result) & (result.result <= 1))


@pytest.mark.parametrize("matrix_shape,syndrome_size", [((5, 10), 5),
                                                        ((15, 30), 15),
                                                        ((20, 40), 20)])
def test_decoder_different_matrix_sizes(matrix_shape, syndrome_size):
    np.random.seed(42)
    H = np.random.randint(0, 2, matrix_shape).astype(np.uint8)
    syndrome = np.random.random(syndrome_size).tolist()

    decoder = qec.get_decoder('example_byod', H)
    convergence, result, opt = decoder.decode(syndrome)

    assert len(result) == syndrome_size
    assert convergence is True
    assert isinstance(result, np.ndarray)
    assert np.all((0 <= result) & (result <= 1))


# FIXME add this back
# def test_decoder_error_handling():
#     H = Tensor(create_test_matrix())
#     decoder = qec.get_decoder('example_byod', H)

#     # Test with incorrect syndrome size
#     with pytest.raises(ValueError):
#         wrong_syndrome = np.random.random(15).tolist()  # Wrong size
#         decoder.decode(wrong_syndrome)

#     # Test with invalid syndrome type
#     with pytest.raises(TypeError):
#         wrong_type_syndrome = "invalid"
#         decoder.decode(wrong_type_syndrome)


def test_decoder_reproducibility():
    decoder = qec.get_decoder('example_byod', H)

    np.random.seed(42)
    convergence1, result1, opt1 = decoder.decode(create_test_syndrome())

    np.random.seed(42)
    convergence2, result2, opt2 = decoder.decode(create_test_syndrome())

    np.testing.assert_array_equal(result1, result2)
    assert convergence1 == convergence2


def test_pass_weights():
    error_probability = 0.1
    weights = np.ones(H.shape[1]) * np.log(
        (1 - error_probability) / error_probability)
    decoder = qec.get_decoder('example_byod', H, weights=weights)
    # Test is that no error is thrown


def test_sort_pcm_columns_non_decreasing_column_weight():
    # Create a test parity-check matrix with random binary values.
    # yapf: disable
    H = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1]],
                 dtype=np.uint8)
    # yapf: enable

    H_calculated = qec.sort_pcm_columns(H)

    # yapf: disable
    H_expected = np.array(
        [[1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 1, 0],
         [0, 1, 1, 0, 0, 0, 1, 1, 1]],
        dtype=np.uint8)
    # yapf: enable

    assert np.array_equal(H_calculated, H_expected)

    col_order = qec.get_sorted_pcm_column_indices(H)
    expected_order = [1, 8, 4, 0, 3, 2, 6, 7, 5]
    assert col_order == expected_order

    # Now check that reordering the columns of H yields H_expected
    H_reordered = qec.reorder_pcm_columns(H, col_order)
    assert np.array_equal(H_reordered, H_expected)


def test_sort_pcm_columns_invalid_input():
    # Test that passing a non-2D array raises an error.
    H_invalid = np.array([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError):
        qec.get_sorted_pcm_column_indices(H_invalid)


def test_gen_random_pcm():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    is_sorted = qec.pcm_is_sorted(pcm)
    assert is_sorted is False
    pcm = qec.sort_pcm_columns(pcm)
    is_sorted = qec.pcm_is_sorted(pcm)
    assert is_sorted is True
    print('')
    qec.dump_pcm(pcm)
    print('')
    assert pcm.shape == (100, 200)


def test_get_pcm_for_rounds():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    pcm = qec.sort_pcm_columns(pcm)
    pcm_for_rounds, first_column, last_column = qec.get_pcm_for_rounds(
        pcm, 10, 0, 1)
    assert pcm_for_rounds.shape == (20, 30)
    print('')
    qec.dump_pcm(pcm_for_rounds)
    print('')


def test_get_pcm_for_rounds_boundary():
    # Boundary layout [B | S | S | B]: boundary width B=3, interior width S=5,
    # two interior rounds => 4 detector layers over 2*B + 2*S = 16 rows. Each
    # column carries a single nonzero placed in exactly one round.
    B, S = 3, 5
    round_starts = [0, 3, 8, 13]  # first row of rounds 0..3
    num_rows = 16
    pcm = np.zeros((num_rows, len(round_starts)), dtype=np.uint8)
    for col, start in enumerate(round_starts):
        pcm[start, col] = 1

    # Boundary-aware slice of rounds 0..1 keeps rows [0, 8) = B + S and only the
    # two columns whose nonzeros fall in those rounds.
    sub, first_column, last_column = qec.get_pcm_for_rounds(
        pcm, S, 0, 1, num_boundary_syndromes=B)
    assert sub.shape == (B + S, 2)
    assert (first_column, last_column) == (0, 1)
    assert sub[0, 0] == 1 and sub[3, 1] == 1 and sub.sum() == 2

    # Interior-only span (rounds 1..2) keeps rows [3, 13) = 2*S.
    sub_interior, _, _ = qec.get_pcm_for_rounds(pcm,
                                                S,
                                                1,
                                                2,
                                                num_boundary_syndromes=B)
    assert sub_interior.shape == (2 * S, 2)

    # Omitting num_boundary_syndromes reinterprets the same matrix as a uniform
    # S-wide layout, which slices rounds 0..1 to rows [0, 10) and selects a
    # different set of columns -- confirming the parameter changes behavior.
    sub_uniform, _, _ = qec.get_pcm_for_rounds(pcm, S, 0, 1)
    assert sub_uniform.shape == (2 * S, 3)

    # An out-of-range end_round is rejected for the boundary layout (4 layers).
    with pytest.raises(ValueError, match="end_round"):
        qec.get_pcm_for_rounds(pcm, S, 0, 4, num_boundary_syndromes=B)


def test_shuffle_pcm_columns():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    sorted_pcm = qec.sort_pcm_columns(pcm)
    shuffled_pcm = qec.shuffle_pcm_columns(sorted_pcm, seed=13)

    # They should not be equal here
    assert not np.array_equal(sorted_pcm, shuffled_pcm)

    # They should be equal after sorting
    assert np.array_equal(qec.sort_pcm_columns(shuffled_pcm), sorted_pcm)


def test_reorder_pcm_columns_scipy_sparse():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)
    column_order = [2, 0]

    dense_reordered = qec.reorder_pcm_columns(H, column_order)
    sparse_reordered = qec.reorder_pcm_columns(scipy_sparse.csr_matrix(H),
                                               column_order)

    assert scipy_sparse.issparse(sparse_reordered)
    assert scipy_sparse.isspmatrix_csc(sparse_reordered)
    assert np.array_equal(sparse_reordered.toarray(), dense_reordered)


def test_reorder_pcm_columns_scipy_sparse_zero_nnz_result():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H = scipy_sparse.csr_matrix((3, 4), dtype=np.uint8)

    sparse_reordered = qec.reorder_pcm_columns(H, [2, 0])

    assert scipy_sparse.isspmatrix_csc(sparse_reordered)
    assert sparse_reordered.shape == (3, 2)
    assert sparse_reordered.nnz == 0


def test_shuffle_pcm_columns_scipy_sparse():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)

    dense_shuffled = qec.shuffle_pcm_columns(H, seed=13)
    sparse_shuffled = qec.shuffle_pcm_columns(scipy_sparse.csc_matrix(H),
                                              seed=13)

    assert scipy_sparse.issparse(sparse_shuffled)
    assert scipy_sparse.isspmatrix_csc(sparse_shuffled)
    assert np.array_equal(sparse_shuffled.toarray(), dense_shuffled)


def test_pcm_to_sparse_vec_scipy_sparse():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)

    sparse_vec = qec.pcm_to_sparse_vec(scipy_sparse.csr_matrix(H))
    assert sparse_vec == qec.pcm_to_sparse_vec(H)


def test_simplify_pcm():
    syndromes_per_round = 10
    pcm = qec.generate_random_pcm(
        n_rounds=10,
        n_errs_per_round=30,
        n_syndromes_per_round=syndromes_per_round,
        weight=1,  # force some duplicate columns for this test
        seed=13)
    weights = np.ones(pcm.shape[1]) * 0.01
    new_pcm, new_weights = qec.simplify_pcm(pcm, weights, syndromes_per_round)
    # qec.dump_pcm(new_pcm)
    print(new_pcm.shape)
    assert new_pcm.shape[0] == pcm.shape[0]
    assert new_pcm.shape[1] < pcm.shape[1]  # we expect fewer columns
    assert new_weights.shape == (new_pcm.shape[1],)

    # Test that the new weights are not all uniform.
    assert not np.allclose(new_weights, new_weights[0])


def test_version():
    decoder = qec.get_decoder('example_byod', H)
    assert "CUDA-Q QEC Base Decoder" in decoder.get_version()


def test_single_error_lut_opt_results():
    # Test with invalid opt_results
    invalid_args = {"opt_results": {"invalid_type": True}}
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder("single_error_lut", H, **invalid_args)
        decoder.decode(create_test_syndrome())
    assert "Requested result types not available" in str(e.value)

    # Test with valid opt_results
    valid_args = {
        "opt_results": {
            "error_probability": True,
            "syndrome_weight": True,
            "decoding_time": False
        }
    }
    decoder = qec.get_decoder("single_error_lut", H, **valid_args)
    result = decoder.decode(create_test_syndrome())

    # Verify opt_results
    assert result.opt_results is not None
    assert "error_probability" in result.opt_results
    assert "syndrome_weight" in result.opt_results
    assert "decoding_time" not in result.opt_results  # Was set to False

    batch_result = decoder.decode_batch(
        [create_test_syndrome(), create_test_syndrome()])
    assert isinstance(batch_result.result, np.ndarray)
    assert batch_result.result.shape == (2, H.shape[1])
    assert isinstance(batch_result.converged, np.ndarray)
    assert batch_result.converged.shape == (2,)
    assert len(batch_result.opt_results) == 2
    for opt_results in batch_result.opt_results:
        assert opt_results is not None
        assert "error_probability" in opt_results
        assert "syndrome_weight" in opt_results
        assert "decoding_time" not in opt_results


def test_decoder_pymatching_results():
    pcm = qec.generate_random_pcm(n_rounds=2,
                                  n_errs_per_round=10,
                                  n_syndromes_per_round=5,
                                  weight=2,
                                  seed=7)
    pcm, _ = qec.simplify_pcm(pcm, np.ones(pcm.shape[1]), 10)
    # Pick 3 random columns from the PCM and XOR them together to get the
    # syndrome.
    columns = np.random.choice(pcm.shape[1], 3, replace=False)
    syndrome = np.sum(pcm[:, columns], axis=1) % 2
    decoder = qec.get_decoder('pymatching', pcm)
    result = decoder.decode(syndrome)
    assert result.converged is True
    assert isinstance(result.result, np.ndarray)
    assert np.all((0 <= result.result) & (result.result <= 1))
    actual_errors = np.zeros(pcm.shape[1], dtype=np.uint8)
    actual_errors[columns] = 1
    assert np.array_equal(result.result, actual_errors)


# --- Sparse matrix (sparse_binary_matrix) tests ---


def test_get_decoder_sparse_csc():
    """get_decoder with a scipy CSC matrix produces a working decoder."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H_dense = create_test_matrix()
    decoder = qec.get_decoder("example_byod", scipy_sparse.csc_matrix(H_dense))
    assert decoder is not None
    assert hasattr(decoder, "decode")
    result = decoder.decode(create_test_syndrome())
    assert hasattr(result, "converged")
    assert hasattr(result, "result")
    assert len(result.result) == H_dense.shape[0]


def test_get_decoder_sparse_csr():
    """get_decoder with a scipy CSR matrix produces a working decoder."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H_dense = create_test_matrix()
    decoder = qec.get_decoder("example_byod", scipy_sparse.csr_matrix(H_dense))
    assert decoder is not None
    assert hasattr(decoder, "decode")
    result = decoder.decode(create_test_syndrome())
    assert hasattr(result, "converged")
    assert hasattr(result, "result")
    assert len(result.result) == H_dense.shape[0]


def test_get_decoder_sparse_vs_dense_same_results():
    """Decoder from scipy sparse H behaves like dense H: same shape and validity."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(123)
    H_dense = np.random.randint(0, 2, (8, 16)).astype(np.uint8)
    syndrome = np.random.random(8).tolist()

    dec_dense = qec.get_decoder("example_byod", H_dense)
    dec_sparse = qec.get_decoder("example_byod",
                                 scipy_sparse.csc_matrix(H_dense))

    r_dense = dec_dense.decode(syndrome)
    r_sparse = dec_sparse.decode(syndrome)

    assert r_dense.converged == r_sparse.converged
    assert len(r_dense.result) == len(r_sparse.result)


def test_get_decoder_sparse_pymatching():
    """Native pymatching decoder accepts scipy sparse H and returns valid result."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pcm = qec.generate_random_pcm(
        n_rounds=2,
        n_errs_per_round=10,
        n_syndromes_per_round=5,
        weight=2,
        seed=7,
    )
    pcm, _ = qec.simplify_pcm(pcm, np.ones(pcm.shape[1]), 10)

    columns = np.random.choice(pcm.shape[1], 3, replace=False)
    syndrome = (np.sum(pcm[:, columns], axis=1) % 2).tolist()

    decoder = qec.get_decoder("pymatching", scipy_sparse.csc_matrix(pcm))
    result = decoder.decode(syndrome)
    assert result.converged is True
    assert len(result.result) == pcm.shape[1]
    assert all(isinstance(x, float) for x in result.result)
    assert all(0 <= x <= 1 for x in result.result)


def test_get_decoder_sparse_from_scipy():
    """get_decoder accepts a scipy sparse matrix directly."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(42)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    decoder = qec.get_decoder("example_byod", scipy_sparse.csr_matrix(H_dense))
    assert decoder is not None
    syndrome = np.random.random(6).tolist()
    result = decoder.decode(syndrome)
    assert result.converged is True
    assert len(result.result) == H_dense.shape[0]


def test_get_decoder_scipy_csr_direct():
    """get_decoder accepts a scipy CSR matrix directly (no dict conversion needed)."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(0)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    H_csr = scipy_sparse.csr_matrix(H_dense)
    decoder = qec.get_decoder("single_error_lut_example", H_csr)
    assert decoder is not None
    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)
    result = decoder.decode(syndrome)
    assert result is not None
    assert len(result.result) == H_dense.shape[1]


def test_get_decoder_scipy_csc_direct():
    """get_decoder accepts a scipy CSC matrix directly."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(0)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    H_csc = scipy_sparse.csc_matrix(H_dense)
    decoder = qec.get_decoder("single_error_lut_example", H_csc)
    assert decoder is not None
    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)
    result = decoder.decode(syndrome)
    assert result is not None
    assert len(result.result) == H_dense.shape[1]


def test_get_decoder_scipy_coo_direct():
    """get_decoder accepts a scipy COO matrix, which has no indptr/indices."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(0)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    H_coo = scipy_sparse.coo_matrix(H_dense)
    # COO format does not expose indptr; detection must rely on tocsr().
    assert not hasattr(H_coo, "indptr")
    decoder = qec.get_decoder("single_error_lut_example", H_coo)
    assert decoder is not None
    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)
    result = decoder.decode(syndrome)
    assert result is not None
    assert len(result.result) == H_dense.shape[1]


def test_get_decoder_scipy_coo_same_result_as_dense():
    """A decoder built from a scipy COO matrix matches the dense build."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(7)
    H_dense = np.random.randint(0, 2, (8, 16)).astype(np.uint8)
    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)

    dec_coo = qec.get_decoder("single_error_lut_example",
                              scipy_sparse.coo_matrix(H_dense))
    dec_dense = qec.get_decoder("single_error_lut_example", H_dense)

    res_coo = dec_coo.decode(syndrome)
    res_dense = dec_dense.decode(syndrome)
    np.testing.assert_array_equal(res_coo.result, res_dense.result)


def test_get_decoder_scipy_explicit_zeros_dropped():
    """Explicitly-stored zeros in a scipy matrix are not treated as PCM ones."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H_dense = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    # Build a CSR matrix that structurally stores a zero at (0, 1): the data
    # array carries an explicit 0 whose position would otherwise look like a
    # nonzero PCM entry if indptr/indices were copied verbatim.
    data = np.array([1, 0, 1, 1], dtype=np.uint8)
    indices = np.array([0, 1, 2, 1])
    indptr = np.array([0, 3, 4])
    H_explicit = scipy_sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
    assert H_explicit.nnz == 4  # the explicit zero is stored
    np.testing.assert_array_equal(H_explicit.toarray(), H_dense)

    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)
    dec_explicit = qec.get_decoder("single_error_lut_example", H_explicit)
    dec_dense = qec.get_decoder("single_error_lut_example", H_dense)

    res_explicit = dec_explicit.decode(syndrome)
    res_dense = dec_dense.decode(syndrome)
    np.testing.assert_array_equal(res_explicit.result, res_dense.result)


def test_get_decoder_scipy_duplicate_unsorted_indices():
    """Duplicate/unsorted scipy indices are canonicalized to match the dense
    matrix, and the caller's matrix is left unmodified."""
    scipy_sparse = pytest.importorskip("scipy.sparse")

    # Non-canonical CSR built directly from raw arrays (scipy does not sort or
    # merge on construction):
    #   row 0: cols [2, 1, 1] -> unsorted, with a duplicate at col 1
    #   row 1: cols [0]
    # The equivalent dense matrix sums the duplicate, so its nonzero pattern is
    # [[0, 1, 1], [1, 0, 0]].
    data = np.array([1, 1, 1, 1], dtype=np.uint8)
    indices = np.array([2, 1, 1, 0])
    indptr = np.array([0, 3, 4])
    H_noncanon = scipy_sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
    assert not H_noncanon.has_canonical_format

    H_dense = (H_noncanon.toarray() != 0).astype(np.uint8)

    # Snapshot the caller's buffers to confirm get_decoder does not mutate them.
    before = (H_noncanon.data.copy(), H_noncanon.indices.copy(),
              H_noncanon.indptr.copy())

    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)
    dec_noncanon = qec.get_decoder("single_error_lut_example", H_noncanon)
    dec_dense = qec.get_decoder("single_error_lut_example", H_dense)

    res_noncanon = dec_noncanon.decode(syndrome)
    res_dense = dec_dense.decode(syndrome)
    np.testing.assert_array_equal(res_noncanon.result, res_dense.result)

    # The input matrix must be untouched (canonicalization happens on a copy).
    np.testing.assert_array_equal(H_noncanon.data, before[0])
    np.testing.assert_array_equal(H_noncanon.indices, before[1])
    np.testing.assert_array_equal(H_noncanon.indptr, before[2])
    assert not H_noncanon.has_canonical_format


def test_get_decoder_scipy_csr_csc_same_result():
    """Decoders built from scipy CSR and CSC of the same matrix produce identical results."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(7)
    H_dense = np.random.randint(0, 2, (8, 16)).astype(np.uint8)
    syndrome = np.zeros(H_dense.shape[0], dtype=np.uint8)

    dec_csr = qec.get_decoder("single_error_lut_example",
                              scipy_sparse.csr_matrix(H_dense))
    dec_csc = qec.get_decoder("single_error_lut_example",
                              scipy_sparse.csc_matrix(H_dense))
    dec_dense = qec.get_decoder("single_error_lut_example", H_dense)

    res_csr = dec_csr.decode(syndrome)
    res_csc = dec_csc.decode(syndrome)
    res_dense = dec_dense.decode(syndrome)

    np.testing.assert_array_equal(res_csr.result, res_dense.result)
    np.testing.assert_array_equal(res_csc.result, res_dense.result)


def test_get_decoder_scipy_int64_indices():
    """get_decoder handles scipy matrices whose indptr/indices use int64 dtype."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(0)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    H_csr = scipy_sparse.csr_matrix(H_dense)
    # Force int64 index dtype.
    H_csr = H_csr.astype(H_csr.dtype)
    H_csr.indptr = H_csr.indptr.astype(np.int64)
    H_csr.indices = H_csr.indices.astype(np.int64)
    decoder = qec.get_decoder("single_error_lut_example", H_csr)
    assert decoder is not None
    result = decoder.decode(np.zeros(H_dense.shape[0], dtype=np.uint8))
    assert len(result.result) == H_dense.shape[1]


def test_get_decoder_scipy_python_registered_decoder():
    """Python BYOD factories receive the scipy object unchanged; the decoder
    can call qec.Decoder.__init__(self, H) with a scipy sparse matrix."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(0)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    H_scipy = scipy_sparse.csr_matrix(H_dense)

    @qec.decoder("_test_scipy_byod")
    class _ScipyByod:

        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H

        def decode(self, syndrome):
            res = qec.DecoderResult()
            res.converged = True
            res.result = [0.0] * self.H.shape[1]
            res.opt_results = None
            return res

    decoder = qec.get_decoder("_test_scipy_byod", H_scipy)
    assert decoder is not None
    assert scipy_sparse.issparse(decoder.H)
    result = decoder.decode(np.zeros(H_dense.shape[0]))
    assert len(result.result) == H_dense.shape[1]


def test_get_decoder_sparse_python_registered_decoder():
    """Python BYOD factories receive the scipy object unchanged."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H_dense = create_test_matrix()
    H_scipy = scipy_sparse.csc_matrix(H_dense)
    decoder = qec.get_decoder("example_byod", H_scipy)
    assert hasattr(decoder, "H")
    assert scipy_sparse.issparse(decoder.H)
    result = decoder.decode(create_test_syndrome())
    assert result.converged is True
    assert len(result.result) == H_dense.shape[0]


def test_generate_random_pcm_signed_weight_rejects_negative():
    """`weight=-1` surfaces the C++ guard, not nanobind's marshalling error."""
    with pytest.raises((ValueError, RuntimeError), match="weight"):
        qec.generate_random_pcm(n_rounds=2,
                                n_errs_per_round=3,
                                n_syndromes_per_round=4,
                                weight=-1,
                                seed=1)


def test_get_decoder_accepts_stim_dem_string():
    dem_text = ("error(0.1) D0 L0\n"
                "error(0.1) D1 L0\n"
                "error(0.05) D0 D1\n")

    decoder = qec.get_decoder("single_error_lut", dem_text)
    assert decoder is not None
    assert decoder.get_syndrome_size() == 2
    assert decoder.get_block_size() == 3

    cases = [
        ([0.0, 0.0], [0.0, 0.0, 0.0]),
        ([1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0, 0.0]),
        ([1.0, 1.0], [0.0, 0.0, 1.0]),
    ]
    for syndrome, expected in cases:
        result = decoder.decode(syndrome)
        assert result.converged is True, f"syndrome {syndrome}"
        assert list(result.result) == expected, f"syndrome {syndrome}"


def test_dem_from_stim_text_explicit_parse_then_get_decoder():
    dem_text = ("error(0.1) D0 L0\n"
                "error(0.1) D1 L0\n"
                "error(0.05) D0 D1\n")

    dem = qec.dem_from_stim_text(dem_text)
    assert isinstance(dem, qec.DetectorErrorModel)
    assert dem.num_detectors() == 2
    assert dem.num_error_mechanisms() == 3
    assert dem.num_observables() == 1
    assert dem.detector_error_matrix.shape == (2, 3)

    decoder = qec.get_decoder("single_error_lut", dem.detector_error_matrix)
    assert decoder.get_syndrome_size() == 2
    assert decoder.get_block_size() == 3


def test_dem_from_stim_text_use_decomp_suggestions():
    dem_text = ("error(0.05) D0 D1 L0\n"
                "error(0.03) D2 L1\n"
                "error(0.1) D0 D2 ^ D1 D3\n")

    # ── use_decomp_suggestions=False
    dem_no = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=False)
    assert dem_no.num_detectors() == 4
    assert dem_no.num_observables() == 2
    assert dem_no.num_error_mechanisms() == 3

    # Also confirm that the default matches explicit False.
    assert qec.dem_from_stim_text(dem_text).num_error_mechanisms() == 3

    explicit_H_no = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]],
                             dtype=np.uint8)
    explicit_O_no = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(
        np.array(dem_no.detector_error_matrix, dtype=np.uint8), explicit_H_no)
    np.testing.assert_array_equal(
        np.array(dem_no.observables_flips_matrix, dtype=np.uint8),
        explicit_O_no)
    np.testing.assert_allclose(dem_no.error_rates, [0.05, 0.03, 0.1],
                               atol=1e-12)

    # ── use_decomp_suggestions=True
    dem_yes = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=True)
    assert dem_yes.num_detectors() == 4
    assert dem_yes.num_observables() == 2
    assert dem_yes.num_error_mechanisms() == 4  # instruction 3 splits into 2

    explicit_H_yes = np.array(
        [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]],
        dtype=np.uint8)
    explicit_O_yes = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(
        np.array(dem_yes.detector_error_matrix, dtype=np.uint8), explicit_H_yes)
    np.testing.assert_array_equal(
        np.array(dem_yes.observables_flips_matrix, dtype=np.uint8),
        explicit_O_yes)
    # Each decomposed component inherits the parent instruction's probability.
    np.testing.assert_allclose(dem_yes.error_rates, [0.05, 0.03, 0.1, 0.1],
                               atol=1e-12)


def test_dem_from_stim_text_use_decomp_suggestions_edge_cases():
    A = lambda d: np.array(d, dtype=np.uint8)

    # 1. No '^' in DEM — use_decomp_suggestions=True must be a no-op.
    dem_text = "error(0.1) D0 D1 L0\nerror(0.2) D1 D2\n"
    no = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=False)
    yes = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=True)
    np.testing.assert_array_equal(A(no.detector_error_matrix),
                                  A(yes.detector_error_matrix))
    np.testing.assert_array_equal(A(no.observables_flips_matrix),
                                  A(yes.observables_flips_matrix))
    np.testing.assert_allclose(no.error_rates, yes.error_rates, atol=1e-12)

    # 2. Observable flips split across components — each L stays with its '^' segment.
    # error(0.1) D0 L0 ^ D1 L1  →  col0: D0+L0, col1: D1+L1
    dem_text = "error(0.1) D0 L0 ^ D1 L1\n"
    dem = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=True)
    assert dem.num_error_mechanisms() == 2
    np.testing.assert_array_equal(A(dem.detector_error_matrix),
                                  A([[1, 0], [0, 1]]))
    np.testing.assert_array_equal(A(dem.observables_flips_matrix),
                                  A([[1, 0], [0, 1]]))

    # 3. Repeated detector within one component XOR-cancels to 0.
    # error(0.1) D0 D0 ^ D1  →  component 0 has no net detectors after
    # cancellation so it is dropped entirely; only D1 remains as one column.
    dem_text = "error(0.1) D0 D0 ^ D1\n"
    dem = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=True)
    assert dem.num_error_mechanisms() == 1
    np.testing.assert_array_equal(A(dem.detector_error_matrix), A([[0], [1]]))


def test_get_decoder_rejects_malformed_stim_dem_text():
    with pytest.raises(RuntimeError):
        qec.get_decoder("single_error_lut", "not a valid DEM")


def test_get_decoder_rejects_unknown_decoder_for_stim_dem_text():
    with pytest.raises(RuntimeError, match="__no_such_decoder__"):
        qec.get_decoder("__no_such_decoder__", "error(0.1) D0 L0\n")


def test_decoder_cuda_device_id_invalid_raises():
    H = create_test_matrix()
    # A negative id is rejected before reaching the C++ guard: the kwargs
    # marshalling layer stores Python ints as size_t, so nanobind refuses the
    # negative value with a bare RuntimeError ("std::bad_cast").
    with pytest.raises(RuntimeError):
        qec.get_decoder("single_error_lut", H, cuda_device_id=-2)
    # An out-of-range id flows through kwargs to decoder::get(), which raises
    # a runtime_error naming the offending parameter.
    with pytest.raises(RuntimeError, match="cuda_device_id"):
        qec.get_decoder("single_error_lut", H, cuda_device_id=1 << 20)


def test_decoder_cuda_device_id_valid():
    H = create_test_matrix()
    try:
        d = qec.get_decoder("single_error_lut", H, cuda_device_id=0)
    except RuntimeError as e:
        if "out of range" in str(e):
            pytest.skip("no CUDA device visible")
        raise
    syndrome = create_test_syndrome()
    result = d.decode(syndrome)
    assert len(result.result) == H.shape[1]


def test_get_decoder_user_O_wins_over_dem_derived():
    dem_text = ("error(0.1) D0 L0\n"
                "error(0.1) D1 L0\n"
                "error(0.05) D0 D1\n")
    bad_O = np.zeros((1, 4), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        qec.get_decoder("pymatching", dem_text, O=bad_O)


def test_get_decoder_stim_dem_without_observables_returns_errors():
    decoder = qec.get_decoder("pymatching", "error(0.1) D0\n")

    result = decoder.decode([1.0])
    assert result.converged is True
    assert list(result.result) == [1.0]


if __name__ == "__main__":
    pytest.main()

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
"""Post-processing for the nv-qldpc-decoder ``relay_solutions`` records.

When the decoder is configured with ``opt_results={"relay_solutions": True}``
(sequential relay, ``composition=1``), every relay convergence is recorded and
returned through ``BatchDecoderResult.batch_opt_results`` as flat arrays. This
module reconstructs, offline, what the decoder would have returned for any
``stop_nconv`` setting, so a single recording run answers the whole
RelayBP-N sweep.

Public API:
    unpack(batch_opt_results, num_shots) -> RelaySolutionRecords
    stop_nconv_sweep(results, obs_truth, percentiles=..., n_values=None, observables=None) -> StopNConvSweep

Requirements on the recording run (documented, not detectable from the data):
the sweep is only meaningful if the run did not itself stop early, i.e.
``srelay_config={"stopping_criterion": "All"}`` and ``relay_solutions=True``
(uncapped). A capped run (``relay_solutions=<int>``) is detected and rejected
for any N beyond the cap.
"""

from dataclasses import dataclass

import numpy as np

_WORD_BYTES = 4  # records are packed into 32-bit words, little-endian bits


def _words_per_record(width):
    return (int(width) + 31) // 32


def _unpack_words(words, width):
    """(..., W) int32 words -> (..., width) 0/1 uint8, little-endian bits."""
    words = np.ascontiguousarray(np.asarray(words, dtype=np.int32))
    as_bytes = words.view(np.uint8).reshape(words.shape[:-1] +
                                            (words.shape[-1] * _WORD_BYTES,))
    return np.unpackbits(as_bytes, axis=-1, bitorder="little")[..., :width]


def _pack_words(bits, width):
    """(..., width) 0/1 -> (..., W) int32 words, little-endian bits."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != width:
        raise ValueError(f"expected {width} bits per row, got {bits.shape[-1]}")
    as_bytes = np.packbits(bits, axis=-1, bitorder="little")
    pad = _words_per_record(width) * _WORD_BYTES - as_bytes.shape[-1]
    if pad:
        pad_widths = [(0, 0)] * (as_bytes.ndim - 1) + [(0, pad)]
        as_bytes = np.pad(as_bytes, pad_widths)
    return np.ascontiguousarray(as_bytes).view(np.int32)


@dataclass(frozen=True)
class RelaySolutionRecords:
    """The ``relay_solutions_*`` arrays with the record axes reconstructed.

    Rows beyond a shot's own record count are padding: iteration count -1,
    weight +inf, packed bits all zero.
    """
    width: int  #: bits per record (block_size, or num_obs when O was given)
    max_records: int  #: the record axis (R) of the padded arrays
    counts: np.ndarray  #: (num_shots,) records stored per shot
    totals: np.ndarray  #: (num_shots,) convergences seen per shot (uncapped)
    iters: np.ndarray  #: (num_shots, R) cumulative iterations at each record
    weight: np.ndarray  #: (num_shots, R) LLR weight of each record
    packed: np.ndarray  #: (num_shots, R, W) int32 words, 32 bits per word

    def bits(self, shot, record):
        """One record's hard decision as a 0/1 vector of length ``width``."""
        return _unpack_words(self.packed[shot, record], self.width)


def unpack(batch_opt_results, num_shots):
    """Reshape the flat ``relay_solutions_*`` arrays.

    Accepts either the batched key set (``relay_solutions_counts``/``totals``,
    from decode_batch) or the single-shot one (``relay_solutions_count``/
    ``total``, from decode, with ``num_shots == 1``).
    """
    m = batch_opt_results
    if m is None:
        raise ValueError("no relay_solutions records: run the decoder with "
                         'opt_results={"relay_solutions": True} and read '
                         "BatchDecoderResult.batch_opt_results")
    width = int(m["relay_solutions_width"])
    R = int(m["relay_solutions_max_records"])
    W = _words_per_record(width)

    if "relay_solutions_counts" in m:
        counts = np.asarray(m["relay_solutions_counts"])
        totals = np.asarray(m["relay_solutions_totals"])
    else:
        if num_shots != 1:
            raise ValueError(
                f"single-shot relay_solutions keys but num_shots={num_shots}")
        counts = np.asarray([m["relay_solutions_count"]])
        totals = np.asarray([m["relay_solutions_total"]])
    if counts.shape != (num_shots,) or totals.shape != (num_shots,):
        raise ValueError(
            f"relay_solutions counts/totals have shape {counts.shape}/"
            f"{totals.shape}, expected ({num_shots},)")

    def reshape(key, shape):
        arr = np.asarray(m[key])
        expected = int(np.prod(shape))
        if arr.shape != (expected,):
            raise ValueError(
                f"{key} has shape {arr.shape}, expected ({expected},)")
        return arr.reshape(shape)

    return RelaySolutionRecords(
        width=width,
        max_records=R,
        counts=counts,
        totals=totals,
        iters=reshape("relay_solutions_iters", (num_shots, R)),
        weight=reshape("relay_solutions_weight", (num_shots, R)),
        packed=reshape("relay_solutions_result",
                       (num_shots, R, W)).astype(np.int32),
    )


@dataclass(frozen=True)
class StopNConvSweep:
    """Per-N results of a ``stop_nconv`` sweep. Arrays are indexed by ``n``."""
    n: np.ndarray  #: (n_N,) the stop_nconv values swept
    ler: np.ndarray  #: (n_N,) logical error rate
    num_errors: np.ndarray  #: (n_N,) raw logical error counts
    avg_iters: np.ndarray  #: (n_N,) mean iterations run
    #: iteration percentiles; (n_N,) for a scalar ``percentiles`` input,
    #: (n_p, n_N) for a list (np.percentile convention)
    iters_percentiles: np.ndarray
    percentiles: np.ndarray  #: the percentile values requested
    num_shots: int
    num_unconverged: int  #: shots with zero recorded convergences
    frac_exhausted: np.ndarray  #: (n_N,) shots with fewer than N convergences


def _class_words(bits, observables, width, k):
    """Hard decisions -> packed observable-class words.

    ``bits`` is (..., width). With ``observables`` (k, width) the class is
    ``observables @ bits % 2``; without, the bits already are the class.
    """
    if observables is None:
        return _pack_words(bits, k)
    cls = (bits.reshape(-1, width).astype(np.int64) @ observables.T) % 2
    return _pack_words(cls.astype(np.uint8), k).reshape(bits.shape[:-1] + (-1,))


def stop_nconv_sweep(results,
                     obs_truth,
                     *,
                     percentiles,
                     n_values=None,
                     observables=None):
    """Reconstruct LER and iteration statistics for a range of ``stop_nconv``.

    For each N, each shot's prediction is the minimum-weight record among its
    first ``min(N, count)`` convergences (weight ties resolve to the earliest
    record), and its iteration count is the cumulative count at the Nth
    convergence -- or the shot's full-schedule ``num_iter`` when it produced
    fewer than N convergences, exactly what stop_nconv=N would have run.
    Shots that never converged are scored with the decoder's returned
    (fallback) result, which is identical for every N.

    Args:
        results: The ``BatchDecoderResult`` of a recording run. Beyond the
            records themselves, per-shot ``num_iter`` is read from
            ``results.opt_results`` when any shot exhausts the schedule
            within the sweep range, and ``results.result`` supplies the
            fallback prediction for never-converged shots.
        obs_truth: (num_shots, k) 0/1 array of actual observable flips.
        percentiles: Iteration percentile(s) to report, in (0, 100]. A scalar
            yields ``iters_percentiles`` of shape (n_N,); a list yields
            (n_p, n_N).
        n_values: Iterable of stop_nconv values to sweep; defaults to
            ``1..max_records``.
        observables: (k, width) 0/1 observables matrix. Required when the
            decoder ran without one (records are corrections); must be
            omitted when it ran with one (records already are observable
            classes).

    Returns:
        A :class:`StopNConvSweep`.
    """
    batch = getattr(results, "batch_opt_results", None)
    obs_truth = np.asarray(obs_truth)
    if obs_truth.ndim != 2:
        raise ValueError(f"obs_truth must be 2-D (num_shots, k), got shape "
                         f"{obs_truth.shape}")
    num_shots, k = obs_truth.shape
    recs = unpack(batch, num_shots)

    if observables is not None:
        observables = (np.asarray(observables) % 2).astype(np.int64)
        if observables.shape != (k, recs.width):
            raise ValueError(
                f"observables has shape {observables.shape}, expected "
                f"({k}, {recs.width}) for {k} observables over records of "
                f"width {recs.width}")
    elif recs.width != k:
        raise ValueError(
            f"records have width {recs.width} but obs_truth has {k} "
            "observables: the decoder ran without an observables matrix, so "
            "one must be passed as `observables` to compute the LER")

    n_values = (np.arange(1, recs.max_records + 1) if n_values is None else
                np.asarray(list(n_values), dtype=np.int64))
    if n_values.size == 0 or np.any(n_values < 1):
        raise ValueError(f"n_values must be positive integers, got {n_values}")
    n_max = int(n_values.max())

    percentiles_arr = np.atleast_1d(np.asarray(percentiles, dtype=np.float64))
    if np.any((percentiles_arr <= 0) | (percentiles_arr > 100)):
        raise ValueError(f"percentiles must be in (0, 100], got {percentiles}")

    # A capped recording (relay_solutions=<int>) dropped records; the sweep is
    # only faithful up to the cap.
    truncated = recs.totals > recs.counts
    if truncated.any():
        cap = int(recs.counts[truncated].min())
        if n_max > cap:
            raise ValueError(
                f"records were capped at {cap} but the sweep requests "
                f"N={n_max}: rerun with relay_solutions=True (or a cap >= N)")

    # Per-shot inputs beyond the records themselves.
    unconverged = recs.counts == 0
    exhausted_somewhere = bool((recs.counts < n_max).any())
    num_iter = None
    if exhausted_somewhere:
        opt_results = list(results.opt_results)
        num_iter = np.empty(num_shots, dtype=np.int64)
        for i, opt in enumerate(opt_results):
            if opt is None or "num_iter" not in opt:
                raise ValueError(
                    "some shots exhaust the relay schedule within the sweep "
                    "range, so per-shot num_iter is needed: rerun with "
                    'opt_results={"relay_solutions": True, "num_iter": True}')
            num_iter[i] = int(opt["num_iter"])

    truth_words = _pack_words((obs_truth % 2).astype(np.uint8), k)

    # Observable class of every record, packed to words so a prediction
    # comparison is a few int32 equalities. Chunk over shots to bound the
    # unpacked-bits intermediate in correction mode (~width bytes per record).
    R, W = recs.max_records, recs.packed.shape[2]
    class_words = np.empty((num_shots, R, truth_words.shape[1]), np.int32)
    chunk = max(1, (1 << 28) // max(1, R * recs.width))
    for s in range(0, num_shots, chunk):
        p = recs.packed[s:s + chunk]
        class_words[s:s + chunk] = _class_words(_unpack_words(p, recs.width),
                                                observables, recs.width, k)

    # Never-converged shots return the fallback hard decision, which is not
    # among the records; score it from the decoder's returned result.
    fallback_err = None
    if unconverged.any():
        fb = np.asarray(results.result)[unconverged] > 0.5
        if fb.shape[1] != recs.width:
            raise ValueError(
                f"results.result rows have length {fb.shape[1]}, expected "
                f"{recs.width} to score never-converged shots")
        fb_words = _class_words(fb.astype(np.uint8), observables, recs.width, k)
        fallback_err = np.any(fb_words != truth_words[unconverged], axis=-1)

    # One pass over the record axis maintains the prefix argmin (padding is
    # +inf, so exhausted shots keep their own minimum); snapshots are taken at
    # the requested N.
    n_sorted = np.sort(np.unique(n_values))
    best_idx_at = {}
    best_idx = np.zeros(num_shots, dtype=np.int64)
    next_snap = 0
    if R > 0:
        best_w = recs.weight[:, 0].copy()
        for r in range(min(n_max, R)):
            if r > 0:
                better = recs.weight[:, r] < best_w
                best_w[better] = recs.weight[better, r]
                best_idx[better] = r
            while next_snap < n_sorted.size and n_sorted[next_snap] == r + 1:
                best_idx_at[r + 1] = best_idx.copy()
                next_snap += 1
    for n in n_sorted[next_snap:]:  # N > R: identical to the full-record min
        best_idx_at[int(n)] = best_idx

    shot_rows = np.arange(num_shots)
    num_errors = np.empty(n_values.size, dtype=np.int64)
    avg_iters = np.empty(n_values.size, dtype=np.float64)
    iters_pcts = np.empty((percentiles_arr.size, n_values.size),
                          dtype=np.float64)
    frac_exhausted = np.empty(n_values.size, dtype=np.float64)
    for j, n in enumerate(n_values):
        if R > 0:
            sel = class_words[shot_rows, best_idx_at[int(n)]]
            err = np.any(sel != truth_words, axis=-1)
        else:
            err = np.zeros(num_shots, dtype=bool)
        if fallback_err is not None:
            err[unconverged] = fallback_err
        num_errors[j] = int(err.sum())

        reached = recs.counts >= n
        it = (np.where(reached, recs.iters[:, min(int(n), R) - 1], 0)
              if R > 0 else np.zeros(num_shots, dtype=np.int64))
        if not reached.all():
            it = np.where(reached, it, num_iter)
        avg_iters[j] = it.mean()
        iters_pcts[:, j] = np.percentile(it, percentiles_arr)
        frac_exhausted[j] = 1.0 - reached.mean()

    if np.isscalar(percentiles) or np.ndim(percentiles) == 0:
        iters_pcts = iters_pcts[0]
    return StopNConvSweep(
        n=n_values,
        ler=num_errors / num_shots,
        num_errors=num_errors,
        avg_iters=avg_iters,
        iters_percentiles=iters_pcts,
        percentiles=percentiles_arr,
        num_shots=num_shots,
        num_unconverged=int(unconverged.sum()),
        frac_exhausted=frac_exhausted,
    )

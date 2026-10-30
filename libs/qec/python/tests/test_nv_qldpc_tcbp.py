# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory-Calibrated BP (TCBP) in the nv-qldpc-decoder.

Fits the TCBP model on a small rotated surface code circuit (Stim file fetched from the
public tesseract-decoder repository, local fallback) and checks the decoder's
``tcbp_model`` / ``tcbp_strict`` / ``osd_init_method="tcbp"`` options:
bytes and path forms, every error path, strict mode, decode() vs decode_batch()
agreement, repeatability, converged-shot semantics and OSD ordering validity.
"""

import os
import urllib.parse
import urllib.request

import numpy as np
import pytest

stim = pytest.importorskip("stim")
qec = pytest.importorskip("cudaq_qec")
nv_qldpc = pytest.importorskip("cudaq_qec.plugins.nv_qldpc")
try:
    nv_qldpc.inspect(b"")  # loads the plugin library; empty bytes are rejected
except ValueError:
    pass  # library loaded
except (FileNotFoundError, OSError) as e:
    pytest.skip(f"nv-qldpc-decoder plugin library not available: {e}",
                allow_module_level=True)
from cudaq_qec.plugins.nv_qldpc import tying  # noqa: E402

CIRCUITS = {
    "d3": "r=3,d=3,p=0.002,noise=si1000,c=surface_code_Z,q=17,gates=cz.stim",
    "d5": "r=5,d=5,p=0.002,noise=si1000,c=surface_code_Z,q=49,gates=cz.stim",
}
RAW = "https://raw.githubusercontent.com/quantumlib/tesseract-decoder/main/testdata/surfacecodes/"
LOCAL = "/workspaces/tesseract-decoder/testdata/surfacecodes/"

BP = dict(use_sparsity=True,
          max_iterations=10,
          clip_value=15.0,
          bp_method=0,
          proc_float="fp32",
          repeatable=True)


def fetch_circuit(name):
    """Stim circuit text: GitHub raw first, local checkout as fallback."""
    url = RAW + urllib.parse.quote(name)
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return stim.Circuit(r.read().decode())
    except Exception as e:  # offline CI: local copy
        local = os.path.join(LOCAL, name)
        if os.path.exists(local):
            return stim.Circuit.from_file(local)
        pytest.skip(f"cannot fetch {name} ({e}) and no local copy at {local}")


def dem_sample(H, rates, n, seed):
    rng = np.random.default_rng(seed)
    E = (rng.random((n, H.shape[1])) < rates[None, :]).astype(np.uint8)
    S = ((E @ H.T) % 2).astype(np.uint8)
    return E, S


def as_array(data):
    return np.array(bytearray(data), dtype=np.uint8)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = {}
    for key, name in CIRCUITS.items():
        c = fetch_circuit(name)
        dem = c.detector_error_model(flatten_loops=True)
        H, O, rates = tying.dem_arrays(dem)
        out[key] = dict(H=H, O=O, rates=rates)
    H = out["d5"]["H"]
    data = nv_qldpc.init_tcbp_model(H,
                                    seed=1,
                                    steps=1500,
                                    shots=16384,
                                    device_memory_budget_bytes=1 << 30)
    path = tmp_path_factory.mktemp("tcbp") / "d5.tcbp"
    path.write_bytes(data)
    E, S = dem_sample(H, out["d5"]["rates"], 2048, seed=3)
    out["tcbp_model"] = as_array(data)
    out["model_path"] = str(path)
    out["E"], out["S"] = E, S
    out["d3_model"] = as_array(
        nv_qldpc.init_tcbp_model(out["d3"]["H"],
                                 seed=1,
                                 steps=500,
                                 shots=8192,
                                 device_memory_budget_bytes=1 << 30))
    return out


def make(setup, **kw):
    d5 = setup["d5"]
    params = dict(BP, error_rate_vec=d5["rates"])
    params.update(kw)
    return qec.get_decoder("nv-qldpc-decoder", d5["H"], **params)


def predecode(setup, dec):
    res = dec.decode_batch(setup["S"])
    return np.asarray(res.result), np.asarray(res.converged)


# --------------------------------------------------------------------------- #
# predecoder semantics
# --------------------------------------------------------------------------- #


def test_predecoder_probabilities(setup):
    dec = make(setup,
               tcbp_model=setup["tcbp_model"],
               use_osd=False,
               bp_batch_size=2048)
    P, conv = predecode(setup, dec)
    E = setup["E"].astype(bool)
    assert P.shape == setup["E"].shape
    assert np.all((P >= 0) & (P <= 1))
    assert conv.sum() > 0 and (~conv).sum() > 0
    nc = ~conv
    # calibrated P(error): large on the true error columns, small elsewhere
    assert P[nc][E[nc]].mean() > 0.3
    assert P[nc][~E[nc]].mean() < 0.02
    # converged shots: BP's posterior probabilities (soft, not hard 0/1),
    # converged stays true
    assert conv.dtype == bool
    pc = P[conv]
    assert np.all((pc >= 0) & (pc <= 1))
    assert len(
        np.unique(pc)) > 10, "converged shots must return soft posteriors"
    assert pc[E[conv]].mean() > 0.5 and pc[~E[conv]].mean() < 0.01


def test_converged_shots_are_bp_posteriors(setup):
    dec = make(setup,
               tcbp_model=setup["tcbp_model"],
               use_osd=False,
               bp_batch_size=2048)
    P, conv = predecode(setup, dec)
    # BP's final marginal LLRs from the plain decoder (last iteration of the
    # LLR history, single-shot path; fp64 because the Python result caster
    # does not carry fp32 histories), P(error) = logistic(-LLR)
    ref = make(setup,
               use_osd=False,
               bp_batch_size=1,
               proc_float="fp64",
               opt_results={
                   "bp_llr_history": 10,
                   "num_iter": True
               })
    checked = 0
    for i in np.where(conv)[0][:48]:
        r = ref.decode(setup["S"][i])
        if not bool(r.converged):
            continue  # fp64 BP may converge differently on a rare shot
        hist = np.asarray(r.opt_results["bp_llr_history"], dtype=np.float64)
        p_bp = 1.0 / (1.0 + np.exp(np.clip(hist[-1], -60, 60)))
        assert np.allclose(P[i], p_bp, atol=5e-3), i
        checked += 1
    assert checked >= 16


def test_path_equals_bytes(setup):
    a = make(setup,
             tcbp_model=setup["tcbp_model"],
             use_osd=False,
             bp_batch_size=512)
    b = make(setup,
             tcbp_model=setup["model_path"],
             use_osd=False,
             bp_batch_size=512)
    Pa, _ = predecode(setup, a)
    Pb, _ = predecode(setup, b)
    assert np.array_equal(Pa, Pb)


def test_repeatable(setup):
    dec = make(setup,
               tcbp_model=setup["tcbp_model"],
               use_osd=False,
               bp_batch_size=1024)
    P1, c1 = predecode(setup, dec)
    P2, c2 = predecode(setup, dec)
    assert np.array_equal(P1, P2) and np.array_equal(c1, c2)


def test_decode_matches_decode_batch(setup):
    batch = make(setup,
                 tcbp_model=setup["tcbp_model"],
                 use_osd=False,
                 bp_batch_size=2048)
    single = make(setup,
                  tcbp_model=setup["tcbp_model"],
                  use_osd=False,
                  bp_batch_size=1)
    P, conv = predecode(setup, batch)
    idx = list(np.where(~conv)[0][:12]) + list(np.where(conv)[0][:4])
    for i in idx:
        r = single.decode(setup["S"][i])
        p = np.asarray(r.result)
        assert bool(r.converged) == bool(conv[i])
        # same algorithm on both paths: same hard decisions, close probabilities
        assert np.array_equal(p > 0.5, P[i] > 0.5)
        assert np.abs(p - P[i]).max() < 2e-2


# --------------------------------------------------------------------------- #
# OSD ordering
# --------------------------------------------------------------------------- #


def test_osd_tcbp_ordering(setup):
    H, O, E, S = setup["d5"]["H"], setup["d5"]["O"], setup["E"], setup["S"]
    outs = {}
    for method in ("final_llr", "tcbp"):
        kw = dict(use_osd=True,
                  osd_method=1,
                  osd_init_method=method,
                  bp_batch_size=2048,
                  osd_batch_size=16)
        if method == "tcbp":
            kw["tcbp_model"] = setup["tcbp_model"]
        dec = make(setup, **kw)
        res = dec.decode_batch(S)
        X = (np.asarray(res.result) > 0.5).astype(np.uint8)
        assert np.array_equal((X @ H.T) % 2, S), f"{method}: invalid correction"
        outs[method] = (X, ((((X ^ E) @ O.T) % 2) != 0).any(1).sum())
    # the ordering is in effect: some non-converged shot decodes differently
    assert not np.array_equal(outs["final_llr"][0], outs["tcbp"][0])
    # and does not make things worse by a wide margin on this small sample
    assert outs["tcbp"][1] <= 1.5 * outs["final_llr"][1] + 5


# --------------------------------------------------------------------------- #
# errors, strict mode, removed keys
# --------------------------------------------------------------------------- #


def expect_error(setup, fragment, **kw):
    with pytest.raises(Exception) as ei:
        make(setup, **kw)
    assert fragment in str(ei.value), str(ei.value)


def test_errors(setup):
    model = setup["tcbp_model"]
    expect_error(setup, "unset tcbp_model", tcbp_model=model, use_osd=True)
    expect_error(setup,
                 "unset tcbp_model",
                 tcbp_model=model,
                 use_osd=True,
                 osd_init_method="min_llr")
    expect_error(setup,
                 'osd_init_method="tcbp" requires tcbp_model',
                 use_osd=True,
                 osd_init_method="tcbp")
    expect_error(setup,
                 'osd_init_method="tcbp" requires use_osd=true',
                 tcbp_model=model,
                 osd_init_method="tcbp")
    expect_error(setup, "tcbp_strict requires tcbp_model", tcbp_strict=True)
    expect_error(setup,
                 "fitted at max_iterations 10, the decoder has 12",
                 tcbp_model=model,
                 max_iterations=12)
    expect_error(setup,
                 "fitted with proc_float fp32, the decoder has fp64",
                 tcbp_model=model,
                 proc_float="fp64")
    expect_error(setup,
                 "fitted with clip_value",
                 tcbp_model=model,
                 clip_value=10.0)
    expect_error(setup, "fitted with bp_method", tcbp_model=model, bp_method=1)
    expect_error(setup,
                 "requires use_sparsity=true",
                 tcbp_model=model,
                 use_sparsity=False)
    expect_error(setup,
                 "not supported for composition=1",
                 tcbp_model=model,
                 composition=1,
                 bp_method=3,
                 gamma0=0.0,
                 explicit_gammas=[[0.0] * setup["d5"]["H"].shape[1]],
                 srelay_config={
                     "pre_iter": 0,
                     "num_sets": 1,
                     "stopping_criterion": "All"
                 })
    expect_error(setup,
                 "bp_llr_history == max_iterations",
                 tcbp_model=model,
                 opt_results={"bp_llr_history": 3})
    expect_error(setup,
                 "cannot open file",
                 tcbp_model=setup["model_path"] + ".missing")
    expect_error(setup, "must be the TCBP model bytes", tcbp_model=3.5)
    expect_error(setup, "truncated", tcbp_model=model[:100].copy())
    # removed private keys of the development branches are rejected
    for key in ("scoreboard", "scoreboard_min", "output_logit"):
        expect_error(setup, key, **{key: True})


def test_strict_and_transfer(setup, capfd):
    # a TCBP model fitted on d=3 is accepted on d=5 with a one-time notice
    dec = make(setup,
               tcbp_model=setup["d3_model"],
               use_osd=False,
               bp_batch_size=256)
    P, conv = predecode(setup, dec)
    assert np.all((P >= 0) & (P <= 1))
    expect_error(setup,
                 "different detector error model structure",
                 tcbp_model=setup["d3_model"],
                 tcbp_strict=True)


def test_hash_known_answers(setup):
    """The blob's CRC-32 and the training-rate SHA-256 against the standard
    library (the plugin's own implementations are not exported)."""
    import hashlib, struct, zlib
    blob = bytes(setup["tcbp_model"])
    assert blob[:8] == b"NVQLSB01"
    length, crc = struct.unpack_from("<II", blob, 12)
    assert len(blob) == 20 + length
    assert zlib.crc32(blob[20:]) == crc
    H, rates = setup["d3"]["H"], setup["d3"]["rates"]
    data = nv_qldpc.init_tcbp_model(H,
                                    error_rate_vec=rates,
                                    seed=1,
                                    steps=500,
                                    shots=8192,
                                    device_memory_budget_bytes=1 << 30)
    p32 = np.asarray(rates, np.float32)
    want = hashlib.sha256(struct.pack("<Q", p32.size) +
                          p32.tobytes()).hexdigest()
    assert nv_qldpc.inspect(data)["training_rate_fingerprint"] == want


def test_sparse_input_same_model(setup):
    """H may be given as any scipy sparse matrix: the model is byte-identical
    to the dense-input one (CSR input is normalised to the dense constructor's
    CSC layout inside the plugin) and the sparse input is never densified."""
    sp = pytest.importorskip("scipy.sparse")

    class NoDense(sp.csr_matrix):

        def toarray(self, *a, **k):
            raise AssertionError("sparse H was densified")

        todense = toarray

    H = setup["d3"]["H"]
    want = bytes(setup["d3_model"])
    kw = dict(seed=1, steps=500, shots=8192, device_memory_budget_bytes=1 << 30)
    for make in (NoDense, sp.csc_matrix, sp.coo_matrix):
        got = nv_qldpc.init_tcbp_model(make(H), **kw)
        assert got == want, make.__name__
    assert nv_qldpc.inspect(want)["n"] == H.shape[1]


def test_c_abi_rejects_malformed_sparse_matrix():
    """The plugin validates the compressed matrix itself (the Python layer
    canonicalises, so these cases are built directly with ctypes)."""
    import ctypes as C
    lib = nv_qldpc._load()
    H = np.array([[1, 1, 0], [0, 1, 1]], np.uint8)
    rows, cols = np.nonzero(H)
    ptr = np.ascontiguousarray(np.r_[0, np.cumsum(np.bincount(rows))],
                               np.uint32)
    idx = np.ascontiguousarray(cols, np.uint32)

    def call(m, n, nnz, layout, ptr, idx):
        d = nv_qldpc._SparseMatrix(m, n, nnz, layout,
                                   ptr.ctypes.data_as(C.POINTER(C.c_uint32)),
                                   idx.ctypes.data_as(C.POINTER(C.c_uint32)))
        model = C.POINTER(C.c_uint8)()
        model_len = C.c_size_t()
        rc = lib.nvqldpc_init_tcbp_model(C.byref(d), b"{}", C.byref(model),
                                         C.byref(model_len), None)
        assert rc < 0
        return lib.nvqldpc_last_error().decode()

    assert "layout" in call(2, 3, 4, 2, ptr, idx)
    assert "ptr[last]" in call(2, 3, 3, 0, ptr, idx)
    bad_ptr = np.array([0, 3, 2], np.uint32)  # ptr[last] == nnz, but decreasing
    assert "non-decreasing" in call(2, 3, 2, 0, bad_ptr, idx)
    bad_idx = idx.copy()
    bad_idx[0] = 3
    assert "out of range" in call(2, 3, 4, 0, ptr, bad_idx)
    bad_idx = idx.copy()
    bad_idx[0], bad_idx[1] = bad_idx[1], bad_idx[0]
    assert "strictly increasing" in call(2, 3, 4, 0, ptr, bad_idx)

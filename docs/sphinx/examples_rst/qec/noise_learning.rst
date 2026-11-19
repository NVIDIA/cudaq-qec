:tocdepth: 1

.. _noise_learning_example:

Learning Noise and Decoding with Learned Rates
==============================================

Estimate the probabilities of a detector error model's existing error
mechanisms from detector samples, then use those probabilities to decode
fresh shots. This example compares three priors: a uniform starting guess,
the learned rates, and the known rates used to simulate the data. The known
rates are used only for comparison; they are not supplied to the learning
step. See :ref:`qldpc_noise_learning` for the method and its assumptions.

Learning uses a TCBP model and does not require JOBS. This example uses JOBS
in the final decoding step to compare the three priors.

Prerequisites
-------------

Use a CUDA-Q QEC installation with the ``nv-qldpc-decoder`` plugin, a
compatible NVIDIA GPU, NumPy, and Stim. Internet access is needed to download
the bivariate-bicycle memory circuit. The script writes
``bb144_r12_si1000_p0.004.stim`` to the current directory.

Run the three blocks below as one Python script, in order. They use the
``p = 0.004`` circuit, 65,536 learning shots, and 4,096 decoding shots. The
original run took about 13 minutes on one GB200, mostly decoding with the
uniform prior; runtime and memory needs depend on your GPU. The
:ref:`performance study <noise_learning_user_guide>` uses up to 2,097,152
learning shots and reports its full measurement settings.

Step 1: Get the Circuit and Fit the TCBP Model
----------------------------------------------

The circuit is downloaded from the ``bivariatebicyclecodes`` test data of the
`tesseract-decoder <https://github.com/quantumlib/tesseract-decoder>`__ repository. The TCBP model is fitted
once per code family and needs no information about the device.

.. code-block:: python

    import urllib.parse, urllib.request
    import numpy as np, stim
    import cudaq_qec as qec
    from cudaq_qec.plugins import nv_qldpc
    from cudaq_qec.plugins.nv_qldpc import tying

    # si1000 memory circuit: [[144,12,12]] code, 12 rounds, Z basis, p = 0.004
    name = ("r=12,d=12,p=0.004,noise=si1000,c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,"
            "iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim")
    url = ("https://raw.githubusercontent.com/quantumlib/tesseract-decoder/"
           "98c663db388ab095939647a165c87b4d31cdd6b1/testdata/bivariatebicyclecodes/"
           + urllib.parse.quote(name))
    urllib.request.urlretrieve(url, "bb144_r12_si1000_p0.004.stim")

    circuit = stim.Circuit.from_file("bb144_r12_si1000_p0.004.stim")
    H, O, true_rates = tying.dem_arrays(tying.circuit_dem(circuit))   # true_rates: scoring only

    tcbp_model = nv_qldpc.init_tcbp_model(H, seed=1)                  # about 10 s

Step 2: Learn the Rates From Detector Data
------------------------------------------

``learn_noise`` takes the structure ``H``, the recorded detector samples, and the TCBP model, and
returns the learned rates together with a report.

.. code-block:: python

    dets = circuit.compile_detector_sampler(seed=10001).sample(2**16).astype(np.uint8)   # [N shots, 1728]
    learned_rates, report = nv_qldpc.learn_noise(H, dets, tcbp_model, passes=16, tol=0)  # about 2 min
    uniform_rate = report["start_rate"]                    # the uniform prior used below

``passes`` is the maximum number of passes and ``tol`` is an early-stop threshold. Without them the
defaults apply (``passes=4``, ``tol=0.1``): learning runs at most 4 passes and stops earlier if a pass
changes the rates by less than ``tol``. ``tol=0`` disables the early stop so that exactly ``passes``
passes run.

Step 3: Decode With Each Prior
------------------------------

Fresh detector shots are sampled together with their observable flips. The same shots are decoded
three times, changing only ``error_rate_vec``, and each prediction is compared with the sampled
observables:

.. code-block:: python

    syndromes, observables = circuit.compile_detector_sampler(seed=778).sample(
        4096, separate_observables=True)
    syndromes, observables = syndromes.astype(np.uint8), observables.astype(np.uint8)

    tcbp = np.array(bytearray(tcbp_model), dtype=np.uint8)
    bp = dict(use_sparsity=True, bp_method=0, max_iterations=10, clip_value=15.0,
              proc_float="fp32", use_osd=True, osd_method=4, osd_init_method="tcbp",
              tcbp_model=tcbp, bp_batch_size=4096)
    priors = {"uniform": np.full(H.shape[1], uniform_rate),
              "learned": learned_rates,
              "oracle":  true_rates}

    for name, rates in priors.items():
        decoder = qec.get_decoder("nv-qldpc-decoder", H,
                                  error_rate_vec=np.clip(rates, 1e-9, 0.45), **bp)
        results = decoder.decode_batch(syndromes)
        errors = (np.asarray(results.result) > 0.5).astype(np.uint8)   # [shots, mechanisms]
        predicted = (errors @ O.T.astype(np.int32)) % 2                # predicted observable flips
        failures = int((predicted != observables).any(axis=1).sum())
        print(f"{name:8s} LER = {failures / len(observables):.3e} "
              f"({failures} failures in {len(observables)} shots)")

Inspect the Results
-------------------

Each line reports the fraction of held-out shots with at least one incorrect
observable prediction. The original GB200 run produced:

.. code-block:: text

    uniform  LER = 5.710e-01 (2339 failures in 4096 shots)
    learned  LER = 6.152e-02 (252 failures in 4096 shots)
    oracle   LER = 5.640e-02 (231 failures in 4096 shots)

The decoders above are built without ``repeatable=True``, so the counts can differ by a few failures
from run to run (floating-point non-determinism).


Next Steps
----------

* Replace the simulated learning samples with recorded detector data in the
  same column order. Keep a separate set of shots to evaluate decoding.
* See :ref:`qldpc_noise_learning` for the learning modes and their assumptions.
* See :ref:`noise_learning_user_guide` for learning time, rate accuracy, and
  logical error rates across larger learning samples.
* See :doc:`Decoding with TCBP and JOBS <tcbp_jobs>` to request observable
  output and complementary-gap information directly from the decoder.

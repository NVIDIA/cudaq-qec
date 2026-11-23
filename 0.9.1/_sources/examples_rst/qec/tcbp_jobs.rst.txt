:tocdepth: 1

.. _tcbp_jobs_example:

Decoding with TCBP and JOBS
===========================

Fit a Trajectory-Calibrated BP (TCBP) model, use its probabilities to guide
Jittered OSD Beam Search (JOBS), and inspect the predicted observable flips
and the confidence information returned for each shot. Both run through the
NVIDIA QLDPC decoder. See :ref:`qldpc_tcbp_jobs` for how they work and when to
use them.

Prerequisites
-------------

Use a CUDA-Q QEC installation with the ``nv-qldpc-decoder`` plugin, a
compatible NVIDIA GPU, NumPy, and Stim. The example below uses a small
simulated surface-code memory experiment. Its purpose is to demonstrate the
API; the :ref:`performance study <tcbp_jobs_user_guide>` uses larger circuits
and many more shots.

Prepare a Detector Error Model and Samples
------------------------------------------

Run the following blocks in the same Python session. Each column of ``H``
represents an error mechanism, and the corresponding column of ``O`` records
which logical observables it flips. ``rates`` contains the prior probability
of each mechanism. The detector samples use the same ordering as the rows of
``H``.

.. code-block:: python

   import numpy as np
   import stim
   from cudaq_qec.plugins.nv_qldpc import tying

   circuit = stim.Circuit.generated(
       "surface_code:rotated_memory_z",
       distance=3,
       rounds=3,
       after_clifford_depolarization=0.001,
   )
   H, O, rates = tying.dem_arrays(tying.circuit_dem(circuit))
   syndromes, observables = circuit.compile_detector_sampler(seed=11).sample(
       64, separate_observables=True)
   syndromes = syndromes.astype(np.uint8)

Fit the Model and Decode
------------------------

The named presets encode the matching BP, TCBP and JOBS configuration. The
following example fits a model for a code family, constructs a decoder that
searches every shot, and requests the complementary gap:

.. code-block:: python

   import numpy as np
   import cudaq_qec as qec
   from cudaq_qec.plugins import nv_qldpc

   # H: detector-error matrix
   # O: observable-flips matrix
   # rates: prior probability of each error mechanism
   model, report = nv_qldpc.init_tcbp_model(H, return_report=True)

   decoder = qec.get_decoder(
       "nv-qldpc-decoder",
       H,
       O=O,
       output="observables",
       error_rate_vec=rates,
       bp_batch_size=2048,
       **nv_qldpc.preset("tcbp-jobs-sc-gap", tcbp_model=model),
   )

   results = decoder.decode_batch(syndromes)
   predicted = np.asarray(results.result)
   gap = np.asarray([r["jobs_gap"] for r in results.opt_results])
   competitor = np.asarray(
       [r["jobs_gap_class"] for r in results.opt_results])

``tcbp-jobs`` omits search on BP-converged shots. ``tcbp-jobs-sc`` searches
them without allocating the logical-class table, and
``tcbp-jobs-sc-gap`` requests both behaviors. The presets use sparse
sum-product BP, 10 iterations, an LLR clip of 15, FP32 processing and
repeatable execution. JOBS then uses its default search parameters unless
they are overridden. The options of ``init_tcbp_model``, its report and
recommendations for fitting a model are in the Trajectory-Calibrated BP
section of the :ref:`API reference <nv_qldpc_decoder_api_python>`.

Inspect the Results
-------------------

``predicted`` contains one row of observable-flip predictions per shot;
``gap`` and ``competitor`` each contain one value per shot. For this circuit,
there is one logical observable:

.. code-block:: python

   assert predicted.shape == observables.shape == (64, 1)
   assert gap.shape == competitor.shape == (64,)
   print("Decoded 64 shots with one observable per shot.")

A larger gap means the search found a greater cost difference between the
returned correction and the best competing logical class it visited. It is
not a calibrated probability of correctness. An infinite gap and competitor
``-1`` mean no competing class was visited; they do not certify the answer.

Next Steps
----------

* Use :ref:`qldpc_tcbp_jobs` to choose the search budget and output mode.
* See :ref:`tcbp_jobs_user_guide` for measured accuracy, latency, and the
  trade-off between retaining shots and reducing logical errors.
* Learn priors from detector data with :doc:`Learning Noise and Decoding with
  Learned Rates <noise_learning>`.

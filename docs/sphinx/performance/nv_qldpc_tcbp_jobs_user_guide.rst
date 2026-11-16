.. _tcbp_jobs_user_guide:

Trajectory-Calibrated BP and Jittered OSD Beam Search
=====================================================

Belief propagation (BP) produces more information than its final hard
decision. At every iteration, it updates a marginal log-likelihood ratio
(LLR) for every error mechanism. A successful BP run turns those marginals
into a syndrome-consistent correction. An unsuccessful run still leaves a
trajectory: which mechanisms appeared likely to be in error, when that
happened, whether the estimate was stable or oscillatory, and how it was
moving when BP stopped.

Conventional BP+OSD decoders [1]_ retain only one number from each
trajectory. Most order the columns of the detector error matrix by the final
LLR; minimum-LLR variants
(:ref:`minllr_osd_user_guide`) retain the minimum value reached during BP.
Both give that single summary ordering to an ordered-statistics decoder
(OSD), which discards valuable information. The final LLR can miss a strong
signal from an earlier iteration, while a minimum LLR records the strength
of an excursion but not its timing or stability. OSD then explores
candidates around only that one summary ordering.

Starting in CUDA-Q QEC 0.9.0, the
:ref:`NV-qLDPC decoder <qldpc_decoder>` provides two components that use
this information more fully:

* **Trajectory-Calibrated BP (TCBP)** is a soft reweighting predecoder. It
  maps each mechanism's BP trajectory to a calibrated probability that the
  mechanism fired.
* **Jittered OSD Beam Search (JOBS)** is a GPU-parallel OSD
  post-processor. It searches many perturbed column orderings and refines
  each OSD-0 solution with a bounded beam search.

This page summarizes how to use the two components and what to expect from
them. The algorithms, the training procedure and the full evaluation are in
the TCBP-JOBS research article [2]_.

TCBP uses AI in a deliberately lightweight form: a small neural network
learns how BP trajectory shapes relate to error probabilities. Training
uses simulations generated from the parity-check matrix (PCM), requires no
experimental training set, and completed in under a minute in our
development runs on NVIDIA GPUs. The default model has only 529 trainable
weights. It is designed for reuse across physical error rates and code
distances within the same code family, so one fit can support many decoding
runs. The same model is what noise learning uses to estimate error rates
from detector data; see :ref:`noise_learning_user_guide`.

TCBP also works well with short BP trajectories. Every experiment in this
study uses 10 BP iterations. Larger
iteration counts give BP more opportunities to explore different beliefs
and give TCBP a longer trajectory to interpret, at the cost of additional
BP work. Because the iteration count is part of the fitted model
configuration, changing it requires a matching TCBP model.

The components are useful separately, but their combination is the focus of
this study. TCBP gives JOBS high-quality, syndrome-specific probability
estimates, which JOBS uses to explore multiple syndrome-consistent
corrections in parallel. During the search it can also retain the cheapest
candidate found in every logical class, providing complementary information
about the returned answer.

.. figure:: ../../../assets/docs/nv_qldpc_decoder_pipeline.png
   :align: center
   :alt: NV-qLDPC pipeline from BP trajectories through TCBP calibrated scores, column ordering and JOBS

   TCBP is both a soft-output predecoder and a source of calibrated column
   scores for OSD. JOBS is the GPU-parallel hard-decoding path that consumes
   those scores.

Key Decoder Options
+++++++++++++++++++

The main options selecting and tuning the TCBP path are:

* ``tcbp_model`` supplies the model bytes returned by
  ``init_tcbp_model`` or the path to a saved model. Its BP method, iteration
  count, clipping value and processing precision must match the decoder.
* ``use_osd=False`` returns TCBP's soft probabilities.
  ``use_osd=True, osd_init_method="tcbp"`` instead uses its scores as the
  column ordering for OSD or JOBS.

JOBS is selected with ``use_osd=True, osd_method=4``. Its principal search
options are:

* ``jobs_draws`` (default 16) sets the number of orderings searched per
  round; ``jobs_rounds`` (default 1) permits additional rounds while they
  continue to improve the answer.
* ``jobs_draw_noise`` (default 1.0) sets the Gumbel perturbation applied
  after draw zero.
* ``jobs_beam_width`` (default 32), ``jobs_beam_depth`` (default 40) and
  ``jobs_beam_patience`` (default 8) bound the search from each OSD-0
  solution.
* ``jobs_search_converged`` (default false) also searches BP-converged
  shots and returns the cheaper correction.
* ``osd_sparsify_pad`` (default 8) sizes JOBS's mandatory
  :math:`K^*`-based window. :math:`K^*` is the smallest number of leading
  columns of the ordering whose span contains the syndrome; JOBS searches
  the first :math:`\lceil` ``osd_sparsify_pad`` :math:`\times K^*\rceil`
  columns. ``bp_batch_size`` controls the shots in flight; the search's
  device memory is allocated at construction for
  ``bp_batch_size * jobs_draws`` items.
* ``opt_results={"jobs_gap": True}`` returns ``jobs_gap`` and
  ``jobs_gap_class`` for every shot. The JOBS gap is the cost, in nats, of
  the cheapest correction the search visited in a different logical class
  minus the cost of the returned correction; it is a search-limited
  approximation of the complementary gap, and it is ``+inf`` (class ``-1``)
  when no competing class was visited. This requires an observables matrix
  and supports at most 12 observables.

TCBP and JOBS require the sparse GPU path (``use_sparsity=True``) and
``composition=0``; JOBS supports at most 8192 detector rows and a beam width
of 64. See the complete ``nv-qldpc-decoder`` option reference for
:ref:`Python <nv_qldpc_decoder_api_python>` or
:ref:`C++ <nv_qldpc_decoder_api_cpp>` for types, constraints and memory
behavior.

Using TCBP and JOBS
+++++++++++++++++++

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
   predicted = np.asarray([r.result for r in results])
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

Comparison With Tesseract
+++++++++++++++++++++++++

We compare TCBP-JOBS with Tesseract [3]_, the search-based
most-likely-error decoder from Google Quantum AI. Tesseract runs on CPUs,
needs neither a GPU nor a fitted model, and is the accuracy reference for
this study. We use its ``tesseract-long-beam``
configuration, one of the two recommended starting points in the
`Tesseract repository <https://github.com/quantumlib/tesseract-decoder>`__:
beam width 20, 21 detector orderings, a priority-queue limit of 1,000,000,
beam climbing and no detector revisits. The Tesseract measurements are the
results in that repository [4]_ and are treated as source data
rather than rerun as part of this study.

The scope of the comparison is this one configuration. Tesseract also
offers a short-beam configuration and error sparsification, which trade
some accuracy for speed and are not shown here, so the ratios below should
not be read as applying to Tesseract in general. The results in the
repository do not record the Tesseract version or the hardware they were measured on, and
other versions or machines may perform differently. The research
article [2]_ has the fuller comparison: both measurement bases, further
physical error rates and additional Tesseract configurations.

The comparison uses the circuit-level SI1000 memory circuits in
the tesseract-decoder repository, at physical error rate ``p = 0.002`` in
the Z basis. They cover rotated surface codes, superdense color codes and
bivariate-bicycle codes over several distances. Detector error models are
flattened with ``approximate_disjoint_errors=True``. TCBP-JOBS runs with
the ``tcbp-jobs-sc`` preset: sparse sum-product BP for 10 iterations with
clip value 15 and FP32 processing, a TCBP model fitted from simulated data
before decoding, and the default JOBS search of 16 draws, one round, beam
width 32, beam depth 40, patience 8 and noise scale 1.0. With this preset
JOBS searches every shot, including the shots BP converges on
(``jobs_search_converged=True``). TCBP-JOBS decodes the same number of
shots per circuit as the Tesseract results in the repository, sampled
independently
with Stim: the two decoders see the same distribution, not the same shots.

Accuracy is reported as shot failure probability divided by the number of
syndrome-extraction rounds. For a circuit with ``r`` rounds and
per-experiment logical error rate ``LER``, the plotted value is

.. math::

   p_{\mathrm{fail,round}}
     = \frac{\mathrm{LER}}{r}.

Runtime is likewise divided by the number of rounds. TCBP-JOBS serial
latency is the mean over 4,096 shots per circuit, each decoded by its own
``decode()`` call on one NVIDIA GB200. The Tesseract values are the
per-shot decode times recorded in the repository's results [4]_, summed
over worker
threads. That data is from a machine that had at least 30 active threads.
Timings from an otherwise quiet machine would differ somewhat; we do not
estimate by how much. The horizontal axis therefore compares one CPU thread
with one whole GPU: it shows time to an answer, not performance on
equivalent hardware. Batched GPU runs (``decode_batch()`` calls of 16,384
shots) are used separately to measure logical error rate and total compute.
Those aggregate GPU-hours must not be interpreted as the serial response
latency shown on the horizontal axis.

The figure connects the two decoders at each code point. Filled colored
markers show TCBP-JOBS on one GB200; hollow markers show the long-beam CPU
result. The horizontal coordinate is time per round and the vertical
coordinate is failure probability per round, so movement toward the lower
left is better. An arrow therefore shows the change in both runtime and
accuracy for the same circuit rather than a scaling trend across different
codes.

.. figure:: ../../../assets/docs/nv_qldpc_tcbp_jobs_tradeoff.png
   :align: center
   :alt: Accuracy and latency tradeoff between TCBP plus JOBS on a GB200 and Tesseract long beam on CPUs

   Failure probability and decode time per syndrome-extraction round at
   ``p = 0.002``, Z basis, with TCBP-JOBS in the ``tcbp-jobs-sc`` preset.
   Each arrow connects the Tesseract and TCBP-JOBS result for the same
   circuit. The annotations report the corresponding time and error ratios;
   lower and farther left are better. Tesseract times are per CPU thread
   from the results on the tesseract-decoder repository; TCBP-JOBS times are serial latency on one
   GB200.

Across the 14 circuits, TCBP-JOBS stays in a narrow sub-millisecond latency
band per round, while Tesseract's time per round grows with code size, from
below that band on the distance-3 surface code to several seconds on the
larger bivariate-bicycle codes. The logical error rates are of the same
order throughout. On two circuits the 95% intervals of the two decoders do
not overlap: TCBP-JOBS has fewer failures on the distance-11 color code and
more on the distance-9 color code. On the other twelve circuits the two
decoders are indistinguishable at this sample size. Because the two
decoders saw independent samples, a difference on a single circuit should
be read together with its interval.

The compute behind the points differs as well. The
TCBP-JOBS study decoded 12.0 million shots in 0.36 GB200 GPU-hours using
batched execution. The long-beam results on the tesseract-decoder repository
cover the same 12.0 million shots in 63,334 CPU core-hours. These totals describe the cost of
collecting the data, not a hardware-normalized speed ratio: GPU-hours and
CPU core-hours are different resources, the timing modes differ, and the
GPU-hours count the decode calls only, not shot sampling, detector error
model construction or TCBP model fitting. They do show what GPU-parallel
search offers for large simulation studies: an accuracy sweep of this size
fits in under an hour on one GPU.

Complementary-Gap Post-Selection
++++++++++++++++++++++++++++++++

The complementary-gap experiment is separate from the Tesseract comparison.
It uses the ``[[144,12,12]]`` bivariate-bicycle (gross) code at
``p = 0.003`` and ``p = 0.004``, with the same BP, TCBP and JOBS search
settings and ``opt_results={"jobs_gap": True}``. We sweep a minimum accepted
gap and report both the fraction of shots retained and the logical error
rate per shot among retained shots. At both physical error rates,
increasing the threshold generally lowers the logical error rate. The
trade-off is especially favorable over the first several tens of nats: many
failures are concentrated among the low-gap shots, before the retained
fraction begins to fall rapidly.

.. figure:: ../../../assets/docs/nv_qldpc_jobs_gap_gross.png
   :align: center
   :alt: Logical error rate and retained-shot fraction versus minimum JOBS gap for the gross code

   Post-selection with the JOBS gap for the ``[[144,12,12]]``
   bivariate-bicycle code. The upper panel shows logical error rate per shot
   among retained shots, with 95% Wilson intervals; dashed segments are
   upper bounds where no retained shot failed. The lower panel shows the
   percentage of shots retained.

Limitations and Practical Guidance
++++++++++++++++++++++++++++++++++

TCBP-JOBS is a bounded learned search, not maximum-likelihood decoding.
TCBP calibration depends on the training distribution and matching BP
configuration. JOBS explores only its sampled orderings, window and beam,
and the reported complementary gap covers only candidates it visited.
Changing code families or BP settings requires validating or refitting the
TCBP model; changing the JOBS budget changes both memory use and search
quality.

Latency and throughput answer different deployment questions. Serial
``decode()`` measurements approximate the response time of one available
syndrome, while ``decode_batch()`` measures how efficiently a study or
offline workload can fill the GPU. ``bp_batch_size``, ``jobs_draws`` and
the detector-matrix dimensions determine whether the device has enough work
and memory to reach its throughput regime. Results should always identify
which mode was measured.

The gap should be calibrated operationally rather than treated as a
universal probability. A threshold that gives a useful acceptance/error
tradeoff for one code and noise regime need not transfer unchanged to
another. ``+inf`` is especially important to interpret correctly: it can
mean that the bounded search saw no competing class, or that the search did
not run on a BP-converged shot.

Within those limits, the components form a useful hierarchy. TCBP alone
exposes calibrated soft information. TCBP ordering can improve an existing
OSD post-processor. TCBP-JOBS spends parallel GPU work on alternative valid
corrections. Enabling search on converged shots and the complementary gap
adds an all-shot confidence signal when the extra search and class-table
memory are justified.

References
++++++++++

.. [1] J. Roffe, D. R. White, S. Burton and E. T. Campbell, "Decoding across the
  quantum LDPC code landscape," *Phys. Rev. Research* 2, 043423 (2020).
  `doi:10.1103/PhysRevResearch.2.043423 <https://doi.org/10.1103/PhysRevResearch.2.043423>`__

.. [2] B. Howe, S. Maurya, J. Olle, I. Basov, M. Li, J. G. Lietz, K. Young and
  K. M. Svore, "TCBP-JOBS: GPU-Accelerated Quantum Error Correction Decoding
  and Noise Learning with Near-Optimal Accuracy," NVIDIA Research (2026).

.. [3] L. Aghababaie Beni, O. Higgott and N. Shutty, "Tesseract: A search-based
  decoder for quantum error correction" (2025).
  `arXiv:2503.10988 <https://arxiv.org/abs/2503.10988>`__

.. [4] Results file ``benchmarking/sparsify_errors/aggregated_results.jsonl``
  (error sparsification benchmarks) in the
  `tesseract-decoder repository <https://github.com/quantumlib/tesseract-decoder/tree/main/benchmarking/sparsify_errors>`__;
  the rows without error sparsification are used here.

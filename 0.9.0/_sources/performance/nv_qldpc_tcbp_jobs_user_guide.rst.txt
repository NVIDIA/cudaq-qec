:tocdepth: 1

.. _tcbp_jobs_user_guide:

Accuracy and Latency of TCBP-JOBS
=================================

This study measures how Trajectory-Calibrated BP (TCBP) and Jittered OSD Beam
Search (JOBS) extend BP+OSD decoding [1]_ on NVIDIA GPUs. It compares their
accuracy and serial latency with a Tesseract long-beam configuration, then
measures how selecting shots by the JOBS gap changes the logical error rate
and the fraction of shots retained. The algorithms and wider evaluation are
described in the TCBP-JOBS research article [2]_.

For the decoding pipeline and configuration choices, see :ref:`qldpc_tcbp_jobs`.
For a small usage example, see :doc:`Decoding with TCBP and JOBS
</examples_rst/qec/tcbp_jobs>`. To run the Tesseract comparison or regenerate
its trade-off figure from the recorded data, see the `benchmark README
<https://github.com/NVIDIA/cudaq-qec/blob/main/benchmarks/qec/tesseract_longbeam_vs_tcbp_jobs/README.md>`_.

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
  K. M. Svore, `TCBP+JOBS: GPU-Accelerated Quantum Error Correction Decoding
  and Noise Learning with Near-Optimal Accuracy
  <https://research.nvidia.com/publication/_tcbpjobs-gpu-accelerated-quantum-error-correction-decoding-and-noise-learning-near>`__,
  NVIDIA Research (2026).

.. [3] L. Aghababaie Beni, O. Higgott and N. Shutty, "Tesseract: A search-based
  decoder for quantum error correction" (2025).
  `arXiv:2503.10988 <https://arxiv.org/abs/2503.10988>`__

.. [4] Results file ``benchmarking/sparsify_errors/aggregated_results.jsonl``
  (error sparsification benchmarks) in the
  `tesseract-decoder repository <https://github.com/quantumlib/tesseract-decoder/tree/main/benchmarking/sparsify_errors>`__;
  the rows without error sparsification are used here.

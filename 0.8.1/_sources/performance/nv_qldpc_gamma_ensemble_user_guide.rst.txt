.. The data and figures on this page are reproducible with the scripts in
   benchmarks/qec/relaybp_gamma_ensemble/ (see the README there for the
   parameter table, required Relay-BP testdata and expected runtime).
   Tested at:
     cudaqx  5140d8930f3e8a5a9a74b63c93dcbd0cda85dc48
     stim    e2fc1eca7fd21684d433aa5f10f4504ea4860d07  (v1.16.0)

.. _ensemble_gamma_user_guide:

Improving Relay BP Decoding With Gamma Ensembles
=================================================

The NV-qLDPC Relay BP decoder supports ensembles of independent gamma distribution trajectories (N "lanes"), run in parallel on a single GPU. When any lane has satisfied the stopping criterion, then the decoder may exit early. This can lead to significantly improved decoding performance as increased parallelism allows for faster exploration of decoding solutions, and therefore lower latency for hard-to-decode syndromes which may stall Relay BP. 

While setting multiple lanes of ensembled gammas can reduce the number of Relay BP iterations needed to converge to a solution, it can also lead to increased time per iteration; each leg of Relay BP synchronizes between lanes to check for convergence. Multiple lanes share a fixed GPU allocation, so as the number of edges in the Tanner graph increases, the GPU eventually must serialize the execution of each lane at each iteration, and then block for every lane to complete. These competing effects offset each other to some extent, so the mean latency per iteration may or may not be lower than the un-ensembled mean latency. However, the main benefit of ensembling is that the decoder can accept the fastest converging lane, and terminate slow-converging lanes early. The slow-converging lanes then do not contribute to the overall decoding time, narrowing the distribution of decode latencies. This can in turn lead to substantially improved LER under hard deadline constraints, where decoding failures are dominated by deadline misses on slow decodes. We will see below examples of bivariate-bicycle codes (BB) seeing up to 50-90x improved logical error rate, depending on the code and deadline.

The first experiment we perform is to investigate the competing effects of reducing the mean number of iterations to converge versus increasing the time per iteration; we build the Z-component of the circuit-level detector error model (DEM) (i.e. the Z-stabilizer detectors only) for the bivariate-bicycle (BB) codes ``[[72,12,6]]``, ``[[144,12,12]]`` and ``[[288,12,18]]`` and, for each, construct the decoder with ``gamma_ensemble_size`` set to N in {1, 2, 4, 8}. We then decode many sampled syndromes, timing each ``decode()`` call and reading back the number of Relay BP iterations the decoder used to converge:

.. code-block:: python

    import cudaq_qec as qec

    # H, error_rates, syndromes come from a circuit-level DEM of the code
    decoder = qec.get_decoder(
        "nv-qldpc-decoder", H, error_rate_vec=error_rates,
        use_sparsity=True, bp_method=3, composition=1, max_iterations=50,
        gamma0=0.125, gamma_dist=[-0.24, 0.66], clip_value=200.0, repeatable=True,
        proc_float="fp32", gamma_ensemble_size=N,    # sweep N in {1, 2, 4, 8}
        srelay_config={"pre_iter": 1, "num_sets": 600,
                       "stopping_criterion": "FirstConv"},
        opt_results={"num_iter": True})              # return the iteration count

    for syndrome in syndromes:
        result = decoder.decode(syndrome)            # timed for the per-iteration cost
        num_iter = result.opt_results["num_iter"]    # Relay BP iterations to converge

Performance Comparison
++++++++++++++++++++++

All experiments below use the same circuit-level noise and decoder settings, and are run on a single GB200 using CUDA-Q QEC 0.7.0. The parameters used for each experiment are listed below.

.. list-table:: Decoder and experiment parameters
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Value
   * - GPU
     - NVIDIA GB200
   * - Codes
     - ``[[72,12,6]]``, ``[[144,12,12]]``, ``[[288,12,18]]`` (Z-component DEM)
   * - Rounds ``r``
     - 6, 12, 18 for ``[[72,12,6]]``, ``[[144,12,12]]``, ``[[288,12,18]]`` respectively
   * - Noise model
     - uniform circuit-level, ``p = 0.002`` 
   * - Shots
     - 150,000
   * - ``bp_method``
     - 3
   * - ``composition``
     - 1
   * - ``gamma0``
     - 0.125
   * - ``gamma_dist``
     - ``[-0.24, 0.66]`` (for ``[[72,12,6]]``, ``[[144,12,12]]``); ``[-0.161, 0.815]`` (for ``[[288,12,18]]``)
   * - ``clip_value``
     - 200.0
   * - ``max_iterations``
     - 50
   * - ``use_sparsity``
     - True
   * - ``repeatable``
     - True
   * - ``proc_float``
     - ``"fp32"``
   * - ``pre_iter``
     - 1
   * - ``num_sets``
     - 600
   * - ``stopping_criterion``
     - ``FirstConv``

.. image:: ../../../assets/docs/relaybp_gamma_ensemble_perf.png
   :align: center
   :alt: Iterations to converge, time per iteration, and mean latency versus ensemble size, for several DEMs

As the ensemble size N grows, the median number of iterations to converge (left) falls for every code — more lanes explore more gamma sets in parallel, so the fastest-converging one wins sooner. At N = 8 the median iteration count drops to 0.67×–0.71× of its single-lane value across the three codes. The time per iteration (middle) rises with N — each lane's parity-check rows are spread over co-resident GPU blocks, so a lane must serially process more work per iteration — growing to 1.53×–1.66× at N = 8, largest for the ``[[288,12,18]]`` DEM. The net mean decode latency (right) combines the two effects; here the higher per-iteration cost roughly cancels the iteration savings, so mean latency stays close to its single-lane value at N = 8, ranging from 0.88× to 1.10×.

Latency Distribution
++++++++++++++++++++

One of the benefits of ensembling is that the latency distribution becomes narrower, leading to more consistent latency for hard-to-decode syndromes. Using the same decoders as above, we now focus on the behavior of the decoding latencies at various percentiles, from the median through the extreme tail.

.. image:: ../../../assets/docs/relaybp_latency_percentiles.png
   :align: center
   :alt: Latency percentiles (p50, p90, p99.9, p99.99) versus ensemble size N, per code

The worst-case (i.e. p99.99) latency improves with N for every code — at N = 8 it improves by ~2.70× for ``[[72,12,6]]``, ~5.55× for ``[[144,12,12]]`` and ~4.17× for ``[[288,12,18]]``. The median (p50) rises with N (by ~1.12×–1.23× at N = 8), as the per-iteration overhead is paid on decodes that would have converged quickly anyway. 

Logical Error Rate Under Hard Deadlines
+++++++++++++++++++++++++++++++++++++++

The narrower tail can improve logical error rates considerably for decoders under hard deadlines. Suppose each decode must finish within a wall-clock budget ``t``; a decode is a success only if it both finishes within ``t`` and returns the correct logical outcome, so the logical error rate under that deadline is ``LER(t) = P(latency > t or logical error)``. Using the same decoders as above, we record both the per-syndrome latency and whether the decoded logical is correct. The plot below shows the LER versus deadline for un-ensembled Relay BP (i.e. N = 1) and for each ensemble size N; each curve is measured over 150,000 sampled syndromes per configuration at the circuit-level noise strength given above (``p = 0.002``), using the ``FirstConv`` stopping criterion.

.. image:: ../../../assets/docs/relaybp_hard_deadline_ler.png
   :align: center
   :alt: Deadline vs LER — bicycle codes

Ensembling contracts the latency tail for all three codes, resulting in substantially lower LER for certain deadlines: beyond a crossover at the tightest deadlines (where the higher per-iteration cost makes the larger ensembles slightly worse), a larger ensemble misses fewer deadlines at looser budgets and reaches a lower floor. The regime under which ensembling is beneficial for these codes depends on the code and deadline, and likely requires specific tuning based on the problem. Below is a plot of the factor by which each ensemble size lowers LER relative to un-ensembled Relay BP as a function of the deadline ``t``; values above 1 mean a lower LER than N = 1. 

.. image:: ../../../assets/docs/relaybp_ler_multiplier.png
   :align: center
   :alt: LER improvement multiplier over un-ensembled Relay BP versus hard deadline, per code

At the tightest deadlines the larger ensembles are briefly worse (their higher per-iteration cost delays the fastest decodes), but past that crossover a larger ensemble lowers LER substantially — by up to ~41× for ``[[144,12,12]]`` and ~89× for ``[[288,12,18]]`` at deadlines of ~1-5 ms. For those two codes the multiplier then falls back toward N = 1 at loose deadlines, where both the ensemble and N = 1 reach their near-zero logical error floors; for ``[[72,12,6]]`` it settles at ~2×.

See Also
++++++++

* :ref:`Quantum Low-Density Parity-Check Decoder <qldpc_decoder>` -- the nv-qldpc-decoder overview
* :ref:`Getting Started with the NVIDIA QLDPC Decoder <examples_rst/qec/decoders:Getting Started with the NVIDIA QLDPC Decoder>` -- runnable example
* :ref:`C++ <nv_qldpc_decoder_api_cpp>` and :ref:`Python <nv_qldpc_decoder_api_python>` API reference

.. The data on this page is reproducible with the parameter table below and
   the code snippets on this page, using the vendored Z-only gross-code
   circuit assets/benchmarks/bb144_memory_Z.stim (from the Relay-BP
   reference implementation, https://github.com/trmue/relay; run
   `git lfs pull` if it appears as a pointer stub). Syndromes sampled with
   stim's compile_detector_sampler(seed=13).
   Tested at:
     cudaqx  73b8fcec

.. _relay_solutions_user_guide:

Sweeping Relay BP Stopping Criteria From a Single Run
=====================================================

The NV-qLDPC Relay BP decoder walks through a schedule of gamma sets ("legs") and can converge to a valid solution multiple times along the way; the ``stopping_criterion`` and ``stop_nconv`` parameters decide how many of those convergences to collect before returning the minimum-weight one (the RelayBP-N algorithm). ``stop_nconv`` is a genuine tuning knob: a larger N can lower the logical error rate (more candidate solutions to pick the minimum-weight answer from) but costs more BP iterations per shot, so choosing it means measuring the whole accuracy/latency trade-off curve.

Measured directly, that curve is expensive: every candidate N requires re-decoding the same syndromes in a separate full GPU run, and each run discards everything about the convergences except the single winning solution. The ``relay_solutions`` option removes this waste. When the decoder is constructed with ``opt_results={"relay_solutions": True}``, one run records *every* relay convergence — the cumulative iteration count at which it happened, the LLR weight of its hard decision, and the hard decision itself — and the :mod:`cudaq_qec.relay_solutions` post-processing helper replays the entire ``stop_nconv`` sweep offline from that recording. One GPU run, plus milliseconds of NumPy, answers every N at once. The reconstruction is exact, not approximate: with ``repeatable=True`` the reconstructed logical error rate and iteration statistics match a real ``stop_nconv=N`` run bit-for-bit, for every N.

Recording a run
+++++++++++++++

The recording run must not itself stop early — it uses ``stopping_criterion="All"`` so every convergence in the schedule is observed — and it should return per-shot iteration counts (``num_iter``), which the sweep uses for shots that exhaust the schedule. Everything else is the decoder configuration you are tuning. The example below uses the ``[[144,12,12]]`` bivariate-bicycle (gross) code with the canonical Relay BP settings for that code:

.. code-block:: python

    import cudaq_qec as qec
    from cudaq_qec import relay_solutions

    # H, O, error_rates, syndromes, obs_truth come from a circuit-level DEM
    decoder = qec.get_decoder(
        "nv-qldpc-decoder", H, error_rate_vec=error_rates,
        use_sparsity=True, bp_method=3, composition=1, max_iterations=60,
        gamma0=0.125, gamma_dist=[-0.24, 0.66], clip_value=200.0,
        repeatable=True, proc_float="fp32",
        O=O,                                      # records become obs flips
        srelay_config={"pre_iter": 80, "num_sets": 60,
                       "stopping_criterion": "All"},   # observe every leg
        opt_results={"relay_solutions": True,          # record all solutions
                     "num_iter": True},
        bp_batch_size=1000)

    results = decoder.decode_batch(syndromes)

    # Replay the whole RelayBP-N sweep from the one recording. n_values
    # defaults to every N up to the deepest recording (the largest number
    # of convergences any shot produced).
    sweep = relay_solutions.stop_nconv_sweep(
        results, obs_truth, percentiles=[50, 99])

    for j, n in enumerate(sweep.n):
        print(f"stop_nconv={n}: LER={sweep.ler[j]:.2e}, "
              f"mean iters={sweep.avg_iters[j]:.1f}, "
              f"p99 iters={sweep.iters_percentiles[1, j]:.0f}")

For every N, :func:`~cudaq_qec.relay_solutions.stop_nconv_sweep` reproduces exactly what ``stopping_criterion="NConv", stop_nconv=N`` would have returned per shot: the prediction is the minimum-weight record among the first N convergences, and the iteration count is the cumulative count at the Nth convergence (or the shot's full-schedule ``num_iter`` when it produced fewer than N). It returns the logical error rate, mean iterations, any requested iteration percentiles, and the fraction of shots that exhausted the schedule, for each N. The raw records are also available directly — :func:`~cudaq_qec.relay_solutions.unpack` reconstructs the per-shot record axes from ``results.batch_opt_results`` — for custom analyses such as weight-spectrum or convergence-time studies.

Example: RelayBP-N on the gross code
++++++++++++++++++++++++++++++++++++

We decode the ``[[144,12,12]]`` gross code at ``p = 0.002`` with the configuration above, using split decoding: the Z-only memory circuit from the Relay-BP reference implementation (vendored under ``assets/benchmarks/``), whose detector error model covers the Z-stabilizer detectors only.

.. list-table:: Decoder and experiment parameters
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Value
   * - GPU
     - NVIDIA RTX PRO 3000 (Blackwell, laptop)
   * - Code
     - ``[[144,12,12]]`` (Z-only split circuit), 12 rounds
   * - Noise model
     - uniform circuit-level, ``p = 0.002``
   * - Shots
     - 100,000
   * - ``bp_method``
     - 3
   * - ``composition``
     - 1
   * - ``gamma0``
     - 0.125
   * - ``gamma_dist``
     - ``[-0.24, 0.66]``
   * - ``clip_value``
     - 200.0
   * - ``max_iterations``
     - 60 (per leg)
   * - ``use_sparsity`` / ``repeatable``
     - True / True
   * - ``proc_float``
     - ``"fp32"``
   * - ``pre_iter`` / ``num_sets``
     - 80 / 60
   * - ``bp_batch_size``
     - 1,000

The single recording run took **256 s** (2.6 ms/shot) for the 100,000 shots, and the offline sweep over every N from 1 to 61 took **414 ms**. A few points on the curve:

.. list-table:: RelayBP-N sweep, all reconstructed from the one recording run
   :header-rows: 1

   * - ``stop_nconv``
     - Logical errors (of 100,000)
     - Mean iterations
     - p50 iterations
     - p99 iterations
     - Schedule exhausted
   * - 1
     - 5
     - 12.9
     - 10
     - 51
     - 0%
   * - 2
     - 3
     - 26.6
     - 20
     - 118
     - 0%
   * - 3
     - 2
     - 39.5
     - 31
     - 162
     - 0%
   * - 5
     - 1
     - 65.5
     - 52
     - 243
     - 0.01%
   * - 15
     - 1
     - 193.0
     - 164
     - 646
     - 0.06%
   * - 50
     - 1
     - 622.5
     - 554
     - 1890
     - 1.51%

The sweep shows the trade-off directly: the logical error rate falls roughly 5x between N = 1 and N = 5 (5e-5 to 1e-5) and then saturates, while the iteration cost keeps growing linearly in N — so for this code and noise, ``stop_nconv=5`` buys all of the measured accuracy at a small fraction of the cost of larger N. Every row of this table — and the 55 other N values on the curve — comes from the same single GPU run. Measuring the curve directly would instead cost one full decode of the same 100,000 syndromes per N — cheap for small N, increasingly close to the recording run's cost as N grows, and always another full GPU pass for every additional point on the curve. The recording also captures strictly more information than the sweep consumes — per-record weights and convergence times for every shot — so later questions (a different percentile, a subset of shots, a custom tie-breaking rule) are answered from the same data with no further GPU time.

Practical notes
+++++++++++++++

* **Record observables, not corrections.** Constructing the decoder with the observables matrix ``O`` makes each record ``O @ correction (mod 2)`` — for the gross-code DEM above this is 12 bits per record instead of 8,784, shrinking readback and post-processing cost by orders of magnitude. Without ``O``, pass the matrix to ``stop_nconv_sweep(..., observables=O)`` instead so it can score the logical error rate.
* **Memory.** ``relay_solutions=True`` records every convergence, bounded by ``num_sets`` (+1 if ``pre_iter > 0``) records per shot. To bound memory on long schedules, pass an integer cap (e.g. ``relay_solutions=16``); the sweep will refuse N values beyond a cap that actually truncated records, so cap at or above the largest N you intend to sweep.
* **Repeatability.** With ``repeatable=True`` (and a non-zero ``clip_value``) the reconstruction is bit-for-bit identical to real ``stop_nconv=N`` runs. Without it, runs differ by floating-point non-determinism, and the reconstruction matches a real run only statistically.

See Also
++++++++

* :ref:`Quantum Low-Density Parity-Check Decoder <qldpc_decoder>` -- the nv-qldpc-decoder overview
* :ref:`C++ <nv_qldpc_decoder_api_cpp>` and :ref:`Python <nv_qldpc_decoder_api_python>` API reference -- the ``relay_solutions`` option under ``opt_results``
* :ref:`ensemble_gamma_user_guide` -- a complementary Relay BP performance study

.. class:: nv_qldpc_decoder

    The ``nv-qldpc-decoder`` plugin provides GPU-accelerated decoders for
    quantum low-density parity-check (QLDPC) codes. Every configuration runs
    belief propagation (BP) on the GPU. The options select how the BP output
    is used:

    - as a hard decision (BP or Relay-BP);
    - as a calibrated error probability for each error mechanism, computed
      with a Trajectory-Calibrated BP (TCBP) model;
    - as the column ordering for an ordered-statistics decoding (OSD)
      post-processor: OSD-0, exhaustive search, combination sweep or
      Jittered OSD Beam Search (JOBS).

    The ``cudaq_qec.plugins.nv_qldpc`` module provides named presets for
    common configurations. See "Presets" below.

    An [[n,k,d]] quantum error correction (QEC) code encodes k logical qubits
    into an n qubit data block, with a code distance d. Quantum low-density
    parity-check (QLDPC) codes are characterized by sparse parity-check matrices
    (or Tanner graphs), corresponding to a bounded number of parity checks per
    data qubit.

    Requires a CUDA-Q compatible GPU. See the `CUDA-Q GPU Compatibility
    List <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_
    for a list of valid GPU configurations.

    References:

    - `Decoding Across the Quantum LDPC Code Landscape <https://arxiv.org/pdf/2005.07016>`_
    - `TCBP+JOBS: GPU-Accelerated Quantum Error Correction Decoding and Noise Learning with Near-Optimal Accuracy <https://research.nvidia.com/publication/_tcbpjobs-gpu-accelerated-quantum-error-correction-decoding-and-noise-learning-near>`_

    .. note::
      It is required to create decoders with the `get_decoder` API from the CUDA-Q QEC
      extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np
            H = np.array([[1, 0, 0, 1, 0, 1, 1],
                          [0, 1, 0, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8) # sample 3x7 PCM
            O = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
            opts = dict() # see below for options
            # H may also be a scipy.sparse matrix (CSR, CSC, COO, or any
            # other scipy.sparse format), which avoids a full dense rows×cols
            # allocation for large PCMs.  Any format is normalised to CSR
            # internally; no call to .toarray() or .todense() is needed.
            nvdec = qec.get_decoder('nv-qldpc-decoder', H, O=O,
                                    output='observables', **opts)

      .. tab:: C++

        .. code-block:: cpp

            std::size_t block_size = 7;
            std::size_t syndrome_size = 3;
            cudaqx::tensor<uint8_t> H, O;

            std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1, 
                                          0, 1, 0, 1, 1, 0, 1,
                                          0, 0, 1, 0, 1, 1, 1};
            H.copy(H_vec.data(), {syndrome_size, block_size});
            std::vector<uint8_t> O_vec = {1, 1, 1, 0, 0, 0, 0};
            O.copy(O_vec.data(), {1, block_size});

            cudaqx::heterogeneous_map nv_custom_args;
            nv_custom_args.insert("use_osd", true);
            // See below for options

            std::vector<double> error_rates(block_size, 0.001);
            cudaq::qec::decoder_init inputs(
                cudaq::qec::sparse_binary_matrix(H),
                cudaq::qec::sparse_binary_matrix(O),
                error_rates);
            auto nvdec = cudaq::qec::get_decoder(
                "nv-qldpc-decoder", std::move(inputs),
                cudaq::qec::decode_result_type::observables, nv_custom_args);
      
    .. note::
      The `"nv-qldpc-decoder"` implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    :param H: Parity-check matrix with shape ``(syndrome_size, block_size)``.
    :param O: Optional observable matrix with shape
      ``(num_observables, block_size)``. It is model data and does not select
      the result basis. Python accepts it as a keyword adapter; C++ supplies it
      through :cpp:class:`cudaq::qec::decoder_init`.
    :param output: Result basis, ``"errors"`` or ``"observables"`` in Python
      and :cpp:enum:`cudaq::qec::decode_result_type` in C++. When omitted, the
      decoder returns error frames even if ``O`` is present.
    :param error_rate_vec: Optional probability per error mechanism, with
      length ``block_size``. Python accepts it as a keyword adapter; C++
      supplies it through :cpp:class:`cudaq::qec::decoder_init`. It overrides
      the scalar ``error_rate`` parameter.

    .. rubric:: Decoder pipeline

    The decoder runs BP first. The BP result is either returned directly or
    used to order the columns of the parity-check matrix for an OSD
    post-processor. When a TCBP model is supplied, the decoder computes a
    calibrated error probability for each mechanism from the BP LLR history.
    These probabilities are either returned as the result
    (``use_osd=false``) or used as the OSD column ordering
    (``osd_init_method="tcbp"``).

    .. image:: ../../../../assets/docs/nv_qldpc_decoder_pipeline.png
       :align: center
       :alt: Block diagram of the nv-qldpc decoder pipeline: BP front end, TCBP calibrated scores, column ordering, OSD post-processing, and the four results.

    The following table lists the supported configurations, the presets that
    select them and the options that distinguish them.

    .. list-table::
       :header-rows: 1
       :widths: 34 30 36

       * - Configuration
         - Presets
         - Options
       * - BP hard decision
         - ``bp60``, ``relay-bp``
         - ``use_osd=false`` (``relay-bp``: ``composition=1``)
       * - TCBP error probabilities
         - ``tcbp``
         - ``tcbp_model``, ``use_osd=false``
       * - Final-LLR ordering with OSD-0 or combination sweep
         - ``bp60-osd0``, ``bp60-osd-cs1000``
         - ``use_osd=true``, ``osd_method`` 1 or 3
       * - Minimum-LLR ordering with OSD-0, combination sweep or JOBS
         - ``bp10-minllr-osd0``, ``bp10-minllr-osd-cs1000``, ``bp10-minllr-jobs``
         - ``osd_init_method="min_llr"``, ``osd_method`` 1, 3 or 4
       * - TCBP ordering with combination sweep or JOBS
         - ``tcbp-osd-cs1000``, ``tcbp-jobs``, ``tcbp-jobs-sc``, ``tcbp-jobs-sc-gap``
         - ``tcbp_model``, ``osd_init_method="tcbp"``, ``osd_method`` 3 or 4

    :param params: Heterogeneous map of decoder options. The options are
      grouped below by pipeline stage. Every option is optional unless its
      description says otherwise.

    .. rubric:: Device and execution

    Options that select the device, the floating-point precision, the batch
    sizes and repeatability.

    - `cuda_device_id` (int): Zero-based CUDA device ordinal on which to
      construct the decoder and run every decode. Must be ``>= 0`` and less
      than the number of visible GPUs. When omitted, the decoder is not
      pinned to a specific device and runs on the default device (GPU 0).
      Introduced in 0.7.0.
    - `use_sparsity` (bool): Whether or not to use a sparse matrix solver
    - `n_threads` (int): Number of CUDA threads to use for the GPU decoder
      (defaults to smart selection based on parity matrix size)
    - `proc_float` (string): The processing float type to use. Defaults to
      "fp64". Valid values are "fp32" and "fp64". Introduced in 0.5.0.
    - `repeatable` (bool): Whether to make the BP algorithm (and Relay BP
      algorithm if enabled) bit-for-bit repeatable. Defaults to False. You
      must set `clip_value` to a non-zero value to use this option. Setting
      this option to True makes it run approximately 5-10% slower, but you
      are guaranteed to get repeatable results, which is often useful for
      both timing and detailed syndrome analysis. Introduced in 0.6.0.
    - `bp_seed` (int): Seed for random number generation used in `bp_method=3` or
      `bp_method=5` (disordered memory BP), or in `composition=1` (sequential relay).
      Optional parameter, defaults to 42 if not provided. Introduced in 0.5.0.
    - `bp_batch_size` (int): Number of syndromes that will be decoded in
      parallel for the BP decoder (defaults to 1)
    - `osd_batch_size` (int): Number of syndromes that will be decoded in
      parallel by the host OSD (defaults to the number of concurrent threads
      supported by the hardware). Not applicable with `osd_method=4`, which
      searches the whole `bp_batch_size` chunk on the device; setting both
      is an error.

    .. rubric:: Belief propagation

    Options for the BP stage, which runs in every configuration.
    ``bp_method`` values 2 through 5 require ``use_sparsity=true``.
    ``scale_factor`` applies only to the min-sum methods.

    - `error_rate` (double): Probability of an error (in 0-1 range) on a
      block data bit (defaults to 0.001)
    - `max_iterations` (int): Maximum number of BP iterations to perform
      (defaults to 30)
    - `iter_per_check` (int): Number of iterations between BP convergence checks
      (defaults to 1, and max is `max_iterations`). Introduced in 0.4.0.
    - `clip_value` (float): Value to clip the BP messages to. Should be a
      non-negative value (defaults to 0.0, which disables clipping). Introduced in
      0.4.0.
    - `bp_method` (int): Core BP algorithm to use (defaults to 0). Introduced in 0.4.0,
      expanded in 0.5.0 and 0.7.0:

      - 0: sum-product
      - 1: min-sum (introduced in 0.4.0)
      - 2: min-sum+mem (uniform memory strength, requires `use_sparsity=True`. Introduced in 0.5.0)
      - 3: min-sum+dmem (disordered memory strength, requires `use_sparsity=True`. Introduced in 0.5.0)
      - 4: sum-product+mem (uniform memory strength, requires `use_sparsity=True`. Introduced in 0.7.0)
      - 5: sum-product+dmem (disordered memory strength, requires `use_sparsity=True`. Introduced in 0.7.0)
    - `scale_factor` (float): The scale factor to use for min-sum. Defaults to 1.0.
      When set to 0.0, the scale factor is dynamically computed based on the
      number of iterations. Introduced in 0.4.0.

    .. rubric:: Memory BP and Relay-BP

    Options for BP with memory (``bp_method`` values 2 through 5) and for
    Relay-BP (``composition=1``). All of them require ``use_sparsity=true``.

    - `composition` (int): Iteration strategy (defaults to 0). Introduced in 0.5.0:

      - 0: Standard (single run)
      - 1: Sequential relay (multiple gamma legs). Requires: `bp_method=3` (min-sum+dmem) or
        `bp_method=5` (sum-product+dmem), `use_sparsity=True`, and `srelay_config`. Support for
        `bp_method=5` was added in 0.7.0.
    - `gamma0` (float): Memory strength parameter. Required for `bp_method=2` (min-sum+mem)
      and `bp_method=4` (sum-product+mem), and for `composition=1` (sequential relay).
      Introduced in 0.5.0; extended in 0.7.0 for `bp_method=4`.
    - `gamma_dist` (vector<float>): Gamma distribution interval [min, max] for disordered
      memory strength. Required for `bp_method=3` (min-sum+dmem) or `bp_method=5`
      (sum-product+dmem) if `explicit_gammas` not provided. Introduced in 0.5.0; extended in
      0.7.0 for `bp_method=5`.
    - `explicit_gammas` (vector<vector<float>>): Explicit gamma values for each variable node.
      For `bp_method=3` or `bp_method=5` with `composition=0`, provide a 2D vector where each
      row has `block_size` columns. For `composition=1` (Sequential relay), provide `num_sets`
      rows (one per relay leg). Overrides `gamma_dist` if provided. Introduced in 0.5.0;
      extended in 0.7.0 for `bp_method=5`.
    - `srelay_config` (heterogeneous_map): Sequential relay configuration (required for
      `composition=1`). Contains the following parameters. Introduced in 0.5.0:

      - `pre_iter` (int): Number of pre-iterations to run before relay legs
      - `num_sets` (int): Number of relay sets (legs) to run
      - `stopping_criterion` (string): When to stop relay legs:

        - "All": Run all legs
        - "FirstConv": Stop relay after first convergence
        - "NConv": Stop after N convergences (requires `stop_nconv` parameter)
      - `stop_nconv` (int): Number of convergences to wait for before stopping
        (required only when `stopping_criterion="NConv"`)

      .. note::
         Starting in version 0.6.0, convergence during the ``pre_iter`` phase counts as a
         successful convergence towards the stopping criteria. Prior to 0.6.0, convergence
         during pre-iterations did not count.
    - `gamma_ensemble_size` (int): Number of parallel gamma trajectories
      ("lanes") run per sequential-relay BP iteration. Allowed values are
      1, 2, 4, and 8 (defaults to 1, which disables the ensemble). Each
      lane explores a distinct gamma set drawn from `gamma_dist` (or
      `explicit_gammas`). The constructor requires
      `num_sets >= gamma_ensemble_size` so the N per-lane gamma-set
      offsets stay pairwise distinct. Introduced in 0.7.0.

      **Design semantics (race-to-fastest):** the ensemble is optimized
      for decode latency. The kernel runs the N lanes in parallel and
      applies the user-supplied `stopping_criterion` across the
      ensemble: the first lane to satisfy the criterion stops the
      others. The winning lane is the converged lane with the
      lowest-weight correction (sum of error-rate LLRs over bits
      decoded as 1). If no lane converges across all `num_sets` legs
      the decoder falls back to lane 0's last-leg marginals. The
      speedups are maximized for the long-tail (p99) syndromes: easy
      syndromes converge in the first leg either way, while hard
      syndromes that would otherwise walk through many sequential legs
      get them raced in parallel instead.

      **Warm-up (`pre_iter`) semantics:** with the ensemble, the search
      diversity comes entirely from the relay legs, because each lane
      runs its own sequence of gamma sets. The `pre_iter` warm-up is
      the opposite: every lane would run it with the same uniform
      `gamma0` value, so the kernel computes it once (on one lane) and
      broadcasts the warmed-up marginals to every lane before the legs
      start. The warm-up helps convergence by settling the messages
      before the disordered gammas are applied and by letting easy
      syndromes converge early (which counts toward the stopping
      criterion), at the cost of delaying the start of the legs by the
      warm-up iterations. Ensemble speedups are therefore maximized
      when `pre_iter` is small or zero.

      The gamma ensemble is supported on the sparse GPU single-decode
      path with `composition=1` and `bp_method=3` (min-sum + dmem) or
      `bp_method=5` (sum-product + dmem). Passing
      `gamma_ensemble_size > 1` with any other `bp_method` or with the
      CPU or dense GPU path raises `std::invalid_argument` at
      construction. The batched relay path (`decode_batch()`) and the
      realtime / graph-dispatch path (`capture_decode_graph()`) do not
      yet support ensembles and raise at call time; support for
      `gamma_ensemble_size > 1` with `capture_decode_graph()` is
      intended in a future release.

    .. rubric:: Trajectory-Calibrated BP (TCBP)

    TCBP computes, for each error mechanism, a calibrated probability that
    the mechanism occurred. It uses the marginal LLR of the mechanism at
    every BP iteration and a TCBP model.
    ``cudaq_qec.plugins.nv_qldpc.init_tcbp_model`` fits the model once per
    code family from the detector error matrix.
    :ref:`Noise learning <qldpc_noise_learning>` uses the same model. TCBP
    requires ``use_sparsity=true`` and ``composition=0``.

    - `tcbp_model` (bytes or str): TCBP model returned by
      `cudaq_qec.plugins.nv_qldpc.init_tcbp_model`, or the path of a file
      that contains it. Supplying this option enables TCBP. In Python, pass
      the model as a writable 1-D `uint8` NumPy array
      (`np.array(bytearray(model), dtype=np.uint8)`). In the YAML schema,
      and therefore for real-time decoding, only the file path is accepted.
      The `bp_method`, `max_iterations`, `clip_value` and `proc_float` of
      the decoder must equal the values the model was fitted with. If
      `opt_results.bp_llr_history` is set, it must equal `max_iterations`.
    - `tcbp_strict` (bool): If true, a `tcbp_model` that was fitted on a
      detector error model with a different structure is a construction
      error. If false (default), the decoder prints a notice once and uses
      the model, so a model fitted at another physical error rate or code
      distance of the same code family can be used. Setting `tcbp_strict`
      without `tcbp_model` is a construction error.

    The result depends on ``use_osd``:

    - ``use_osd=false``: ``decode()`` and ``decode_batch()`` return the
      calibrated error probability of each mechanism. For shots on which BP
      converges, the result is the BP posterior probability and
      ``converged`` is true.
    - ``use_osd=true`` with ``osd_init_method="tcbp"``: the calibrated
      probabilities determine the OSD column ordering, and the result is the
      OSD correction. Supplying ``tcbp_model`` with ``use_osd=true`` and any
      other ``osd_init_method`` is a construction error.

    .. code-block:: python

        import numpy as np
        import cudaq_qec as qec
        from cudaq_qec.plugins import nv_qldpc

        model = nv_qldpc.init_tcbp_model(H)     # once per code family; bytes
        # H may be a dense uint8 array or any scipy sparse matrix (never densified)
        # optional: fit on the code's nominal error rates instead of the default draw
        # (the rates must vary across mechanisms; see error_rate_vec below)
        # model = nv_qldpc.init_tcbp_model(H, error_rate_vec=nominal)
        tcbp = np.array(bytearray(model), dtype=np.uint8)
        bp = dict(use_sparsity=True, max_iterations=10, clip_value=15.0,
                  bp_method=0, proc_float="fp32", error_rate_vec=rates)
        soft = qec.get_decoder("nv-qldpc-decoder", H, tcbp_model=tcbp,
                               use_osd=False, **bp)          # P(error) per mechanism
        hard = qec.get_decoder("nv-qldpc-decoder", H, tcbp_model=tcbp,
                               use_osd=True, osd_init_method="tcbp", **bp)

    **Fitting a TCBP model.** In Python,
    ``cudaq_qec.plugins.nv_qldpc.init_tcbp_model(H, return_report=False, **options)``
    fits a TCBP model on the GPU and returns it as ``bytes``. With
    ``return_report=True`` it returns ``(model, report)``. ``H`` is the
    detector error matrix, with one row per detector and one column per
    error mechanism, as a 2-D ``uint8`` array or a ``scipy.sparse`` matrix.
    The function samples errors from ``H`` at the training rates, runs BP on
    them and fits the model to the result. It needs no data from a device.
    Fitting took between 9 and 24 seconds per circuit on one GB200 for the
    circuits of :ref:`tcbp_jobs_user_guide`.

    Options of ``init_tcbp_model``:

    - `min_p`, `max_p` (float): Bounds of the training error rates. The
      function draws one rate for each mechanism, log-uniformly between
      `min_p` and `max_p` (defaults to 1e-4 and 1e-2). Requires
      0 < `min_p` < `max_p` < 0.5.
    - `error_rate_vec` (array of float): One training rate per mechanism,
      each greater than 0 and less than 0.5, used in place of the drawn
      rates. The rates must vary across mechanisms: the largest must be at
      least 1.5 times the smallest, and a uniform vector is rejected. Cannot
      be combined with `min_p` or `max_p`.
    - `seed` (int): Seed for the training rates, the sampling and the fit
      (defaults to 1).
    - `shots` (int): Number of sampled shots used for the fit. With 0
      (default), the function chooses the number so that the shots contain
      about 5 million sampled errors in total: 5,000,000 divided by the sum
      of the training rates, which is the expected number of errors per
      shot. The result is limited to the range from 16,384 to 1,048,576.
      A further 15% of that number is sampled as held-out shots.
    - `steps` (int): Number of fitting steps (defaults to 25000).
    - `device_memory_budget_bytes` (int): Device memory that the fit may
      use. With 0 (default), the fit may use half of the device memory that
      is free when the function is called.
    - `verbosity` (int): With 0 (default), the function prints nothing.
      With a positive value, it prints the report when the fit ends.
    - `bp_method`, `max_iterations`, `clip_value`, `proc_float`: BP settings
      that the model is fitted for (default to 0, 10, 15.0 and "fp32").
      `bp_method` must be 0 or 1, and `max_iterations` must be between 2 and
      32. A decoder that uses the model must be constructed with the same
      values.

    The report is a dictionary. Its main entries are:

    - `bce_model`, `brier_model`, `ece_model`: cross-entropy, Brier score
      and expected calibration error of the model's probabilities on
      held-out shots. Lower is better.
    - `bce_baseline`, `brier_baseline`, `ece_baseline`: the same metrics
      without the model.
    - `prob_regression_flag`: true if the model is worse than the baseline
      on the held-out shots.
    - `converged`: true if the fit converged.
    - `shots_used`, `holdout_shots`, `steps_run`, `seed`, `wall_s`: size,
      seed and wall time of the fit.
    - `budget_bytes`, `device_highwater_mib`: the device memory budget that
      was applied and the peak device memory that the fit used.

    ``cudaq_qec.plugins.nv_qldpc.inspect(model)`` returns the report stored
    in a model, together with the dimensions of ``H`` and fingerprints of
    its structure and of the training rates. The timing and memory entries
    are not stored in the model, so ``inspect`` returns them as 0.

    The following example fits a model, checks the report, saves the model
    to a file and constructs a decoder from the file:

    .. code-block:: python

        model, report = nv_qldpc.init_tcbp_model(H, return_report=True)
        assert not report["prob_regression_flag"]
        with open("code_family.tcbp", "wb") as f:
            f.write(model)

        dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates,
                              **nv_qldpc.preset("tcbp-jobs",
                                                tcbp_model="code_family.tcbp"))

    Recommendations:

    - Fit one model per code family with the default options, and reuse it
      at the other physical error rates and code distances of the family.
      This is the recommended starting point.
    - Fit a new model when `bp_method`, `max_iterations`, `clip_value` or
      `proc_float` changes.
    - Check `prob_regression_flag` after every fit.
    - If the error rates of the circuit lie outside the range from `min_p`
      to `max_p`, set `min_p` and `max_p` so that the range covers them.
    - For decoding, any training rates can be used, including rates passed
      as `error_rate_vec`. A model that is passed to ``learn_noise`` must
      not be fitted on the rates of the device whose noise is being learned,
      because the learned rates would then not be an independent estimate.
    - Save the model to a file to reuse it.
    - To obtain byte-identical models from repeated fits, pass the same
      `seed` and an explicit `device_memory_budget_bytes`.

    .. rubric:: Column ordering

    ``osd_init_method`` selects the values that order the columns of the
    parity-check matrix for the OSD post-processor. Preset names start with
    the ordering: ``bp60-*`` (final LLR), ``bp10-minllr-*`` (minimum LLR)
    and ``tcbp-*`` (TCBP).

    - `osd_init_method` (string): Which LLRs initialize the OSD column
      ordering when BP fails to converge (defaults to "final_llr").
      Introduced in 0.8.0.

      - "final_llr": the final-iteration marginal LLRs.
      - "min_llr": the per-column minimum marginal LLR over all BP
        iterations, so columns that were at any iteration confident that
        an error occurred sort first.
        Requires `use_osd`, the sparse GPU implementation
        (`use_sparsity=true`), and `composition=0`.
      - "tcbp": the TCBP error probabilities. Requires `tcbp_model`, in
        addition to the requirements of "min_llr".

    .. rubric:: OSD post-processing: OSD-0, exhaustive and combination sweep

    Options for the OSD post-processor, which runs on the shots for which BP
    does not converge. ``osd_method`` selects OSD-0, exhaustive search,
    combination sweep or JOBS. The options specific to JOBS are listed in
    the next section.

    - `use_osd` (bool): Whether or not to use an OSD post processor if the
      initial BP algorithm fails to converge on a solution
    - `osd_method` (int): 1=OSD-0, 2=Exhaustive, 3=Combination Sweep,
      4=JOBS (defaults to 1). Ignored unless `use_osd` is true. JOBS
      (Jittered OSD Beam Search) is described in the next section and was
      introduced in 0.9.0.
    - `osd_order` (int): OSD postprocessor order (defaults to 0). Ref:
      `Decoding Across the Quantum LDPC Code Landscape <https://arxiv.org/pdf/2005.07016>`_

      - For `osd_method=2` (Exhaustive), the number of possible
        permutations searched after OSD-0 grows by 2^osd_order.
      - For `osd_method=3` (Combination Sweep), this is the λ parameter. All
        weight 1 permutations and the first λ bits worth of weight 2
        permutations are searched after OSD-0. This is (syndrome_length -
        block_size + λ * (λ - 1) / 2) additional permutations.
      - For other `osd_method` values, this is ignored.
    - `osd_sparsify` (bool): Run OSD on a prefix of the ordered columns
      instead of all of them (defaults to false). The prefix is sized from
      K*, the smallest number of leading columns (in the `osd_init_method`
      ordering) whose span contains the syndrome; the OSD solution and the
      higher-order sweep then only involve those columns, which makes OSD
      much faster when K* is small compared to the number of columns.
      Requires `use_osd`.
    - `osd_sparsify_pad` (float >= 1): With `osd_sparsify`, OSD uses the
      first ceil(`osd_sparsify_pad` * K*) columns, clamped to the number
      of columns (defaults to 8.0). Mutually exclusive with
      `osd_sparsify_exact`. Except with `osd_method=4`, setting
      `osd_sparsify_pad` or `osd_sparsify_exact` without
      `osd_sparsify=true` is a construction error.
    - `osd_sparsify_exact` (int >= 1): With `osd_sparsify`, OSD uses
      exactly this many leading columns, which bounds the per-shot OSD
      work. A shot whose K* exceeds it cannot be solved inside the window:
      it receives an all-zero correction and a warning is printed (one per
      `decode()`, one with the number of affected shots per
      `decode_batch()`). Nothing in the result marks such a shot; to find
      them, request `opt_results.kstar` and look for
      `kstar > osd_sparsify_exact`.

    .. rubric:: OSD post-processing: JOBS

    JOBS (Jittered OSD Beam Search, ``osd_method=4``) is an OSD
    post-processor that runs on the GPU. For each shot, it searches
    ``jobs_draws`` column orderings and returns the lowest-cost correction
    that it finds. The cost of a correction is the sum of the prior
    log-likelihood ratios of its error mechanisms. The options in this
    section require ``use_osd=true`` and ``osd_method=4``. Setting any of
    them otherwise is a construction error. For measured accuracy and
    latency, see :ref:`tcbp_jobs_user_guide`.

    - `jobs_draws` (int >= 1): Number of column orderings ("draws")
      searched per round for each shot (defaults to 16). Introduced in
      0.9.0.
    - `jobs_rounds` (int >= 1): Maximum number of rounds of `jobs_draws`
      draws per shot (defaults to 1). The search of a shot stops after the
      first round that does not lower its cost. Introduced in 0.9.0.
    - `jobs_beam_width` (int, 1..64): Number of states the beam search
      keeps per step (defaults to 32). Introduced in 0.9.0.
    - `jobs_beam_depth` (int >= 0): Maximum number of beam search steps
      (defaults to 40). With 0, no beam search runs and each draw returns
      its OSD-0 solution. Introduced in 0.9.0.
    - `jobs_beam_patience` (int >= 1): Number of consecutive beam search
      steps without improvement after which the search of a draw stops
      (defaults to 8). Introduced in 0.9.0.
    - `jobs_draw_noise` (float > 0): Scale of the Gumbel noise added to the
      ordering scores of every draw except draw 0 (defaults to 1.0).
      Introduced in 0.9.0.
    - `jobs_search_converged` (bool): If true, JOBS also searches the shots
      for which BP converged and returns the lower-cost of the BP
      correction and the JOBS correction (defaults to false). `converged`
      still reports whether BP converged. With
      `osd_init_method="final_llr"`, `opt_results.bp_llr_history`, if set,
      must equal `max_iterations`. Introduced in 0.9.0.

    The search has the following steps:

    1. Draw 0 uses the ``osd_init_method`` ordering. Every other draw adds
       Gumbel noise, scaled by ``jobs_draw_noise``, to the ordering scores
       and sorts the columns again.
    2. Each draw computes an OSD-0 solution on the first W columns of its
       own ordering, where W = ceil(``osd_sparsify_pad`` x K*). K* is the
       smallest number of leading columns of draw 0 whose span contains the
       syndrome, so every draw of a shot uses the same W.
    3. A beam search starts from each OSD-0 solution and looks for
       lower-cost corrections that satisfy the syndrome. It keeps
       ``jobs_beam_width`` states per step. It stops after
       ``jobs_beam_depth`` steps, or after ``jobs_beam_patience`` steps
       without improvement.
    4. If ``jobs_rounds`` is greater than 1, further rounds of draws run
       while a round lowers the cost of the shot.

    By default, JOBS runs only on the shots for which BP does not converge.
    Set ``jobs_search_converged`` to search every shot.

    Requirements and limits:

    - JOBS requires ``use_osd=true``, ``use_sparsity=true`` and
      ``composition=0``.
    - Every error rate must be at most 0.5. A larger rate is a construction
      error.
    - ``H`` can have at most 8192 detectors (rows), and ``jobs_beam_width``
      can be at most 64. A value outside a limit is a construction error
      that reports the value.
    - JOBS always restricts the search to the K* window, so ``osd_sparsify``
      is not needed. ``osd_order`` is ignored. Setting
      ``osd_sparsify=false``, ``osd_batch_size`` or ``osd_sparsify_exact`` is
      a construction error.

    Device memory: the decoder allocates all device memory for the search at
    construction, sized for ``bp_batch_size`` x ``jobs_draws`` concurrent
    searches. If the free device memory is insufficient, the decoder reduces
    the number of concurrent searches at the largest window sizes and prints
    a warning. Results are unchanged, and throughput can be lower.
    Construction fails only if the minimum configuration does not fit. The
    error message reports the number of bytes required.

    Repeatability: the noise is seeded from the syndrome, so ``decode()``
    and ``decode_batch()`` search the same draws, and repeated calls return
    identical results. ``opt_results.kstar`` reports K* of draw 0.

    The following example constructs a JOBS decoder with the default search
    options:

    .. code-block:: python

        dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates,
                              use_sparsity=True, max_iterations=10,
                              clip_value=15.0, use_osd=True, osd_method=4,
                              osd_init_method="min_llr", bp_batch_size=2048)
        # defaults: jobs_draws=16, jobs_beam_width=32, jobs_beam_depth=40,
        #           jobs_beam_patience=8, jobs_draw_noise=1.0, osd_sparsify_pad=8

    **JOBS gap.** With ``opt_results.jobs_gap``, the decoder returns two
    additional values per shot:

    - ``jobs_gap``: the cost of the lowest-cost correction found in a
      different observable class, minus the cost of the returned correction,
      in nats. The observable class of a correction is the set of
      observables that it flips.
    - ``jobs_gap_class``: the observable class of that competing correction,
      as an integer bit mask in which bit ``o`` is set if the correction
      flips observable ``o``, or -1 if none was found.

    The gap is computed only over the corrections that the search evaluated.
    It is therefore an approximation of the complementary gap and not the
    exact value. The gap is ``+inf`` when the search found no correction in
    another class. It is also ``+inf`` for shots on which JOBS did not run,
    that is, shots for which BP converged when ``jobs_search_converged`` is
    false. A small gap indicates a shot that is more likely to be decoded
    incorrectly, so the gap can be used for post-selection.

    ``jobs_gap`` requires a decoder constructed with an observable matrix
    that has at most 12 observables. For k observables, the option adds
    ``bp_batch_size`` x 2^k x 4 bytes to the device memory of the search.

    The following example requests the JOBS gap and keeps the shots whose
    gap is at least 5 nats:

    .. code-block:: python

        dec = qec.get_decoder("nv-qldpc-decoder", H, O=O, error_rate_vec=rates,
                              use_sparsity=True, max_iterations=10,
                              clip_value=15.0, use_osd=True, osd_method=4,
                              osd_init_method="min_llr", bp_batch_size=2048,
                              opt_results=dict(jobs_gap=True))
        res = dec.decode_batch(syndromes)
        gap = np.array([o["jobs_gap"] for o in res.opt_results])
        keep = gap >= 5.0

    .. rubric:: Optional results

    Additional outputs that the decoder returns with each result.

    - `opt_results` (heterogeneous_map): Optional results to return. This field can be
      left empty if no additional results are desired. Choices are:

      - `bp_llr_history` (int): Return the last `bp_llr_history` iterations
        of the BP LLR history. Minimum value is 0 and maximum value is
        max_iterations. The actual number of returned iterations might be fewer
        than `bp_llr_history` if BP converges before the requested number of
        iterations. Introduced in 0.4.0. Note: Not supported for `composition=1`.
      - `num_iter` (bool): If true, return the number of BP iterations run.
        Introduced in 0.5.0.
      - `kstar` (bool): If true, return K* per shot: the smallest number of
        leading columns of the OSD ordering whose span contains the
        syndrome; -1 when OSD did not run because BP converged (with
        `jobs_search_converged`, converged shots are searched and report
        the K* of that search). With `osd_sparsify_exact`, a value above
        `osd_sparsify_exact` marks a shot the window could not solve.
        Requires `use_osd`.
      - `jobs_gap` (bool): If true, return per shot the JOBS gap
        (`jobs_gap`, nats) and the competing observable class
        (`jobs_gap_class`, int; -1 when none). See "JOBS gap" in
        the JOBS section. Requires `osd_method=4` and a decoder
        constructed with observables (at most 12).
      - `relay_solutions` (bool or int): Record every relay convergence
        ("solution"), not just the winning one. `True` records all of
        them; a positive integer caps the number of records kept per shot
        (convergences beyond the cap still increment the per-shot total
        but are not stored). Requires `composition=1` (sequential relay)
        on the sparse GPU path (`use_sparsity=True`); other backends
        reject the option at construction. Not yet supported with
        `gamma_ensemble_size > 1`. Compatible with `use_osd=True` (OSD
        post-processes non-converged shots and does not affect the
        records). Introduced in 0.8.0.

        Each record holds the cumulative BP iteration count at which the
        convergence occurred, the LLR weight of its hard decision (the sum
        of error-rate LLRs over bits decoded as 1), and the hard decision
        itself, bit-packed 32 bits per little-endian word. The hard
        decision follows the explicitly selected result basis: the
        correction vector for error output, or its observable flips for
        observable output.

        The records describe a batch as a whole, so `decode_batch()`
        returns them through its batch-level results (the
        `BatchDecoderResult.batch_opt_results` attribute in Python; the
        optional `batch_opt_results` output parameter in C++) as flat
        arrays under the keys `relay_solutions_width`,
        `relay_solutions_max_records`, `relay_solutions_counts`,
        `relay_solutions_totals`, `relay_solutions_iters`,
        `relay_solutions_weight`, and `relay_solutions_result`.
        (`decode()` returns the same keys through
        `DecoderResult.opt_results`, with scalar `relay_solutions_count` /
        `relay_solutions_total`.) The `cudaq_qec.relay_solutions` Python
        module post-processes the records: `unpack()` reconstructs the
        per-shot record axes, and `stop_nconv_sweep()` replays an entire
        RelayBP-N `stop_nconv` sweep — logical error rate, mean
        iterations, and iteration percentiles for every N — from a single
        recording run. See :ref:`relay_solutions_user_guide` for a usage
        and performance walkthrough.

    .. rubric:: Presets

    The ``cudaq_qec.plugins.nv_qldpc`` module defines named presets for the
    configurations in the table above. ``preset(name, **overrides)`` returns
    the options of a preset as a dictionary, with ``overrides`` applied.
    Pass the dictionary to ``get_decoder`` as keyword arguments.

    .. list-table::
       :header-rows: 1
       :widths: 28 30 20 22

       * - Preset
         - BP
         - Column ordering
         - Post-processor
       * - ``bp60``
         - sum-product, 60 iterations
         - n/a
         - none (BP hard decision)
       * - ``bp60-osd0``
         - sum-product, 60 iterations
         - final LLR
         - OSD-0
       * - ``bp60-osd-cs1000``
         - sum-product, 60 iterations
         - final LLR
         - combination sweep, ``osd_order=1000``
       * - ``bp10-minllr-osd0``
         - sum-product, 10 iterations
         - minimum LLR
         - OSD-0
       * - ``bp10-minllr-osd-cs1000``
         - sum-product, 10 iterations
         - minimum LLR
         - combination sweep, ``osd_order=1000``
       * - ``bp10-minllr-jobs``
         - sum-product, 10 iterations
         - minimum LLR
         - JOBS
       * - ``relay-bp``
         - Relay-BP (``bp_method=3``), 80 pre-iterations, 300 relay legs
         - n/a
         - none (BP hard decision)
       * - ``tcbp``
         - sum-product, 10 iterations
         - n/a
         - none (TCBP error probabilities)
       * - ``tcbp-osd-cs1000``
         - sum-product, 10 iterations
         - TCBP
         - combination sweep, ``osd_order=1000``
       * - ``tcbp-jobs``
         - sum-product, 10 iterations
         - TCBP
         - JOBS
       * - ``tcbp-jobs-sc``
         - sum-product, 10 iterations
         - TCBP
         - JOBS with ``jobs_search_converged``
       * - ``tcbp-jobs-sc-gap``
         - sum-product, 10 iterations
         - TCBP
         - JOBS with ``jobs_search_converged`` and ``opt_results.jobs_gap``

    All presets set ``use_sparsity=true``, ``proc_float="fp32"`` and
    ``repeatable=true``. All presets except ``relay-bp`` also set
    ``bp_method=0`` and ``clip_value=15.0``.

    Notes:

    - Presets do not set error rates. Pass ``error_rate_vec`` or
      ``error_rate`` to ``get_decoder``.
    - The ``tcbp*`` presets require a ``tcbp_model`` override. The override
      accepts the bytes returned by ``init_tcbp_model``, a ``uint8`` array
      or a file path.
    - The ``tcbp*`` presets use the BP settings that ``init_tcbp_model``
      fits with by default. To use a different ``bp_method``,
      ``max_iterations``, ``clip_value`` or ``proc_float``, fit the model
      with those settings and pass the same values as overrides.
    - ``tcbp-jobs-sc-gap`` requires a decoder constructed with an observable
      matrix that has at most 12 observables.

    Options that are commonly overridden:

    - ``bp_batch_size``: number of shots decoded in parallel.
    - ``osd_sparsify=True`` and ``osd_sparsify_pad``: restrict OSD-0 and the
      combination sweep to the K* window, which reduces their cost. The
      window is off by default for these methods, and ``osd_sparsify_pad``
      defaults to 8.0. JOBS always uses the window.
    - ``jobs_draws``, ``jobs_beam_width``, ``jobs_beam_depth``,
      ``jobs_beam_patience`` and ``jobs_draw_noise``: amount of search that
      JOBS performs.
    - ``gamma0``, ``gamma_dist`` and ``srelay_config``: Relay-BP parameters.
    - ``bp_method=1``: min-sum BP in place of sum-product (``scale_factor``
      applies).

    The following example constructs a decoder from an OSD preset with the
    K* window enabled and a pad of 4.0 in place of the default 8.0, and a
    second decoder from a TCBP+JOBS preset that
    returns the JOBS gap:

    .. code-block:: python

        import numpy as np
        import cudaq_qec as qec
        from cudaq_qec.plugins import nv_qldpc

        dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates,
                              bp_batch_size=2048,
                              **nv_qldpc.preset("bp10-minllr-osd-cs1000",
                                                osd_sparsify=True,
                                                osd_sparsify_pad=4.0))
        model = nv_qldpc.init_tcbp_model(H)          # once per code family
        dec = qec.get_decoder("nv-qldpc-decoder", H, O=O, error_rate_vec=rates,
                              bp_batch_size=2048,
                              **nv_qldpc.preset("tcbp-jobs-sc-gap",
                                                tcbp_model=model))
        res = dec.decode_batch(syndromes)
        gap = np.array([o["jobs_gap"] for o in res.opt_results])

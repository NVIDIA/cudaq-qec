.. class:: nv_qldpc_decoder

    A general purpose Quantum Low-Density Parity-Check Decoder (QLDPC)
    decoder based on GPU accelerated belief propagation (BP). Since belief
    propagation is an iterative method, decoding can be improved with a
    second-stage post-processing step. Optionally, ordered statistics decoding
    (OSD) can be chosen to perform the second stage of decoding.

    An [[n,k,d]] quantum error correction (QEC) code encodes k logical qubits
    into an n qubit data block, with a code distance d. Quantum low-density
    parity-check (QLDPC) codes are characterized by sparse parity-check matrices
    (or Tanner graphs), corresponding to a bounded number of parity checks per
    data qubit.

    Requires a CUDA-Q compatible GPU. See the `CUDA-Q GPU Compatibility
    List <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_
    for a list of valid GPU configurations.

    References:
    `Decoding Across the Quantum LDPC Code Landscape <https://arxiv.org/pdf/2005.07016>`_

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
    :param params: Heterogeneous map of parameters:

        - `cuda_device_id` (int): Zero-based CUDA device ordinal on which to
          construct the decoder and run every decode. Must be ``>= 0`` and less
          than the number of visible GPUs. When omitted, the decoder is not
          pinned to a specific device and runs on the default device (GPU 0).
          Introduced in 0.7.0.
        - `use_sparsity` (bool): Whether or not to use a sparse matrix solver
        - `error_rate` (double): Probability of an error (in 0-1 range) on a
          block data bit (defaults to 0.001)
        - `max_iterations` (int): Maximum number of BP iterations to perform
          (defaults to 30)
        - `n_threads` (int): Number of CUDA threads to use for the GPU decoder
          (defaults to smart selection based on parity matrix size)
        - `use_osd` (bool): Whether or not to use an OSD post processor if the
          initial BP algorithm fails to converge on a solution
        - `osd_method` (int): 1=OSD-0, 2=Exhaustive, 3=Combination Sweep
          (defaults to 1). Ignored unless `use_osd` is true.
        - `osd_order` (int): OSD postprocessor order (defaults to 0). Ref:
          `Decoding Across the Quantum LDPC Code Landscape <https://arxiv.org/pdf/2005.07016>`_

          - For `osd_method=2` (Exhaustive), the number of possible
            permutations searched after OSD-0 grows by 2^osd_order.
          - For `osd_method=3` (Combination Sweep), this is the λ parameter. All
            weight 1 permutations and the first λ bits worth of weight 2
            permutations are searched after OSD-0. This is (syndrome_length -
            block_size + λ * (λ - 1) / 2) additional permutations.
          - For other `osd_method` values, this is ignored.
        - `bp_batch_size` (int): Number of syndromes that will be decoded in
          parallel for the BP decoder (defaults to 1)
        - `osd_batch_size` (int): Number of syndromes that will be decoded in
          parallel for OSD (defaults to the number of concurrent threads supported
          by the hardware)
        - `iter_per_check` (int): Number of iterations between BP convergence checks
          (defaults to 1, and max is `max_iterations`). Introduced in 0.4.0.
        - `clip_value` (float): Value to clip the BP messages to. Should be a
          non-negative value (defaults to 0.0, which disables clipping). Introduced in
          0.4.0.
        - `repeatable` (bool): Whether to make the BP algorithm (and Relay BP
          algorithm if enabled) bit-for-bit repeatable. Defaults to False. You
          must set `clip_value` to a non-zero value to use this option. Setting
          this option to True makes it run approximately 5-10% slower, but you
          are guaranteed to get repeatable results, which is often useful for
          both timing and detailed syndrome analysis. Introduced in 0.6.0.
        - `bp_method` (int): Core BP algorithm to use (defaults to 0). Introduced in 0.4.0,
          expanded in 0.5.0 and 0.7.0:

          - 0: sum-product
          - 1: min-sum (introduced in 0.4.0)
          - 2: min-sum+mem (uniform memory strength, requires `use_sparsity=True`. Introduced in 0.5.0)
          - 3: min-sum+dmem (disordered memory strength, requires `use_sparsity=True`. Introduced in 0.5.0)
          - 4: sum-product+mem (uniform memory strength, requires `use_sparsity=True`. Introduced in 0.7.0)
          - 5: sum-product+dmem (disordered memory strength, requires `use_sparsity=True`. Introduced in 0.7.0)
        - `composition` (int): Iteration strategy (defaults to 0). Introduced in 0.5.0:

          - 0: Standard (single run)
          - 1: Sequential relay (multiple gamma legs). Requires: `bp_method=3` (min-sum+dmem) or
            `bp_method=5` (sum-product+dmem), `use_sparsity=True`, and `srelay_config`. Support for
            `bp_method=5` was added in 0.7.0.
        - `scale_factor` (float): The scale factor to use for min-sum. Defaults to 1.0.
          When set to 0.0, the scale factor is dynamically computed based on the
          number of iterations. Introduced in 0.4.0.
        - `proc_float` (string): The processing float type to use. Defaults to
          "fp64". Valid values are "fp32" and "fp64". Introduced in 0.5.0.
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
        - `bp_seed` (int): Seed for random number generation used in `bp_method=3` or
          `bp_method=5` (disordered memory BP), or in `composition=1` (sequential relay).
          Optional parameter, defaults to 42 if not provided. Introduced in 0.5.0.
        - `opt_results` (heterogeneous_map): Optional results to return. This field can be
          left empty if no additional results are desired. Choices are:

          - `bp_llr_history` (int): Return the last `bp_llr_history` iterations
            of the BP LLR history. Minimum value is 0 and maximum value is
            max_iterations. The actual number of returned iterations might be fewer
            than `bp_llr_history` if BP converges before the requested number of
            iterations. Introduced in 0.4.0. Note: Not supported for `composition=1`.
          - `num_iter` (bool): If true, return the number of BP iterations run.
            Introduced in 0.5.0.
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

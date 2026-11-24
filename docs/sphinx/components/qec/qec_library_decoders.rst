:tocdepth: 2

.. _prebuilt_qec_decoders:

QEC Library Decoders
====================

CUDA-Q QEC provides pre-built decoders for a variety of use cases.
The Python, C++ and realtime columns describe their integration with CUDA-Q
QEC. The **Standalone Python** column lists decoder packages provided by CUDA-Q
that run without the CUDA-Q runtime.

.. list-table::
   :header-rows: 1
   :widths: 18 24 7 6 10 16 29
   :class: decoder-catalog

   * - Decoder
     - Decoder String Identifier
     - Python
     - C++
     - Realtime enabled
     - Standalone Python
     - Notes
   * - NVIDIA QLDPC Decoder¹
     - `"nv-qldpc-decoder"`
     - Yes
     - Yes
     - Yes
     - |nv-qldpc-version|

       :ref:`Installation <nv_qldpc_standalone_installation>`
     - Supports Relay BP, BP+OSD, and :ref:`TCBP+JOBS <qldpc_tcbp_jobs>`.
   * - NVIDIA Fusion Decoder
     - `"nv-fusion-decoder"`
     - Yes
     - Yes
     - Yes
     - —
     - Multithreaded CPU MWPM decoder optimized for low-latency streaming
   * - Tensor Network Decoder¹
     - `"tensor_network_decoder"`
     - Yes²
     - No
     - No
     - —
     - Exact Maximum Likelihood Decoder
   * - TensorRT Decoder¹
     - `"trt_decoder"`
     - Yes³
     - Yes
     - Yes
     - —
     - GPU inference with a compatible trained model; decodes once the configured input is complete
   * - PyMatching Decoder
     - `"pymatching"`
     - Yes
     - Yes
     - Yes
     - —
     - MWPM decoder for matchable codes such as the surface code
   * - Chromobius Decoder
     - `"chromobius"`
     - Yes
     - Yes
     - Yes
     - —
     - Color-code (Möbius) decoder; constructed from Stim DEM text
   * - Look-Up Table Decoder
     - `"single_error_lut"` / `"multi_error_lut"`
     - Yes
     - Yes
     - Yes
     - —
     - Simple LUT decoders; ``multi_error_lut`` handles up to ``lut_error_depth`` errors
   * - Sliding Window Decoder
     - `"sliding_window"`
     - Yes
     - Yes
     - No
     - —
     - Decodes syndromes in a sliding window; pairs with any inner decoder except the TensorRT Decoder

| ¹ GPU-accelerated decoder
| ² Requires installation with `pip install cudaq-qec[tensor-network-decoder]` for Python
| ³ Requires installation with `pip install cudaq-qec[trt-decoder]` for Python

Here's a detailed overview of each:

.. _qldpc_decoder:

Quantum Low-Density Parity-Check Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Quantum Low-Density Parity-Check (QLDPC) decoder leverages GPU-accelerated belief propagation (BP) for efficient error correction.
Since belief propagation is an iterative method which may not converge, decoding can be improved with a second-stage post-processing step. The `nv-qldpc-decoder`
API provides various post-processing options, which can be selected through its parameters.

For every option, its default and its constraints, see the full ``nv-qldpc-decoder`` API reference for
:ref:`Python <nv_qldpc_decoder_api_python>` or :ref:`C++ <nv_qldpc_decoder_api_cpp>`.
The :doc:`standalone Python API </api/decoders/nv_qldpc>` documents its own
constructors, methods, and result types.

**Decoder Pipeline:**

The decoder runs BP first. The BP result is either returned directly or used to order the columns of the
parity-check matrix for an ordered-statistics decoding (OSD) post-processor. When a Trajectory-Calibrated BP
(TCBP) model is supplied, the decoder computes a calibrated error probability for each error mechanism, which
is either returned as the result or used as the OSD column ordering.

.. image:: ../../../../assets/docs/nv_qldpc_decoder_pipeline.png
   :align: center
   :alt: Block diagram of the nv-qldpc decoder pipeline: BP front end, TCBP calibrated scores, column ordering, OSD post-processing, and the four results.

The API reference lists the supported configurations of this pipeline and the options and presets that select
them.

**Belief propagation methods:**

The decoder supports several belief-propagation algorithms -- sum-product, min-sum, and memory-based variants -- selected via ``bp_method``, with optional BP+OSD post-processing. For the complete list of methods, parameters, and defaults, see the ``nv-qldpc-decoder`` entries in the :ref:`C++ <nv_qldpc_decoder_api_cpp>` and :ref:`Python <nv_qldpc_decoder_api_python>` API reference.

**Highlighted features:**

* **Sequential Relay BP** (``composition=1``): chains multiple "relay legs" -- sequential BP runs with different gamma configurations -- to decode syndromes that stall a single BP pass. **Requires:** ``bp_method=3``, ``gamma0``, ``srelay_config``, and either ``gamma_dist`` OR ``explicit_gammas``.
* **Gamma ensembling** (``gamma_ensemble_size``): an extension of Relay BP that explores multiple sets of gamma values in parallel on a single GPU (N independent "lanes"). The first lane to converge lets the decoder exit early, so slow lanes are terminated without adding to the decode time -- narrowing the latency distribution and improving performance on hard-to-decode syndromes that would otherwise stall Relay BP. See :doc:`Improving Relay BP decoding with gamma ensembles </performance/nv_qldpc_gamma_ensemble_user_guide>` for a performance study.
* **Trajectory-Calibrated BP (TCBP)** (``tcbp_model``): computes a calibrated error probability for each error mechanism from its LLR at every BP iteration, using a model that is fitted once per code family. The probabilities are returned directly (``use_osd=false``) or used to order the OSD columns (``osd_init_method="tcbp"``). Introduced in 0.9.0.
* **Jittered OSD Beam Search (JOBS)** (``osd_method=4``): an OSD post-processor that runs on the GPU. For each shot it searches several perturbed column orderings and refines each OSD-0 solution with a beam search. It can also return a per-shot confidence value, the JOBS gap, for post-selection. Introduced in 0.9.0. See :ref:`TCBP and JOBS <qldpc_tcbp_jobs>` for the decoding pipeline and :ref:`tcbp_jobs_user_guide` for measured accuracy and latency.
* **Presets** (``cudaq_qec.plugins.nv_qldpc.preset``): named option sets for common configurations, such as ``bp60-osd0``, ``relay-bp`` and ``tcbp-jobs``. The API reference lists all of them.
* **Noise learning** (``learn_noise``, CUDA-Q QEC only): learns the error rate of every mechanism of a detector error model from recorded detector data alone, using the TCBP model. See :ref:`Noise Learning <qldpc_noise_learning>`.

The QLDPC decoder `nv-qldpc-decoder` requires a CUDA-Q compatible GPU. See the `CUDA-Q dependencies and compatibility <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_ list.

The decoder is based on the following references:

* J. Roffe, D. R. White, S. Burton and E. T. Campbell,
  `Decoding Across the Quantum LDPC Code Landscape
  <https://arxiv.org/pdf/2005.07016>`__,
  Physical Review Research 2, 043423 (2020).
* T. Müller, T. Alexander, M. E. Beverland, M. Bühler, B. R. Johnson,
  T. Maurer and D. Vandeth, `Improved belief propagation is sufficient for
  real-time decoding of quantum memory <https://arxiv.org/pdf/2506.01779>`__,
  arXiv:2506.01779 (2025).
* B. Howe, S. Maurya, J. Olle, I. Basov, M. Li, J. G. Lietz, K. Young and
  K. M. Svore, `TCBP+JOBS: GPU-Accelerated Quantum Error Correction Decoding
  and Noise Learning with Near-Optimal Accuracy
  <https://research.nvidia.com/publication/_tcbpjobs-gpu-accelerated-quantum-error-correction-decoding-and-noise-learning-near>`__,
  NVIDIA Research (2026).

Software repositories:

* J. Roffe, `LDPC: Python tools for low density parity check codes
  <https://github.com/quantumgizmos/ldpc>`__, GitHub (2022).
* `Relay-BP decoder <https://github.com/trmue/relay>`__, GitHub
  (``trmue/relay``).


Usage:

Choose the Python package you installed. Both use the same decoder options;
the standalone package takes matrices directly without a decoder-name argument.

.. tab:: Python — CUDA-Q QEC

    .. code-block:: python

        import cudaq_qec as qec
        import numpy as np

        H_list = [
                    [1, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1]
                 ]

        H_np = np.array(H_list, dtype=np.uint8)

        decoder = qec.get_decoder("nv-qldpc-decoder", H_np)

.. tab:: Python — standalone

    .. code-block:: python

        from cudaq_decoding import nv_qldpc
        import numpy as np

        H_np = np.array([
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ], dtype=np.uint8)

        decoder = nv_qldpc.create_decoder(H_np)

.. tab:: C++ — CUDA-Q QEC

    .. code-block:: cpp

        std::size_t block_size = 7;
        std::size_t syndrome_size = 3;
        cudaqx::tensor<uint8_t> H;

        std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1,
                                      0, 1, 0, 1, 1, 0, 1,
                                      0, 0, 1, 0, 1, 1, 1};
        H.copy(H_vec.data(), {syndrome_size, block_size});

        cudaqx::heterogeneous_map nv_custom_args;
        nv_custom_args.insert("use_osd", true);

        auto d1 = cudaq::qec::get_decoder("nv-qldpc-decoder", H, nv_custom_args);

        // Alternatively, configure the decoder without instantiating a heterogeneous_map
        auto d2 = cudaq::qec::get_decoder("nv-qldpc-decoder", H, {{"use_osd", true}, {"bp_batch_size", 100}});

For a runnable example, see :ref:`Getting started with the NVIDIA QLDPC Decoder <qldpc_decoder_example>`.
For the standalone interface, see the
:doc:`NVIDIA QLDPC Python API </api/decoders/nv_qldpc>`.
The :doc:`Standalone Decoders <standalone_decoders>` guide covers installation
and the :ref:`differences between the Python packages <nv_qldpc_package_comparison>`.
For the CUDA-Q QEC interfaces and shared decoder options, see the
:ref:`Python <nv_qldpc_decoder_api_python>` or :ref:`C++ <nv_qldpc_decoder_api_cpp>`
API reference.

.. _qldpc_tcbp_jobs:

Trajectory-Calibrated BP and Jittered OSD Beam Search
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Belief propagation (BP) produces more information than its final hard
decision. At every iteration, it updates a marginal log-likelihood ratio
(LLR) for every error mechanism. A successful BP run turns those marginals
into a syndrome-consistent correction. An unsuccessful run still leaves a
trajectory: which mechanisms appeared likely to be in error, when that
happened, whether the estimate was stable or oscillatory, and how it was
moving when BP stopped.

Conventional BP+OSD decoders retain only one number from each
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

TCBP uses AI in a deliberately lightweight form: a small neural network
learns how BP trajectory shapes relate to error probabilities. Training
uses simulations generated from the parity-check matrix (PCM), requires no
experimental training set, and completed in under a minute in our
development runs on NVIDIA GPUs. The default model has only 529 trainable
weights. It is designed for reuse across physical error rates and code
distances within the same code family, so one fit can support many decoding
runs. The same model is what noise learning uses to estimate error rates
from detector data; see :ref:`qldpc_noise_learning`.

The :ref:`performance study <tcbp_jobs_user_guide>` uses 10 BP iterations. Larger
iteration counts give BP more opportunities to explore different beliefs
and give TCBP a longer trajectory to interpret, at the cost of additional
BP work. Because the iteration count is part of the fitted model
configuration, changing it requires a matching TCBP model.

The components are useful separately, but their combination is the focus of
the :ref:`performance study <tcbp_jobs_user_guide>`. TCBP gives JOBS
syndrome-specific probability estimates, which JOBS uses to explore multiple syndrome-consistent
corrections in parallel. During the search it can also retain the cheapest
candidate found in every logical class, providing complementary information
about the returned answer.

.. rubric:: Configuration Choices

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
``composition=0``. Starting in 0.9.1, JOBS supports models with more than
8192 detector rows. The maximum detector count is limited by the memory of
the GPU; a model that is too large for the device is a construction error.
The maximum beam width is 64.
See the complete ``nv-qldpc-decoder`` option reference for
:ref:`Python <nv_qldpc_decoder_api_python>` or
:ref:`C++ <nv_qldpc_decoder_api_cpp>` for types, constraints and memory
behavior.

For a step-by-step example, see :doc:`Decoding with TCBP and JOBS
</examples_rst/qec/tcbp_jobs>`. For measured accuracy, serial latency, and
post-selection results, see :ref:`tcbp_jobs_user_guide`.

NVIDIA Fusion Decoder
^^^^^^^^^^^^^^^^^^^^^

The ``nv-fusion-decoder`` is a multithreaded CPU decoder using minimum-weight
perfect matching (MWPM) for graphlike error models. It partitions detector data into
temporal blocks and fuses their solutions to reduce latency in streaming,
realtime decoding environments. It also supports offline batch decoding.

See the :ref:`Python <nv_fusion_decoder_api_python>` and
:ref:`C++ <nv_fusion_decoder_api_cpp>` API references for configuration and
usage details. For benchmark results and reproduction instructions, see
:doc:`/performance/nv_fusion_latency_user_guide`.

Tensor Network Decoder
^^^^^^^^^^^^^^^^^^^^^^

The ``tensor_network_decoder`` constructs a tensor network representation of a quantum code given its parity check matrix, logical observable(s), and noise model. It can decode individual syndromes or batches of syndromes, returning the probability that a logical observable has flipped.

Due to the additional dependencies of the Tensor Network Decoder, you must
specify the optional pip package when installing CUDA-Q QEC in order to use this
decoder. Use `pip install cudaq-qec[tensor-network-decoder]` in order to use
this decoder.

Key steps:

1. **Define the parity check matrix**: This matrix encodes the structure of the quantum code. In the example, a simple [3,1] repetition code is used.

2. **Specify the logical observable**: This is typically a row vector indicating which qubits participate in the logical operator.

3. **Set the noise model**: The example uses a factorized noise model with independent bit-flip probability for each error mechanism.

4. **Instantiate the decoder**: Create a decoder object using ``qec.get_decoder("tensor_network_decoder", ...)`` with the code parameters.

5. **Decode syndromes**: Use the ``decode`` method for single syndromes or ``decode_batch`` for multiple syndromes.


Usage:

.. tab:: Python

    .. code-block:: python

        # This example demonstrates how to use the get_decoder("tensor_network_decoder", ...) API
        # from the ``cudaq_qec`` library to decode syndromes for a simple
        # quantum error-correcting code using tensor networks.

        import cudaq_qec as qec
        import numpy as np

        # Define code parameters
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        logical_obs = np.array([[1, 1, 1]], dtype=np.uint8)
        noise_model = [0.1, 0.1, 0.1]

        decoder = qec.get_decoder("tensor_network_decoder", H, logical_obs=logical_obs, noise_model=noise_model)

        # Decode a single syndrome
        syndrome = [0.0, 1.0]
        result = decoder.decode(syndrome)
        print(result.result)

        # Decode a batch of syndromes
        syndrome_batch = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        batch_results = decoder.decode_batch(syndrome_batch)
        for res in batch_results:
            print(res.result)

.. tab:: C++

    The ``tensor_network_decoder`` is a Python-only implementation and it requires Python 3.11 or higher. C++ APIs are not available for this decoder.

Output:

The decoder returns the probability that the logical observable has flipped for each syndrome. This can be used to assess the performance of the code and the decoder under different error scenarios.

.. note::

    In general, the Tensor Network Decoder has the same GPU support as the
    :ref:`Quantum Low-Density Parity-Check Decoder <qldpc_decoder>`.
    However, if you are using the V100 GPU (SM70), you will need to pin your
    cuTensor version to 2.2 by running `pip install cutensor_cu12==2.2`.

For a runnable example, see :ref:`Exact maximum likelihood decoding with NVIDIA Tensor Network Decoder <tensor_network_decoder_example>`.


TensorRT Decoder
^^^^^^^^^^^^^^^^

The ``trt_decoder`` deploys a trained neural-network decoder (an ONNX model) through NVIDIA TensorRT for optimized GPU inference. Unlike the algorithmic decoders, it is trained on a specific code and noise model — you bring your own model. Python use requires ``pip install cudaq-qec[trt-decoder]``. See the :ref:`TensorRT Decoder API <trt_decoder_api_python>` for configuration options, and the :ref:`Deploying AI decoders with TensorRT example <deploying-ai-decoders>` for the full train-to-deploy workflow.

For realtime use, configure the measurement mapping and a model whose input and
output dimensions match the decoding problem. The standard enqueue path invokes
inference once the configured input is complete; this does not provide native
incremental streaming or guarantee a particular feedback latency. A compatible
GPU and the TensorRT plugin are required.

For a neural predecoder followed by a conventional decoder, see
:doc:`AI Predecoding <ai_predecoding>` and
:doc:`Comparing Ising and Chromobius for Color-Code Decoding
</examples_rst/qec/color_code_demo>`.

PyMatching Decoder
^^^^^^^^^^^^^^^^^^

The ``pymatching`` decoder is a minimum-weight perfect matching (MWPM) decoder built on the open-source `PyMatching <https://github.com/oscarhiggott/PyMatching>`_ library, suitable for matchable codes such as the surface code. It is selected by name through ``get_decoder`` and takes a parity-check matrix whose columns each have one or two set entries; per-edge priors are supplied via ``error_rate_vec``. See the :ref:`PyMatching Decoder API <pymatching_decoder_api_python>` and the :ref:`Matching-based decoding with PyMatching example <pymatching_decoder_example>`.

Chromobius Decoder
^^^^^^^^^^^^^^^^^^

The ``chromobius`` decoder is a color-code decoder built on the open-source `Chromobius <https://github.com/quantumlib/chromobius>`_ Möbius decoder. Unlike the matrix-based decoders, it is constructed from Stim detector-error-model (DEM) text rather than a parity-check matrix, and predicts logical observable flips directly. See the :ref:`Chromobius Decoder API <chromobius_decoder_api_python>` and the :ref:`Color-code decoding with Chromobius example <chromobius_decoder_example>`.

Chromobius can also decode the residual detector events left by an AI
predecoder. See :doc:`Comparing Ising and Chromobius for Color-Code Decoding
</examples_rst/qec/color_code_demo>` for a comparison with Chromobius alone.

Sliding Window Decoder
^^^^^^^^^^^^^^^^^^^^^^

Sliding-window decoding handles **circuit-level noise** across several syndrome
rounds by processing syndromes **before the full measurement sequence arrives**,
which **reduces latency** at the cost of **higher logical error rates** than
decoding the entire sequence at once.

Whether that tradeoff is worthwhile depends on the **noise model**, **code
parameters**, and **latency budget**. Since **CUDA-Q 0.5.0**, you can use **any
CUDA-Q decoder** as the **inner** decoder and tune behavior mainly via **window
size** and the other settings below. Each round must yield the **same
number of syndrome measurements**; the decoder assumes **no particular temporal
structure** of the noise, so you can still vary noise **from round to round** in
experiments.

Key steps:

1. **Obtain a detector error matrix and rates**: Pass the parity check matrix
   ``H`` (for example ``dem.detector_error_matrix``) and ``error_rate_vec`` with
   one entry per column of ``H`` (for example ``dem.error_rates`` from the same
   DEM). The matrix must be in the sorted form expected by :code:`pcm_is_sorted`
   for your ``num_syndromes_per_round``; DEMs from :code:`dem_from_memory_circuit`
   (and its single-basis variants :code:`z_dem_from_memory_circuit` /
   :code:`x_dem_from_memory_circuit`) are canonicalized. Hand-built matrices may
   need :code:`simplify_pcm`.
2. **Set the schedule and window**: Provide ``num_syndromes_per_round`` (the number of
   syndrome measurements per round) and ``num_boundary_syndromes`` (the number of
   stabilizer syndromes fixed by the state-prep at the beginning and end of the circuit).
   Choose ``window_size`` and ``step_size`` so ``window_size`` and
   ``step_size`` stay within valid bounds and ``num_rounds - window_size`` is
   divisible by ``step_size``, with ``num_rounds`` inferred from ``H`` and
   ``num_syndromes_per_round``.
3. **Pick an inner decoder**: Use ``inner_decoder_name`` and
   ``inner_decoder_params`` for the decoder that runs inside each window (for
   example :code:`nv-qldpc-decoder`). Optional ``straddle_start_round`` /
   ``straddle_end_round`` control cross-round mechanisms at window edges.
4. **Construct and run**: Call :code:`get_decoder("sliding_window", H, opts)`,
   then ``decode`` or ``decode_batch``. Partial syndromes leave the decoder in an
   intermediate state until enough bits arrive; full parameter lists and
   behavior are in the :ref:`Python <sliding_window_decoder_api_python>` and
   :ref:`C++ <sliding_window_decoder_api_cpp>` sliding-window API references.

Background: `Toward Low-latency Iterative Decoding of QLDPC Codes Under Circuit-Level Noise <https://arxiv.org/abs/2403.18901>`__.

Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq
        import cudaq_qec as qec
        import numpy as np

        cudaq.set_target('stim')
        num_rounds = 5
        code = qec.get_code('surface_code', distance=num_rounds)
        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.001), 1)
        statePrep = qec.operation.prep0
        dem = qec.dem_from_memory_circuit(code, statePrep, num_rounds, noise)
        inner_decoder_params = {'use_osd': True, 'max_iterations': 50, 'use_sparsity': True}
        opts = {
            'error_rate_vec': np.array(dem.error_rates),
            'window_size': 1,
            'num_syndromes_per_round': code.get_num_z_stabilizers() + code.get_num_x_stabilizers(),
            'num_boundary_syndromes': code.get_num_z_stabilizers(),
            'inner_decoder_name': 'nv-qldpc-decoder',
            'inner_decoder_params': inner_decoder_params,
        }
        swdec = qec.get_decoder('sliding_window', dem.detector_error_matrix, **opts)

.. tab:: C++

    .. code-block:: cpp

        #include "cudaq/qec/code.h"
        #include "cudaq/qec/decoder.h"
        #include "cudaq/qec/experiments.h"
        #include "common/NoiseModel.h"

        int main() {
            int num_rounds = 5;
            auto code = cudaq::qec::get_code(
                "surface_code", cudaqx::heterogeneous_map{{"distance", num_rounds}});
            cudaq::noise_model noise;
            noise.add_all_qubit_channel("x", cudaq::depolarization2(0.001), 1);
            auto statePrep = cudaq::qec::operation::prep0;
            auto dem = cudaq::qec::dem_from_memory_circuit(*code, statePrep, num_rounds,
                                                            noise);
            auto inner_decoder_params = cudaqx::heterogeneous_map{
                {"use_osd", true}, {"max_iterations", 50}, {"use_sparsity", true}};
            auto opts = cudaqx::heterogeneous_map{
                {"window_size", 1},
                {"num_syndromes_per_round", code->get_num_z_stabilizers() + code->get_num_x_stabilizers()},
                {"num_boundary_syndromes", code->get_num_z_stabilizers()},
                {"inner_decoder_name", "nv-qldpc-decoder"},
                {"inner_decoder_params", inner_decoder_params}};
            auto swdec = cudaq::qec::get_decoder("sliding_window",
                                                 cudaq::qec::decoder_init{dem}, opts);
            return 0;
        }

Output:

Once a decode step completes, results use the same types as other pre-built
decoders (:class:`cudaq_qec.Decoder` in Python, :cpp:class:`cudaq::qec::decoder`
in C++).

QEC Decoders
============

The ``cudaq-qec`` decoder interface (:code:`cudaq::qec::decoder`) turns syndromes into corrections. This page covers the framework — the class structure and how to implement a decoder — together with a catalog of the decoders that ship with the library. Read it to choose a built-in decoder or to write your own. For runnable programs, see the :doc:`Decoders examples </examples_rst/qec/decoders>`.

Decoder Framework :code:`cudaq::qec::decoder`
----------------------------------------------

The CUDA-Q QEC decoder framework provides an extensible system for implementing
quantum error correction decoders through the :code:`cudaq::qec::decoder` base class.

Class Structure
^^^^^^^^^^^^^^^

The decoder base owns immutable model data and fixes the result basis when an
instance is constructed. The relevant interface is:

.. code-block:: cpp

    class decoder {
    public:
        decoder(decoder_init inputs,
                decode_result_type requested_output =
                    decode_result_type::errors);

        virtual decoder_result decode(
            const std::vector<float_t>& syndrome) = 0;

        virtual std::vector<decoder_result> decode_batch(
            const std::vector<std::vector<float_t>>& syndrome);

        decode_result_type get_result_type() const noexcept;

    protected:
        const decoder_init& get_inputs() const noexcept;
        std::size_t block_size;
        std::size_t syndrome_size;
    };

Key Components:

* **Construction inputs**: :code:`decoder_init` owns ``H`` and optional ``O``,
  ``D``, error rates, and authoritative Stim DEM text
* **Fixed result basis**: An instance returns either error frames or observable
  flips for its entire lifetime
* **Block Size**: Number of modeled error mechanisms (columns of ``H``)
* **Syndrome Size**: Number of detector values (rows of ``H``)
* **Decoder Result**: Contains convergence status, values in the configured
  error or observable basis, and optional decoder-specific metadata
* **Multiple Decoding Modes**: Single syndrome or batch processing

Migrating Existing C++ Decoder Plugins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This decoder construction contract is a breaking C++ source and ABI change.
Out-of-tree decoder plugins must be rebuilt and migrated as follows:

* Replace constructors that take ``H`` with a constructor that takes
  :cpp:class:`cudaq::qec::decoder_init` by value and passes it to
  ``decoder(std::move(inputs), requested_output)``.
* Read model data through ``get_inputs()``. ``H``, ``O``, ``D``, and
  ``error_rate_vec`` are framework model data and must not be passed in the
  heterogeneous custom-parameter map or registered as plugin schema keys.
* Change custom creator signatures to accept ``decoder_init`` by value,
  ``std::optional<decode_result_type>``, and the custom parameter map. The
  removed ``make_pcm_decoder`` helper must not be used.
* Choose the plugin's default result basis when the optional request is empty,
  and reject unsupported explicit requests during construction. The presence
  of ``O`` does not select observable output.
* ``observables_and_residual_detectors`` is a multi-output composition
  contract, not a third standalone result basis. A plugin accepts it only if
  it returns ``observables`` in ``decoder_result::result`` and residual detectors in
  ``decoder_result::opt_results["residual_detectors"]``; all other plugins must
  reject it during construction.
* Remove calls to the deleted ``set_O_sparse`` and ``set_D_sparse`` methods.
  The base constructor now derives the corresponding model state and buffer
  sizes. A streaming decoder supplies only its layer geometry through
  ``initialize_streaming_layout`` during construction.

The main source migration maps old symbols as follows:

.. list-table:: Decoder plugin API migration
   :header-rows: 1

   * - Removed API
     - Replacement
   * - ``decoder::decode_result_type::decode_to_errs``
     - ``cudaq::qec::decode_result_type::errors``
   * - ``decoder::decode_result_type::decode_to_obs``
     - ``cudaq::qec::decode_result_type::observables``
   * - ``decoder(H)`` and protected ``H``
     - ``decoder(std::move(inputs), requested_output)`` and
       ``get_inputs().detector_error_matrix()``
   * - protected ``O_sparse`` / ``D_sparse``
     - ``get_inputs().observable_flips_matrix()`` /
       ``get_inputs().measurement_to_detectors()``
   * - ``set_result_type(...)``
     - Choose and validate ``requested_output`` in the constructor; it is
       immutable afterward
   * - ``set_O_sparse(...)`` / ``set_D_sparse(...)``
     - Put ``O`` / ``D`` in ``decoder_init`` before factory construction
   * - ``on_o_sparse_configured()`` / ``on_d_sparse_configured()``
     - Perform model-dependent setup in the derived constructor after the base
       constructor; use ``initialize_streaming_layout(...)`` only for decoder
       layer geometry
   * - ``make_pcm_decoder<DecoderT>(...)``
     - Implement the registered creator with ``decoder_init`` and inspect
       ``inputs.source()`` / ``inputs.has_stim_dem()`` when source-specific
       handling is required

For direct callers, construct ``decoder_init`` with the model data and use the
``get_decoder`` overload taking :cpp:enum:`cudaq::qec::decode_result_type` when
the result basis must be explicit. Python callers use ``output="errors"`` or
``output="observables"``. Python intentionally does not expose
``observables_and_residual_detectors``; composition infrastructure requests
that contract from a compatible decoder. See the PyMatching API for the
migration from its former ``O``-implies-observables behavior.

Implementing a New Decoder in C++
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To implement a new decoder:

1. **Create Decoder Class**:

.. code-block:: cpp

    class my_decoder : public cudaq::qec::decoder {
    public:
        my_decoder(cudaq::qec::decoder_init inputs,
                   cudaq::qec::decode_result_type requested_output,
                   const cudaqx::heterogeneous_map& params)
            : cudaq::qec::decoder(std::move(inputs), requested_output) {
            if (requested_output !=
                cudaq::qec::decode_result_type::errors)
                throw std::invalid_argument(
                    "my_decoder produces error frames only");

            const auto& H = get_inputs().detector_error_matrix();
            // Initialize from H and decoder-specific params.
        }

        cudaq::qec::decoder_result decode(
            const std::vector<cudaq::qec::float_t>& syndrome) override {
            return {/*converged=*/true,
                    std::vector<cudaq::qec::float_t>(block_size, 0.0)};
        }
    };

2. **Register Extension Point**:

.. code-block:: cpp

    class my_decoder : public cudaq::qec::decoder {
        // ... constructor and decode() from above ...

        CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
            my_decoder,
            static std::unique_ptr<cudaq::qec::decoder> create(
                cudaq::qec::decoder_init inputs,
                std::optional<cudaq::qec::decode_result_type> output,
                const cudaqx::heterogeneous_map& params) {
                return std::make_unique<my_decoder>(
                    std::move(inputs),
                    output.value_or(
                        cudaq::qec::decode_result_type::errors),
                    params);
            })
    };

    CUDAQ_EXT_PT_REGISTER_TYPE(my_decoder)

The creator receives an owning ``decoder_init`` handle. Matrix-based plugins
read its sparse matrix projection; DEM-native plugins first require
``has_stim_dem()`` and then read ``stim_dem()``.

Example: Lookup Table Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a simple lookup table decoder for the Steane code:

.. code-block:: cpp

    class single_error_lut : public cudaq::qec::decoder {
    private:
        std::map<std::string, std::size_t> single_qubit_err_signatures;

    public:
        single_error_lut(cudaq::qec::decoder_init inputs,
                         cudaq::qec::decode_result_type requested_output,
                         const cudaqx::heterogeneous_map& params)
            : cudaq::qec::decoder(std::move(inputs), requested_output) {
            if (requested_output !=
                cudaq::qec::decode_result_type::errors)
                throw std::invalid_argument(
                    "single_error_lut produces error frames only");

            const auto& H = get_inputs().detector_error_matrix();
            // Canonicalize before using each sparse column as an error
            // signature so duplicate row indices cancel over GF(2).
            auto H_e2d = H.canonicalize().to_nested_csc();

            for (std::size_t qErr = 0; qErr < block_size; qErr++) {
                std::string err_sig(syndrome_size, '0');
                for (std::uint32_t row : H_e2d[qErr])
                    err_sig[row] = '1';
                single_qubit_err_signatures.insert({err_sig, qErr});
            }
        }

        cudaq::qec::decoder_result decode(
            const std::vector<cudaq::qec::float_t>& syndrome) override {
            cudaq::qec::decoder_result result{false,
                std::vector<cudaq::qec::float_t>(block_size, 0.0)};

            // Convert syndrome to string
            std::string syndrome_str(syndrome_size, '0');
            for (std::size_t i = 0; i < syndrome_size; i++)
                syndrome_str[i] = (syndrome[i] >= 0.5) ? '1' : '0';

            // Lookup error location
            auto it = single_qubit_err_signatures.find(syndrome_str);
            if (it != single_qubit_err_signatures.end()) {
                result.converged = true;
                result.result[it->second] = 1.0;
            }

            return result;
        }

        CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
            single_error_lut,
            static std::unique_ptr<cudaq::qec::decoder> create(
                cudaq::qec::decoder_init inputs,
                std::optional<cudaq::qec::decode_result_type> output,
                const cudaqx::heterogeneous_map& params) {
                return std::make_unique<single_error_lut>(
                    std::move(inputs),
                    output.value_or(
                        cudaq::qec::decode_result_type::errors),
                    params);
            })
    };

    CUDAQ_EXT_PT_REGISTER_TYPE(single_error_lut)

Implementing a Decoder in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-Q QEC supports implementing decoders in Python using the :code:`@qec.decoder` decorator:

1. **Create Decoder Class**:

.. code-block:: python

    @qec.decoder("my_decoder")
    class MyDecoder:
        def __init__(self, H, **kwargs):
            # H is a scipy.sparse matrix or a dense numpy uint8 array,
            # mirroring whatever was passed to qec.get_decoder().
            # Pass it unchanged to Decoder.__init__ so the C++ base class
            # stores a compact sparse representation without a dense allocation.
            qec.Decoder.__init__(self, H)
            self.H = H
            # Initialize with optional kwargs

        def decode(self, syndrome):
            # Create result object
            result = qec.DecoderResult()

            # Implement decoding logic
            # ...

            # Set results
            result.converged = True
            result.result = [0.0] * self.get_block_size()

            return result

2. **Using Custom Parameters**:

.. code-block:: python

    # Create decoder with custom parameters
    decoder = qec.get_decoder("my_decoder",
                            parity_check_matrix,
                            custom_param=42)

Key Features
^^^^^^^^^^^^^

* **Soft Decision Decoding**: Results are probabilities in [0,1]
* **Batch Processing**: Support for decoding multiple syndromes
* **Asynchronous Decoding**: Optional async interface for parallel processing
* **Custom Parameters**: Flexible configuration via heterogeneous_map
* **Python Integration**: First-class support for Python implementations

Usage Example
^^^^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Get a code instance
        steane = qec.get_code("steane")

        # Create decoder with code's parity matrix
        decoder = qec.get_decoder('single_error_lut', steane.get_parity())

        # Run stabilizer measurements
        syndromes, dataQubitResults = qec.sample_memory_circuit(steane, numShots=1, numRounds=1)

        # Decode a syndrome
        result = decoder.decode(syndromes[0])
        if result.converged:
            print("Error locations:",
                [i for i,p in enumerate(result.result) if p > 0.5])
            # No errors as we did not include a noise model and
            # thus prints:
            # Error locations: []

.. tab:: C++

    .. code-block:: cpp

        using namespace cudaq;

        // Get a code instance
        auto code = qec::get_code("steane");

        // Create decoder with code's parity matrix
        auto decoder = qec::get_decoder("single_error_lut",
                                code->get_parity());

        // Run stabilizer measurements
        auto [syndromes, dataQubitResults] = qec::sample_memory_circuit(*code, /*numShots*/ numShots, /*numRounds*/ 1);

        // Copy a single shot syndrome and decode
        std::vector<qec::float_t> syndrome(
            syndromes.data(), syndromes.data() + syndromes.shape()[1]);
        auto result = decoder->decode(syndrome);


.. _detector_error_model:

Detector Error Model
--------------------

A detector error model (DEM) captures how the physical errors in a QEC circuit map to the detectors (syndrome bits) that observe them. CUDA-Q QEC represents it with the ``cudaq.qec.detector_error_model`` type, built from a QEC circuit and a noise model via functions like ``dem_from_memory_circuit()``. For circuit-level noise, the DEM can be put into a canonical form organized by measurement rounds, making it suitable for multi-round decoding.

The parity check matrix a decoder consumes is derived from the DEM: each row is a detector and each column a possible error mechanism. For a runnable example that generates a DEM from a surface code and decodes with it, see the :doc:`Experiments and Noise Modeling </examples_rst/qec/modeling_noise>` example.

.. _decoding_from_stim_dem_text:

Decoding from Stim DEM Text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A DEM does not have to be produced inside CUDA-Q. Decoders can be constructed from either a parity-check matrix or raw Stim detector error model (DEM) text, which is useful when the model is already available in Stim's ``.dem`` format — from a saved file, a Stim workflow, or CUDA-Q DEM generation.

For native decoders, raw DEM text is authoritative: :cpp:class:`cudaq::qec::decoder_init` retains the text and exposes its parsed ``H``, ``O``, and error rates, and custom parameters cannot override those model inputs. Python-registered decoders cannot consume ``decoder_init`` directly, so the binding instead passes the parsed ``H`` and supplies DEM-derived ``O`` and ``error_rate_vec`` keyword defaults when the Python caller did not provide them. By default, ``get_decoder(..., dem_text)`` and ``dem_from_stim_text(dem_text)`` parse with ``use_decomp_suggestions=False`` — Stim ``^`` decomposition hints are ignored and each ``error(...)`` instruction becomes one matrix column; passing ``use_decomp_suggestions=True`` splits ``^``-separated components into separate columns.

For a runnable example, see :ref:`Decoding From Stim DEM Text <stim_dem_text_example>`.

.. _dem_sampling:

DEM Sampling
^^^^^^^^^^^^

The ``dem_sampling`` function samples errors and syndromes from a detector error model, which is useful for generating synthetic syndrome data to exercise a decoder. Given a binary check matrix :math:`H` of shape ``[num_checks x num_error_mechanisms]`` and a vector of per-mechanism Bernoulli probabilities, it generates random error vectors and computes :math:`\text{syndromes} = \text{errors} \cdot H^T \pmod{2}`.

In Python, the ``backend`` parameter (``"auto"``, ``"gpu"``, or ``"cpu"``) controls whether sampling runs on the GPU via cuStabilizer or on the CPU. The function accepts NumPy arrays and PyTorch CUDA tensors. In C++ the CPU and GPU paths live in separate namespaces (``cudaq::qec::dem_sampler::cpu`` and ``cudaq::qec::dem_sampler::gpu``).

For a complete, runnable walkthrough — including GPU acceleration and input-type handling — see the :ref:`DEM Sampling example <dem_sampling_example>`.


.. _dynamic_dem_construction:

Dynamic DEM Construction
^^^^^^^^^^^^^^^^^^^^^^^^^

When a Stim circuit is not available — or when the round count must stay flexible until decoder construction — CUDA-Q QEC can build a detector error model directly from CSS generator matrices instead. ``dem_from_css_matrices`` produces a :math:`T`-round code-capacity or phenomenological DEM from ``css_code_matrices`` (Python ``CssCodes``) and ``css_noise_params`` (Python ``CssNoise``).

For decoders whose round count is chosen at run time, the same model is expressed as composable per-round chunks: ``extended_dem_from_css_matrices`` builds a one-round :math:`\text{ExtendedDem}` chunk, and ``dem_stitch`` / ``dem_close_all`` stitch and close chunks into the same flat DEM. Real-time decoder configs accept this as a YAML ``dem_chunks`` block with ``init`` / ``bulk`` / ``final`` phases, expanded during decoder construction by ``expand_dem_chunks``; omit ``num_rounds`` for streaming decoders.

For a complete, runnable walkthrough — matrix construction, chunk stitching, the YAML ``dem_chunks`` layout, and closing rules — see the :ref:`Dynamic DEM Construction example <dyn_dem_example>`.


.. _prebuilt_qec_decoders:

Pre-built QEC Decoders
----------------------

CUDA-Q QEC provides pre-built decoders for a variety of use cases.

.. list-table::
   :header-rows: 1
   :widths: 20 26 8 8 12 40

   * - Decoder
     - Decoder String Identifier
     - Python
     - C++
     - Realtime Enabled
     - Notes
   * - NVIDIA QLDPC Decoder¹
     - `"nv-qldpc-decoder"`
     - Yes
     - Yes
     - Yes
     - Supports Relay BP and BP+OSD
   * - NVIDIA Fusion Decoder
     - `"nv-fusion-decoder"`
     - Yes
     - Yes
     - Yes
     - Multithreaded MWPM decoder optimized for low-latency streaming
   * - Tensor Network Decoder¹
     - `"tensor_network_decoder"`
     - Yes²
     - No
     - No
     - Exact Maximum Likelihood Decoder
   * - TensorRT Decoder¹
     - `"trt_decoder"`
     - Yes³
     - Yes
     - No
     - AI decoder. Bring your own model.
   * - PyMatching Decoder
     - `"pymatching"`
     - Yes
     - Yes
     - Yes
     - MWPM decoder for matchable codes such as the surface code
   * - Chromobius Decoder
     - `"chromobius"`
     - Yes
     - Yes
     - No
     - Color-code (Möbius) decoder; constructed from Stim DEM text
   * - Look-Up Table Decoder
     - `"single_error_lut"` / `"multi_error_lut"`
     - Yes
     - Yes
     - Yes
     - Simple LUT decoders; ``multi_error_lut`` handles up to ``lut_error_depth`` errors
   * - Sliding Window Decoder
     - `"sliding_window"`
     - Yes
     - Yes
     - No
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

**Belief Propagation Methods:**

The decoder supports several belief-propagation algorithms -- sum-product, min-sum, and memory-based variants -- selected via ``bp_method``, with optional BP+OSD post-processing. For the complete list of methods, parameters, and defaults, see the ``nv-qldpc-decoder`` entries in the :ref:`C++ <nv_qldpc_decoder_api_cpp>` and :ref:`Python <nv_qldpc_decoder_api_python>` API reference.

**Highlighted Features:**

* **Sequential Relay BP** (``composition=1``): chains multiple "relay legs" -- sequential BP runs with different gamma configurations -- to decode syndromes that stall a single BP pass. **Requires:** ``bp_method=3``, ``gamma0``, ``srelay_config``, and either ``gamma_dist`` OR ``explicit_gammas``.
* **Gamma ensembling** (``gamma_ensemble_size``): an extension of Relay BP that explores multiple sets of gamma values in parallel on a single GPU (N independent "lanes"). The first lane to converge lets the decoder exit early, so slow lanes are terminated without adding to the decode time -- narrowing the latency distribution and improving performance on hard-to-decode syndromes that would otherwise stall Relay BP. See :doc:`Improving Relay BP Decoding With Gamma Ensembles </performance/nv_qldpc_gamma_ensemble_user_guide>` for a performance study.

The QLDPC decoder `nv-qldpc-decoder` requires a CUDA-Q compatible GPU. See the `CUDA-Q dependencies and compatibility <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_ list.

The decoder is based on the following references:

* https://arxiv.org/pdf/2005.07016 
* https://github.com/quantumgizmos/ldpc 
* https://arxiv.org/pdf/2506.01779 
* https://github.com/trmue/relay 


Usage:

.. tab:: Python

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

.. tab:: C++

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

For a runnable example, see :ref:`Getting Started with the NVIDIA QLDPC Decoder <qldpc_decoder_example>`.

NVIDIA Fusion Decoder
^^^^^^^^^^^^^^^^^^^^^

The ``nv-fusion-decoder`` is a multithreaded minimum-weight perfect matching
(MWPM) decoder for graphlike error models. It partitions detector data into
temporal blocks and fuses their solutions to reduce latency in streaming,
realtime decoding environments. It also supports offline batch decoding.

See the :ref:`Python <nv_fusion_decoder_api_python>` and
:ref:`C++ <nv_fusion_decoder_api_cpp>` API references for configuration and
usage details. For benchmark results and reproduction instructions, see
:doc:`NV-Fusion Decoder Latency </performance/nv_fusion_latency_user_guide>`.

Tensor Network Decoder
^^^^^^^^^^^^^^^^^^^^^^

The ``tensor_network_decoder`` constructs a tensor network representation of a quantum code given its parity check matrix, logical observable(s), and noise model. It can decode individual syndromes or batches of syndromes, returning the probability that a logical observable has flipped.

Due to the additional dependencies of the Tensor Network Decoder, you must
specify the optional pip package when installing CUDA-Q QEC in order to use this
decoder. Use `pip install cudaq-qec[tensor-network-decoder]` in order to use
this decoder.

Key Steps:

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

For a runnable example, see :ref:`Exact Maximum Likelihood Decoding with NVIDIA Tensor Network Decoder <tensor_network_decoder_example>`.


TensorRT Decoder
^^^^^^^^^^^^^^^^

The ``trt_decoder`` deploys a trained neural-network decoder (an ONNX model) through NVIDIA TensorRT for optimized GPU inference. Unlike the algorithmic decoders, it is trained on a specific code and noise model — you bring your own model. Python use requires ``pip install cudaq-qec[trt-decoder]``. See the :ref:`TensorRT Decoder API <trt_decoder_api_python>` for configuration options, and the :ref:`Deploying AI Decoders with TensorRT example <deploying-ai-decoders>` for the full train-to-deploy workflow.

PyMatching Decoder
^^^^^^^^^^^^^^^^^^

The ``pymatching`` decoder is a minimum-weight perfect matching (MWPM) decoder built on the open-source `PyMatching <https://github.com/oscarhiggott/PyMatching>`_ library, suitable for matchable codes such as the surface code. It is selected by name through ``get_decoder`` and takes a parity-check matrix whose columns each have one or two set entries; per-edge priors are supplied via ``error_rate_vec``. See the :ref:`PyMatching Decoder API <pymatching_decoder_api_python>` and the :ref:`Matching-Based Decoding with PyMatching example <pymatching_decoder_example>`.

Chromobius Decoder
^^^^^^^^^^^^^^^^^^

The ``chromobius`` decoder is a color-code decoder built on the open-source `Chromobius <https://github.com/quantumlib/chromobius>`_ Möbius decoder. Unlike the matrix-based decoders, it is constructed from Stim detector-error-model (DEM) text rather than a parity-check matrix, and predicts logical observable flips directly. See the :ref:`Chromobius Decoder API <chromobius_decoder_api_python>` and the :ref:`Color-Code Decoding with Chromobius example <chromobius_decoder_example>`.

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

Key Steps:

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
   behavior are in :doc:`/api/qec/python_api` and :doc:`/api/qec/cpp_api`.

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

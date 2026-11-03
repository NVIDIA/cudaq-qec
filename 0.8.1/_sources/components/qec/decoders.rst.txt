QEC Decoders
============

The ``cudaq-qec`` decoder interface (:code:`cudaq::qec::decoder`) turns syndromes into corrections. This page covers the framework — the class structure and how to implement a decoder — together with a catalog of the decoders that ship with the library. Read it to choose a built-in decoder or to write your own. For runnable programs, see the :doc:`Decoders examples </examples_rst/qec/decoders>`.

Decoder Framework :code:`cudaq::qec::decoder`
----------------------------------------------

The CUDA-Q QEC decoder framework provides an extensible system for implementing
quantum error correction decoders through the :code:`cudaq::qec::decoder` base class.

Class Structure
^^^^^^^^^^^^^^^

The decoder base class defines the core interface for syndrome decoding:

.. code-block:: cpp

    class decoder {
    protected:
        std::size_t block_size;       // For [n,k] code, this is n
        std::size_t syndrome_size;    // For [n,k] code, this is n-k
        sparse_binary_matrix H;       // Parity check matrix

    public:
        struct decoder_result {
            bool converged;                 // Decoder convergence status
            std::vector<float_t> result;    // Soft error probabilities
        };

        virtual decoder_result decode(
            const std::vector<float_t>& syndrome) = 0;

        virtual std::vector<decoder_result> decode_batch(
            const std::vector<std::vector<float_t>>& syndrome);
    };

Key Components:

* **Parity Check Matrix**: Defines the code structure via the sparse :code:`H` member
* **Block Size**: Number of physical qubits in the code
* **Syndrome Size**: Number of stabilizer measurements
* **Decoder Result**: Contains convergence status and error probabilities
* **Multiple Decoding Modes**: Single syndrome or batch processing

Implementing a New Decoder in C++
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To implement a new decoder:

1. **Create Decoder Class**:

.. code-block:: cpp

    class my_decoder : public qec::decoder {
    private:
        // Decoder-specific members

    public:
        my_decoder(const qec::sparse_binary_matrix& H,
                  const cudaqx::heterogeneous_map& params)
            : decoder(H) {
            // Initialize decoder
        }

        decoder_result decode(
            const std::vector<float_t>& syndrome) override {
            // Implement decoding logic
        }
    };

2. **Register Extension Point**:

.. code-block:: cpp

    class my_decoder : public qec::decoder {
        // ... constructor and decode() from above ...

        CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
            my_decoder,
            static std::unique_ptr<decoder> create(
                const qec::decoder_init& init,
                const cudaqx::heterogeneous_map& params) {
                return qec::make_pcm_decoder<my_decoder>(init, params);
            })
    };

    CUDAQ_EXT_PT_REGISTER_TYPE(my_decoder)

The :code:`make_pcm_decoder` helper dispatches :code:`decoder_init`. It
passes a stored sparse PCM directly to the decoder constructor; when the
variant contains Stim DEM text, it parses the DEM and constructs the sparse
detector matrix before invoking the same constructor.

Example: Lookup Table Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a simple lookup table decoder for the Steane code:

.. code-block:: cpp

    class single_error_lut : public decoder {
    private:
        std::map<std::string, std::size_t> single_qubit_err_signatures;

    public:
        single_error_lut(const qec::sparse_binary_matrix& H,
                          const cudaqx::heterogeneous_map& params)
            : decoder(H) {
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

        decoder_result decode(
            const std::vector<float_t>& syndrome) override {
            decoder_result result{false,
                std::vector<float_t>(block_size, 0.0)};

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
    };

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

For PCM-based decoders, CUDA-Q QEC parses the DEM text into a detector error matrix and supplies DEM-derived ``O`` and ``error_rate_vec`` defaults when the user does not provide them. C++ decoder plugins that need full Stim DEM metadata can consume the raw DEM string from the decoder construction input. By default, ``get_decoder(..., dem_text)`` and ``dem_from_stim_text(dem_text)`` parse with ``use_decomp_suggestions=False`` — Stim ``^`` decomposition hints are ignored and each ``error(...)`` instruction becomes one matrix column; passing ``use_decomp_suggestions=True`` splits ``^``-separated components into separate columns.

For a runnable example, see :ref:`Decoding From Stim DEM Text <stim_dem_text_example>`.

.. _dem_sampling:

DEM Sampling
^^^^^^^^^^^^

The ``dem_sampling`` function samples errors and syndromes from a detector error model, which is useful for generating synthetic syndrome data to exercise a decoder. Given a binary check matrix :math:`H` of shape ``[num_checks x num_error_mechanisms]`` and a vector of per-mechanism Bernoulli probabilities, it generates random error vectors and computes :math:`\text{syndromes} = \text{errors} \cdot H^T \pmod{2}`.

In Python, the ``backend`` parameter (``"auto"``, ``"gpu"``, or ``"cpu"``) controls whether sampling runs on the GPU via cuStabilizer or on the CPU. The function accepts NumPy arrays and PyTorch CUDA tensors. In C++ the CPU and GPU paths live in separate namespaces (``cudaq::qec::dem_sampler::cpu`` and ``cudaq::qec::dem_sampler::gpu``).

For a complete, runnable walkthrough — including GPU acceleration and input-type handling — see the :ref:`DEM Sampling example <dem_sampling_example>`.


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
                {"error_rate_vec", dem.error_rates},
                {"window_size", 1},
                {"num_syndromes_per_round", code->get_num_z_stabilizers() + code->get_num_x_stabilizers()},
                {"num_boundary_syndromes", code->get_num_z_stabilizers()},
                {"inner_decoder_name", "nv-qldpc-decoder"},
                {"inner_decoder_params", inner_decoder_params}};
            auto swdec = cudaq::qec::get_decoder("sliding_window",
                                                 dem.detector_error_matrix, opts);
            return 0;
        }

Output:

Once a decode step completes, results use the same types as other pre-built
decoders (:class:`cudaq_qec.Decoder` in Python, :cpp:class:`cudaq::qec::decoder`
in C++).


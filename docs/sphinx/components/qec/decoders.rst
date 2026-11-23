:tocdepth: 3

QEC Decoders
============

Decoders turn syndromes into corrections. Choose a pre-built decoder below,
or use the CUDA-Q QEC decoder framework to implement your own in C++ or
Python. For runnable programs, see the
:doc:`Decoders </examples_rst/qec/decoders>` examples.

.. _prebuilt_decoders:

Pre-built Decoders
------------------

Use QEC library decoders with the CUDA-Q QEC framework, or install a
standalone decoder package for Python decoding without CUDA-Q.

.. toctree::
   :maxdepth: 1

   qec_library_decoders
   standalone_decoders

Build Your Own Decoder
-----------------------

To add a decoding algorithm, implement the CUDA-Q QEC decoder interface and
register your implementation. The following sections describe the framework,
show C++ and Python implementations, and cover migration of existing C++ plugins.

Decoder Framework :code:`cudaq::qec::decoder`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CUDA-Q QEC decoder framework provides an extensible system for implementing
quantum error correction decoders through the :code:`cudaq::qec::decoder` base class.

Class Structure
"""""""""""""""

The decoder base owns immutable model data and fixes the primary and auxiliary
output contract when an instance is constructed. The relevant interface is:

.. code-block:: cpp

    class decoder {
    public:
        decoder(decoder_init inputs,
                decoder_output_request requested_output =
                    decoder_output_request{decode_result_type::errors});

        virtual decoder_result decode(
            const std::vector<float_t>& syndrome) = 0;

        virtual std::vector<decoder_result> decode_batch(
            const std::vector<std::vector<float_t>>& syndrome);

        decode_result_type get_result_type() const noexcept;
        decoder_auxiliary_result_type
        get_auxiliary_result_type() const noexcept;

    protected:
        const decoder_init& get_inputs() const noexcept;
        std::size_t block_size;
        std::size_t syndrome_size;
    };

Key components:

* **Construction inputs**: :code:`decoder_init` owns ``H`` and optional ``O``,
  ``D``, error rates, and authoritative Stim DEM text
* **Fixed output contract**: An instance returns either error frames or
  observable flips as its primary result and may return requested standardized
  auxiliary outputs
* **Block size**: Number of modeled error mechanisms (columns of ``H``)
* **Syndrome size**: Number of detector values (rows of ``H``)
* **Decoder result**: Contains convergence status, primary values in the
  configured error or observable basis, requested standardized auxiliary
  outputs in ``opt_results``, and optional decoder-specific metadata
* **Multiple decoding modes**: Single syndrome or batch processing

.. _migrating-existing-c-decoder-plugins:
.. _qec_decoder_plugin_migration:

Migrating C++ Decoder Plugins from 0.8.x to 0.9.0
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

CUDA-Q QEC 0.9.0 changes the C++ decoder construction API. Developers of
out-of-tree C++ plugins must update their source for the new interface.
Python users do not need to implement these C++ interfaces. However, Python
code that relied on PyMatching selecting observable output when ``O`` was
supplied must now pass ``output="observables"`` explicitly; see the
:ref:`PyMatching API <pymatching_decoder_api_python>`.

To migrate a C++ plugin:

* Replace constructors that take ``H`` with constructors that take
  :cpp:class:`cudaq::qec::decoder_init` by value and
  ``decoder_output_request``. Forward both to the base constructor as
  ``decoder(std::move(inputs), request)``.
* Read model data through ``get_inputs()``. ``H``, ``O``, ``D``, and
  ``error_rate_vec`` are framework model data and must not be passed in the
  heterogeneous custom-parameter map or registered as plugin schema keys.
* Change custom creator signatures to accept ``decoder_init`` by value,
  ``std::optional<cudaq::qec::decoder_output_request>``, and the custom
  parameter map. The removed ``make_pcm_decoder`` helper must not be used.
* Choose the plugin's default primary output when the optional request is
  empty, and validate explicit requests using ``request.primary``. The
  presence of ``O`` does not select observable output.
* Reject unsupported auxiliary requests during construction, for example with
  ``if (request.has_auxiliary_outputs()) throw std::invalid_argument(...)``.
  A plugin that accepts ``residual_detectors`` must return them in
  ``decoder_result::opt_results["residual_detectors"]``.
* Remove calls to the deleted ``set_O_sparse`` and ``set_D_sparse`` methods.
  The base constructor now derives the corresponding model state and buffer
  sizes. A streaming decoder supplies only its layer geometry through
  ``initialize_streaming_layout`` during construction.
* Override the cancellable ``decode`` and ``decode_batch`` overloads only if
  the decoder can honor a stop; the defaults ignore the token.

The following table maps the 0.8.x API to its 0.9.0 replacement:

.. _qec_decoder_plugin_api_migration:

.. list-table:: Decoder plugin API migration
   :header-rows: 1

   * - 0.8.x API
     - 0.9.0 replacement
   * - ``decoder::decode_result_type::decode_to_errs``
     - ``cudaq::qec::decode_result_type::errors``
   * - ``decoder::decode_result_type::decode_to_obs``
     - ``cudaq::qec::decode_result_type::observables``
   * - Creator arguments ``const decoder_init &`` and the parameter map
     - ``decoder_init`` by value, ``std::optional<decoder_output_request>``,
       and the parameter map
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

For direct callers, construct ``decoder_init`` with the model data and pass a
:cpp:enum:`cudaq::qec::decode_result_type` to ``get_decoder`` when the result
basis must be explicit. It converts to a ``decoder_output_request`` with no
auxiliary outputs. Python callers use ``output="errors"`` or
``output="observables"``. Python intentionally exposes only the primary result
basis; composition infrastructure uses ``decoder_output_request`` to request
standardized auxiliary outputs from a compatible decoder. See the PyMatching
API for the migration from its former ``O``-implies-observables behavior.

.. _qec-custom-decoder:

Implementing a New Decoder in C++
""""""""""""""""""""""""""""""""""

To implement a new decoder:

1. **Create decoder class**:

.. code-block:: cpp

    class my_decoder : public cudaq::qec::decoder {
    public:
        my_decoder(cudaq::qec::decoder_init inputs,
                   cudaq::qec::decoder_output_request request,
                   const cudaqx::heterogeneous_map& params)
            : cudaq::qec::decoder(std::move(inputs), request) {
            if (request.has_auxiliary_outputs())
                throw std::invalid_argument(
                    "my_decoder does not produce auxiliary outputs");
            if (request.primary !=
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

2. **Register extension point**:

.. code-block:: cpp

    class my_decoder : public cudaq::qec::decoder {
        // ... constructor and decode() from above ...

        CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
            my_decoder,
            static std::unique_ptr<cudaq::qec::decoder> create(
                cudaq::qec::decoder_init inputs,
                std::optional<cudaq::qec::decoder_output_request> output,
                const cudaqx::heterogeneous_map& params) {
                const auto request = output.value_or(
                    cudaq::qec::decoder_output_request{
                        cudaq::qec::decode_result_type::errors});
                return std::make_unique<my_decoder>(
                    std::move(inputs), request, params);
            })
    };

    CUDAQ_EXT_PT_REGISTER_TYPE(my_decoder)

The creator receives an owning ``decoder_init`` handle. Matrix-based plugins
read its sparse matrix projection; DEM-native plugins first require
``has_stim_dem()`` and then read ``stim_dem()``.

Example: Lookup Table Decoder
""""""""""""""""""""""""""""""

Here's a simple lookup table decoder for the Steane code:

.. code-block:: cpp

    class single_error_lut : public cudaq::qec::decoder {
    private:
        std::map<std::string, std::size_t> single_qubit_err_signatures;

    public:
        single_error_lut(cudaq::qec::decoder_init inputs,
                         cudaq::qec::decoder_output_request request,
                         const cudaqx::heterogeneous_map& params)
            : cudaq::qec::decoder(std::move(inputs), request) {
            if (request.has_auxiliary_outputs())
                throw std::invalid_argument(
                    "single_error_lut does not produce auxiliary outputs");
            if (request.primary !=
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
                std::optional<cudaq::qec::decoder_output_request> output,
                const cudaqx::heterogeneous_map& params) {
                const auto request = output.value_or(
                    cudaq::qec::decoder_output_request{
                        cudaq::qec::decode_result_type::errors});
                return std::make_unique<single_error_lut>(
                    std::move(inputs), request, params);
            })
    };

    CUDAQ_EXT_PT_REGISTER_TYPE(single_error_lut)

Implementing a Decoder in Python
""""""""""""""""""""""""""""""""

CUDA-Q QEC supports implementing decoders in Python using the :code:`@qec.decoder` decorator:

1. **Create decoder class**:

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

2. **Using custom parameters**:

.. code-block:: python

    # Create decoder with custom parameters
    decoder = qec.get_decoder("my_decoder",
                            parity_check_matrix,
                            custom_param=42)

Key Features
"""""""""""""

* **Soft decision decoding**: Results are probabilities in [0,1]
* **Batch processing**: Support for decoding multiple syndromes
* **Asynchronous decoding**: Optional async interface for parallel processing
* **Custom parameters**: Flexible configuration via heterogeneous_map
* **Python integration**: First-class support for Python implementations

Usage Example
""""""""""""""

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

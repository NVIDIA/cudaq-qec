The Realtime Decoding API enables low-latency error correction on quantum hardware by allowing CUDA-Q quantum kernels to interact with decoders during circuit execution. This API is designed for use cases where corrections must be calculated and applied within qubit coherence times.

The real-time decoding system supports simulation environments for local testing and hardware integration (e.g., on
`Quantinuum's Helios QPU
<https://www.quantinuum.com/products-solutions/quantinuum-systems/helios>`_).

Core Decoding Functions
------------------------

These functions can be called from within CUDA-Q quantum kernels (``@cudaq.kernel`` decorated functions) to interact with real-time decoders.

.. py:function:: cudaq_qec.qec.enqueue_syndromes(decoder_id, syndromes, tag=0)

   Enqueue syndrome measurements for decoding.

   :param decoder_id: Unique identifier for the decoder instance (matches configured decoder ID)
   :param syndromes: List of syndrome measurement results from stabilizer measurements
   :param tag: Optional tag for logging and debugging (default: 0)

   **Example:**

   .. code-block:: python

      import cudaq
      import cudaq_qec as qec
      from cudaq_qec import patch

      @cudaq.kernel
      def measure_and_decode(logical: patch, decoder_id: int):
          syndromes = measure_stabilizers(logical)
          qec.enqueue_syndromes(decoder_id, syndromes, 0)

.. py:function:: cudaq_qec.qec.get_corrections(decoder_id, return_size, reset=False)

   Retrieve calculated corrections from the decoder.

   :param decoder_id: Unique identifier for the decoder instance
   :param return_size: Number of correction bits to return (typically equals number of logical observables)
   :param reset: Whether to reset accumulated corrections after retrieval (default: False)
   :returns: List of boolean values indicating detected bit flips for each logical observable

   **Example:**

   .. code-block:: python

      @cudaq.kernel
      def apply_corrections(logical: patch, decoder_id: int):
          corrections = qec.get_corrections(decoder_id, 1, False)
          if corrections[0]:
              x(logical.data)  # Apply transversal X correction

.. py:function:: cudaq_qec.qec.reset_decoder(decoder_id)

   Reset decoder state, clearing all queued syndromes and accumulated corrections.

   :param decoder_id: Unique identifier for the decoder instance to reset

   **Example:**

   .. code-block:: python

      @cudaq.kernel
      def run_experiment(decoder_id: int):
          qec.reset_decoder(decoder_id)  # Reset at start of each shot
          # ... perform experiment ...

Configuration API
-----------------

The configuration API enables setting up decoders before circuit execution. Decoders are configured using YAML files or programmatically constructed configuration objects.

Decoder Parameters
^^^^^^^^^^^^^^^^^^

Decoder-specific parameters (``decoder_config.decoder_custom_args``) are
plain dicts. The set of accepted keys, their types, and which are required
are defined by the *parameter schema* each decoder registers -- including
out-of-tree decoder plugins. Use ``cudaq_qec.decoder_param_schema(name)`` to
inspect a decoder's parameters and ``cudaq_qec.registered_decoder_schemas()``
to list all decoders with registered schemas.

For example, model error rates may be assigned to the top-level
``decoder_config.error_rate_vec`` field, while the ``pymatching`` decoder's
custom ``merge_strategy`` is one of ``"disallow"``, ``"independent"``,
``"smallest_weight"``, ``"keep_original"``, or ``"replace"``. Emitted YAML
retains the established spelling under ``decoder_custom_args``. Input YAML
accepts either location, but rejects a configuration that supplies both:

.. code-block:: python

   config.type = "pymatching"
   config.error_rate_vec = [0.1, 0.1, 0.1]
   config.decoder_custom_args = {
       "merge_strategy": "smallest_weight",
   }

The corresponding YAML keeps the error rates in the compatibility location:

.. code-block:: yaml

   decoder_custom_args:
     error_rate_vec: [0.1, 0.1, 0.1]
     merge_strategy: smallest_weight

The ``trt_decoder`` accepts ``onnx_load_path`` or ``engine_load_path``
(mutually exclusive), required ``engine_output_format`` (``"errors"``,
``"residual_detectors"``, ``"observables"``, or
``"observables_and_residual_detectors"``), ``engine_save_path``, ``precision``
(``"best"``, ``"tf32"``, or ``"noTF32"``; legacy ``"fp16"``, ``"bf16"``,
``"int8"``, and ``"fp8"`` values are accepted but ignored for strongly typed
networks),
``memory_workspace`` (bytes), ``batch_size``, ``use_cuda_graph``, and an
optional global decoder attached via ``global_decoder`` plus
``global_decoder_params`` (a nested dict whose keys follow the schema of the
named global decoder).

.. py:function:: cudaq_qec.decoder_param_schema(decoder_name)

   Return the registered parameter schema for a decoder as a list of
   descriptors (``key``, ``kind``, ``required``, and, for nested sections,
   ``subschema`` or ``discriminator``), or ``None`` when the decoder has not
   registered one.

.. py:function:: cudaq_qec.registered_decoder_schemas()

   Names of all decoders (and nested parameter sections) with registered
   parameter schemas.

.. py:function:: cudaq_qec.decoder_config_json_schema()

   Return a JSON Schema (draft 2020-12) document, as a string, that
   validates ``multi_decoder_config`` YAML files. Generated from the decoder
   parameter schemas registered in this installation (including loaded
   third-party decoder plugins), for use with standard tools such as
   ``check-jsonschema``, the python ``jsonschema`` package, or editor YAML
   language servers. Schema validation hooks are not representable in JSON
   Schema, so a passing document may still be rejected when parsed.

.. py:method:: decoder_config.validate_custom_args()

   Validate ``decoder_custom_args`` against the parameter schema registered
   for this decoder ``type``: unknown keys, missing required keys, and the
   schema's own validation hook. Raises ``RuntimeError`` on the first
   violation. YAML parsing applies the same checks automatically; call this
   to vet a configuration built programmatically before using it. Also
   available on ``multi_decoder_config`` to validate every decoder at once.

Deprecated Typed Configuration Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The typed configuration classes from earlier releases
(``nv_qldpc_decoder_config``, ``trt_decoder_config``, ``pymatching_config``,
``chromobius_config``, ``multi_error_lut_config``, and the
``qecrt.config``-level ``single_error_lut_config``, ``sliding_window_config``,
and ``srelay_bp_config``) remain available as deprecated compatibility shims.
They emit a ``DeprecationWarning`` on construction and will be removed in a
future release; existing code that builds one and passes it to
``decoder_config.set_decoder_custom_args`` (or assigns it to
``decoder_config.decoder_custom_args``) continues to work unchanged. Note
that *reading* ``decoder_custom_args`` now always returns a plain dict, never
a typed object. New code should assign dicts directly, as shown above.

Configuration Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: cudaq_qec.configure_decoders(config)

   Configure decoders from a multi_decoder_config object.

   :param config: multi_decoder_config object containing decoder specifications
   :returns: 0 on success, non-zero error code on failure

.. py:function:: cudaq_qec.configure_decoders_from_file(config_file)

   Configure decoders from a YAML file.

   :param config_file: Path to YAML configuration file
   :returns: 0 on success, non-zero error code on failure

.. py:function:: cudaq_qec.configure_decoders_from_str(config_str)

   Configure decoders from a YAML string.

   :param config_str: YAML configuration as a string
   :returns: 0 on success, non-zero error code on failure

.. py:function:: cudaq_qec.finalize_decoders()

   Finalize and clean up decoder resources. Should be called before program exit.

Helper Functions
----------------

Realtime decoding requires converting matrices to sparse format for efficient decoder configuration. The following utility functions are essential:

.. py:function:: cudaq_qec.pcm_to_sparse_vec(pcm)

   Convert a parity check matrix (PCM) to sparse vector representation for decoder configuration.

   :param pcm: Dense binary matrix as numpy array (e.g., ``dem.detector_error_matrix`` or ``dem.observables_flips_matrix``)
   :returns: Sparse vector (list of integers) where -1 separates rows

   **Usage in real-time decoding:**

   .. code-block:: python

      config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
      config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)

.. py:function:: cudaq_qec.pcm_from_sparse_vec(sparse_vec, num_rows, num_cols)

   Convert sparse vector representation back to a dense parity check matrix.

   :param sparse_vec: Sparse representation (from YAML or decoder config)
   :param num_rows: Number of rows in the output matrix
   :param num_cols: Number of columns in the output matrix
   :returns: Dense binary matrix as numpy array

.. py:function:: cudaq_qec.d_sparse(m2d)

   Flatten a measurement-to-detector map into the ``-1``-terminated sparse vector a
   realtime decoder config expects for its ``D_sparse``.

   :param m2d: List of lists of measurement indices. ``m2d[d]`` contains the measurement
               indices whose XOR forms detector ``d``. Obtain this from the second element
               of the tuple returned by :meth:`DecoderContext.x_component`,
               :meth:`DecoderContext.z_component`, or :meth:`DecoderContext.full_component`.
   :returns: ``-1``-terminated sparse vector suitable for ``decoder_config.D_sparse``

   **Usage in real-time decoding:**

   .. code-block:: python

      ctx = qec.decoder_context_from_memory_circuit(code, statePrep, num_rounds, noise)
      dem, m2d, m2o = ctx.z_component()  # or x_component() / full_component()
      config.D_sparse = qec.d_sparse(m2d)

See also :ref:`parity_check_matrix_utilities_python` for additional PCM manipulation functions.

The Realtime Decoding API enables low-latency error correction on quantum hardware by allowing CUDA-Q quantum kernels to interact with decoders during circuit execution. This API is designed for use cases where corrections must be calculated and applied within qubit coherence times.

The real-time decoding system supports simulation environments for local testing and hardware integration (e.g., on
`Quantinuum's Helios QPU
<https://www.quantinuum.com/products-solutions/quantinuum-systems/helios>`_).

Core Decoding Functions
------------------------

These functions can be called from within CUDA-Q quantum kernels (``__qpu__`` functions) to interact with real-time decoders.

.. doxygenfunction:: cudaq::qec::decoding::enqueue_syndromes
.. doxygenfunction:: cudaq::qec::decoding::get_corrections
.. doxygenfunction:: cudaq::qec::decoding::reset_decoder


Configuration API
-----------------

The configuration API enables setting up decoders before circuit execution. Decoders are configured using YAML files or programmatically constructed configuration objects.

.. doxygenstruct:: cudaq::qec::realtime::graph_resources
   :members:

.. doxygenenum:: cudaq::qec::decoding::config::DecoderDispatch

.. doxygenstruct:: cudaq::qec::decoding::config::decoder_config
   :members:

.. doxygenstruct:: cudaq::qec::decoding::config::transport_shape_override
   :members:

.. doxygenstruct:: cudaq::qec::decoding::config::transport_config
   :members:

.. doxygenclass:: cudaq::qec::decoding::config::multi_decoder_config
   :members:

.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders(multi_decoder_config &)
.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders(multi_decoder_config &, const std::filesystem::path &)
.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders_from_file
.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders_from_str
.. doxygenfunction:: cudaq::qec::decoding::config::finalize_decoders
.. doxygenfunction:: cudaq::qec::decoding::config::expand_dem_chunks

Helper Functions
----------------

Realtime decoding requires converting matrices to sparse format for efficient decoder configuration. The following utility functions are essential:

- :cpp:func:`cudaq::qec::pcm_to_sparse_vec` for converting a dense PCM to a sparse PCM.
- :cpp:func:`cudaq::qec::pcm_from_sparse_vec` for converting a sparse PCM to a dense PCM.
- :cpp:func:`cudaq::qec::d_sparse` for converting an ``M2DSparseMatrix`` (obtained from
  a :cpp:struct:`cudaq::qec::decoder_inputs` component) into the ``-1``-terminated sparse
  vector a decoder config expects for ``D_sparse``.

   **Usage in real-time decoding:**

   .. code-block:: cpp

      auto ctx = cudaq::qec::decoder_context_from_memory_circuit(
          code, statePrep, numRounds, noise);
      auto inputs = ctx.z_component(); // or x_component() / full_component()
      config.H_sparse = cudaq::qec::pcm_to_sparse_vec(inputs.dem.detector_error_matrix);
      config.O_sparse = cudaq::qec::pcm_to_sparse_vec(inputs.dem.observables_flips_matrix);
      config.D_sparse = cudaq::qec::d_sparse(inputs.m2d);

See also :ref:`parity_check_matrix_utilities` for additional PCM manipulation functions.

CUDA-Q QEC C++ API
******************************

Code
=============

.. doxygenclass:: cudaq::qec::code
    :members:

.. doxygenstruct:: cudaq::qec::patch
    :members:

.. doxygenclass:: cudaq::qec::repetition::repetition
    :members:

.. doxygenclass:: cudaq::qec::steane::steane
    :members:

.. _qec_stabilizer_grid_cpp:

.. doxygenenum:: cudaq::qec::surface_code::surface_role
.. doxygenenum:: cudaq::qec::surface_code::sc_orientation
.. doxygenstruct:: cudaq::qec::surface_code::vec2d
   :members:

.. doxygenclass:: cudaq::qec::surface_code::stabilizer_grid
    :members:

.. doxygenclass:: cudaq::qec::surface_code::surface_code
    :members:

Detector Error Model
====================

.. doxygenstruct:: cudaq::qec::detector_error_model
    :members:

.. doxygenstruct:: cudaq::qec::decoder_context
    :members:

.. doxygenfunction:: cudaq::qec::dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::x_dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::z_dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::decoder_context_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::dem_from_stim_text(const std::string &, bool)

.. _dyn_dem_cpp_api:

Dynamic DEM Construction
========================

Build a code-capacity or phenomenological DEM from CSS generator matrices
(no Stim circuit required), or compose per-round DEM chunks that can be
stitched and closed into a flat :cpp:struct:`cudaq::qec::detector_error_model`.
See :doc:`/examples_rst/qec/dyn_dem` for a walkthrough.

CSS matrices and noise
----------------------

.. doxygenstruct:: cudaq::qec::css_code_matrices
    :members:

.. doxygenstruct:: cudaq::qec::css_noise_params
    :members:

.. doxygenfunction:: cudaq::qec::css_matrices_from_code
.. doxygenfunction:: cudaq::qec::dem_from_css_matrices(const css_code_matrices &, const css_noise_params &, std::size_t)
.. doxygenfunction:: cudaq::qec::dem_from_css_matrices(const code &, const css_noise_params &, std::size_t)

Extended DEM chunks
-------------------

Seam and phase identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: cudaq::qec::seam_id
    :members:

.. doxygentypedef:: cudaq::qec::phase_id

.. doxygennamespace:: cudaq::qec::seam_name
    :members:

.. doxygennamespace:: cudaq::qec::phase_name
    :members:

Chunk types
~~~~~~~~~~~

.. doxygenstruct:: cudaq::qec::extended_dem
    :members:

.. doxygenfunction:: cudaq::qec::extended_dem_from_css_matrices

.. doxygenstruct:: cudaq::qec::dem_seam_spec
    :members:

.. doxygenstruct:: cudaq::qec::seam_spec_entry
    :members:

.. doxygenstruct:: cudaq::qec::dem_chunk_spec
    :members:

.. doxygenstruct:: cudaq::qec::seam_connection
    :members:

.. doxygenstruct:: cudaq::qec::phase_connection
    :members:

.. doxygenstruct:: cudaq::qec::phase_spec_entry
    :members:

.. doxygenstruct:: cudaq::qec::dem_chunks_spec
    :members:

.. doxygenfunction:: cudaq::qec::dem_chunk_from_spec
.. doxygenfunction:: cudaq::qec::dem_chunks_from_spec

Stitch, close, and merge
------------------------

.. doxygenenum:: cudaq::qec::prior_combine_mode

.. doxygenfunction:: cudaq::qec::dem_stitch
.. doxygenfunction:: cudaq::qec::dem_stitch_all
.. doxygenfunction:: cudaq::qec::dem_stitch_merged
.. doxygenfunction:: cudaq::qec::dem_close
.. doxygenfunction:: cudaq::qec::dem_close_all
.. doxygenfunction:: cudaq::qec::dem_merge_duplicate_columns
.. doxygenfunction:: cudaq::qec::are_dem_columns_unique
.. doxygenfunction:: cudaq::qec::assert_dem_columns_unique

Streaming decoder maps
----------------------

.. doxygenfunction:: cudaq::qec::dem_chunk_rounds
.. doxygenfunction:: cudaq::qec::dem_chunks_to_rounds
.. doxygenfunction:: cudaq::qec::dem_chunks_to_detector_round
.. doxygenfunction:: cudaq::qec::dem_chunks_to_d_sparse
.. doxygenfunction:: cudaq::qec::dem_chunks_to_o_sparse
.. doxygenfunction:: cudaq::qec::dem_chunks_to_pcm

.. _dem_sampling_cpp_api:

Detector Error Model (DEM) Sampling
===================================

CPU sampling (host tensors):

.. doxygenfunction:: cudaq::qec::dem_sampler::cpu::sample_dem(const cudaqx::tensor<uint8_t> &, std::size_t, const std::vector<double> &)
.. doxygenfunction:: cudaq::qec::dem_sampler::cpu::sample_dem(const cudaqx::tensor<uint8_t> &, std::size_t, const std::vector<double> &, unsigned)

GPU sampling (device pointers, requires cuStabilizer):

.. doxygenfunction:: cudaq::qec::dem_sampler::gpu::sample_dem

Legacy convenience wrappers (delegate to ``cpu::sample_dem``; prefer the
``dem_sampler::cpu::sample_dem`` overloads above):

.. doxygenfunction:: cudaq::qec::dem_sampling(const cudaqx::tensor<uint8_t> &, std::size_t, const std::vector<double> &)
.. doxygenfunction:: cudaq::qec::dem_sampling(const cudaqx::tensor<uint8_t> &, std::size_t, const std::vector<double> &, unsigned)

Decoder Interfaces
==================

.. doxygenenum:: cudaq::qec::decode_result_type

.. doxygenenum:: cudaq::qec::decoder_auxiliary_result_type

.. doxygenstruct:: cudaq::qec::decoder_output_request
   :members:

.. doxygenenum:: cudaq::qec::decoder_model_source

.. doxygenstruct:: cudaq::qec::decoder_inputs
    :members:

.. doxygenclass:: cudaq::qec::decoder_init
    :members:

.. doxygenfunction:: cudaq::qec::d_sparse(const cudaq::M2DSparseMatrix &)

.. doxygenenum:: cudaq::qec::cancellation_level

.. doxygenclass:: cudaq::qec::cancellation_token
    :members:

.. doxygenclass:: cudaq::qec::cancellation_source
    :members:

.. doxygenclass:: cudaq::qec::decoder
    :members:

.. doxygenstruct:: cudaq::qec::decoder_result
    :members:

Built-in Decoders
=================

.. _nv_qldpc_decoder_api_cpp:

NVIDIA QLDPC Decoder
--------------------

.. include:: nv_qldpc_decoder_api.rst

.. _nv_fusion_decoder_api_cpp:

NVIDIA Fusion Decoder
---------------------

.. include:: nv_fusion_decoder_api.rst

Sliding Window Decoder
----------------------

.. include:: sliding_window_api.rst

.. _trt_decoder_api_cpp:

TensorRT Decoder
----------------

.. include:: trt_decoder_api.rst

.. _pymatching_decoder_api_cpp:

PyMatching Decoder
------------------

.. include:: pymatching_api.rst

.. _chromobius_decoder_api_cpp:

Chromobius Decoder
------------------

.. include:: chromobius_api.rst

.. _cpp_realtime_decoding_api:

Realtime Decoding
=================

.. include:: cpp_realtime_decoding_api.rst

.. include:: realtime_pipeline_api.rst

.. _parity_check_matrix_utilities:

Parity Check Matrix Utilities
=============================

The utilities below create, convert, inspect, and transform parity-check
matrices (PCMs). CUDA-Q QEC supports dense matrices as
``cudaqx::tensor<std::uint8_t>`` and sparse matrices as
``sparse_binary_matrix``. Decoder entry points accept either representation
and store the PCM internally as a sparse matrix.

``sparse_binary_matrix`` stores a binary matrix in compressed sparse column
(CSC) or compressed sparse row (CSR) layout without storing values for its
nonzero entries. Input indices are preserved as supplied. Use
:cpp:func:`cudaq::qec::sparse_binary_matrix::canonicalize` when duplicate
indices should be combined over GF(2) and each compressed row or column should
be sorted. Canonicalization preserves the matrix layout. The matrix uses
``std::uint32_t`` indices, so each dimension and the number of stored entries
must fit in that type.

Sparse utility overloads operate without materializing the full input as a
dense tensor. Use
:cpp:func:`cudaq::qec::generate_random_pcm_sparse` when a generated PCM would
be impractical to allocate densely. The dense generator remains available and
rejects dimensions whose products overflow ``std::size_t``.

.. doxygenenum:: cudaq::qec::sparse_binary_matrix_layout

.. doxygenclass:: cudaq::qec::sparse_binary_matrix
    :members:

.. doxygenfunction:: cudaq::qec::to_parity_matrix(const std::vector<cudaq::spin_op_term> &, stabilizer_type)
.. doxygenfunction:: cudaq::qec::to_parity_matrix(const std::vector<std::string> &, stabilizer_type)

.. doxygenfunction:: cudaq::qec::dense_to_sparse(const cudaqx::tensor<uint8_t> &)
.. doxygenfunction:: cudaq::qec::generate_random_pcm(std::size_t, std::size_t, std::size_t, int, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::generate_random_pcm_sparse(std::size_t, std::size_t, std::size_t, int, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, bool include_first_round = false)
.. doxygenfunction:: cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, std::vector<std::int64_t> first_round_matrix)
.. doxygenfunction:: cudaq::qec::get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &, std::uint32_t, std::uint32_t, std::uint32_t, bool, bool, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_pcm_for_rounds(const sparse_binary_matrix &, std::uint32_t, std::uint32_t, std::uint32_t, bool, bool, bool, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &, std::size_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_from_sparse_string(const std::string &, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::pcm_from_sparse_vec(const std::vector<std::int64_t>& sparse_vec, std::size_t num_rows, std::size_t num_cols)
.. doxygenfunction:: cudaq::qec::pcm_is_sorted(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_is_sorted(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_to_sparse_string(const cudaqx::tensor<uint8_t> &)
.. doxygenfunction:: cudaq::qec::pcm_to_sparse_string(const sparse_binary_matrix &)
.. doxygenfunction:: cudaq::qec::pcm_to_sparse_vec(const cudaqx::tensor<uint8_t>& pcm)
.. doxygenfunction:: cudaq::qec::pcm_to_sparse_vec(const sparse_binary_matrix &)
.. doxygenfunction:: cudaq::qec::reorder_pcm_columns(const cudaqx::tensor<uint8_t> &, const std::vector<std::uint32_t> &, uint32_t, uint32_t);
.. doxygenfunction:: cudaq::qec::reorder_pcm_columns(const sparse_binary_matrix &, const std::vector<std::uint32_t> &, uint32_t, uint32_t);
.. doxygenfunction:: cudaq::qec::shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::shuffle_pcm_columns(const sparse_binary_matrix &, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::simplify_pcm(const cudaqx::tensor<uint8_t> &, const std::vector<double> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::sort_pcm_columns(const cudaqx::tensor<uint8_t> &, std::uint32_t);

Logger
=============

The QEC logger API currently lives in ``cudaq::qec::detail`` and is used by
the ``CUDA_QEC_*`` macros exposed in ``cudaq/qec/logger.h``.

.. doxygenenum:: cudaq::qec::detail::log_level
.. doxygenenum:: cudaq::qec::detail::forward_drop_policy

.. doxygenstruct:: cudaq::qec::detail::forwarded_log_record
    :members:

.. doxygenvariable:: cudaq::qec::detail::realtime_forwarder_default_message_capacity

.. doxygenstruct:: cudaq::qec::detail::forwarder_config
    :members:

.. doxygenstruct:: cudaq::qec::detail::forwarder_stats
    :members:

.. doxygenfunction:: cudaq::qec::detail::should_log
.. doxygenfunction:: cudaq::qec::detail::set_forwarder(forwarder_config)
.. doxygenfunction:: cudaq::qec::detail::set_forwarder()
.. doxygenfunction:: cudaq::qec::detail::clear_forwarder()
.. doxygenfunction:: cudaq::qec::detail::is_forwarder_enabled()
.. doxygenfunction:: cudaq::qec::detail::get_forwarder_message_capacity()
.. doxygenfunction:: cudaq::qec::detail::get_forwarder_stats()
.. doxygenfunction:: cudaq::qec::detail::set_log_level
.. doxygenfunction:: cudaq::qec::detail::get_log_level
.. doxygenfunction:: cudaq::qec::detail::flush_logs()
.. doxygenfunction:: cudaq::qec::detail::trace
.. doxygenfunction:: cudaq::qec::detail::info
.. doxygenfunction:: cudaq::qec::detail::debug
.. doxygenfunction:: cudaq::qec::detail::warn
.. doxygenfunction:: cudaq::qec::detail::error
.. doxygenfunction:: cudaq::qec::detail::path_to_file_name

Common
=============

.. doxygentypedef:: cudaq::qec::float_t

.. doxygenenum:: cudaq::qec::operation
.. doxygenenum:: cudaq::qec::stabilizer_type

.. doxygenfunction:: cudaq::qec::sample_code_capacity(const cudaqx::tensor<uint8_t> &, std::size_t, double)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const cudaqx::tensor<uint8_t> &, std::size_t, double, unsigned)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const code &, std::size_t, double)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const code &, std::size_t, double, unsigned)

.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, std::size_t, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, operation, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, operation, std::size_t, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::x_sample_memory_circuit
.. doxygenfunction:: cudaq::qec::z_sample_memory_circuit

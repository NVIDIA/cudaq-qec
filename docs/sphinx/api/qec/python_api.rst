CUDA-Q QEC Python API
******************************

.. automodule:: cudaq_qec

Code
=============

.. autoclass:: cudaq_qec.Code
    :members:

Surface code layout
===================

.. _qec_stabilizer_grid_python:

The rotated surface code exposes a grid helper for stabilizer and data-qubit
indexing. In Python it is available as :class:`cudaq_qec.stabilizer_grid` (call
``cudaq_qec.stabilizer_grid(distance)``). The C++ type is
:cpp:class:`cudaq::qec::surface_code::stabilizer_grid` (:ref:`API <qec_stabilizer_grid_cpp>`).

.. autoclass:: cudaq_qec.stabilizer_grid
    :members:

Detector Error Model
====================

.. autoclass:: cudaq_qec.DetectorErrorModel
    :members:

.. autoclass:: cudaq_qec.DecoderContext
    :members:

.. note::

   The ``x_component()``, ``z_component()``, and ``full_component()`` methods each
   return a ``(dem, m2d, m2o)`` tuple:

   - ``dem`` (:class:`DetectorErrorModel`) — canonicalized detector error model
   - ``m2d`` (``list[list[int]]``) — measurement-to-detector map; ``m2d[d]`` lists
     the measurement indices whose XOR forms detector ``d``
   - ``m2o`` (``list[list[int]]``) — measurement-to-observable map

   Pass ``m2d`` to :func:`d_sparse` to produce the ``D_sparse`` vector for a
   real-time decoder config.

.. autofunction:: cudaq_qec.dem_from_memory_circuit
.. autofunction:: cudaq_qec.x_dem_from_memory_circuit
.. autofunction:: cudaq_qec.z_dem_from_memory_circuit
.. autofunction:: cudaq_qec.decoder_context_from_memory_circuit
.. autofunction:: cudaq_qec.dem_from_stim_text
.. autofunction:: cudaq_qec.d_sparse

.. _dyn_dem_python_api:

Dynamic DEM Construction
========================

Build a code-capacity or phenomenological DEM from CSS generator matrices
(no Stim circuit required), or compose per-round DEM *chunks* that can be
stitched and closed into a flat :class:`~cudaq_qec.DetectorErrorModel`.
See :doc:`/examples_rst/qec/dyn_dem` for a walkthrough.

CSS matrices and noise
----------------------

.. autoclass:: cudaq_qec.CssCodes
    :members:

.. autoclass:: cudaq_qec.CssNoise
    :members:

.. autofunction:: cudaq_qec.css_matrices_from_code
.. autofunction:: cudaq_qec.dem_from_css_matrices

Extended DEM chunks
-------------------

Seam and phase identifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cudaq_qec.SeamId
    :members:

.. autodata:: cudaq_qec.seam_name
.. autodata:: cudaq_qec.phase_name

Chunk types
^^^^^^^^^^^

.. autoclass:: cudaq_qec.ExtendedDemSeam
    :members:

.. autoclass:: cudaq_qec.ExtendedDem
    :members:

.. autofunction:: cudaq_qec.extended_dem_from_css_matrices

.. autoclass:: cudaq_qec.DemSeamSpec
    :members:

.. autoclass:: cudaq_qec.SeamSpecEntry
    :members:

.. autoclass:: cudaq_qec.DemChunkSpec
    :members:

.. autoclass:: cudaq_qec.SeamConnection
    :members:

.. autoclass:: cudaq_qec.PhaseConnection
    :members:

.. autoclass:: cudaq_qec.PhaseSpecEntry
    :members:

.. autoclass:: cudaq_qec.DemChunksSpec
    :members:

.. autofunction:: cudaq_qec.dem_chunk_from_spec
.. autofunction:: cudaq_qec.dem_chunks_from_spec

Stitch, close, and merge
------------------------

.. autoclass:: cudaq_qec.PriorCombineMode
    :members:

.. autofunction:: cudaq_qec.dem_stitch
.. autofunction:: cudaq_qec.dem_stitch_all
.. autofunction:: cudaq_qec.dem_stitch_merged
.. autofunction:: cudaq_qec.dem_close
.. autofunction:: cudaq_qec.dem_close_all
.. autofunction:: cudaq_qec.dem_merge_duplicate_columns
.. autofunction:: cudaq_qec.are_dem_columns_unique
.. autofunction:: cudaq_qec.assert_dem_columns_unique

Streaming decoder maps
----------------------

.. autofunction:: cudaq_qec.dem_chunk_rounds
.. autofunction:: cudaq_qec.dem_chunks_to_rounds
.. autofunction:: cudaq_qec.dem_chunks_to_detector_round
.. autofunction:: cudaq_qec.dem_chunks_to_d_sparse
.. autofunction:: cudaq_qec.dem_chunks_to_o_sparse

Decoder Interfaces
==================

.. autoclass:: cudaq_qec.Decoder
    :members:

.. autoclass:: cudaq_qec.DecoderResult
    :members:

.. autoclass:: cudaq_qec.BatchDecoderResult
    :members:

.. autoclass:: cudaq_qec.AsyncDecoderResult
    :members:

.. note::
   **NumPy result arrays** — As of 0.7.0, the ``result`` field of
   :class:`cudaq_qec.DecoderResult` (and the per-shot results returned by
   :class:`cudaq_qec.BatchDecoderResult` and
   :class:`cudaq_qec.AsyncDecoderResult`) is a 1-D NumPy array rather than a
   Python ``list``. Indexing and iteration are unchanged, but code that relied
   on the result being a ``list`` specifically (for example ``isinstance(res,
   list)`` or ``list``-only methods) should be updated.

.. autofunction:: cudaq_qec.get_decoder

.. note::
   **scipy.sparse interop** — :func:`cudaq_qec.get_decoder` and
   :class:`cudaq_qec.Decoder` accept a ``scipy.sparse`` matrix (CSR, CSC,
   COO, or any other ``scipy.sparse`` format) as the parity-check matrix
   ``H``.  This is the preferred form for large PCMs because no dense
   ``rows x cols`` allocation is made — the matrix is normalised to CSR
   internally.  Dense NumPy ``uint8`` arrays remain supported.
   The PCM utilities :func:`cudaq_qec.reorder_pcm_columns`,
   :func:`cudaq_qec.shuffle_pcm_columns`, and
   :func:`cudaq_qec.pcm_to_sparse_vec` also accept SciPy sparse matrices
   without creating a dense ``cudaqx::tensor``. Reordering and shuffling a
   sparse input returns a ``scipy.sparse.csc_matrix``; a dense input continues
   to return a NumPy array.

   ``scipy`` is an optional dependency; if it is not installed, pass a dense
   NumPy array instead.

Built-in Decoders
=================

.. _nv_qldpc_decoder_api_python:

NVIDIA QLDPC Decoder
--------------------

.. include:: nv_qldpc_decoder_api.rst

Relay Solutions Post-Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cudaq_qec.relay_solutions

.. autofunction:: cudaq_qec.relay_solutions.unpack
.. autofunction:: cudaq_qec.relay_solutions.stop_nconv_sweep
.. autoclass:: cudaq_qec.relay_solutions.RelaySolutionRecords
    :members:
.. autoclass:: cudaq_qec.relay_solutions.StopNConvSweep
    :members:

.. _nv_fusion_decoder_api_python:

NVIDIA Fusion Decoder
---------------------

.. include:: nv_fusion_decoder_api.rst

Sliding Window Decoder
----------------------

.. include:: sliding_window_api.rst

.. _trt_decoder_api_python:

TensorRT Decoder
----------------

.. include:: trt_decoder_api.rst

.. _tensor_network_decoder_api_python:

Tensor Network Decoder
----------------------

.. include:: tensor_network_decoder_api.rst

.. _pymatching_decoder_api_python:

PyMatching Decoder
------------------

.. include:: pymatching_api.rst

.. _chromobius_decoder_api_python:

Chromobius Decoder
------------------

.. include:: chromobius_api.rst

.. _python_realtime_decoding_api:

Realtime Decoding
=================

.. include:: python_realtime_decoding_api.rst


Common
=============

.. autofunction:: cudaq_qec.sample_memory_circuit
.. autofunction:: cudaq_qec.x_sample_memory_circuit
.. autofunction:: cudaq_qec.z_sample_memory_circuit

.. _syndrome_measurement_layout:

.. note::
   **Syndrome measurement layout** — ``sample_memory_circuit`` returns a tuple
   ``(syndromes, data)``. The ``syndromes`` tensor has shape
   ``(num_shots, num_detectors)`` with columns laid out as ``[ B  S  S  …  S  B ]``:

   - ``B`` (boundary block) = ``code.get_num_z_stabilizers()`` for Z-basis
     preparations (``prep0``/``prep1``), or ``code.get_num_x_stabilizers()`` for
     X-basis preparations (``prepp``/``prepm``).
   - ``S`` (inter-round block) = ``num_z_stabilizers + num_x_stabilizers``
     detectors per round transition (``num_rounds - 1`` blocks total).
   - Total: ``num_detectors = 2*B + (num_rounds - 1)*S``.

   The ``data`` tensor has shape ``(num_shots, block_size)`` and holds the final
   data-qubit measurements used to verify logical-state preservation.

.. autofunction:: cudaq_qec.sample_code_capacity

.. _dem_sampling_python_api:

Detector Error Model (DEM) Sampling
===================================

.. autofunction:: cudaq_qec.dem_sampling

.. _parity_check_matrix_utilities_python:

Parity Check Matrix Utilities
=============================

.. autofunction:: cudaq_qec.generate_random_pcm
.. autofunction:: cudaq_qec.generate_timelike_sparse_detector_matrix
.. autofunction:: cudaq_qec.get_pcm_for_rounds
.. autofunction:: cudaq_qec.get_sorted_pcm_column_indices
.. autofunction:: cudaq_qec.pcm_extend_to_n_rounds
.. autofunction:: cudaq_qec.pcm_is_sorted
.. autofunction:: cudaq_qec.pcm_to_sparse_vec
.. autofunction:: cudaq_qec.reorder_pcm_columns
.. autofunction:: cudaq_qec.shuffle_pcm_columns
.. autofunction:: cudaq_qec.simplify_pcm
.. autofunction:: cudaq_qec.sort_pcm_columns

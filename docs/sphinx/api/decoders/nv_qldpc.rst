:tocdepth: 2

.. _nv_qldpc_standalone_api:

Standalone NVIDIA QLDPC Python API
==================================

.. py:module:: cudaq_decoding.nv_qldpc

This page documents ``cudaq_decoding.nv_qldpc`` for standalone package
version |nv-qldpc-version|. See :ref:`nv_qldpc_standalone_installation` for
installation and :ref:`nv_qldpc_package_comparison` when moving between the
standalone and CUDA-Q QEC packages.

The module provides synchronous GPU decoding. Its classes do not inherit
from CUDA-Q QEC classes, and it has no decoder registry, asynchronous
decoding, task-graph, or realtime-server interface.

First Decode
------------

.. code-block:: python

    import numpy as np
    from cudaq_decoding import nv_qldpc

    H = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ], dtype=np.uint8)
    syndromes = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint8)

    decoder = nv_qldpc.create_decoder(
        H, error_rate=0.01,
        **nv_qldpc.preset("bp60-osd0", bp_batch_size=2),
    )
    results = decoder.decode_batch(syndromes)
    print(results.result)     # uint8 array, shape (2, 7)
    print(results.converged)  # bool array, shape (2,)

Construction
------------

.. py:function:: create_decoder(H, O=None, output="errors", **options)

   Construct a :class:`Decoder`. This is the preferred factory; the module
   already identifies the decoder, so no decoder-name argument is needed.

   :param H: Binary parity-check matrix with shape ``(m, n)``: one row per
      detector and one column per error mechanism. Accepts a dense array-like
      object, a SciPy sparse matrix, or a :class:`CSRMatrix`-compatible object.
   :param O: Optional binary observable matrix with shape ``(k, n)``, in the
      same formats as ``H``. It must have the same number of columns as ``H``.
   :param output: ``"errors"`` (default) returns one value per column of
      ``H``. ``"observables"`` returns one flip bit per row of ``O`` and
      requires ``O``. Supplying ``O`` alone does not change the output basis.
   :param options: Decoder options passed as keyword arguments. For example,
      ``error_rate_vec`` supplies one probability per column of ``H`` and
      overrides the scalar ``error_rate``. See `Decoder Options`_ below.
   :returns: A configured :class:`Decoder`.
   :raises ValueError: For invalid matrix dimensions, output mode, or decoder
      configuration.
   :raises RuntimeError: For native execution failures, including CUDA
      allocation failures.

   Dense inputs are copied into contiguous ``uint8`` arrays. Sparse inputs
   are copied into CSR index arrays and remain sparse. Binary entries must
   be 0 or 1; matrices are not weighted adjacency matrices.

   If construction fails because GPU memory allocation fails, retry with a
   smaller ``bp_batch_size``. Recovery from other CUDA errors is not guaranteed.

.. py:function:: get_decoder(H, O=None, output="errors", **options)

   Compatibility factory with the same behavior as :func:`create_decoder`.

.. py:class:: Decoder(H, O=None, output="errors", **options)

   Direct construction is equivalent to :func:`create_decoder`. Instances
   retain their construction inputs. Pickling stores the matrices and
   options; unpickling creates a new native decoder and requires a compatible
   GPU. Changing the caller's original matrices or options after construction
   does not change this reconstruction. Dense matrices retain one byte per
   entry, including inputs already stored as contiguous ``uint8`` arrays.
   Sparse matrices retain copied CSR index arrays and remain sparse during
   pickling.

   .. py:method:: decode(syndrome)

      Decode one binary syndrome with shape ``(syndrome_size,)``. Input is
      converted to a contiguous ``uint8`` array. Returns a
      :class:`DecoderResult`; a wrong shape raises ``ValueError``.

      Sequential Relay BP (``composition=1``) requires ``bp_batch_size=1``
      for this call. Use a separately configured decoder for parallel batches.

   .. py:method:: decode_batch(syndromes)

      Decode binary syndromes with shape ``(shots, syndrome_size)``. Input
      is converted to a contiguous ``uint8`` array. Returns a
      :class:`BatchDecoderResult`; a wrong shape raises ``ValueError``.

      An empty input with shape ``(0, syndrome_size)`` returns arrays with
      shapes ``(0, result_size)`` and ``(0,)``, retaining the selected result
      dtype. Its ``opt_results`` is ``None``.

   .. py:attribute:: block_size

      Number of columns in ``H`` (``n``).

   .. py:attribute:: syndrome_size

      Required syndrome width, reported by the native decoder.

   .. py:attribute:: num_observables

      Number of rows in ``O`` (``k``), or zero if ``O`` was omitted.

   .. py:attribute:: result_size

      Result width: ``n`` for error output, ``k`` for observable output.

   .. py:attribute:: result_dtype

      NumPy dtype selected at construction: ``float64`` for TCBP error
      probabilities without OSD, otherwise ``uint8``. Observable output
      always contains hard flip bits.

   .. py:attribute:: output

      Selected result basis: ``"errors"`` or ``"observables"``.

   .. py:attribute:: options

      A copy of the supplied construction options, with ``tcbp_model``
      normalized to bytes when present. Omitted backend defaults are not
      added. Editing this dictionary does not reconfigure the decoder.

   .. py:attribute:: version

      Native decoder identity, including its Git revision when recorded by
      the build. This is distinct from the installed package's
      :data:`__version__`.

Results
-------

.. py:class:: DecoderResult(result, converged, opt_results=None)

   Dataclass returned by :meth:`Decoder.decode`.

   .. py:attribute:: result

      NumPy array with shape ``(decoder.result_size,)`` and dtype
      ``decoder.result_dtype``.

   .. py:attribute:: converged

      Python ``bool`` reporting decoder convergence.

   .. py:attribute:: opt_results

      Dictionary of requested optional results for this shot, or ``None``.
      For example, requesting ``opt_results={"num_iter": True}`` adds the
      ``"num_iter"`` entry.

.. py:class:: BatchDecoderResult(result, converged, opt_results=None)

   Dataclass returned by :meth:`Decoder.decode_batch`. Access its arrays
   directly; it does not implement iteration or indexing over per-shot
   result objects.

   .. py:attribute:: result

      NumPy array with shape ``(shots, decoder.result_size)`` and dtype
      ``decoder.result_dtype``.

   .. py:attribute:: converged

      Boolean NumPy array with shape ``(shots,)``.

   .. py:attribute:: opt_results

      ``None`` when no optional results were returned, otherwise a dictionary
      with ``"shots"`` (a list of per-shot dictionaries) and ``"batch"``
      (a dictionary of batch-wide results). Numeric array entries remain
      NumPy arrays. There is no ``batch_opt_results`` attribute.

Optional results preserve nonfinite scalar values, such as an infinite
``jobs_gap``, for both single-shot and batch decoding.

For example, using the ``H`` and ``syndromes`` from `First Decode`_:

.. code-block:: python

    decoder = nv_qldpc.create_decoder(
        H, error_rate=0.01,
        **nv_qldpc.preset("bp60-osd0", bp_batch_size=2,
                          opt_results={"num_iter": True}),
    )
    results = decoder.decode_batch(syndromes)
    iterations = [shot["num_iter"] for shot in results.opt_results["shots"]]
    for prediction, converged, count in zip(
            results.result, results.converged, iterations):
        print(prediction, converged, count)

Decoder Options
---------------

The keyword options select the shared NVIDIA QLDPC backend. The
:ref:`decoder option reference <nv_qldpc_decoder_api_python>` lists the full
BP, Relay BP, OSD, JOBS, and optional-result settings, their defaults, and
constraints. Its QEC constructors and result-access examples apply to the
QEC interface; use the factories and result types on this page for standalone
code. Historical "introduced in" versions in that reference refer to CUDA-Q
QEC releases.

In standalone Python, pass option maps such as ``srelay_config`` and
``opt_results`` as dictionaries. ``tcbp_model`` accepts bytes, a bytearray,
a NumPy ``uint8`` array, or a file path (string or ``os.PathLike``).
An empty model is invalid.

Use :func:`preset` to select a configuration, then override individual
options. With ``output="errors"``, a TCBP model and ``use_osd=False`` return
probabilities without thresholding. With OSD or JOBS, the backend uses the
TCBP scores for column ordering and returns hard decisions. The
:ref:`pipeline overview <qldpc_decoder>` explains these configurations.

Presets
-------

.. py:function:: names()

   Return the preset names as a list of strings, in documentation order.

.. py:function:: preset(name, **overrides)

   Return a preset option dictionary with overrides applied. Pass it to
   :func:`create_decoder` with ``**options``. The options are copied from
   the preset, so editing them does not change the preset definition.

   :raises KeyError: For an unknown preset name.
   :raises ValueError: For a TCBP preset without a ``tcbp_model``.

   The preset families are ``bp60``, ``bp10-minllr``, ``relay-bp``, and
   ``tcbp``. Suffixes select OSD-0 (``osd0``), combination sweep
   (``osd-cs1000``), or JOBS (``jobs``). JOBS modifiers ``sc`` and ``gap``
   enable searching converged shots and reporting the confidence gap.
   Call :func:`names` for the available combinations.

TCBP Model Fitting and Inspection
---------------------------------

.. py:function:: init_tcbp_model(H, return_report=False, **params)

   Fit a TCBP model on the GPU using the structure of ``H``. Dense and sparse
   inputs use the same formats as :func:`create_decoder`; sparse inputs
   remain sparse. Training samples are generated from the matrix, so no
   recorded detector samples are needed.

   Return model bytes, or ``(model, report)`` when ``return_report=True``.
   Save the bytes for reuse, or pass them directly as ``tcbp_model`` to a
   decoder. Fit a new model when the BP settings change.

   ``params`` accepts training rates (``min_p`` and ``max_p``, or
   ``error_rate_vec``), ``seed``, ``shots``, ``steps``,
   ``device_memory_budget_bytes``, ``verbosity``, and the BP settings
   ``bp_method``, ``max_iterations``, ``clip_value``, and ``proc_float``.
   Defaults and constraints are listed under TCBP in the
   :ref:`shared option reference <nv_qldpc_decoder_api_python>`.

   A decoder using the model must match its BP settings. The default fitting
   settings match the ``tcbp`` presets. To make repeated fits byte-identical,
   supply the same seed and an explicit device memory budget.

.. py:function:: inspect(tcbp_model)

   Inspect model bytes and return a dictionary containing the stored report,
   fitted decoder settings, matrix dimensions, and fingerprints. Raises
   ``ValueError`` for a malformed model. Load a model file as bytes before
   calling this function; it does not accept a file path.

   Timing and memory measurements are not stored in the model and are
   reported as zero by inspection. Use the report returned by the fitting
   call for those measurements.

See the :doc:`TCBP/JOBS tutorial </examples_rst/qec/tcbp_jobs>` for a complete
workflow with standalone code tabs. Noise learning is a CUDA-Q QEC feature
and is not exported by this module.

Sparse Matrices
---------------

.. py:class:: CSRMatrix(shape, indptr, indices)

   A binary matrix in compressed sparse row form. ``shape`` is ``(m, n)``,
   ``indptr`` contains ``m + 1`` row offsets starting at zero, and ``indices``
   contains the column indices of the ones. Index arrays use ``int32`` or
   ``int64``. Construction copies both arrays and validates dimensions,
   offsets, and index bounds.

   The decoder also accepts other objects with these three attributes.
   SciPy sparse matrices are converted through ``tocsr()``; SciPy is not
   required to use this class. Repeated indices represent a single set bit.

   .. py:attribute:: nnz

      Number of stored column indices.

   .. py:method:: toarray()

      Return a dense ``uint8`` copy with the matrix's shape.

Package Information
-------------------

.. py:data:: __version__

   Installed distribution version. Source-tree imports without distribution
   metadata report ``"0.0.0"``. Record this value and
   :attr:`Decoder.version` when reporting a benchmark or issue.

.. py:function:: plugin_path()

   Return the filesystem path of the native library bundled with this
   package. The C interface is internal to the Python bindings.

See :doc:`Decoder Release History </releases/decoders>` for package changes.

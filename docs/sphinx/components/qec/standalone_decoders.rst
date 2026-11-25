:tocdepth: 2

Standalone Decoders
===================

Standalone decoder packages provide Python decoding without installing CUDA-Q
or CUDA-Q QEC. Each decoder has its own package, API reference, and release
version. Use CUDA-Q QEC when you also need code construction, detector error
models, noise learning, task graphs, or realtime integration.

.. list-table::
   :header-rows: 1
   :widths: 30 15 30 25

   * - Decoder
     - Version
     - Python API
     - Release history
   * - NVIDIA QLDPC
     - |nv-qldpc-version|
     - :doc:`API reference </api/decoders/nv_qldpc>`
     - :ref:`Release notes <nv_qldpc_standalone_releases>`

.. _nv_qldpc_standalone:

NVIDIA QLDPC Decoder
--------------------

The standalone NVIDIA QLDPC decoder provides BP, Relay BP, OSD, JOBS, presets,
and TCBP model fitting and inspection. It uses the same decoding algorithms
as the :ref:`CUDA-Q QEC plugin <qldpc_decoder>`, with a separate Python interface.

.. _nv_qldpc_package_comparison:

Comparing the Python Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`standalone Python API </api/decoders/nv_qldpc>` accepts matrices
directly and returns NumPy arrays. Its ``Decoder`` class is independent of
``cudaq_qec.Decoder``.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Operation
     - CUDA-Q QEC
     - Standalone
   * - Construct a decoder
     - ``qec.get_decoder("nv-qldpc-decoder", H, ...)``
     - ``nv_qldpc.create_decoder(H, ...)``
   * - Presets and TCBP module
     - ``cudaq_qec.plugins.nv_qldpc``
     - ``cudaq_decoding.nv_qldpc``
   * - Decode one shot or a batch
     - ``decoder.decode(s)`` / ``decoder.decode_batch(S)``
     - Same calls
   * - Read predictions and convergence
     - ``results.result`` / ``results.converged``
     - Same fields
   * - Optional per-shot batch results
     - ``results.opt_results``
     - ``results.opt_results["shots"]``
   * - Optional batch-wide results
     - ``results.batch_opt_results``
     - ``results.opt_results["batch"]``
   * - Iterate over a batch
     - Per-shot result objects or result arrays
     - Read the result arrays directly
   * - Async decoding, task graphs and realtime integration
     - Provided by CUDA-Q QEC
     - Synchronous decoding only
   * - Noise learning and DEM utilities
     - Provided by CUDA-Q QEC
     - Not included

The optional-result entries apply when those results were requested. In the
standalone package, ``opt_results`` is otherwise ``None``. For both packages,
batch predictions have shape ``(shots, result_size)`` and convergence has
shape ``(shots,)``. This array-based loop works with either package:

.. code-block:: python

    results = decoder.decode_batch(syndromes)
    for prediction, converged in zip(results.result, results.converged):
        print(prediction, converged)

Supply ``H`` and optional ``O`` as dense or sparse matrices. The standalone
package does not provide the QEC decoder registry or helpers such as
``tying.dem_arrays`` and ``relay_solutions.stop_nconv_sweep``.
LUT and sliding-window decoders are not included. The standalone and QEC
packages can be used in the same Python process in either import order;
each uses its own native decoder library.
The :ref:`getting-started example <qldpc_decoder_example>` and
:doc:`TCBP/JOBS tutorial </examples_rst/qec/tcbp_jobs>` have tabs for each
package.

.. _nv_qldpc_standalone_installation:

Installation
^^^^^^^^^^^^

Use Linux x86_64 or aarch64, Python 3.11 or later, and an NVIDIA GPU with a
driver compatible with the chosen CUDA variant. NumPy 1.24 or later is the
only Python runtime dependency. SciPy is optional for sparse-matrix inputs.
The wheel includes the CUDA runtime; a separate CUDA Toolkit installation
is not required to run it.

Install one CUDA variant per Python environment. For a wheel supplied
directly, pass its local ``.whl`` path to ``python -m pip install``.
To install version |nv-qldpc-version| from your package index, use:

.. tab:: CUDA 13

    .. parsed-literal::

        python -m pip install cudaq-decoders-nv-qldpc-cu13==\ |nv-qldpc-version|

.. tab:: CUDA 12

    .. parsed-literal::

        python -m pip install cudaq-decoders-nv-qldpc-cu12==\ |nv-qldpc-version|

The distribution name uses ``cudaq-decoders``; the Python import is
``cudaq_decoding.nv_qldpc``. See the
:doc:`standalone Python API </api/decoders/nv_qldpc>` for a complete first
decoding example.

Package Version and Backend Revision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package version identifies the standalone release. The native decoder's
version string includes its Git revision when recorded by the build. After
constructing a decoder:

.. code-block:: python

    from cudaq_decoding import nv_qldpc

    print(nv_qldpc.__version__)  # installed standalone package version
    print(decoder.version)      # native decoder identity and available revision

Record both when reporting an issue or a benchmark, together with the CUDA
variant and GPU. The standalone release number is independent of the CUDA-Q
QEC version shown in this site's documentation selector. See
:doc:`Decoder Release History </releases/decoders>` for versioned release notes.

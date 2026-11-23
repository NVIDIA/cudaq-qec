:tocdepth: 2

Decoder Release History
=======================

Standalone decoder packages are versioned independently of CUDA-Q QEC and
of each other. CUDA 12 and CUDA 13 variants of a decoder share the same
package version. See the :doc:`Standalone Decoders
</components/qec/standalone_decoders>` guide for installation and the
differences from CUDA-Q QEC.

.. _nv_qldpc_standalone_releases:

NVIDIA QLDPC Decoder
--------------------

Distributions: ``cudaq-decoders-nv-qldpc-cu12`` and
``cudaq-decoders-nv-qldpc-cu13``. Python module: ``cudaq_decoding.nv_qldpc``.

0.1.0 — In Preparation
^^^^^^^^^^^^^^^^^^^^^^^

Initial standalone Python release. This version has not yet been published.

* GPU decoding with BP, Relay BP, OSD, and JOBS, plus named presets and
  TCBP model fitting and inspection.
* Direct construction from dense or sparse parity-check and observable
  matrices, with synchronous single-shot and batch decoding.
* NumPy result arrays and standalone result classes. Optional batch results
  are grouped under ``opt_results["shots"]`` and ``opt_results["batch"]``.
* Linux x86_64 and aarch64 wheels for CUDA 12 and CUDA 13, with Python 3.11
  or later and NumPy 1.24 or later. No CUDA-Q or CUDA-Q QEC runtime dependency.

See the :doc:`NVIDIA QLDPC Python API </api/decoders/nv_qldpc>` for the
standalone interface and :ref:`nv_qldpc_package_comparison` for the
differences from CUDA-Q QEC. Noise learning, QEC DEM utilities, task graphs,
and realtime execution are outside this package's API.

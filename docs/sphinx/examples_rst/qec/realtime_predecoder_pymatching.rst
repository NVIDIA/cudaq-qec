AI Predecoder with CUDA-Q Realtime
==================================

.. note::

  The following information is about a C++ demonstration that must be built
  from source and is not part of any distributed CUDA-Q QEC binaries.

This guide explains how to build and run the hybrid AI predecoder + PyMatching
streaming benchmark. The benchmark uses a TensorRT-accelerated neural network
(the *predecoder*) to reduce syndrome density on the GPU, then feeds the
residual detectors to a pool of PyMatching MWPM decoders on the CPU. A
software data injector streams pre-generated syndrome shots through the
``RealtimePipeline`` at a configurable rate and collects latency, throughput,
syndrome density, and logical error rate statistics.

The benchmark binary is
``test_realtime_predecoder_w_pymatching``, built from
`libs/qec/unittests/realtime/test_realtime_predecoder_w_pymatching.cpp
<https://github.com/NVIDIA/cudaq-qec/blob/main/libs/qec/unittests/realtime/test_realtime_predecoder_w_pymatching.cpp>`_.


Prerequisites
-------------

Hardware
^^^^^^^^

- CUDA-capable GPU (NVIDIA Grace Blackwell / GB200 recommended)
- Sufficient GPU memory for the TensorRT engine (the d13_r104 model requires
  approximately 11 MB per predecoder instance)

Software
^^^^^^^^

- **CUDA Toolkit** 12.6 or later
- **TensorRT** 10.x (headers and libraries)
- **CUDA-Q SDK** pre-installed (provides ``libcudaq``, ``libnvqir``, ``nvq++``)
- **CUDA-Q Realtime** libraries (``libcudaq-realtime``,
  ``libcudaq-realtime-dispatch``, ``libcudaq-realtime-host-dispatch``) built
  and installed to a known prefix (e.g. ``/tmp/cudaq-realtime``)

Additional inputs:

.. note::

  Exercising the following test program requires data files that are generated
  using the Ising-Decoding repository. Please see the instructions in the
  `Ising-Decoding README
  <https://github.com/NVIDIA/Ising-Decoding/blob/main/README.md#generating-data-for-cuda-q-qec-realtime-predecoder-test-application>`_
  for how to generate these files.

- **Predecoder ONNX model** (e.g. ``predecoder_memory_d13_T104_X.onnx``)
  placed under ``libs/qec/lib/realtime/``. A cached TensorRT ``.engine`` file
  with the same base name is loaded automatically if present; otherwise the
  engine is built from the ONNX file on first run (this can take 1--2 minutes
  for large models).
- **Syndrome data directory** containing pre-generated detector samples,
  observables, and matching graph data (see :ref:`Data Directory Layout <sw_data_directory_layout>`).


.. _sw_data_directory_layout:

Data Directory Layout
---------------------

The ``--data-dir`` flag points to a directory with the following files.
All binary files use little-endian format.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``detectors.bin``
     - Detector samples. Header: ``uint32 num_samples``, ``uint32 num_detectors``;
       body: ``int32[num_samples * num_detectors]``.
   * - ``observables.bin``
     - Observable ground-truth labels. Header: ``uint32 num_samples``,
       ``uint32 num_observables``; body: ``int32[num_samples * num_observables]``.
   * - ``H_csr.bin``
     - Sparse CSR parity check matrix. Header: ``uint32 nrows``,
       ``uint32 ncols``, ``uint32 nnz``; body: ``int32 indptr[nrows+1]``,
       ``int32 indices[nnz]``.
   * - ``O_csr.bin``
     - Sparse CSR observables matrix (same format as ``H_csr.bin``).
   * - ``priors.bin``
     - Per-edge error probabilities. Header: ``uint32 num_edges``; body:
       ``float64[num_edges]``.
   * - ``metadata.txt``
     - Human-readable parameters (``distance``, ``n_rounds``, ``p_error``,
       etc.). Not read by the binary; included for reference.


Building
--------

The benchmark requires two CMake targets:

- ``test_realtime_predecoder_w_pymatching`` -- the benchmark binary
- ``cudaq-qec-pymatching`` -- the PyMatching decoder plugin (loaded at runtime)

Configure and build:

.. code-block:: bash

   cd /path/to/cudaq-qec

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
     -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq \
     -DCUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime \
     -DCUDAQ_QEC_BUILD_TRT_DECODER=ON \
     -DCUDAQX_ENABLE_LIBS=qec \
     -DCUDAQX_INCLUDE_TESTS=ON \
     -DCUDAQX_QEC_INCLUDE_TESTS=ON

   cmake --build build -j$(nproc) --target \
     test_realtime_predecoder_w_pymatching \
     cudaq-qec-pymatching

.. note::

   The ``test_realtime_predecoder_w_pymatching`` target requires TensorRT
   headers and libraries to be discoverable. CMake searches standard system
   paths (e.g. ``/usr/include/aarch64-linux-gnu``,
   ``/usr/lib/aarch64-linux-gnu``). If TensorRT is installed elsewhere, set
   ``-DTENSORRT_ROOT=/path/to/tensorrt``.

   The ``cudaq-qec-pymatching`` shared library is written to
   ``build/lib/decoder-plugins/``. If the benchmark fails with
   ``invalid decoder requested: pymatching``, verify that this file exists.


Running
-------

.. code-block:: text

   test_realtime_predecoder_w_pymatching <config> [rate_us] [duration_s] [flags]

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Argument
     - Description
     - Default
   * - ``config``
     - Pipeline configuration name (see table below)
     - ``d7``
   * - ``rate_us``
     - Inter-arrival time in microseconds. ``0`` runs open-loop (as fast as
       possible).
     - ``0``
   * - ``duration_s``
     - Test duration in seconds
     - ``5``

Named Flags
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--data-dir <path>``
     - Path to syndrome data directory (see :ref:`Data Directory Layout <sw_data_directory_layout>`). When
       omitted, random syndromes with 1% error rate are generated.
   * - ``--num-gpus <n>``
     - Number of GPUs to use. Currently clamped to 1 (multi-GPU dispatch is
       not yet supported).

Pipeline Configurations
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 10 10 38 10 10 10

   * - Config
     - Distance
     - Rounds
     - ONNX Model
     - Pre-decoders
     - Workers
     - Decode Workers
   * - ``d13_r104``
     - 13
     - 104
     - ``predecoder_memory_d13_T104_X.onnx``
     - 8
     - 8
     - 16

Example
^^^^^^^

Run the d13_r104 configuration at 500 req/s for 2 minutes with real syndrome
data:

.. code-block:: bash

   ./build/libs/qec/unittests/realtime/test_realtime_predecoder_w_pymatching \
       d13_r104 2000 120 \
       --data-dir /path/to/syndrome_data/p0.003


Changing the Predecoder Model
-----------------------------

The ONNX model file for each configuration is set in the ``PipelineConfig``
factory methods in
``libs/qec/unittests/realtime/predecoder_pipeline_common.h``. To use a
different model, edit the ``onnx_filename`` field and rebuild.

.. code-block:: cpp

   static PipelineConfig d13_r104() {
       return {
           "d13_r104_X", 13, 104,
           "predecoder_memory_model_4_d13_T104_X.onnx",  // changed model
           8, 8, 16};
   }

Then rebuild:

.. code-block:: bash

   cmake --build build -j$(nproc) --target test_realtime_predecoder_w_pymatching

ONNX model files and their corresponding ``.engine`` caches live in
``libs/qec/lib/realtime/``. If a cached engine exists with the same base name
as the ONNX file, TensorRT loads it directly. Otherwise, the engine is built
from the ONNX file on the first run.


Reading the Output
------------------

The benchmark prints a structured report after the streaming run completes.

Throughput and Timing
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Submitted:          60001
   Completed:          60001
   Throughput:         500.0 req/s
   Backpressure stalls:       0

``Backpressure stalls`` counts how many times the producer had to spin because
all pipeline slots were occupied. Zero stalls means the pipeline kept up with
the injection rate.

Latency Distribution
^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Latency (us)  [steady-state, 59981 requests after 20 warmup]
     min    =      154.8
     p50    =      203.9
     mean   =      215.5
     p99    =      363.4

End-to-end latency measured from ``injector.submit()`` to the completion
callback. Includes GPU inference, CPU-side PyMatching decode, and all pipeline
overhead. The first 20 requests are excluded as warmup.

PyMatching Average Time
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   PyMatching decode:         75.6 us

Average time for the PyMatching MWPM decoder to process a single residual
syndrome.

Syndrome Density
^^^^^^^^^^^^^^^^

.. code-block:: text

   Input:  931.0 / 17472  (0.0533)
   Output: 16.0 / 17472  (0.0009)
   Reduction: 98.3%

Average nonzero detectors before the predecoder (input) and after (residual
output). Higher reduction means the predecoder is removing more syndrome
weight, which reduces PyMatching decode time.

Correctness Verification
^^^^^^^^^^^^^^^^^^^^^^^^

Printed only when ``--data-dir`` is provided:

.. code-block:: text

   Pipeline (pred+pymatch) mismatches: 108  LER: 0.0018

- **Pipeline LER**: logical error rate of the full predecoder + PyMatching
  chain compared to ground-truth observables.

.. note::

   Syndrome samples are cycled when the run exceeds the dataset size.
   For example, if the dataset has 10,000 shots and the test runs 60,000
   requests, each shot is replayed approximately 6 times. Correctness
   verification still compares against the correct ground truth for each
   replayed shot.

See Also
--------

* :doc:`Realtime Decoding </components/qec/realtime_decoding>` -- concept and workflow
* :doc:`Getting Started with Realtime Decoding </examples_rst/qec/getting_started_realtime_decoding>`
* :ref:`C++ <cpp_realtime_decoding_api>` and :ref:`Python <python_realtime_decoding_api>` realtime decoding API

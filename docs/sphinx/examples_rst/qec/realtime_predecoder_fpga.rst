AI Predecoder with CUDA-Q Realtime (with FPGA Data Injection)
=============================================================

.. note::

   The following information is about a C++ demonstration that must be built
   from source and is not part of any distributed CUDA-Q QEC binaries.

This guide explains how to build, test, and run the AI predecoder + PyMatching
pipeline over GPU RoCE RDMA using the cudaq-qec realtime pipeline over the
``GpuRoceTransceiver`` RDMA transport.
The pipeline runs a TensorRT-accelerated neural network (the *predecoder*) on
the GPU to reduce syndrome density, then feeds the residual detectors to a
pool of PyMatching MWPM decoders on the CPU.  It operates in two
configurations:

- **Emulated end-to-end test** -- software FPGA emulator replaces real hardware
- **FPGA end-to-end test** -- real FPGA connected via ConnectX RDMA/RoCE

For the software-only benchmark (no FPGA or network hardware), see
:doc:`/examples_rst/qec/realtime_predecoder_pymatching`.


Prerequisites
-------------

Hardware
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 20

   * - Configuration
     - GPU
     - ConnectX NIC
     - FPGA
   * - Emulated E2E
     - CUDA GPU with GPUDirect RDMA
     - Required (loopback cable)
     - Not required
   * - FPGA E2E
     - CUDA GPU with GPUDirect RDMA
     - Required
     - Required

Tested platforms: GB200.

Software
^^^^^^^^

- **CUDA Toolkit**: 12.6 or later
- **TensorRT**: 10.x (headers and libraries)
- **CUDA-Q SDK**: pre-installed (provides ``libcudaq``, ``libnvqir``, ``nvq++``)
- **DOCA**: 3.4 or later (for ``gpu_roce_transceiver`` RDMA transport)
- **PyMatching decoder plugin**: the ``cudaq-qec-pymatching`` shared library
  (``libcudaq-qec-pymatching.so``).  Built as part of the CUDA-Q QEC build and
  required at runtime.
- **Predecoder ONNX model** (e.g. ``predecoder_memory_d13_T104_X.onnx``)
  placed under ``libs/qec/lib/realtime/``.  A cached TensorRT ``.engine`` file
  with the same base name is loaded automatically if present; otherwise the
  engine is built from the ONNX file on first run (this can take 1--2 minutes
  for large models).

Source Repositories
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Repository
     - URL
     - Version
   * - **cudaq-qec**
     - https://github.com/NVIDIA/cudaq-qec
     - ``main`` branch (or your feature branch)
   * - **cuda-quantum** (realtime)
     - https://github.com/NVIDIA/cuda-quantum
     - ``main`` at the ref recorded in cudaq-qec's ``.cudaq_version``
   * - **holoscan-sensor-bridge**
     - https://github.com/nvidia-holoscan/holoscan-sensor-bridge
     - Branch ``EA_CUDAQ_10_26`` (cloned and built by
       ``realtime/scripts/install_devdeps.sh``)

``cuda-quantum`` provides ``libcudaq-realtime`` (ring buffer helpers and the
GPU dispatch kernels) and the GpuRoce bridge library.  The host dispatch loop
that drives the predecoder graphs lives in cudaq-qec's
``libcudaq-realtime-pipeline``.  ``holoscan-sensor-bridge`` provides the
``GpuRoceTransceiver`` library for RDMA transport.

.. note::

   The FPGA emulator (``hsb_fpga_emulator``) is built from the
   ``cuda-quantum`` repository and is only needed for the emulated test.


Repository Layout
-----------------

Key files within ``cudaq-qec``:

.. code-block:: text

   libs/qec/
     unittests/
       realtime/
         gpu_roce_predecoder_bridge.cpp        # Bridge tool (RDMA <-> AI predecoder + PyMatching)
         gpu_roce_predecoder_test.sh           # Orchestration script
         predecoder_pipeline_common.h          # Pipeline config and shared utilities
         predecoder_pipeline_common.cpp        # Data loading (detectors, H, O, priors)
         test_realtime_predecoder_w_pymatching.cpp  # Software-only benchmark
       utils/
         hsb_fpga_syndrome_playback.cpp   # Playback tool (loads syndromes into FPGA)

The FPGA emulator is in the ``cuda-quantum`` repository:

.. code-block:: text

   cuda-quantum/realtime/
     unittests/utils/
       hsb_fpga_emulator.cpp              # Software FPGA emulator


Data Directory Layout
---------------------

.. note::

  Exercising the following test program requires data files that are generated
  using the Ising-Decoding repository. Please see the instructions in the
  `Ising-Decoding README
  <https://github.com/NVIDIA/Ising-Decoding/blob/main/README.md#generating-data-for-cuda-q-qec-realtime-predecoder-test-application>`_
  for how to generate these files.

The syndrome data directory follows the same format as the software benchmark.
See :doc:`/examples_rst/qec/realtime_predecoder_pymatching` for the full
specification.  In summary, it must contain:

- ``detectors.bin`` -- detector samples (binary, int32)
- ``observables.bin`` -- observable ground-truth labels (binary, int32)

The orchestration script automatically converts ``detectors.bin`` to the text
format that ``hsb_fpga_syndrome_playback`` expects.

.. note::

   **FPGA BRAM constraints**: The FPGA BRAM has a fixed depth
   (``RAM_DEPTH=512`` lines of 64 bytes each = 32 KB).  For large configs
   like d13_r104 (frame size 17,536 bytes = 274 lines per shot), only
   **1 shot** fits in BRAM per playback.  The ``--num-shots`` flag in the
   orchestration script controls how many shots are loaded; the script
   applies config-appropriate defaults automatically.


Building
--------

Building the FPGA demo requires ``holoscan-sensor-bridge`` and
``libcudaq-realtime`` with HSB tools enabled.  The steps below are the ones
cudaq-qec's CI runs (``.github/actions/build-lib/build_qec.sh``).  The
``install_devdeps.sh`` script in the ``cuda-quantum`` realtime tree installs
the DOCA GPUNetIO and Holoscan SDK packages with ``apt-get`` (it expects a
host or container that already ships Mellanox OFED), then clones, patches
and builds ``holoscan-sensor-bridge`` at branch ``EA_CUDAQ_10_26`` under
``$HSB_ROOT``.

.. code-block:: bash

   # 1. Clone cuda-quantum (realtime) at the ref recorded in cudaq-qec/.cudaq_version
   CUDAQ_REF=$(jq -r '.cudaq.ref' cudaq-qec/.cudaq_version)
   git clone --filter=blob:none --no-checkout \
     https://github.com/NVIDIA/cuda-quantum.git cudaq-realtime-src
   cd cudaq-realtime-src
   git sparse-checkout init --cone
   git sparse-checkout set realtime cmake
   git checkout "$CUDAQ_REF"
   cd ..

   # 2. Install DOCA/Holoscan SDK and build holoscan-sensor-bridge (EA_CUDAQ_10_26)
   export HSB_ROOT=/tmp/holoscan-sensor-bridge
   bash cudaq-realtime-src/realtime/scripts/install_devdeps.sh

   # 3. Build and install libcudaq-realtime with HSB tools enabled
   cmake -S cudaq-realtime-src/realtime -B cudaq-realtime-src/realtime/build \
     -G Ninja -DCMAKE_INSTALL_PREFIX=/tmp/cudaq-realtime \
     -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
     -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT \
     -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_ROOT/build
   cmake --build cudaq-realtime-src/realtime/build --target install

   # 4. Build cudaq-qec with HSB tools enabled
   cmake -S cudaq-qec -B cudaq-qec/build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAQ_DIR=/path/to/cudaq-install/lib/cmake/cudaq/ \
     -DCUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime \
     -DCUDAQ_QEC_BUILD_TRT_DECODER=ON \
     -DCUDAQX_ENABLE_LIBS="qec" \
     -DCUDAQX_INCLUDE_TESTS=ON \
     -DCUDAQX_QEC_ENABLE_HSB_TOOLS=ON \
     -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT \
     -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_ROOT/build
   cmake --build cudaq-qec/build --target \
     gpu_roce_predecoder_bridge \
     hsb_fpga_syndrome_playback \
     cudaq-qec-pymatching

.. note::

   The ``gpu_roce_predecoder_bridge`` target is skipped with a CMake warning
   (``gpu_roce_predecoder_bridge: skipped (missing deps)``) when the GpuRoce
   bridge libraries from step 3, TensorRT, or the ``cudaq-realtime-pipeline``
   library are not found.  Check the configure output of step 4 for that
   warning, and for ``RealtimePipeline: building``, before building the
   targets.

Emulated End-to-End Test
------------------------

The emulated test replaces the physical FPGA with a software emulator.  Three
processes run concurrently:

1. **Emulator** -- receives syndromes via the UDP control plane, sends them
   to the bridge via RDMA, and captures corrections
2. **Bridge** (``gpu_roce_predecoder_bridge``) -- receives RDMA data, runs the
   AI predecoder (TensorRT CUDA graph) and PyMatching decode via the
   ``realtime_pipeline``
3. **Playback** (``hsb_fpga_syndrome_playback``) -- loads syndrome data
   into the emulator's BRAM and triggers playback

Requirements
^^^^^^^^^^^^

- ConnectX NIC with a loopback cable connecting both ports
- Software dependencies (DOCA, Holoscan SDK, etc.) as described in the
  cuda-quantum realtime build guide
- All three tools built (bridge, playback, emulator)

Running
^^^^^^^

.. code-block:: bash

   ./libs/qec/unittests/realtime/gpu_roce_predecoder_test.sh \
     --emulate \
     --setup-network \
     --cuda-quantum-dir /path/to/cuda-quantum \
     --cuda-qx-dir /path/to/cudaq-qec \
     --data-dir /path/to/syndrome_data

The ``--setup-network`` flag configures the ConnectX interface with the
appropriate IP addresses and MTU.  It only needs to be run once per boot.

After the initial network setup, subsequent runs are faster:

.. code-block:: bash

   ./libs/qec/unittests/realtime/gpu_roce_predecoder_test.sh \
     --emulate \
     --cuda-quantum-dir /path/to/cuda-quantum \
     --cuda-qx-dir /path/to/cudaq-qec \
     --data-dir /path/to/syndrome_data


Bounded decoding-server regression (software emulator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The decoding-server regression harness can replay one BRAM-sized dataset for
longer endurance checks without loading more distinct samples. Use the
counter-capable software HSB emulator from the repository's ``.cudaq_version``
and an already configured RoCE loopback connection. The server, emulator,
syndrome player and surface-code data generator must already be built.

.. code-block:: bash

   ./libs/qec/unittests/utils/hsb_fpga_decoding_server_test.sh \
     --emulate --decoder pymatching \
     --cudaqx-dir /path/to/cudaq-qec \
     --cuda-quantum-dir /path/to/cuda-quantum \
     --hsb-dir /path/to/holoscan-sensor-bridge \
     --loop-seconds 30 --min-decodes 1000

The harness first verifies the corrections from one finite ILA capture, then
replays that same load for the requested duration. Acceptance requires at
least the requested number of completed measurements, matching sent/received
frame counts, and no transport or RPC-status failures. The initial verification
pass is excluded from the loop's completed-measurement count. Repeated
playback tests endurance, not accuracy on additional independent samples;
it also does not increase the maximum size of a single BRAM load.

``--loop-seconds`` and ``--min-decodes`` are emulator-only harness options;
they are rejected for real-FPGA mode before any network setup. The player's
``--loop`` option additionally allows manual continuous playback until Ctrl+C.
On hardware, only the finite preflight is correction-verified: emulator-only
cumulative counters must not be assumed to exist in the FPGA bitstream.
Without loop options, the existing single-pass playback path is unchanged.


FPGA End-to-End Test
--------------------

The FPGA test uses a real FPGA connected to the GPU via a ConnectX NIC.  Two
processes run:

1. **Bridge** (``gpu_roce_predecoder_bridge``) -- same as emulated mode
2. **Playback** (``hsb_fpga_syndrome_playback``) -- loads syndromes into
   the FPGA's BRAM and triggers RDMA playback to the bridge

Requirements
^^^^^^^^^^^^

- FPGA programmed with the HSB IP bitfile, connected to a ConnectX NIC via
  direct cable or switch
- FPGA IP and bridge IP on the same subnet
- ConnectX device name (e.g., ``mlx5_4``)

Running
^^^^^^^

.. code-block:: bash

   ./libs/qec/unittests/realtime/gpu_roce_predecoder_test.sh \
     --cuda-quantum-dir /path/to/cuda-quantum \
     --cuda-qx-dir /path/to/cudaq-qec \
     --data-dir /path/to/syndrome_data \
     --device mlx5_4 \
     --bridge-ip 192.168.0.1 \
     --fpga-ip 192.168.0.2 \
     --gpu 2 \
     --config d13_r104 \
     --timeout 60

Expected output:

.. code-block:: text

   ========================================
     GPU RoCE Predecoder + PyMatching Bridge Test
   ========================================

       Mode: Real FPGA (2-tool)
       Config: d13_r104
   ...
   [RDMA+TRT] Shot 0: received 17472 detectors (input_nonzero=939),
              predecoder logical_pred=1, residual_nonzero=23

   ========================================
     PREDECODER BRIDGE TEST: PASS
   ========================================

   === Results ===
     Total completed: 1
     Avg PyMatching decode: 211.0 us (1 samples)
     Shot 0: logical_pred=1 total_corrections=64 converged=1

This confirms: FPGA RDMA receipt (939 nonzero detectors), TensorRT inference
(reduced to 23 residuals), and PyMatching decode (64 corrections, converged).

Key parameters for FPGA mode:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``--device``
     - ConnectX IB device name (e.g., ``mlx5_4``)
   * - ``--bridge-ip``
     - IP address assigned to the ConnectX interface
   * - ``--fpga-ip``
     - FPGA's IP address
   * - ``--gpu``
     - GPU device ID (choose NUMA-local GPU for lowest latency)
   * - ``--config``
     - Pipeline configuration (e.g., ``d13_r104``)
   * - ``--data-dir``
     - Path to syndrome data directory
   * - ``--page-size``
     - Ring buffer slot size in bytes (auto-set per config by default; at
       least 512, the playback tool's minimum)
   * - ``--num-shots``
     - Number of syndrome shots to play back (limited by FPGA BRAM)

.. note::

   For d13_r104 (frame size 17,536 bytes), the default page size is 32,768
   bytes and the maximum number of shots per playback is 1 due to FPGA BRAM
   constraints.  Smaller configs (e.g. d7) can fit more shots.


GPU Selection
^^^^^^^^^^^^^

For lowest latency, choose a GPU that is NUMA-local to the ConnectX NIC.
For example, on a GB200 system where ``mlx5_4`` is on NUMA node 1,
use ``--gpu 2`` or ``--gpu 3``.  Check NUMA locality with:

.. code-block:: bash

   cat /sys/class/infiniband/<device>/device/numa_node

Network Sanity Check
^^^^^^^^^^^^^^^^^^^^

Before running, verify that the bridge IP is assigned to exactly one interface:

.. code-block:: bash

   ip addr show | grep 192.168.0.1

If multiple interfaces show the same IP, remove the duplicate to avoid
routing ambiguity that silently drops RDMA packets.


Changing the Predecoder Model
-----------------------------

The ONNX model file for each configuration is set in the ``PipelineConfig``
factory methods in
``libs/qec/unittests/realtime/predecoder_pipeline_common.h``.  To use a
different model, edit the ``onnx_filename`` field and rebuild:

.. code-block:: cpp

   static PipelineConfig d13_r104() {
       return {
           "d13_r104_X", 13, 104,
           "predecoder_memory_model_4_d13_T104_X.onnx",  // changed model
           8, 8, 16};
   }

Then rebuild:

.. code-block:: bash

   cmake --build build --target gpu_roce_predecoder_bridge


Orchestration Script Reference
------------------------------

.. code-block:: text

   gpu_roce_predecoder_test.sh [options]

Modes
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Flag
     - Description
   * - ``--emulate``
     - Use FPGA emulator (no real FPGA needed)
   * - *(default)*
     - FPGA mode (requires real FPGA)

Actions
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Flag
     - Description
   * - ``--setup-network``
     - Configure ConnectX network interfaces
   * - ``--no-run``
     - Skip running the test

Directory Options
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Flag
     - Default
     - Description
   * - ``--cuda-quantum-dir DIR``
     - ``/workspaces/cuda-quantum``
     - cuda-quantum source directory
   * - ``--cuda-qx-dir DIR``
     - ``/workspaces/cudaqx``
     - cudaq-qec source directory
   * - ``--data-dir DIR``
     - Per-config default
     - Syndrome data directory (expects ``detectors.bin``)

Network Options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Flag
     - Default
     - Description
   * - ``--device DEV``
     - auto-detect
     - ConnectX IB device name
   * - ``--bridge-ip ADDR``
     - ``10.0.0.1``
     - Bridge tool IP address
   * - ``--emulator-ip ADDR``
     - ``10.0.0.2``
     - Emulator IP (emulate mode only)
   * - ``--fpga-ip ADDR``
     - ``192.168.0.2``
     - FPGA IP address
   * - ``--mtu N``
     - ``4096``
     - MTU size

Run Options
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--config NAME``
     - ``d13_r104``
     - Pipeline config (``d7``, ``d13``, ``d13_r104``, ``d21``, ``d31``)
   * - ``--gpu N``
     - ``0``
     - GPU device ID
   * - ``--timeout N``
     - ``60``
     - Timeout in seconds
   * - ``--num-shots N``
     - Per-config
     - Number of syndrome shots (limited by FPGA BRAM)
   * - ``--page-size N``
     - Per-config
     - Ring buffer slot size in bytes (at least 512)
   * - ``--num-pages N``
     - ``64``
     - Number of ring buffer slots (power of two; also the pipeline ring
       depth)
   * - ``--spacing N``
     - *(unset)*
     - Inter-shot spacing in microseconds
   * - ``--control-port N``
     - ``8193``
     - UDP control port for emulator

See Also
--------

* :doc:`Realtime Decoding </components/qec/realtime_decoding>` -- concept and workflow
* :doc:`Getting Started with Realtime Decoding </examples_rst/qec/getting_started_realtime_decoding>`
* :ref:`C++ <cpp_realtime_decoding_api>` and :ref:`Python <python_realtime_decoding_api>` realtime decoding API

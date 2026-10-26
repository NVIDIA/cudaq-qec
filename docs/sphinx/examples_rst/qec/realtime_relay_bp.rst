Relay BP Decoding with CUDA-Q Realtime
========================================

.. note::

  The following information is about a C++ demonstration that must be built
  from source and is not part of any distributed CUDA-Q QEC binaries.

This guide explains how to build, test, and run the nv-qldpc-decoder Relay BP
decoder using CUDA-Q's realtime dispatch system.  The decoder is driven by a
**self-relaunching device-graph scheduler** and can operate in two
configurations:

- **Emulated end-to-end test** -- software FPGA emulator replaces real hardware
- **FPGA end-to-end test** -- real FPGA connected via ConnectX RDMA/RoCE

.. tip::

   To run the nv-qldpc Relay BP decoder end-to-end against the delivered
   ``decoding_server`` -- from either a real FPGA (GPU device-graph path) or a
   lowered QPU kernel over UDP (GPU host-call path) -- use the
   :doc:`/examples_rst/qec/realtime_decoding_demo` example with
   ``--decoder nv-qldpc-decoder --gpu 0``. That example compiles against the
   installed SDK and resolves the server + playback tool from the install
   prefix. This page documents the underlying device-graph scheduler and its
   emulator-based testing.

Decode dispatch architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The realtime path uses the per-round decode-server protocol with three RPCs:
``enqueue_syndromes`` (append one round of syndromes), ``get_corrections``
(read the logical correction for a completed shot), and ``reset_decoder``.
These are serviced by a single GPU **device-graph scheduler** -- a persistent,
self-relaunching CUDA graph:

- All three RPCs are ``DEVICE_CALL`` handlers.  ``enqueue_syndromes``
  accumulates a round's syndromes into the decoder's device-resident state;
  when a full window has accumulated it returns a sentinel
  (``CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH``) that tells the scheduler to fire
  the decode.
- The Relay BP decode is captured as a **device-launchable cooperative CUDA
  graph** and launched *fire-and-forget* from the scheduler when a window is
  ready.  ``get_corrections`` then reads the result.
- After firing a decode the scheduler **tail self-relaunches**
  (``cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch)``),
  which resets the 120 fire-and-forget-launch-per-parent-execution budget --
  so an unbounded number of decodes can be dispatched without the host in the
  loop.  The tail launch is ordered after the fired decode, so
  ``get_corrections`` always observes the finished result.

This replaces the earlier ``HOST_LOOP`` design (a CPU thread launching one
graph per request).  ``libcudaq-realtime`` provides the scheduler
(``cudaq_create_dispatch_graph_regular`` / ``cudaq_launch_dispatch_graph`` in
``dispatch_kernel.cu``); the closed-source proprietary archive provides the
``DEVICE_CALL`` handlers (see *Obtaining the proprietary components* below).

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

Tested platforms: DGX Spark, GB200.

Software
^^^^^^^^

- **CUDA Toolkit**: 12.6 or later
- **CUDA-Q SDK**: pre-installed (provides ``libcudaq``, ``libnvqir``, ``nvq++``)
- **nv-qldpc-decoder plugin**: the proprietary nv-qldpc-decoder shared library
  (``libcudaq-qec-nv-qldpc-decoder.so``).  Required at runtime for all
  three configurations -- see *Obtaining the nv-qldpc-decoder plugin* below
  for how to install it.

Obtaining the proprietary components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The realtime decode path uses **two** closed-source artifacts that are not
built from this (cudaq-qec) repository:

- ``libcudaq-qec-nv-qldpc-decoder.so`` -- the Relay BP decoder **plugin**,
  ``dlopen``'d at runtime.  It supplies the device-launchable cooperative
  decode graph (``capture_decode_graph``).
- ``libcudaq-qec-realtime-cudevice-proprietary.a`` -- a static **archive**
  needed at **build** time.  It contains the ``enqueue_syndromes`` /
  ``get_corrections`` / ``reset_decoder`` ``DEVICE_CALL`` handlers (the device
  functions the scheduler dispatches).  It is linked ``WHOLE_ARCHIVE`` and
  device-linked into the bridge, and is pointed at via the
  ``-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=<path>`` CMake variable.
  Both artifacts come from the same closed-source decoder package; build the
  ``cudaq-qec-realtime-cudevice-proprietary`` target from the proprietary
  decoder sources to produce the ``.a``.

The plugin must be obtained as a pre-built binary as shown below.

.. important::

   The ``cudaq-qec`` PyPI wheel ships a version of this plugin that does
   **not** support the realtime graph-dispatch path used by these tests.
   The wheel build pipeline does not enable ``CUDAQ_REALTIME_ROOT``, so the
   graph-dispatch overrides are compiled out.  Using the wheel's plugin
   causes the bridge tool to abort with::

      ERROR: nv-qldpc-decoder does not support graph dispatch

   Use the version from the ``ghcr.io/nvidia/cudaqx`` container instead, as
   shown below.

Extract the plugin from the container without running it:

.. code-block:: bash

   IMAGE=ghcr.io/nvidia/cudaqx:cu13-latest    # or cu12-latest

   docker pull "$IMAGE"
   CID=$(docker create "$IMAGE")
   docker cp "$CID:/opt/nvidia/cudaq/lib/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so" .
   docker rm "$CID"

``docker pull`` automatically selects the correct image variant for the host
CPU architecture (``amd64`` or ``arm64``) -- no manual override is needed.

The runtime plugin loader searches for decoder plugins in the
``decoder-plugins/`` subdirectory next to ``libcudaq-qec.so``.  For a source
build of cudaq-qec, that path is ``<cudaq-qec-build>/lib/decoder-plugins/``:

.. code-block:: bash

   mkdir -p <cudaq-qec-build>/lib/decoder-plugins
   cp libcudaq-qec-nv-qldpc-decoder.so \
      <cudaq-qec-build>/lib/decoder-plugins/

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
     - Branch ``releases/v0.15.1``
   * - **holoscan-sensor-bridge**
     - https://github.com/nvidia-holoscan/holoscan-sensor-bridge
     - Tag ``2.6.0-EA2``

``cuda-quantum`` provides ``libcudaq-realtime`` (the dispatch kernel, ring
buffer management, and the device-graph scheduler).  ``holoscan-sensor-bridge``
provides the ``GpuRoceTransceiver`` library for RDMA transport.

.. note::

   The self-relaunching device-graph scheduler is provided by the
   ``releases/v0.15.1`` branch of ``cuda-quantum`` (the extension that adds the
   ``CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH`` sentinel, the triggered
   fire-and-forget decode launch, and tail self-relaunch on top of the
   device-side graph dispatch).

Repository Layout
-----------------

Key files within ``cudaq-qec``:

.. code-block:: text

   libs/qec/
     unittests/
       realtime/
         qec_roce_decode_test/
           data/
             config_nv_qldpc_relay.yml              # Relay BP decoder config
             syndromes_nv_qldpc_relay.txt           # 100 test syndrome shots
       utils/
         gpu_roce_qldpc_graph_decoder_bridge.cpp    # Bridge tool (RDMA <-> decoder)
         gpu_roce_qldpc_graph_decoder_test.sh       # Orchestration script
         hsb_fpga_syndrome_playback.cpp        # Playback tool (loads syndromes)

The FPGA emulator is in the ``cuda-quantum`` repository:

.. code-block:: text

   cuda-quantum/realtime/
     unittests/utils/
       hsb_fpga_emulator.cpp                   # Software FPGA emulator

Building
--------

Build the HSB bridge and playback tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To also build the bridge and playback tools for emulated or FPGA testing:

.. code-block:: bash

   # 1. Clone cuda-quantum (realtime)
   git clone --filter=blob:none --no-checkout \
     https://github.com/NVIDIA/cuda-quantum.git cudaq-realtime-src
   cd cudaq-realtime-src
   git sparse-checkout init --cone
   git sparse-checkout set realtime
   git checkout releases/v0.15.1
   cd ..

   # 2. Build holoscan-sensor-bridge (tag 2.6.0-EA2)
   #    Requires cmake >= 3.30.4 (HSB -> find_package(holoscan) -> rapids_logger).
   #    If your system cmake is older: pip install cmake
   git clone --branch 2.6.0-EA2 \
     https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
   cd holoscan-sensor-bridge

   # Strip operators we don't need to avoid configure failures from missing deps
   sed -i '/add_subdirectory(audio_packetizer)/d; /add_subdirectory(compute_crc)/d;
           /add_subdirectory(csi_to_bayer)/d; /add_subdirectory(image_processor)/d;
           /add_subdirectory(iq_dec)/d; /add_subdirectory(iq_enc)/d;
           /add_subdirectory(linux_coe_receiver)/d; /add_subdirectory(linux_receiver)/d;
           /add_subdirectory(packed_format_converter)/d; /add_subdirectory(sub_frame_combiner)/d;
           /add_subdirectory(udp_transmitter)/d; /add_subdirectory(emulator)/d;
           /add_subdirectory(sig_gen)/d; /add_subdirectory(sig_viewer)/d' \
     src/hololink/operators/CMakeLists.txt

   mkdir -p build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
     -DHOLOLINK_BUILD_PYTHON=OFF \
     -DHOLOLINK_BUILD_TESTS=OFF \
     -DHOLOLINK_BUILD_TOOLS=OFF \
     -DHOLOLINK_BUILD_EXAMPLES=OFF \
     -DHOLOLINK_BUILD_EMULATOR=OFF ..
   cmake --build . --target gpu_roce_transceiver hololink_core
   cd ../..

   # 3. Build libcudaq-realtime with HSB tools enabled
   #    This produces libcudaq-realtime-bridge-gpu-roce.so (needed by the bridge
   #    tool) as well as the FPGA emulator.
   cd cudaq-realtime-src/realtime && mkdir -p build && cd build
   cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/tmp/cudaq-realtime \
     -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
     -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=../../holoscan-sensor-bridge \
     -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=../../holoscan-sensor-bridge/build \
     ..
   ninja && ninja install
   cd ../../..

   # 4. Build cudaq-qec with HSB tools enabled.
   #    CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE supplies the DEVICE_CALL
   #    handlers (WHOLE_ARCHIVE-linked into the bridge + test).
   cmake -S cudaq-qec -B cudaq-qec/build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAQ_DIR=/path/to/cudaq-install/lib/cmake/cudaq/ \
     -DCUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime \
     -DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=/path/to/libcudaq-qec-realtime-cudevice-proprietary.a \
     -DCUDAQX_ENABLE_LIBS="qec" \
     -DCUDAQX_INCLUDE_TESTS=ON \
     -DCUDAQX_QEC_ENABLE_HSB_TOOLS=ON \
     -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/path/to/holoscan-sensor-bridge \
     -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/path/to/holoscan-sensor-bridge/build
   cmake --build cudaq-qec/build --target \
     gpu_roce_qldpc_graph_decoder_bridge \
     hsb_fpga_syndrome_playback

Using the orchestration script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The orchestration script can build everything automatically:

.. code-block:: bash

   ./libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_test.sh \
     --build \
     --hsb-dir /path/to/holoscan-sensor-bridge \
     --cuda-quantum-dir /path/to/cuda-quantum \
     --no-run

Surface Code Test (Relay BP)
----------------------------

The ``surface_code-1-local`` app example runs a surface code memory experiment
with the nv-qldpc-decoder configured for Relay BP.  It simulates a surface code
with ``stim`` and generates syndromes on the fly, so it can run an arbitrary
number of shots.

Build the app example (it links the same plugin + proprietary archive as the
bridge):

.. code-block:: bash

   cmake --build cudaq-qec/build --target surface_code-1-local

Run it in two steps -- generate the decoder config (DEM), then run the shots:

.. code-block:: bash

   cd cudaq-qec/build
   export CUDAQ_DEFAULT_SIMULATOR=stim

   APP=./libs/qec/unittests/realtime/app_examples/surface_code-1-local

   # 1. Generate the Relay BP decoder config (DEM) for a distance-3 surface code
   "$APP" --distance 3 --num_rounds 12 \
          --decoder_type nv-qldpc-decoder \
          --num_shots 1000 --save_dem config.yml

   # 2. Run the shots
   "$APP" --distance 3 --num_rounds 12 \
          --decoder_type nv-qldpc-decoder \
          --num_shots 1000 --load_dem config.yml

A clean run exits ``0`` and reports a small number of non-zero syndrome
measurements alongside a larger number of corrections found.

Emulated End-to-End Test
------------------------

The emulated test replaces the physical FPGA with a software emulator.  Three
processes run concurrently:

1. **Emulator** -- receives syndromes via the UDP control plane, sends them
   to the bridge via RDMA, and captures corrections
2. **Bridge** -- runs the device-graph scheduler on the GPU directly on the
   GpuRoceTransceiver DOCA ring (the scheduler polls the RX flags written by the
   GpuRoceTransceiver RX kernel and writes responses for the TX kernel), firing the
   cooperative Relay BP decode fire-and-forget per completed shot
3. **Playback** -- loads syndrome data into the emulator's BRAM and triggers
   playback in **per-round** mode (``--per-round``: N ``enqueue_syndromes``
   frames + one ``get_corrections`` per shot), then verifies corrections

.. note::

   The orchestration script drives the playback tool in ``--per-round`` mode
   automatically (matching the decode-server protocol the scheduler speaks).
   The playback tool also retains a shot-based default for other decoders; the
   per-round path is opt-in via ``--per-round``.

Requirements
^^^^^^^^^^^^

- ConnectX NIC with a loopback cable connecting both ports (the emulator
  sends RDMA traffic out one port and the bridge receives on the other)
- Software dependencies (DOCA, Holoscan SDK, etc.) as described in the
  `cuda-quantum realtime build guide <https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/building.md>`__
- All three tools built (bridge, playback, emulator)

Running the Emulated Test
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_test.sh \
     --emulate \
     --build \
     --setup-network \
     --hsb-dir /path/to/holoscan-sensor-bridge

The ``--setup-network`` flag configures the ConnectX interface with the
appropriate IP addresses and MTU.  It only needs to be run once per boot.

After the initial build and network setup, subsequent runs are faster:

.. code-block:: bash

   ./libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_test.sh --emulate

FPGA End-to-End Test
--------------------

The FPGA test uses a real FPGA connected to the GPU via a ConnectX NIC.  Two
processes run:

1. **Bridge** -- same as emulated mode
2. **Playback** -- loads syndromes into the FPGA's BRAM and triggers playback,
   then reads back corrections from the FPGA's capture RAM to verify them

FPGA Requirements
^^^^^^^^^^^^^^^^^

- FPGA programmed with the HSB IP bitfile, connected to a ConnectX NIC via
  direct cable or switch.  Bitfiles for supported FPGA vendors are available
  `here <https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/QEC/HSB-2.6.0-EA/>`__.
  See the `cuda-quantum realtime user guide <https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/user_guide.md>`__
  for FPGA setup instructions.
- FPGA IP and bridge IP on the same subnet
- ConnectX device name (e.g., ``mlx5_4``, ``mlx5_5``)

Running the FPGA Test
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_test.sh \
     --build \
     --setup-network \
     --device mlx5_5 \
     --bridge-ip 192.168.0.1 \
     --fpga-ip 192.168.0.2 \
     --gpu 2 \
     --page-size 512 \
     --spacing 100 \
     --hsb-dir /path/to/holoscan-sensor-bridge

``--spacing`` is **important for the FPGA** (it is not needed for the
emulator).  The FPGA's BRAM player is **open-loop** -- it transmits a frame
every ``--spacing`` microseconds on a fixed hardware timer, with no
backpressure -- whereas the emulator naturally paces itself by waiting for each
response.  Without adequate spacing the FPGA outruns the decoder, the input
ring fills, and frames are lost.  See the note below for sizing.

Key parameters for FPGA mode:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``--device``
     - ConnectX IB device name (e.g., ``mlx5_5``)
   * - ``--bridge-ip``
     - IP address assigned to the ConnectX interface
   * - ``--fpga-ip``
     - FPGA's IP address
   * - ``--gpu``
     - GPU device ID (choose NUMA-local GPU for lowest latency)
   * - ``--page-size``
     - Ring buffer slot size in bytes (use ``512`` on GB200 for alignment)
   * - ``--spacing``
     - Inter-**frame** spacing in microseconds (FPGA BRAM-player timer)

.. note::

   **Sizing the spacing.**  In per-round mode each shot is ``rounds + 1``
   frames (N ``enqueue_syndromes`` + one ``get_corrections``) but only one
   decode, so the decoder consumes roughly one ``decode_time`` per shot.  Since
   ``--spacing`` is the gap between *frames*, the sustained-safe value is

   .. code-block:: text

      spacing >= decode_time / (rounds + 1)

   For this ``[[8,3,6]]`` relay-BP config (~200 us decode, 4 rounds -> 5
   frames/shot) that is ``>= ~40 us``.  Start **conservative** (e.g.
   ``--spacing 100``) for the first run to rule out ring overrun while
   confirming corrections, then tune down toward ``~50 us`` for a realistic
   latency profile.  If frames are still dropped/duplicated at generous
   spacing, the cause is *not* ring overrun -- investigate the FPGA capture
   (ILA) side.

GPU Selection
^^^^^^^^^^^^^

For lowest latency, choose a GPU that is NUMA-local to the ConnectX NIC.
For example, on a GB200 system where ``mlx5_5`` is on NUMA node 1,
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

Orchestration Script Reference
------------------------------

.. code-block:: text

   gpu_roce_qldpc_graph_decoder_test.sh [options]

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
   * - ``--build``
     - Build all required tools before running
   * - ``--setup-network``
     - Configure ConnectX network interfaces
   * - ``--no-run``
     - Skip running the test (useful with ``--build``)

Build Options
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Flag
     - Default
     - Description
   * - ``--hsb-dir DIR``
     - ``/workspaces/holoscan-sensor-bridge``
     - holoscan-sensor-bridge source directory
   * - ``--cuda-quantum-dir DIR``
     - ``/workspaces/cuda-quantum``
     - cuda-quantum source directory
   * - ``--cuda-qx-dir DIR``
     - ``/workspaces/cudaqx``
     - cudaq-qec source directory
   * - ``--jobs N``
     - ``nproc``
     - Parallel build jobs

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
     - Emulator IP (emulate mode)
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
   * - ``--gpu N``
     - ``0``
     - GPU device ID
   * - ``--timeout N``
     - ``60``
     - Timeout in seconds
   * - ``--num-shots N``
     - all available
     - Limit number of syndrome shots
   * - ``--page-size N``
     - ``384``
     - Ring buffer slot size in bytes
   * - ``--spacing N``
     - ``10``
     - Inter-shot spacing in microseconds
   * - ``--no-verify``
     - *(verify)*
     - Skip correction verification
   * - ``--control-port N``
     - ``8193``
     - UDP control port for emulator

Ring buffer depth (``num_pages``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ring depth is intentionally **not** a script option and is fixed at
**64** (both the bridge and playback default to it).  This matches the
GpuRoceTransceiver work-queue depth ``WQE_NUM = 64``: the
transceiver posts 64 receive/send WQEs and runs one kernel thread per WQE.

A ring deeper than ``WQE_NUM`` makes a single transceiver thread service more
than one ring slot (slot ``t`` and slot ``t+64`` share one WQE / CQ position),
and the free-running RX/TX kernels then race on that shared resource.  On the
emulator this was observed as a rare (~1-2%) **duplicated frame ``W`` plus a
dropped frame ``W+64``** -- every failure was an exact ``(W, W+64)`` pair on a
single thread.  A 1:1 slot-to-WQE mapping (``num_pages <= WQE_NUM``) is the
only safe configuration and is collision-free.

The bridge enforces this: if ``--num-pages`` is ever passed with a value above
``WQE_NUM``, it clamps to 64 and prints a warning.  Supporting a deeper ring
would require changing ``WQE_NUM`` (and the per-thread WQE striding) in
``holoscan-sensor-bridge``, diverging from the ``2.6.0-EA2`` tag.

See Also
--------

* :doc:`Realtime Decoding </components/qec/realtime_decoding>` -- concept and workflow
* :doc:`Getting Started with Realtime Decoding </examples_rst/qec/getting_started_realtime_decoding>`
* :ref:`C++ <cpp_realtime_decoding_api>` and :ref:`Python <python_realtime_decoding_api>` realtime decoding API

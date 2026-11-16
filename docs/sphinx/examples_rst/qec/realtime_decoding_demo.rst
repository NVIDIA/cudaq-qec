Realtime Decoding from a Quantum Kernel or FPGA
===============================================

This example sends measurements to a standalone decoding server from either
a running quantum kernel or a real FPGA. The quantum-kernel route executes a
surface-code circuit in a simulator, streams its measurements, and receives
corrections during execution. The FPGA route sends pre-generated measurements
over RoCE and checks the returned corrections.

Start with the software UDP route and a CPU decoder; it needs no QPU, FPGA, or
RDMA hardware. The later sections cover RDMA and FPGA input, optional Ising
predecoding, and an additional :ref:`packaged surface-code client
<decoding-server-packaged-client>`.

For why you would deploy a server and how it is configured, see
:doc:`Decoding Server </components/qec/decoding_server>`. To test the server
with scheduled measurements instead of a quantum circuit, see
:doc:`Realtime Decoding with the Playback Emulator <decoding_server_playback>`.

.. note::

   This page describes a C++ example that you compile against the *installed*
   CUDA-Q QEC SDK. The decoding server and the FPGA playback tool it drives are
   **deliverables** — prebuilt and installed alongside the SDK, not built from
   this example. The SDK, both deliverables and this example ship in the
   CUDA-Q decoding container, published on the NGC catalog at
   https://catalog.ngc.nvidia.com/orgs/nvidia/quantum/containers/cudaq-decoding:

   .. code-block:: bash

      docker run -it --gpus all nvcr.io/nvidia/quantum/cudaq-decoding:cu13-latest
      cd ~/examples/realtime_decoding_demo

   Alternatively, use the ``cu12-latest`` tag for CUDA 12. The paths and
   commands on this page are as used inside that container. For the CPU-only
   UDP examples, omit ``--gpus all`` from the container command. The ``cpu_roce``
   and FPGA paths additionally need the host's network and RDMA devices passed
   through, for example:

   .. code-block:: bash

      docker run -it --gpus all --net host --privileged \
          --device /dev/infiniband -v /dev:/dev --shm-size 1g \
          nvcr.io/nvidia/quantum/cudaq-decoding:cu13-latest

This example drives the delivered ``decoding_server`` from two different
syndrome sources, both decoding through the **same prebuilt server**:

- ``--source qpu-kernel`` — a lowered QEC kernel supplies syndromes itself over
  **UDP**, in software. No NIC, no FPGA. This is the portable, hardware-free
  path and is what CI exercises. With ``--wire cpu_roce`` the same kernel
  carries its syndromes over **real RDMA** between two RoCE-capable ports
  instead.
- ``--source fpga`` — the delivered ``hsb_fpga_syndrome_playback`` tool
  streams pre-generated syndromes over **RoCE from a real FPGA** into the
  server's RDMA ring. Requires a ConnectX NIC cabled to the FPGA. There is no
  emulator here.

Both sources decode with any of three decoders — ``pymatching``,
``multi_error_lut`` (CPU lookup table), and ``nv-qldpc-decoder`` (GPU
relay-BP) — plus an opt-in fourth, ``trt_decoder`` (the Ising neural-network
predecoder on TensorRT + a PyMatching global decoder; see
`Ising decoder (TensorRT predecoder + PyMatching)`_). Runs are seeded
(``--seed``, default 42), so the reported counts are reproducible.

For the software-only pymatching benchmark see
:doc:`/examples_rst/qec/realtime_predecoder_pymatching`.


Deliverables versus the example
-------------------------------

**Deliverables** (installed, not built by this example), at these paths in
the container:

- ``/opt/nvidia/cudaq-qec/bin/decoding_server`` — the standalone server that
  owns the decoder instances.
- ``/opt/nvidia/cudaq-qec/bin/hsb_fpga_syndrome_playback`` — the FPGA
  playback tool.
- ``/opt/nvidia/cudaq-qec/lib`` — the QEC libraries, with the decoder plugins
  in ``/opt/nvidia/cudaq-qec/lib/decoder-plugins``.
- ``/opt/nvidia/cudaq-realtime`` — the realtime libraries.
- ``/opt/nvidia/cudaq`` — CUDA-Q itself (``nvq++``, headers and runtime).

The container exports these locations as ``CUDAQX_INSTALL_DIR``,
``CUDAQ_REALTIME_DIR`` and ``CUDA_QUANTUM_PATH``. The build and the run script
below read those variables, so no path flags are needed inside the container.

**The example** (the only thing you compile) is two sources, each built two
ways against the installed SDK — using only installed headers and libraries:

- ``surface_code_realtime_decoding`` — the **generator** (``--target stim``):
  writes the decoder configuration and, for the FPGA source, a syndrome file.
- ``surface_code_realtime_decoding-cqr`` — the **lowered kernel**
  (``-frealtime-lowering -DQEC_APP_CQR``): the live
  syndrome source that streams to the server over UDP or cpu_roce.
- ``surface_code_ising_realtime_decoding`` / ``…-cqr`` — the same pair for the
  opt-in Ising ``trt_decoder`` profile (a separate source whose measurement
  layout matches the Ising artifacts; see the Ising section below).


Building the example
--------------------

In the container the example lives at ``~/examples/realtime_decoding_demo``
and the prefixes come from the environment:

.. code-block:: bash

   cd ~/examples/realtime_decoding_demo
   cmake -S . -B build
   cmake --build build

Against an SDK installed elsewhere, pass the prefixes explicitly:

.. code-block:: bash

   cmake -S . -B build \
     -DCUDAQ_INSTALL_DIR=<cuda-quantum install prefix> \
     -DCUDAQX_INSTALL_DIR=<cuda-qx install prefix>
   cmake --build build

In the decoding container, ``CUDAQ_INSTALL_DIR`` defaults to ``/opt/nvidia/cudaq``
and ``CUDAQX_INSTALL_DIR`` to ``/opt/nvidia/cudaq-qec``.
If the realtime libraries are in a separate prefix, add
``-DCUDAQ_REALTIME_DIR=<realtime prefix>``. The lowered kernel links the
realtime dispatch archive (relocatable CUDA device code), so the build needs a
CUDA toolchain; the device-link architecture defaults to ``80`` (Ampere) —
override with ``-DCMAKE_CUDA_ARCHITECTURES=90`` for Hopper or
``-DCMAKE_CUDA_ARCHITECTURES=100`` for Blackwell (e.g. GB200).


Running
-------

``run_realtime_decoding.sh`` resolves the deliverables from ``--install-prefix``
and the two example binaries from ``--example-build-dir`` (default ``./build``).
In the decoding container ``--install-prefix`` defaults to
``/opt/nvidia/cudaq-qec`` and can be omitted, as in the commands below.

QPU-kernel source (software, UDP, no hardware)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching
   ./run_realtime_decoding.sh --source qpu-kernel --decoder multi_error_lut
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0

The lowered kernel runs the surface-code memory experiment and streams each
shot's syndromes to the server over the ``udp`` wire; every decoder is served
on ``host`` dispatch (a server CPU thread calls the decoder — nv-qldpc still
decodes on its GPU) and the corrections come back before readout.

QPU-kernel source over real RDMA (``--wire cpu_roce``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same lowered kernel can carry its syndromes over the ``cpu_roce`` wire
instead: its channel RDMA-writes each request straight into the server's ring.
This needs two RoCE-capable ports that can reach each other (kernel side =
*channel*, server side = *daemon*) — e.g. loopback-cabled ConnectX ports, or
SoftRoCE (``rdma_rxe``) devices. The topology comes from the same environment
variables as the in-tree cpu_roce tests; ``--setup-network`` assigns the IPs
and waits for the RoCE v2 GIDs:

.. code-block:: bash

   export CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE=<kernel-side ibdev>   # e.g. mlx5_0
   export CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE=<server-side ibdev>    # e.g. mlx5_1
   # IPs default to 10.0.0.1 (channel) / 10.0.0.2 (daemon); override with
   # CUDAQ_CPU_ROCE_TEST_CHANNEL_IP / CUDAQ_CPU_ROCE_TEST_DAEMON_IP.

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching            --wire cpu_roce --setup-network
   ./run_realtime_decoding.sh --source qpu-kernel --decoder multi_error_lut       --wire cpu_roce --setup-network
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --wire cpu_roce --setup-network

The server's READY ``port=`` is then the TCP rendezvous port (the RDMA wire
itself is negotiated via the QP/rkey exchange); dispatch stays ``host``, and
the same pass/fail criteria apply unchanged.

FPGA source (real FPGA over RoCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source fpga --decoder pymatching \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip>

   ./run_realtime_decoding.sh --source fpga --decoder multi_error_lut \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip>

   ./run_realtime_decoding.sh --source fpga --decoder nv-qldpc-decoder --gpu 0 \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip>

``--setup-network`` configures the ConnectX interface (needs ``sudo``);
``--spacing`` (default 10 µs) paces the playback so it does not overrun the
server's 64-slot RDMA RX ring. The playback timer fires once per **frame**
(one BRAM window), so a shot spans frames-per-shot × spacing. The
``nv-qldpc-decoder`` profiles auto-pace slower — 100 µs per frame — because
the GPU decode cannot drain the ring at 10 µs; an explicit ``--spacing``
always wins (``trt_decoder`` auto-paces the same way).

The server has two independent knobs, both derived automatically (override
with ``--wire`` / ``--dispatch``): the **wire** is the bridge-provider library
that carries syndromes into the server, and the **dispatch** is the engine
consuming each decoder's ring. ``pymatching``, ``multi_error_lut`` and
``trt_decoder`` ride the ``cpu_roce`` wire on ``host`` dispatch (a server CPU
thread calls the decoder). ``nv-qldpc-decoder --gpu 0`` defaults to the
``gpu_roce`` wire on ``device_graph`` dispatch, where a GPU device-call
scheduler fires the Relay BP decode graph on-device. That mode needs a
``decoding_server`` built with the device-graph component, which released
builds do not include: against a released server the script reports the
reason and exits ``77`` (skip). Pass ``--dispatch host`` to decode with
nv-qldpc on host dispatch over ``cpu_roce`` instead.

.. note::

   ``decoding_server`` also accepts ``gpu_roce`` as a transport provider for
   host-dispatch rings. No released decoder uses that path; this example
   derives ``gpu_roce`` only for the device_graph mode described above.

The script uses the default host loop. See :ref:`decoding-server-host-loops`
for loop options when launching the server directly.

Both sources apply real pass/fail criteria. The qpu-kernel source uses the
same checks as the in-tree surface-code tests — no decoder errors, a residual
logical-error ceiling of ``num_shots/50``, an in-process dispatch count of 0
(proof the decode stayed in the external server), and a server dispatch-count
floor of ``num_shots * (num_rounds + 3)``. The FPGA source verifies every
shot's corrections against the expected values computed at
syndrome-generation time (the playback tool's per-shot verification).


.. _decoding-server-ising-example:

Ising decoder (TensorRT predecoder + PyMatching)
------------------------------------------------

The ``trt_decoder`` profile decodes with the published **Ising neural-network
predecoder** (a TensorRT engine built from its ONNX export) chained into a
PyMatching global decoder — the same decoder stack, served by the same
``decoding_server``, over any of the wires above (host dispatch only).

This profile is **opt-in** and never runs in CI: the model weights are a gated
Hugging Face download, and the decoder inputs are prepared locally into a
six-file *artifact directory* passed via ``--ising-artifacts-dir`` (or the
``QEC_ISING_ARTIFACTS_DIR`` environment variable):

``model.onnx`` (the exported predecoder), ``H_csr.bin`` / ``O_csr.bin``
(detector-error and observables matrices, Ising detector order),
``priors.bin`` (per-mechanism priors), ``metadata.txt`` (the contract:
``basis=Z``, ``code_rotation=XV``, ``distance=7``, ``n_rounds=7``), and
``D_sparse.txt`` (each Ising detector as a parity over this example's live
measurement stream). A missing or incomplete directory makes the profile
**skip** (exit 77) listing the absent files by name.

The profile is pinned to the published model's operating point: distance 7,
7 rounds, Z basis, XV orientation, SPAM noise ``--p-spam`` (default 0.01,
replacing ``--p-cnot``). It runs its own example pair —
``surface_code_ising_realtime_decoding``/``-cqr``, built from the same
CMake configure — whose measurement layout matches the artifacts' binding
(the sibling ``surface_code_realtime_decoding`` source uses a different
orientation and per-round bit order and cannot consume these artifacts).

.. code-block:: bash

   ./run_realtime_decoding.sh --source qpu-kernel --decoder trt_decoder \
       --ising-artifacts-dir <dir>

   ./run_realtime_decoding.sh --source qpu-kernel --decoder trt_decoder --wire cpu_roce \
       --setup-network --ising-artifacts-dir <dir>

   ./run_realtime_decoding.sh --source fpga --decoder trt_decoder \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --ising-artifacts-dir <dir>

On the FPGA source the playback BRAM (512 frames) caps this geometry at **56
shots** per run (9 frames per shot: 8 syndrome slices + 1 corrections frame);
the script defaults to exactly that, and also raises the per-frame
``--spacing`` to 100 µs for this profile — at the default 10 µs the stream
outruns the served rate and overruns the server's 64-slot RX ring (about 7
shots of buffer).

Preparing the artifact directory (once, on a machine with an NVIDIA GPU —
the pinned workflow runs GPU inference for the export) is automated by the
``prepare_ising_artifacts.py`` script that ships in this example's directory:

1. Accept the gated model terms at
   ``https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast`` and
   authenticate the ``hf`` CLI (``hf auth login``). The script also needs
   ``git``, ``bash``, and the Ising exporter's Python packages
   (``beliefmatching hydra ldpc matplotlib numpy omegaconf onnx pymatching
   safetensors scipy stim torch``; on aarch64 install torch from a CUDA
   wheel index).
2. Run (from this example's directory, after building the example)::

      python prepare_ising_artifacts.py \
          --app build/surface_code_ising_realtime_decoding \
          --artifacts-dir <dir>

   This sparse-checkouts the pinned Ising-Decoding revision, downloads the
   gated weights, exports the Z-basis ONNX model, generates the decoder
   matrices, captures THIS example's printed CNOT schedules to build a
   matching ``D_sparse.txt``, validates the six files, and installs them
   atomically at ``<dir>``.

For a custom Ising export, the script's ``d-sparse`` subcommand regenerates
just the measurement mapping from a saved schedule printout — see
``python prepare_ising_artifacts.py d-sparse --help``.

Decoders
--------

.. list-table::
   :header-rows: 1

   * - decoder
     - qpu-kernel (udp / cpu_roce wire)
     - fpga (real FPGA)
     - extra requirement
   * - ``pymatching``
     - CPU (udp: no hardware)
     - NIC
     - none
   * - ``multi_error_lut``
     - CPU (udp: no hardware)
     - NIC
     - none
   * - ``nv-qldpc-decoder``
     - GPU (host dispatch)
     - NIC + GPU (device_graph dispatch; internal builds only, else ``--dispatch host``)
     - plugin + ``--gpu``
   * - ``trt_decoder``
     - GPU (host dispatch)
     - NIC + GPU (host dispatch)
     - TRT plugin + Ising artifacts (opt-in)

The ``nv-qldpc-decoder`` profile needs the prebuilt plugin — auto-found in the
install prefix (``/opt/nvidia/cudaq-qec/lib/decoder-plugins`` in the
container), else ``--nv-qldpc-plugin <path.so>`` — and a GPU; if the plugin
is unavailable the script exits ``77`` (skip).


.. _decoding-server-packaged-client:

Packaged Surface-Code Client
----------------------------

This additional example uses the ``surface_code-1`` program packaged under
``/root/examples``. It demonstrates manual server/client startup and includes
the sliding-window decoder. Follow :ref:`decoding-server-container-setup` to
create the shared run directory and open the reusable container. These CPU
examples need no GPU. Before starting, run ``cd "$QEC_RUN_DIR"`` in each
container shell; the CMake example's build directory is not used here.

The packaged ``surface_code-1`` program runs a distance-3 surface-code memory
circuit in Stim. Its ``-cqr`` variant streams measurements to the separate
server while the circuit executes. The following examples run 100 shots with
three rounds and CNOT error probability 0.001, using three decoders:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Decoder
     - Configuration
   * - ``pymatching``
     - CPU minimum-weight perfect matching.
   * - ``multi_error_lut``
     - CPU lookup table, using the example's lookup depth 2. Keep this small
       geometry because table size grows with code size and lookup depth.
   * - ``sliding_window``
     - CPU sliding window with PyMatching, window size 3, and step size 1.
       Here the window covers the three-round shot.

The application supplies the measurements and reads the correction through
CUDA-Q's realtime decoding API. The complete program is already included in
the image; only the client application is compiled below.

.. _decoding-server-packaged-client-build:

Build the Packaged Client
^^^^^^^^^^^^^^^^^^^^^^^^^

Compile only the application using the compiler and matching headers inside
the image. The server is already built. In a container shell at the shared run
directory, save the executables alongside the experiment files:

.. code-block:: bash

   BUILD_DIR="$PWD/client-build" bash /root/examples/build.sh surface_code-1

Use the image's example sources and build script together. An arbitrary host
SDK or a newer source checkout may use a different realtime protocol.

Generate Configurations and Select a Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate one server configuration for each decoder. Run this block in the
container shell at the shared run directory:

.. code-block:: bash

   (
     set -e
     for DECODER in pymatching multi_error_lut sliding_window; do
       SW_ARGS=()
       if [ "$DECODER" = sliding_window ]; then
         SW_ARGS=(--sw_inner_decoder pymatching --sw_window_size 3 --sw_step_size 1)
       fi
       ./client-build/surface_code-1-local \
         --distance 3 --num_rounds 3 --num_shots 100 --p_cnot 0.001 \
         --decoder_type "$DECODER" \
         "${SW_ARGS[@]}" \
         --save_dem "$PWD/surface-$DECODER.yaml"
     done
   )

Start one decoder's server:

.. code-block:: bash

   DECODER=pymatching
   QEC_DECODING_SERVER_STATS=1 decoding_server \
     --config="surface-$DECODER.yaml" --transport=udp --port=50000 --timeout=3600

Wait for ``QEC_DECODING_SERVER_READY``. In the second container shell at the
shared run directory, select the same decoder and run its ``-cqr`` client:

.. code-block:: bash

   DECODER=pymatching
   SW_ARGS=()
   if [ "$DECODER" = sliding_window ]; then
     SW_ARGS=(--sw_inner_decoder pymatching --sw_window_size 3 --sw_step_size 1)
   fi
   QEC_DECODING_SERVER_PORT=50000 \
   QEC_DECODING_SERVER_TRANSPORT=udp \
   CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch \
     ./client-build/surface_code-1-cqr \
       --distance 3 --num_rounds 3 --num_shots 100 --p_cnot 0.001 \
       --decoder_type "$DECODER" "${SW_ARGS[@]}" \
       --load_dem "$PWD/surface-$DECODER.yaml"

Stop the server with Ctrl+C after the client completes. Repeat with
``DECODER=multi_error_lut`` and ``DECODER=sliding_window`` in both shells.

Read the Application Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The client prints ``Result size: 100``, the measured logical outcomes and the
number of corrections applied. These are experiment results; a nonzero logical
error count does not by itself indicate a server or transport failure.

To confirm that the separate server handled the shots, check its shutdown
line ``QEC_DECODING_SERVER_DECODER_STATS id=0`` for ``decodes=100`` and
``errors=0``. The client should also print ``CQR service dispatch count: 0``.
The client counter alone is not proof of server routing; check both logs.

These compiled-client examples cover fixed-duration shots. For generated
memory experiments, use :ref:`decoding-server-memory-configuration` rather
than substituting a new YAML into these fixed-model clients. NVIDIA qLDPC and
NV-Fusion require their corresponding plugins in the release image.
Surface-code Ising requires its own model and measurement contract. See
:ref:`decoding-server-ising-example` for the surface-code Ising application and
:ref:`the Fusion API <nv_fusion_decoder_api_cpp>` for Fusion configuration.

Source for the CMake Example
----------------------------

The Ising profile builds its own companion source,
``surface_code_ising_realtime_decoding.cpp`` (plus the shipped
``prepare_ising_artifacts.py`` used during artifact preparation), in
the same example directory. The primary example source:

.. literalinclude:: ../../examples/qec/realtime_decoding_demo/surface_code_realtime_decoding.cpp
   :language: cpp

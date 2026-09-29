Realtime Decoding with CUDA-Q Decoding Server
=============================================

.. note::

   This page describes a C++ example that you compile against the *installed*
   CUDA-Q QEC SDK. The decoding server and the FPGA playback tool it drives are
   **deliverables** — prebuilt and installed alongside the SDK, not built from
   this example.

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

For the nv-qldpc Relay BP device-graph walkthrough see
:doc:`/examples_rst/qec/realtime_relay_bp`; for the software-only pymatching
benchmark see :doc:`/examples_rst/qec/realtime_predecoder_pymatching`.


Deliverables versus the example
-------------------------------

**Deliverables** (installed, not built by this example):

- ``decoding_server`` — the standalone server that owns the decoder instances.
- ``hsb_fpga_syndrome_playback`` — the FPGA playback tool (present only in
  the hololink-enabled deliverable image).
- the QEC + realtime libraries and the decoder plugins.

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

.. code-block:: bash

   cmake -S . -B build \
     -DCUDAQ_INSTALL_DIR=<cuda-quantum install prefix> \
     -DCUDAQX_INSTALL_DIR=<cuda-qx install prefix>
   cmake --build build

In a CUDA-Q QEC container ``CUDAQ_INSTALL_DIR`` defaults to ``/usr/local/cudaq``.
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

QPU-kernel source (software, UDP, no hardware)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching            --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder multi_error_lut       --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --install-prefix <prefix>

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

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching            --wire cpu_roce --setup-network --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder multi_error_lut       --wire cpu_roce --setup-network --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --wire cpu_roce --setup-network --install-prefix <prefix>

The server's READY ``port=`` is then the TCP rendezvous port (the RDMA wire
itself is negotiated via the QP/rkey exchange); dispatch stays ``host``, and
the same pass/fail criteria apply unchanged.

FPGA source (real FPGA over RoCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source fpga --decoder pymatching \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

   ./run_realtime_decoding.sh --source fpga --decoder multi_error_lut \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

   ./run_realtime_decoding.sh --source fpga --decoder nv-qldpc-decoder --gpu 0 \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

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
consuming each decoder's ring. ``pymatching`` and ``multi_error_lut`` ride the
``cpu_roce`` wire on ``host`` dispatch (a server CPU thread calls the
decoder); ``nv-qldpc-decoder --gpu 0`` defaults to the ``hololink`` wire on
``device_graph`` dispatch (the GPU device-call scheduler fires the Relay BP
decode graph on-device) and can be forced onto the host path with
``--dispatch host``.

Both sources apply real pass/fail criteria. The qpu-kernel source uses the
same checks as the in-tree surface-code tests — no decoder errors, a residual
logical-error ceiling of ``num_shots/50``, an in-process dispatch count of 0
(proof the decode stayed in the external server), and a server dispatch-count
floor of ``num_shots * (num_rounds + 3)``. The FPGA source verifies every
shot's corrections against the expected values computed at
syndrome-generation time (the playback tool's per-shot verification).


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
       --ising-artifacts-dir <dir> --install-prefix <prefix>

   ./run_realtime_decoding.sh --source qpu-kernel --decoder trt_decoder --wire cpu_roce \
       --setup-network --ising-artifacts-dir <dir> --install-prefix <prefix>

   ./run_realtime_decoding.sh --source fpga --decoder trt_decoder \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --ising-artifacts-dir <dir> --install-prefix <prefix>

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
     - NIC + GPU (device_graph dispatch)
     - plugin + ``--gpu``
   * - ``trt_decoder``
     - GPU (host dispatch)
     - NIC + GPU (host dispatch)
     - TRT plugin + Ising artifacts (opt-in)

The ``nv-qldpc-decoder`` profile needs the prebuilt plugin — auto-found in the
install prefix, else ``--nv-qldpc-plugin <path.so>`` — and a GPU; if the plugin
is unavailable the script exits ``77`` (skip).


The example source
------------------

The Ising profile builds its own companion source,
``surface_code_ising_realtime_decoding.cpp`` (plus the shipped
``prepare_ising_artifacts.py`` used during artifact preparation), in
the same example directory. The primary example source:

.. literalinclude:: ../../examples/qec/realtime_decoding_demo/surface_code_realtime_decoding.cpp
   :language: cpp

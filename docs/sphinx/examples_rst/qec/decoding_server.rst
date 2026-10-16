.. _decoding-server:

Using the Decoding Server Docker Package
========================================

The CUDA-Q QEC decoding server receives measurement data from a quantum
processor's control system, runs configured decoders, and returns predicted
logical corrections to the runtime executing the experiment. Use the prebuilt
Docker package to deploy this service independently of your quantum application.
You do not need to compile the server or run a CUDA-Q program to use it.

This guide starts with a software playback check, then shows color-code AI
decoding and a CUDA-Q surface-code client. The same server can receive messages
from a third-party runtime connected to a QPU, provided that runtime implements
the message and transport contracts described below.

.. note::

   The dedicated decoding-server image is being prepared for distribution.
   Use the image reference supplied with your package, including its version
   or digest, in ``DECODING_SERVER_IMAGE`` below. The standard CUDA-Q QEC
   0.8.0 and 0.8.1 images do not include this server. A public registry location
   and a tested package version must be confirmed before using this guide as
   release documentation.

How it connects to a QPU
------------------------

The QPU control system measures qubits and delivers the results to its runtime.
The runtime sends decoder requests through a transport provider loaded by the
server. The server owns the decoder instances; the runtime owns the quantum
experiment and decides how to use the returned corrections.

.. code-block:: text

   QPU measurements -> third-party runtime or CUDA-Q application
                               |                       ^
                        measurement stream      logical corrections
                               v                       |
                         transport provider <-> decoding server
                                                      |
                                                 decoder(s)

Before execution, prepare a decoder configuration for the exact circuit,
measurement ordering, observable definitions, and noise model. During execution:

1. Reset the decoder at the start of an independent shot.
2. Send the measurement groups in the order described by the configuration,
   including the final data-qubit measurements required by that circuit.
3. Request the configured number of logical correction bits when the decoder
   has enough input. Combine them with the experiment's logical measurement
   or use them in the runtime's feedback logic.

A successful enqueue acknowledges receipt; it does not mean decoding has
finished. Corrections are predicted observable flips, whose interpretation is
defined by the configuration, rather than a list of physical gates to apply.
For matrix conventions and the measurement-to-detector transformation, see
:doc:`/components/qec/realtime_decoding`.

The server uses CUDA-Q Realtime transport libraries, but a third-party client
does not need the CUDA-Q compiler or programming model. UDP is useful for
functional integration checks. Hardware deployments can use CPU RoCE, GPU RoCE,
or a compatible custom provider. Meeting a QPU's feedback deadline requires
measurement of the complete control-system-to-decoder path on that hardware;
the playback examples below do not establish a latency guarantee.

Get the package ready
---------------------

Use a Linux Docker host and an image matching the host's CPU architecture.
GPU decoding additionally needs a compatible NVIDIA GPU and driver, the NVIDIA
Container Toolkit, and a package variant that includes the selected decoder.
Choose the CUDA variant supported by that driver. The CPU playback check below
does not require a GPU or RDMA hardware.

On the Docker host, set the exact image reference from the package instructions:

.. code-block:: bash

   # Set this to the supplied registry/repository:tag or repository@sha256:digest.
   export DECODING_SERVER_IMAGE='REPLACE_WITH_SUPPLIED_IMAGE_REFERENCE'
   docker pull "$DECODING_SERVER_IMAGE"
   docker run --rm --entrypoint decoding_server "$DECODING_SERVER_IMAGE" --help
   docker run --rm --entrypoint playback-emulator "$DECODING_SERVER_IMAGE" --help

The package supplies the server, the playback CLI, transport libraries, and
decoder plugins. Decoder availability depends on the package variant. In
particular, check that ``trt_decoder`` and TensorRT are included before using
AI decoding; the planned ARM64 CUDA 12 variant does not include TensorRT.
NVIDIA decoder plugins and GPU device-graph dispatch also depend on the
artifacts included in the package.

The runtime image does not supply the CUDA-Q compiler, development headers, or
the ``cudaq_qec`` Python package. Prepare application code and model exports
in a compatible SDK environment, then mount their configuration and artifacts
into the server container. Keep client and server versions compatible with
the package's CUDA-Q Realtime message protocol. Do not copy decoder libraries
from a different release into the container: plugin ABI compatibility matters.

Playback emulator: verify the packaged server
---------------------------------------------

The playback emulator supplies decoder calls from a text schedule in place of
a live runtime. Use it to check configuration, connectivity, and returned
corrections before connecting a QPU.

Download :download:`config.yaml <../../examples/qec/decoding_server/config.yaml>`
and :download:`schedule.txt <../../examples/qec/decoding_server/schedule.txt>`
into an empty directory on the Docker host, and run all commands in this
section from that directory. These files exercise a synthetic three-bit
identity mapping. They are a connectivity check, not a quantum-code experiment.

The configuration selects a CPU lookup-table decoder:

.. literalinclude:: ../../examples/qec/decoding_server/config.yaml
   :language: yaml

The schedule resets decoder 0, sends ``010``, and expects correction ``010``:

.. literalinclude:: ../../examples/qec/decoding_server/schedule.txt
   :language: text

Start the server in terminal 1 on the Docker host:

.. code-block:: bash

   docker run --rm --name qec-decoding-server \
     --mount type=bind,src="$PWD",dst=/work,readonly \
     --workdir /work --entrypoint decoding_server \
     "$DECODING_SERVER_IMAGE" \
     --config=/work/config.yaml --transport=udp --port=50000 --timeout=600

Wait for ``QEC_DECODING_SERVER_READY`` and check that ``ring0=50000`` is
reported. A running container alone is not evidence that the server is ready.
The timeout is the server's run limit in seconds; increase it for longer runs.

In terminal 2 on the same Docker host, set the same
``DECODING_SERVER_IMAGE`` value and run:

.. code-block:: bash

   docker run --rm --network container:qec-decoding-server \
     --mount type=bind,src="$PWD",dst=/work,readonly \
     --workdir /work --entrypoint playback-emulator \
     "$DECODING_SERVER_IMAGE" \
     --schedule=/work/schedule.txt --backend=udp \
     --udp-endpoint=0:127.0.0.1:50000 --tick=1ms > result.csv

The two containers share a network namespace, so ``127.0.0.1`` reaches the
server without publishing a host port. The CSV is written on the host by
shell redirection; the input directory remains read-only in both containers.
This assumes one decoder and an unused port 50000 in the server's namespace.

Inspect ``result.csv``. The reset and correction read should have status
``0`` (``OK`` in the protocol). The enqueue is recorded as a one-round stream
with status ``100`` (completed the requested rounds). The
``get_corrections`` row should have ``read_completed=1``,
``correction_bits=010``, and ``correction_mismatch=0``. A mismatch flag of zero
alone is insufficient: verify that a correction was actually returned.
The one-second delay before the read makes this a functional check, not a
latency benchmark. If the read returns ``NOT_READY``, increase the delay and
rerun the complete schedule.

Stop the server when finished:

.. code-block:: bash

   docker stop qec-decoding-server

Schedules, sources, and sessions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each schedule line is ``<trigger> <operation> [key=value ...]``.
``N`` is an absolute tick, ``+N`` is relative to the preceding dispatch, and
``-`` means dispatch immediately. After switching to relative triggers, keep
subsequent triggers relative. ``signal=NAME`` marks completion of an operation;
``after=NAME`` makes another operation wait for that completion. Schedule
spacing controls when requests are sent, not when the decoder finishes.

Use ``reset``, ``enqueue``, ``stream``, ``enqueue_data``, and
``get_corrections`` to describe the experiment. ``session=N`` selects the
decoder ID and defaults to 0. ``enqueue_data`` supplies a source's final
data-qubit readout; whether it is needed depends on the circuit's measurement
layout. The generic emulator does not choose that layout for you.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Input source
     - Use
   * - Inline bits, ``source=0b010``
     - Small deterministic checks such as the example above.
   * - Static source, ``--source=ID:FILE``
     - Recorded measurement groups, one ``0``/``1`` string per line.
   * - Stim source, ``--stim-source=ID:...``
     - Generated memory-experiment measurements. Requires Stim support in
       the installed tool; see its ``--help`` for the accepted parameters.
   * - Python CUDA-Q memory source
     - Measurements generated from a CUDA-Q memory experiment in an SDK
       environment. Requires Python bindings outside the runtime image.

Select ``--backend=udp`` to exercise this server. ``inproc`` runs the decoder
inside the emulator and bypasses the server; ``null`` measures emulator
overhead without decoding. ``cpu_roce`` exercises an RDMA connection and needs
the RDMA device, local IP, endpoint, and ring geometry to match the server.
The Python entry point is ``cudaq_qec.playback.run`` in an SDK environment;
it can use ``udp_endpoints={0: "host:port"}`` to target the server. The
packaged CLI avoids that Python dependency.

For additional source and session settings, inspect
``playback-emulator --help`` from the selected package.

.. TODO: Link the dedicated playback reference when PR #850 (issue #835)
   lands; keep the full schedule grammar there rather than duplicating it.

.. _decoding-server-color-code:

Playback with the color-code AI decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`color-code example <color_code_demo>` samples superdense memory
circuits with Stim and compares two decoder configurations using the same
measurements:

* ``chromobius``: a color-code decoder running on the CPU.
* ``trt_decoder``: the published Ising-Decoder-ColorCode-1-Fast AI predecoder
  running through TensorRT, followed by Chromobius on the residual detectors.

Use the published color-code model, not a surface-code model. The circuit,
distance, measurement-to-detector mapping, observable support, and exported
model must agree. Model preparation belongs in an SDK environment with the
Python dependencies and model access described in :doc:`color_code_demo`.
The server container needs only the resulting artifacts and decoder runtime.

In that SDK environment, create a scratch directory and run the example from
this checkout, replacing ``/path/to/cudaq-qec`` with its actual location:

.. code-block:: bash

   mkdir -p color-code-package
   cd color-code-package
   python3 /path/to/cudaq-qec/docs/sphinx/examples/qec/color_code_demo/color_code_demo.py \
     --prepare-only --distances 5 --shots 100 --seed 1234

This downloads the pinned public model and Ising source if needed, exports
``pre_d5.onnx``, and writes ``dem_d5.txt``, ``schedule_d5.txt``,
``config_chromobius_d5.yaml``, and ``config_trt_decoder_d5.yaml``.
It does not start a local server or require a locally installed server binary.
Copy these five files together into a directory on the Docker host. All paths
inside the YAML are relative to that directory.

From that directory, start the CPU baseline in terminal 1:

.. code-block:: bash

   docker run --rm --name qec-decoding-server \
     --mount type=bind,src="$PWD",dst=/work,readonly \
     --workdir /work --entrypoint decoding_server \
     "$DECODING_SERVER_IMAGE" \
     --config=/work/config_chromobius_d5.yaml \
     --transport=udp --port=50000 --timeout=600

After readiness, replay in terminal 2 using the same image value and directory:

.. code-block:: bash

   docker run --rm --network container:qec-decoding-server \
     --mount type=bind,src="$PWD",dst=/work,readonly \
     --workdir /work --entrypoint playback-emulator \
     "$DECODING_SERVER_IMAGE" \
     --schedule=/work/schedule_d5.txt --backend=udp \
     --udp-endpoint=0:127.0.0.1:50000 --tick=1us > chromobius.csv
   docker stop qec-decoding-server

Next, start the AI configuration in terminal 1. This needs a package with
``trt_decoder`` and a GPU:

.. code-block:: bash

   docker run --rm --name qec-decoding-server --gpus all \
     --mount type=bind,src="$PWD",dst=/work \
     --workdir /work --entrypoint decoding_server \
     "$DECODING_SERVER_IMAGE" \
     --config=/work/config_trt_decoder_d5.yaml \
     --transport=udp --port=50000 --timeout=600

The AI configuration loads ONNX and builds ``engine_d5.trt`` on the server's
GPU at initialization, so this mount is writable. Wait for readiness; engine
construction can take substantially longer than the CPU baseline. Replay
the same schedule from terminal 2 with the previous playback command, changing
the output filename to ``ising.csv``, then stop the server.

For each run, require 100 completed ``get_corrections`` rows with status
``0`` (``OK``). Here the schedule's expected bits are Stim's true observable flips:
a correction mismatch represents a logical decoding error and can occur in
a valid run. Compare the mismatch fraction over completed shots to assess
logical error rate. Use the full :doc:`color_code_demo` for the per-round
conversion and uncertainty treatment; 100 shots is a connectivity check,
not an accuracy study. Playback timings include emulator and transport work
and should not be presented as standalone neural-network inference latency.

CUDA-Q examples: surface-code programs
--------------------------------------

A CUDA-Q application is another client of the same packaged server. The
:doc:`surface-code example <realtime_decoding_demo>` provides a generator
that characterizes the circuit and a realtime-lowered program that sends
measurements and receives corrections while the circuit executes.

Build only those client programs in a compatible CUDA-Q QEC SDK environment,
following that example's ``Building the example`` section. The runtime image
cannot compile them. Keep the server in Docker; the following sequence uses
the existing example binaries directly instead of its orchestration script,
which launches a server from a local SDK installation.

In the SDK environment, from a directory writable on the Docker host:

.. code-block:: bash

   # Point this at the example build directory from realtime_decoding_demo.
   export EXAMPLE_BUILD_DIR=/path/to/realtime_decoding_demo/build
   "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type pymatching --save_dem "$PWD/surface.yaml"

On the Docker host, from the directory containing ``surface.yaml``, start
the server in terminal 1. This Linux example uses host networking so that
the client's loopback address can reach it:

.. code-block:: bash

   docker run --rm --name qec-decoding-server --network host \
     --mount type=bind,src="$PWD",dst=/work,readonly \
     --workdir /work --entrypoint decoding_server \
     "$DECODING_SERVER_IMAGE" \
     --config=/work/surface.yaml --transport=udp --port=50000 --timeout=600

After readiness, run the client in terminal 2 in the SDK environment, setting
the same ``EXAMPLE_BUILD_DIR`` value and using the directory containing
``surface.yaml``:

.. code-block:: bash

   QEC_DECODING_SERVER_PORT=50000 \
     "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding-cqr" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type pymatching --load_dem "$PWD/surface.yaml"

The SDK process must share the Docker host's network namespace. If the SDK
runs in another Linux container, use host networking for that client container
as well. This example's UDP client hardcodes ``127.0.0.1``; it is not a
remote-host example. Confirm that the server reports dispatched requests at
shutdown and the application completes its shots. The application's local
``CQR service dispatch count`` should be zero because decoding occurred in
the server. Stop the server with ``docker stop qec-decoding-server``.

Using different decoders
^^^^^^^^^^^^^^^^^^^^^^^^

Regenerate ``surface.yaml`` with the desired ``--decoder_type`` and use the
same option when running the client. Stop the previous server and launch a
new one with the regenerated configuration. Decoder choice does not require
rebuilding the server, but may require a different model or error-model
representation; changing only the YAML ``type`` is not generally sufficient.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Decoder
     - Server resources
     - Example and configuration
   * - ``pymatching``
     - CPU
     - Surface-code example above; generates a decomposed DEM.
   * - ``multi_error_lut``
     - CPU
     - Same client example with that decoder type. Start with distance 3;
       lookup-table size grows with code size and configured error depth.
   * - ``nv-qldpc-decoder``
     - GPU and supplied NVIDIA plugin
     - Same client example with that decoder type and ``--gpus all`` on
       the server container. Host dispatch can still invoke GPU decoding.
   * - ``trt_decoder`` with Ising + PyMatching
     - GPU, TensorRT, and model artifacts
     - Use the separate ``surface_code_ising_realtime_decoding`` sources
       and artifact preparation in :doc:`realtime_decoding_demo`. Their
       measurement layout matches the surface-code Ising model.

For NVIDIA Fusion's streaming MWPM configuration, see
:doc:`/api/qec/nv_fusion_decoder_api`. It is a distinct decoder requiring
the matching plugin and configuration; the surface-code client above does
not expose it as a ``--decoder_type`` option.

Connect a third-party runtime
-----------------------------

The runtime integration has two contracts: decoder messages and the transport
that delivers them. Sending arbitrary measurement arrays to an IP address
does not satisfy either contract.

**Decoder messages.** Implement ``reset_decoder``, ``enqueue_syndromes``, and
``get_corrections`` using the
`Decoding Server Message Format <https://github.com/NVIDIA/cudaq-qec/blob/main/libs/qec/tools/decoding-server/MESSAGE_FORMAT.md>`__
and the matching package version of the
`CUDA-Q Realtime message protocol <https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/cudaq_realtime_message_protocol.md>`__.
The protocol defines framing, function identifiers, byte order, packed bits,
request correlation, and status codes. The YAML decoder IDs, incoming
measurement count and order, and returned observable count must agree.
Handle ``NOT_READY`` without treating it as a correction; diagnose malformed
requests or decoder failures before continuing the experiment. A full
``reset_decoder`` clears shot state; the ``reset`` argument on
``get_corrections`` only clears accumulated correction bits after reading.

**Transport.** The server loads a provider library. UDP and CPU RoCE can be
selected on the command line for host-only configurations that do not name a
provider in YAML. Configurations using GPU RoCE or device-graph dispatch must
put the provider and its arguments in the YAML transport section. A conflicting
command-line provider is rejected. A custom provider must implement the
`CUDA-Q Realtime bridge interface <https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/bridge/bridge_interface.h>`__
compatible with the package and have its library and runtime dependencies
available inside the container.

For multiple decoders, read every ``ring<ID>=<endpoint>`` from the readiness
line and route each decoder to its own ring. Do not assume that a single UDP
port serves every decoder. CPU RoCE readiness ports are TCP rendezvous
endpoints, rather than UDP data ports. Ring slot counts and sizes must agree
between the provider and runtime. RDMA deployments also need the appropriate
host devices, network configuration, and GPU/NIC topology exposed to the
container; use the package's deployment instructions for that hardware.

Verify your integration first with known measurements and expected observable
flips, then with live measurements from the QPU. Record end-to-end latency,
correctness, and behavior under the experiment's sustained request rate before
relying on the path for feedback.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Check
   * - Image cannot be pulled
     - Confirm the supplied registry, immutable tag or digest, access, and
       architecture. The dedicated package is separate from the general
       CUDA-Q QEC image.
   * - Container runs but no readiness line appears
     - Read the server output for configuration, plugin, model, or transport
       errors. Use ``--entrypoint decoding_server``; the package's default
       entry point is a shell. Allow for first-time TensorRT engine creation.
   * - Playback times out or produces no correction
     - Confirm readiness, decoder ID and ring endpoint, and network namespace.
       Loopback in an unrelated container cannot reach the server. Check the
       returned status and completed-read fields, not just the exit code.
   * - ``NOT_READY``
     - Check the complete measurement layout, final readout, and decoder
       completion. Schedule time is not a completion barrier for decoding.
   * - Invalid decoder or malformed request
     - Match IDs, message format, packed-bit counts, and observable return
       size to the package version and YAML configuration.
   * - Model file not found or TensorRT engine fails to load
     - Check paths inside the container, working directory, GPU visibility,
       plugin availability, and TensorRT compatibility. Rebuild engines on
       the deployment GPU/runtime when required; do not assume portability.
   * - SDK tools or Python imports are missing
     - Run client compilation and model preparation in the SDK environment.
       Use the playback CLI to drive the runtime-only package.

See also
--------

* :doc:`/components/qec/realtime_decoding` — concepts and configuration.
* :doc:`realtime_decoding_demo` — surface-code and FPGA client examples.
* :doc:`color_code_demo` — color-code model preparation and accuracy study.
* :doc:`/components/qec/decoders` — available decoder implementations.

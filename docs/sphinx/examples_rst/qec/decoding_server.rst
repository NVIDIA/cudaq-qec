.. _decoding-server:

Decoding Server
===============

The CUDA-Q QEC decoding server runs classical decoders independently of the
program controlling a quantum processor (QPU). A runtime sends measurements to
the server and receives predicted logical corrections. The runtime can be
CUDA-Q or a third-party control system that implements the server protocol.

This page shows how to use the **prebuilt Docker package**. You never build the
server. The playback examples run without a QPU. The CUDA-Q examples compile
only the application that sends measurements to the packaged server.

.. contents:: On this page
   :local:
   :depth: 2

Intended use cases
------------------

**Connect a QPU and a third-party runtime.** The control system measures the
QPU's qubits. Its runtime sends those measurements to a configured decoder,
requests logical corrections, and uses them to interpret logical measurements
or make feedback decisions. The server does not control the QPU or choose the
quantum experiment. See `Integrate a third-party runtime`_ for the contracts
that the runtime must implement.

**Test decoding without a QPU.** The playback emulator sends scheduled decoder
requests using recorded or simulated measurements. Use this path to verify
configuration and connectivity, and to evaluate decoding accuracy against
known logical outcomes. Start with `Playback emulator`_.

**Run a CUDA-Q surface-code program.** A CUDA-Q application sends measurements
while its circuit executes and receives corrections from the same Docker
server. The examples here use Stim to simulate the QPU and require no physical
QPU. Start with `CUDA-Q examples: surface-code programs`_.

.. code-block:: text

   Physical QPU or simulator
             |
        measurements
             v
   Third-party runtime or CUDA-Q application
             |                          ^
        decoder requests          logical corrections
             v                          |
       transport provider <-> decoding server <-> configured decoder

Before a run, the application prepares a detector error model (DEM), the
measurement-to-detector mapping, and the observable definitions for its exact
circuit. During each independent shot, it resets the decoder, sends the
measurements in that configured order, and reads the correction. An enqueue
acknowledgement confirms receipt; decoding may still be running. The returned
bits predict logical observable flips. The application decides how to use them.
See :doc:`/components/qec/realtime_decoding` for these matrix conventions.

Before you start
----------------

Use a **Linux Docker host** with Docker access and Python 3.10 or later. Run
all commands below in the same terminal on that host. These examples use UDP
host networking and one decoder with ID 0. They measure functional correctness;
they do not establish a QPU feedback-latency guarantee.

The CPU connectivity, Chromobius, PyMatching, and lookup-table examples do not
need a GPU. The AI and NVIDIA qLDPC examples additionally need a compatible
NVIDIA GPU, driver, NVIDIA Container Toolkit, and a package containing the
required decoder plugin. The ARM64 CUDA 12 package variant in the image work
does not include TensorRT; use a TensorRT-enabled variant for AI decoding.

Get the prebuilt package
^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

   The dedicated server package is currently a private preview from
   `the decoding-server image work <https://github.com/NVIDIA/cudaq-qec/pull/798>`__.
   The preview registry is ``ghcr.io/nvidia/private/cudaq-decoders``; its CUDA
   13 moving tag is ``cu13-latest``. You need package read access. The general
   CUDA-Q QEC 0.8.0 and 0.8.1 images do not contain this server.

   If the pull below returns ``denied``, obtain access from the package owner
   before continuing. Successful registry login alone does not confirm package
   access. A public release URI and an end-to-end tested image digest are still
   required before publishing this page as release documentation.

Pull the CUDA 13 preview, then pin the local reference to its immutable digest:

.. code-block:: bash

   docker pull ghcr.io/nvidia/private/cudaq-decoders:cu13-latest
   export DECODING_SERVER_IMAGE="$(docker image inspect \
     ghcr.io/nvidia/private/cudaq-decoders:cu13-latest \
     --format '{{index .RepoDigests 0}}')"
   docker run --rm --entrypoint decoding_server "$DECODING_SERVER_IMAGE" --help
   docker run --rm --entrypoint playback-emulator "$DECODING_SERVER_IMAGE" --help
   docker run --rm --entrypoint ls "$DECODING_SERVER_IMAGE" \
     /opt/nvidia/cudaqx/lib/decoder-plugins

Both help commands must succeed. Keep the digest for reproducibility. For a
CUDA 12 deployment, use the package owner's corresponding CUDA 12 reference
and check decoder availability for that architecture. Do not mix plugin
libraries from other package versions into this image.

The plugin listing must include ``libcudaq-qec-trt-decoder.so`` for the AI
examples and ``libcudaq-qec-nv-qldpc-decoder.so`` for the NVIDIA qLDPC example.
If either is absent, obtain a package variant containing it before attempting
that example. CPU-only examples can still be run with their required plugins.

Prepare the example directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the CUDA-Q QEC checkout containing this page. From its repository root:

.. code-block:: bash

   export QEC_SOURCE="$PWD"
   export QEC_RUN_DIR="$HOME/qec-decoding-server-runs"
   mkdir -p "$QEC_RUN_DIR"
   cp "$QEC_SOURCE/docs/sphinx/examples/qec/decoding_server/"{run_package_example.py,config.yaml,schedule.txt} \
     "$QEC_RUN_DIR/"
   cd "$QEC_RUN_DIR"

You can also download the three files directly:
:download:`run_package_example.py <../../examples/qec/decoding_server/run_package_example.py>`,
:download:`config.yaml <../../examples/qec/decoding_server/config.yaml>`, and
:download:`schedule.txt <../../examples/qec/decoding_server/schedule.txt>`.
The CUDA-Q and model-preparation sections additionally need the full checkout.

The example launcher starts the Docker server on an available UDP port, waits
for readiness, runs the requested client, and stops and removes its containers.
It saves the client output and a separate ``.server.log`` file. It fails on
startup errors, client errors, incomplete corrections, or server decoder errors.
Output files must not already exist; choose a new output name when repeating a
run. This prevents overwriting previous results.

All configuration and model files must be under ``QEC_RUN_DIR``. The launcher
mounts this directory at the same absolute path inside the container, so the
absolute paths written by SDK tools continue to work. Keep the SDK and client
sources compatible with the server package's QEC configuration and CUDA-Q
Realtime protocol versions.

Playback emulator
-----------------

The playback emulator replaces the runtime and QPU with a schedule of decoder
calls. It still sends those calls to the **separate packaged decoding server**.
The following examples select the UDP backend. Selecting ``inproc`` would run
the decoder inside the emulator and would not test the server.

Check connectivity first
^^^^^^^^^^^^^^^^^^^^^^^^

Run this CPU-only check before preparing a model:

.. code-block:: bash

   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config config.yaml --schedule schedule.txt \
     --shots 1 --expect-no-errors --output connectivity.csv

The supplied configuration uses a synthetic three-bit identity mapping with
``multi_error_lut``. The schedule resets decoder 0, sends ``010``, then reads
three correction bits. This is a connectivity check, not a quantum-code
experiment.

.. literalinclude:: ../../examples/qec/decoding_server/schedule.txt
   :language: text

A successful run prints:

.. code-block:: text

   Completed shots: 1; logical mismatches: 0; fraction: 0
   Server dispatched 3 requests; saved client output: .../connectivity.csv

``connectivity.csv.server.log`` must report one decode, one correction, and zero
errors. In the CSV, reset and read use status ``0``; the enqueue is recorded as
a completed one-round stream with status ``100``. The read contains
``read_completed=1``, ``correction_bits=010``, and ``correction_mismatch=0``.
The launcher checks these completion conditions automatically.

Schedules, sources, and sessions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A schedule line is ``<trigger> <operation> [key=value ...]``:

* ``N`` means absolute tick ``N``; ``+N`` means ``N`` ticks after the preceding
  dispatch; ``-`` means dispatch immediately. Once triggers become relative,
  all following triggers must remain relative.
* ``reset`` starts a fresh shot. ``enqueue`` sends one measurement group;
  ``stream`` sends multiple groups. ``enqueue_data`` sends a source's final
  data-qubit readout. ``get_corrections`` reads logical corrections.
* ``session=N`` selects decoder ID ``N``. ``signal=NAME`` signals operation
  completion; ``after=NAME`` waits for it. An enqueue's completion is an
  acknowledgement, not a decoding-completion barrier.

Inline bits such as ``source=0b010`` are useful for small checks. Static sources
read recorded measurement groups from a file. Stim sources generate simulated
memory-experiment measurements. Python CUDA-Q memory sources require an SDK
with Python bindings. For the available source parameters, run the packaged
``playback-emulator --help``.

The playback sessions are ``udp`` (server over UDP), ``cpu_roce`` (server over
RDMA), ``inproc`` (local decoder), and ``null`` (no decoding). This page uses
``udp``. CPU RoCE additionally needs a configured RDMA device and matching ring
geometry; it is not required for any walkthrough here.

.. TODO: Link the complete playback grammar and Python reference after #850
   lands. That reference owns the full grammar requested in issue #835.

.. _decoding-server-color-code:

Color-code-enabled AI decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Ising-Decoder-ColorCode-1-Fast`` is a published neural-network predecoder for
color codes. In the server it runs through ``trt_decoder`` using TensorRT.
It predicts observable flips and residual detectors; Chromobius decodes the
residual, and the server combines the results into logical corrections.

The example below compares **Chromobius alone** with **Ising + Chromobius**
using exactly the same simulated measurements and expected logical outcomes.
It uses a distance-5, Z-basis superdense color-code memory circuit with five
rounds and physical error probability 0.001. It is a fixed-shot example.
A surface-code model or surface-code measurement mapping cannot replace the
color-code model and mapping.

Color-code example: prepare once, run both decoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preparation needs a CUDA-Q QEC SDK environment on the Docker host with
``cudaq_qec`` and its playback bindings. The runtime-only server image does not
contain that Python SDK. Activate the SDK's Python environment, install the
inference dependencies from the Ising checkout used by the example, and accept
the model terms at
`Ising-Decoder-ColorCode-1-Fast <https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast>`__.
Authenticate the ``hf`` CLI before downloading gated weights.

The following commands fetch the same Ising revision pinned by the example
and install its inference requirements. They do not build the server:

.. code-block:: bash

   mkdir -p model-deps
   git clone https://github.com/NVIDIA/Ising-Decoding.git model-deps/Ising-Decoding
   git -C model-deps/Ising-Decoding checkout 33acb152e403bc189f2effdb07f1a87b34c745f1
   python3 -m pip install -r model-deps/Ising-Decoding/code/requirements_public_inference.txt
   python3 -m pip install huggingface_hub
   hf auth login
   python3 -c 'import cudaq_qec; print(cudaq_qec.playback)'
   python3 "$QEC_SOURCE/docs/sphinx/examples/qec/color_code_demo/color_code_demo.py" \
     --prepare-only --deps "$PWD/model-deps" --distances 5 --shots 100 --seed 1234

Preparation writes five server inputs in the current directory:
``pre_d5.onnx``, ``dem_d5.txt``, ``schedule_d5.txt``,
``config_chromobius_d5.yaml``, and ``config_trt_decoder_d5.yaml``.
It does not start a decoder or server. The example pins the model revision and
measurement mapping; keep these five files together.

Run the CPU baseline:

.. code-block:: bash

   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config config_chromobius_d5.yaml --schedule schedule_d5.txt \
     --shots 100 --output color-chromobius.csv

Check GPU access, then run the AI decoder:

.. code-block:: bash

   docker run --rm --gpus all --entrypoint nvidia-smi "$DECODING_SERVER_IMAGE"
   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config config_trt_decoder_d5.yaml --schedule schedule_d5.txt \
     --gpu --writable --shots 100 --output color-ising.csv

``--gpu`` exposes the GPU to the server. ``--writable`` allows the server to
save ``engine_d5.trt`` while building TensorRT from ONNX. Engine construction
happens before readiness; the launcher's default startup timeout is 600 seconds.
If the log shows ongoing engine construction at that limit, rerun with a new
output name and ``--startup-timeout 1800``. No engine is copied from another GPU.

Both runs must complete 100 correction reads and report zero server errors.
The launcher prints the logical-mismatch count and fraction for each run.
For this quantum-code experiment, a mismatch is a logical decoding error,
not a transport failure; a valid run can contain mismatches. The 100-shot run
checks operation and is too small for an accuracy claim. See
:doc:`color_code_demo` for larger studies, per-round logical error rates, and
uncertainties. This deliberately paced walkthrough is not a latency benchmark.

CUDA-Q examples: surface-code programs
--------------------------------------

These examples run a surface-code circuit in Stim and stream its measurements
to the Docker server through CUDA-Q realtime calls. They cover PyMatching,
lookup-table decoding, NVIDIA qLDPC, and surface-code Ising + PyMatching.
Every run uses a single decoder with ID 0.

Build the client programs once
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use an installed CUDA-Q QEC SDK with realtime support and a CUDA compiler on
this Linux Docker host. The server runtime image does not contain the compiler
or development headers. Activate the SDK environment so that:

* ``CUDA_QUANTUM_PATH`` names the CUDA-Q installation containing ``bin/nvq++``.
* ``CUDAQX_INSTALL_DIR`` names the QEC installation containing
  ``include/cudaq/qec/code.h`` and ``lib/libcudaq-qec.so``.

From ``QEC_RUN_DIR``, verify the installations and compile the client examples:

.. code-block:: bash

   : "${CUDA_QUANTUM_PATH:?Set CUDA_QUANTUM_PATH to your CUDA-Q SDK installation}"
   : "${CUDAQX_INSTALL_DIR:?Set CUDAQX_INSTALL_DIR to your CUDA-Q QEC SDK installation}"
   test -x "$CUDA_QUANTUM_PATH/bin/nvq++"
   test -f "$CUDAQX_INSTALL_DIR/include/cudaq/qec/code.h"
   export EXAMPLE_BUILD_DIR="$QEC_RUN_DIR/client-build"
   cmake -S "$QEC_SOURCE/docs/sphinx/examples/qec/realtime_decoding_demo" \
     -B "$EXAMPLE_BUILD_DIR" \
     -DCUDAQ_INSTALL_DIR="$CUDA_QUANTUM_PATH" \
     -DCUDAQX_INSTALL_DIR="$CUDAQX_INSTALL_DIR"
   cmake --build "$EXAMPLE_BUILD_DIR" --parallel 4

If your SDK installs realtime libraries separately, also pass
``-DCUDAQ_REALTIME_DIR="$CUDAQ_REALTIME_DIR"`` to CMake with that directory
set to the realtime installation prefix. These commands compile the four
example executables, including the two ``-cqr`` clients. They do not compile
the server. All subsequent commands run from ``QEC_RUN_DIR`` on the SDK host;
the launcher passes the discovered server port to the client automatically.

PyMatching on the CPU
^^^^^^^^^^^^^^^^^^^^^

Generate the configuration, then run 100 shots through the packaged server:

.. code-block:: bash

   "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type pymatching --save_dem "$PWD/surface-pymatching.yaml"
   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config surface-pymatching.yaml --shots 100 --output surface-pymatching.log \
     --client "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding-cqr" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type pymatching --load_dem "$PWD/surface-pymatching.yaml"

Lookup-table decoding on the CPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate a separate configuration for the same distance-3 experiment:

.. code-block:: bash

   "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type multi_error_lut --save_dem "$PWD/surface-lut.yaml"
   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config surface-lut.yaml --shots 100 --output surface-lut.log \
     --client "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding-cqr" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type multi_error_lut --load_dem "$PWD/surface-lut.yaml"

This configuration uses lookup depth 2. Keep the small example geometry;
lookup-table memory and setup work grow with the code size and error depth.

NVIDIA qLDPC decoding on the GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The image must contain the ``nv-qldpc-decoder`` plugin. Check GPU access with
the ``nvidia-smi`` command above, then run:

.. code-block:: bash

   "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type nv-qldpc-decoder --save_dem "$PWD/surface-qldpc.yaml"
   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config surface-qldpc.yaml --gpu --shots 100 --output surface-qldpc.log \
     --client "$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding-cqr" \
     --distance 3 --num_rounds 3 --num_shots 100 \
     --decoder_type nv-qldpc-decoder --load_dem "$PWD/surface-qldpc.yaml"

The server uses a CPU host-dispatch thread to invoke the GPU decoder. GPU
computation does not require GPU RoCE or device-graph transport in this example.

Surface-code Ising AI decoder with PyMatching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This path uses the separate ``surface_code_ising_realtime_decoding`` executable
pair. Its circuit is distance 7, seven rounds, Z basis, XV orientation, with
SPAM probability 0.01. The ordinary surface-code executable above has a different
measurement layout and cannot use these model artifacts.

Accept the terms for
`Ising-Decoder-SurfaceCode-1-Fast <https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast>`__
and authenticate ``hf``. Artifact preparation requires a GPU in the SDK
environment and the exporter dependencies below:

.. code-block:: bash

   python3 -m pip install beliefmatching hydra-core ldpc matplotlib numpy \
     omegaconf onnx pymatching safetensors scipy stim torch huggingface_hub
   hf auth login
   python3 "$QEC_SOURCE/docs/sphinx/examples/qec/realtime_decoding_demo/prepare_ising_artifacts.py" \
     --app "$EXAMPLE_BUILD_DIR/surface_code_ising_realtime_decoding" \
     --artifacts-dir "$PWD/surface-ising-artifacts"
   "$EXAMPLE_BUILD_DIR/surface_code_ising_realtime_decoding" \
     --distance 7 --num_rounds 7 --num_shots 100 --p_spam 0.01 \
     --decoder_type trt_decoder --ising-artifacts-dir "$PWD/surface-ising-artifacts" \
     --save_dem "$PWD/surface-ising.yaml"
   python3 run_package_example.py --image "$DECODING_SERVER_IMAGE" \
     --config surface-ising.yaml --gpu --shots 100 --output surface-ising.log \
     --client "$EXAMPLE_BUILD_DIR/surface_code_ising_realtime_decoding-cqr" \
     --distance 7 --num_rounds 7 --num_shots 100 --p_spam 0.01 \
     --yaml "$PWD/surface-ising.yaml"

Preparation produces ``model.onnx``, ``H_csr.bin``, ``O_csr.bin``, ``priors.bin``,
``metadata.txt``, and ``D_sparse.txt``. The generator embeds the matrices in
``surface-ising.yaml`` and records the ONNX path under the mounted directory.
The server builds TensorRT on its deployment GPU. With ``--yaml``, the client
reads the decoder choice from the file; do not also pass ``--decoder_type``.

Verify each surface-code run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each launch must print ``Client completed with external server routing
confirmed`` and report nonzero server dispatches. The launcher checks that
the client exited successfully, its local CQR dispatch count was zero, and
server statistics contain 100 decodes, at least 100 correction requests, and
zero decoder errors.
The ``.log`` file contains the application output; the corresponding
``.log.server.log`` contains the server output. This proves requests reached
the external server. It is not a logical-error-rate or latency benchmark.

These are fixed-shot examples with complete configurations. NVIDIA Fusion's
streaming decoder, chunked DEM execution, and Ising overlap-save task graphs
have additional configuration and model contracts. Do not substitute their
names into these commands or assume that this example demonstrates
unknown-duration shots. See :ref:`the Fusion C++ API <nv_fusion_decoder_api_cpp>` for Fusion's
configuration and package availability; its type is not accepted by the
ordinary surface-code example's ``--decoder_type`` option.

Integrate a third-party runtime
-------------------------------

A real QPU deployment uses the same server but needs a runtime adapter for its
control system. The hardware vendor supplies the QPU connection; the adapter
supplies decoder messages and a compatible transport.

1. Generate the configuration for the actual circuit, noise model,
   measurement ordering, and observables. Match the decoder ID and the number
   of correction bits to the runtime.
2. Implement ``reset_decoder``, ``enqueue_syndromes``, and ``get_corrections``
   using the package version of the
   `Decoding Server Message Format <https://github.com/NVIDIA/cudaq-qec/blob/main/libs/qec/tools/decoding-server/MESSAGE_FORMAT.md>`__
   and `CUDA-Q Realtime protocol <https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/cudaq_realtime_message_protocol.md>`__.
   These specify framing, function IDs, packed-bit order, request correlation,
   and statuses. ``NOT_READY`` is not a valid correction.
3. Connect through the selected provider. UDP supports functional integration;
   CPU RoCE and GPU RoCE require the package's hardware deployment setup.
   A custom provider implements the matching
   `bridge interface <https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/bridge/bridge_interface.h>`__.
   The third-party client does not need the CUDA-Q compiler.
4. Reset at the beginning of an independent shot, send every configured
   measurement group including the final readout, and read the logical
   correction. A full decoder reset clears shot state; the reset argument to
   ``get_corrections`` clears correction bits after reading them.
5. Validate known measurements first, then the live QPU stream. Measure the
   complete control-system-to-correction path under sustained load before
   relying on it for feedback deadlines.

The launcher on this page serves only one UDP host-dispatch decoder. For a
multi-decoder integration, route each decoder to its advertised
``ring<ID>`` endpoint. CPU RoCE ports are TCP rendezvous endpoints, not UDP data
ports. Configurations using GPU RoCE or device-graph dispatch put provider
selection and arguments in YAML; they cannot use this launcher's UDP flags.
Use the hardware deployment configuration supplied with that package.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Failure
     - Action
   * - Registry returns ``denied``
     - Confirm the image reference and package read access. Authenticate to
       the registry as instructed by the package owner. Do not continue until
       ``docker pull`` succeeds.
   * - Server exits before readiness
     - Read the saved ``.server.log``. Fix the reported YAML, missing plugin,
       model path, or GPU error before rerunning.
   * - Startup times out
     - Read the server log. If TensorRT is still building an engine, increase
       ``--startup-timeout``. If initialization failed, fix that error first.
   * - Output file already exists
     - Choose a new ``--output`` filename. The launcher preserves prior runs.
   * - Playback has incomplete or failed reads
     - Check that the schedule, decoder ID, DEM, and measurement mapping belong
       to the same experiment. Inspect the returned status in the CSV.
   * - Client compilation or Python import fails
     - Activate the compatible SDK environment and check its installed
       development headers, realtime libraries, and Python bindings. These
       tools are not included in the server runtime image.
   * - Model or engine initialization fails
     - Check GPU visibility, TensorRT/plugin availability, and paths under
       ``QEC_RUN_DIR``. Build engines on the deployment GPU/runtime.

See also
--------

* :doc:`/components/qec/realtime_decoding` — decoding concepts and configuration.
* :doc:`realtime_decoding_demo` — surface-code and real FPGA client examples.
* :doc:`color_code_demo` — color-code accuracy and performance studies.
* :doc:`/components/qec/decoders` — decoder capabilities and API references.

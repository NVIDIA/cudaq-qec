# CUDA-Q QEC Decoding Server

The decoding server is a standalone process that decodes realtime syndrome
data for a running quantum error correction experiment. The experiment
application and the decoder are deliberately separated into two processes:
the application streams each round of stabilizer measurements (and the final
data measurements) to the server, and the server runs the configured decoder
and returns the predicted logical correction. The decoder setup crosses the
process boundary as **configuration**, not code — the same YAML file that
the application generated configures the server.

The server itself is **transport agnostic**: it contains no networking
code. The wire between the two processes is a *transport provider* — a
shared library loaded at runtime. The same server binary and configuration
format can serve different transports; choose the provider and its arguments
for your hardware and dispatch mode (see [Transport providers](#transport-providers)).

This page is the **from-source guide for power users and integration
partners**: it covers building the decoding server from source, verifying
the build, operating it, and extending it (decoder plugins, transport
providers, wire format). The other realtime decoding documents cover
different viewpoints:

- [Getting started with realtime decoding](../../../../docs/sphinx/examples_rst/qec/getting_started_realtime_decoding.rst)
  — the in-process API: generating a detector error model, configuring
  decoders inside your application, and calling them from quantum kernels
  (Python and C++).
- [Realtime decoding demo](../../../../docs/sphinx/examples_rst/qec/realtime_decoding_demo.rst)
  — using the *delivered* server and the FPGA playback tool inside the
  CUDA-Q decoding container (nothing built but the example), including the
  `cpu_roce` and FPGA syndrome sources.
- [Decoding task graphs](../../../../docs/sphinx/examples_rst/qec/decoding_task_graph.rst)
  — the task-graph execution model in depth.

If you do not need to build anything, start with the container-based demo
page. If you want the decoder *inside* your application process, start
with the getting-started page. This page is for the standalone server you
build and operate yourself.

## What is in this directory

| File | Contents |
|---|---|
| `decoding_server.cpp` | The standalone decoding server |
| `decoding_server_config.yaml` | Minimal PyMatching configuration (identity H/O/D) |
| `decoding_server_config_multi_error_lut.yaml` | The same experiment with the `multi_error_lut` decoder |
| `decoding_server_config_task_graph.yaml` | The same experiment decoded through a [task graph](#decoding-task-graphs) |
| `MESSAGE_FORMAT.md` | Wire-format notes; see [Transport providers](#transport-providers) for the canonical definitions |
| `CMakeLists.txt` | Build rules |

## Building from source

The full build runs in the dev container, which carries a prebuilt CUDA-Q
**with realtime support** at `/usr/local/cudaq` — see the root
[`Building.md`](../../../../Building.md) for the complete story. The
[dev-image recipe](../../../../docker/build_env/cudaqx.dev.Dockerfile) uses
this repository's [`.cudaq_version`](../../../../.cudaq_version) pin.

If you do not have a checkout yet, clone it on the host:

```bash
git clone https://github.com/NVIDIA/cudaq-qec.git
cd cudaq-qec
```

Select the repository revision you intend to build before choosing its image.
Start from the root of that checkout. On a
Linux x86-64 host with Docker and Python 3, these commands select the
CUDA 13.0 dev-image tag for that checkout's CUDA-Q pin and mount the checkout
at the path used throughout this guide:

```bash
cudaq_ref=$(python3 -c 'import json; print(json.load(open(".cudaq_version"))["cudaq"]["ref"])')
cudaq_dev_image="ghcr.io/nvidia/cudaq-qec-dev:$(printf '%.8s' "$cudaq_ref")-amd64-cu13.0"
docker run -it --name cudaq-qec-dev \
    --mount "type=bind,src=$PWD,dst=/workspaces/cudaq-qec" \
    --workdir /workspaces/cudaq-qec "$cudaq_dev_image"
```

This starts the container for the CPU/Stim smoke test. To expose host GPUs,
add `--gpus all` on a host configured with NVIDIA Container Toolkit. GPU
dispatch additionally needs the decoder, provider, and build components
described under [Decoding task graphs](#decoding-task-graphs). For an ARM64 or
CUDA 12.6 environment, select the corresponding image suffix listed in
`Building.md`.

Inside the container:

```bash
cat >> ~/.bashrc << 'EOF'
export CUDAQX_INSTALL_PREFIX=~/.cudaq-qec
export PATH="${CUDAQX_INSTALL_PREFIX}/bin:${PATH}"
export CUDAQ_DEFAULT_SIMULATOR=stim
EOF
source ~/.bashrc
cd /workspaces/cudaq-qec
cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAQ_INSTALL_DIR=/usr/local/cudaq \
    -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq \
    -DCMAKE_INSTALL_PREFIX="$CUDAQX_INSTALL_PREFIX" \
    -DCUDAQX_INCLUDE_TESTS=ON -DCUDAQX_QEC_INCLUDE_TESTS=ON
ninja -C build install
```

The environment block goes into `~/.bashrc` on purpose: every interactive
shell opened afterwards (`docker exec -it cudaq-qec-dev bash`) picks it
up, including the `CUDAQ_DEFAULT_SIMULATOR=stim` setting the example needs.

`ninja -C build install` produces:

- The server and three sample YAML files in `${CUDAQX_INSTALL_PREFIX}/bin/`.
- The decoder runtime, including the built-in LUT and `sliding_window`
  implementations, in `${CUDAQX_INSTALL_PREFIX}/lib/libcudaq-qec-decoders.so`.
- Shared decoder plugins in `${CUDAQX_INSTALL_PREFIX}/lib/decoder-plugins/`,
  including PyMatching, Chromobius, and the example plugin. The TensorRT
  plugin is built when its optional dependencies are available and its build
  option permits it.

CMake may skip optional TensorRT or realtime-pipeline test targets when
their dependencies are absent. Those warnings do not prevent the PyMatching
UDP smoke test; a missing `decoding_server` or `surface_code-1-cqr` target
does prevent it.

**Build without the CUDA-Q compiler/runtime.** Use a separate
build directory and configure `libs/qec` as the top-level project. The
server still requires a compatible **cudaq-realtime** installation, a CUDA
toolkit, and LLVM support libraries for YAML parsing and schema export.
For example, the dev container's `/usr/local/cudaq` prefix can supply
cudaq-realtime:

```bash
cd /workspaces/cudaq-qec
cmake -G Ninja -S libs/qec -B build_decoders_only \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAQ_QEC_DECODERS_ONLY=ON \
    -DCUDAQ_REALTIME_ROOT=/usr/local/cudaq \
    -DCMAKE_INSTALL_PREFIX="$CUDAQX_INSTALL_PREFIX"
cmake --build build_decoders_only --target install
```

Replace `CUDAQ_REALTIME_ROOT` with the prefix of a standalone cudaq-realtime
installation when building outside that container. If cudaq-realtime is not
found, CMake warns and builds only the decoder stack. Decoder-only mode omits
the quantum application examples and their tests, so use the full build for
the two-process smoke test. See
[the decoder-only server instructions](../../../../Building.md#the-decoding-server)
for dependencies and a server startup check.

**Pairing rules.** Match the CUDA-Q pin, architecture, and CUDA flavor of
the checkout and dev image. The
[image workflow](../../../../.github/workflows/build_dev.yaml)
publishes tags containing the first eight characters of `.cudaq_version`'s
CUDA-Q commit; `latest-*` tags move as images are rebuilt. An older image can
be missing headers or APIs even when CMake configuration succeeds. If the
matching image is unavailable, use the pinned CUDA-Q rebuild instructions in
[`Building.md`](../../../../Building.md). Confirm the combination with the
build and smoke test below. Record the repository commit and resolved image
digest with your results; tags can be rebuilt. Build external decoder plugins
and transport providers against the headers and dependencies of their deployment
environment; the [transport-provider interface check](#transport-providers)
applies specifically to transport providers.

**The release image.** The
[release Dockerfile](../../../../docker/release/Dockerfile.decoding-server)
builds an image containing the server, decoder plugins, playback tools, and
examples. The
[release-image workflow](../../../../.github/workflows/build_decoding_server_docker.yaml)
controls its publication and the optional prebuilt decoder artifacts supplied
to it. The
[validation script](../../../../scripts/validation/decoding-server/validate_decoding_server.sh)
exercises an image's installed tools and two-process examples. See the
[container demo](../../../../docs/sphinx/examples_rst/qec/realtime_decoding_demo.rst)
for that workflow; the following sections use the source build.

## Verify your build: the two-process smoke test

The fastest check that a from-source build works end to end is the two-process
flow, using the in-tree UDP transport over loopback with both processes in
one container. This host-dispatch path needs no QPU, GPU, or RDMA device.
The application (`surface_code-1-cqr`) simulates the QPU with Stim, streams
its syndrome data to the server, and applies the corrections it gets back.
Use the full build above: decoder-only mode omits the application examples.
The commands fix seed 42 and a CNOT depolarizing rate of 0.001.

**1. Generate the decoder configuration** in the first container shell
(the app characterizes the detector error model and writes the YAML; no
server needed for this pass). Adjust the paths if you installed elsewhere:

```bash
export CUDAQX_INSTALL_PREFIX="$HOME/.cudaq-qec"
export PATH="$CUDAQX_INSTALL_PREFIX/bin:$PATH"
export CUDAQ_DEFAULT_SIMULATOR=stim
cd /workspaces/cudaq-qec/build/libs/qec/unittests/realtime/app_examples
./surface_code-1-cqr --distance 3 --num_rounds 12 --num_shots 1000 \
    --seed 42 --p_cnot 0.001 --decoder_type pymatching \
    --save_dem config-sc1-pymatching.yml
```

This pass exits after writing the YAML; it does not run shots or open a
realtime channel.

**2. In a second shell, start the server on that configuration.** On the
host, `docker exec -it cudaq-qec-dev bash` opens it. These exports also make
the setup explicit if you are running the commands through automation:

```bash
export CUDAQX_INSTALL_PREFIX="$HOME/.cudaq-qec"
export PATH="$CUDAQX_INSTALL_PREFIX/bin:$PATH"
export CUDAQ_DEFAULT_SIMULATOR=stim
cd /workspaces/cudaq-qec/build/libs/qec/unittests/realtime/app_examples
QEC_DECODING_SERVER_STATS=1 decoding_server \
    --config=config-sc1-pymatching.yml --transport=udp --port=52100 --timeout=3600
```

Wait for `QEC_DECODING_SERVER_READY port=52100 transport=udp ring0=52100`.
If port 52100 is already occupied, choose another port in both commands.

**3. Run the experiment against the server** (back in the first shell):

```bash
QEC_DECODING_SERVER_TRANSPORT=udp QEC_DECODING_SERVER_PORT=52100 \
    ./surface_code-1-cqr --distance 3 --num_rounds 12 --num_shots 1000 \
    --seed 42 --p_cnot 0.001 --load_dem config-sc1-pymatching.yml
```

The application should print `CQR service dispatch count: 0`. After it exits,
press **Ctrl+C in the server shell** to stop the server cleanly and print its
counters. For this successful single-client run, check these values together:

```text
QEC_DECODING_SERVER_DECODER_STATS id=0 decodes=1000 enqueues=13000 corrections=1000 resets=1000 errors=0
QEC_DECODING_SERVER_RING decoder=0 dispatched=15000
QEC_DECODING_SERVER_DISPATCHED count=15000
```

There are 15 RPCs per shot: one reset, 12 stabilizer enqueues, one final-data
enqueue, and one correction request. Thus 1,000 shots require 15,000 RPCs,
independently of the seed. The server's `corrections` counter counts requests;
the application's “Number of corrections decoder found” counts nonzero
logical corrections. A zero client count alone does not establish the remote
path; also check the server counters.

[PR #921](https://github.com/NVIDIA/cudaq-qec/pull/921) reports these reference
results from a prior live two-process run on a GPU host:

| Decoder | Non-zero values measured | Corrections found | Server dispatched |
|---|---:|---:|---:|
| `pymatching` | 1 | 76 | 15000 |

The outcome totals are a reported reference, not acceptance thresholds for
every build. The reported workload used distance 3, 12 rounds, 1,000 shots and
the default seed, 42. Record your simulator/decoder versions and seed when
comparing outcomes; the 15,000-RPC expectation follows from the workload.

The automated two-process tests cover this architecture, including multiple
rings, the unified host loop, and a parallel-window task graph. Run them from
the build directory with `ctest --output-on-failure -R two-process`; they use
outcome thresholds and traffic checks rather than asserting the reference row.

## Configuration

A configuration names one **decoder entry** per decoder instance. Each entry
carries the model and parameters the server needs to reconstruct its decoder
in another process. The example application also constructs local decoder
objects; the realtime channel determines where its decoding RPCs execute.

Here is the minimal PyMatching configuration shipped in this directory. For
the surface-code experiment above, use the YAML generated by the application:

```yaml
decoders:
  - id: 0
    type: pymatching
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args:
      error_rate_vec: [0.1, 0.1, 0.1]
      merge_strategy: smallest_weight
```

- `H_sparse`, `O_sparse`, `D_sparse` describe the detector error model: H
  maps error mechanisms to detectors, O maps them to observable flips, and
  D maps incoming measurements to detectors. Each sparse vector lists the
  nonzero column indices of successive rows, with `-1` terminating each row.
  In this form, `block_size` counts error mechanisms and `syndrome_size`
  counts detectors. The server applies D before decoding. The legacy
  `error_rate_vec` key under `decoder_custom_args` supplies model error rates;
  the framework removes it from the decoder's custom-parameter map.
- `type` names a **registered decoder**, not a library filename. The runtime
  loads `.so` files from `decoder-plugins/` next to `libcudaq-qec-decoders.so`
  when that library initializes. PyMatching and Chromobius are shared
  plugins; `multi_error_lut` and the `type: sliding_window` decoder wrapper
  are built in. That wrapper needs an inner decoder accepting sliced matrix
  inputs and returning error frames, such as PyMatching or `multi_error_lut`.
  Chromobius needs a Stim DEM and returns observable corrections, so it
  cannot be used as this wrapper's inner decoder.
- A top-level `transport:` section names the wire in the file itself
  (instead of `--transport` on the command line); the two are mutually
  exclusive — the server refuses to start if both name a provider.
- A per-decoder `dispatch:` key (default `host`) selects how that decoder's
  RPCs execute; `device_graph` is the GPU scheduler path for graph-capable
  decoders. Its prerequisites are described under [Decoding task graphs](#decoding-task-graphs).

Configurations can also use `dem_chunks` or Stim DEM text (`stim_dem` /
`stim_dem_path`), with different required and derived fields. See the
[configuration API](../../include/cudaq/qec/realtime/decoding_config.h) and
[task-graph guide](../../../../docs/sphinx/examples_rst/qec/decoding_task_graph.rst).

**Adding a decoder.** An out-of-tree plugin is a shared library installed
in `decoder-plugins/`. Register the decoder, then register its parameter
schema under the same name using `register_decoder_schema`; the
[example plugin](../../lib/decoders/plugins/example/single_error_lut_example.cpp)
shows both registrations, and
[`decoder_config_schema.h`](../../include/cudaq/qec/decoder_config_schema.h)
describes the parameter types. Restart the server after installing the library.
Unknown custom parameter keys are rejected against the registered schema.

The shipped example returns error frames only. Direct server use requires
observable corrections. To adapt an error-frame decoder, use
`decoder::project_errors_to_observables` with the supplied observable mapping,
then return the observable result. Supplying the mapping alone does not select
observable output; loading a plugin and validating its schema do not check
this contract.

Build against the headers, C++ ABI and CUDA dependencies of the environment
you will run in. The decoder loader uses `dlopen`; it does not explicitly
check decoder ABI or CUDA major versions. Unloadable libraries are skipped
with an INFO diagnostic. Set `CUDAQ_LOG_LEVEL=info` and check dependencies
with `ldd` when a decoder is unexpectedly unavailable.

Optional prebuilt NVIDIA plugins include `nv-fusion-decoder` (CPU MWPM for
streaming) and `nv-qldpc-decoder` (GPU belief propagation, including Relay BP
and optional OSD). Availability depends on the supplied artifacts. Consult
their accompanying terms; [`libs/qec/LICENSE`](../../LICENSE) includes the
NVIDIA Software License Agreement for the qLDPC plugin.

**Checking a configuration.** Export the schema from the installation you
intend to run, after installing its plugins:

```bash
cd /workspaces/cudaq-qec
decoding_server --export-schema=decoder-config.schema.json
```

The export reflects the parameter names and types registered by that
installation. With PyYAML and `jsonschema` installed, run the
[offline validator](../config-validator/validate_decoder_config.py)
from the repository root:

```bash
python3 libs/qec/tools/config-validator/validate_decoder_config.py \
    --schema decoder-config.schema.json --allow-task-graph \
    libs/qec/tools/decoding-server/decoding_server_config.yaml \
    libs/qec/tools/decoding-server/decoding_server_config_multi_error_lut.yaml \
    libs/qec/tools/decoding-server/decoding_server_config_task_graph.yaml
```

Task-graph configurations require `--allow-task-graph`. Copying the schema,
validator script, and configuration files to another machine lets you check
them without CUDA-Q or a GPU. This validates structure, parameter types,
and the cross-field rules implemented by the validator; it does not execute
decoder validation hooks or construct decoders. A passing offline check
therefore does not guarantee that the server can load and run the config.
See the [validator documentation](../config-validator/README.md) for the
remaining runtime checks and the `--lib-dir` schema-export alternative.

## Running the server

```text
decoding_server --config=<file> [--transport=<name|/path/to/lib.so>] [--timeout=N]
                [--host-loop=threaded|unified]
                [provider args, e.g. --port=N]

decoding_server --export-schema[=FILE]
```

The examples here use **host dispatch**. The server forwards provider
arguments such as `--port` to the selected transport. A configuration that
names its provider in YAML cannot also use `--transport`. Configurations
containing `device_graph` decoders, or using `gpu_roce`, must put the provider
and its arguments in YAML. Serving stops on SIGINT, SIGTERM, or the timeout
(default 60 seconds); use a longer timeout for an interactive experiment.

For host dispatch, the server opens **one ring per configured decoder**,
each with its own transport instance, endpoint and dispatcher. It prints
one `ring<id>=<port>` token per decoder on the READY line after creating
and binding the endpoints; transport connection and dispatch startup follow.

**Several decoders work the same way.** Stop the single-decoder server,
then generate a config with one entry per logical qubit in the first shell.
Keep the smoke test's environment and working directory:

```bash
./surface_code-1-cqr --distance 3 --num_rounds 12 --num_shots 1000 \
    --seed 42 --p_cnot 0.001 --num_logical 2 --decoder_type pymatching \
    --save_dem config-multi.yml
```

In the second shell, using the same environment and working directory:

```bash
QEC_DECODING_SERVER_STATS=1 decoding_server \
    --config=config-multi.yml --transport=udp --port=0 --timeout=3600
```

`--port=0` assigns a separate port to each ring; a fixed UDP port cannot be
shared by these provider instances. Read `ring0` and `ring1` from the READY
line, then enter those numbers in the first shell:

```bash
read -r -p 'ring0 port from READY: ' port0
read -r -p 'ring1 port from READY: ' port1
QEC_DECODING_SERVER_TRANSPORT=udp QEC_DECODING_SERVER_PORT="$port0" \
    QEC_DECODING_SERVER_PORT_1="$port1" \
    ./surface_code-1-cqr --distance 3 --num_rounds 12 --num_shots 1000 \
    --seed 42 --p_cnot 0.001 --num_logical 2 --load_dem config-multi.yml
```

After the application exits, stop the server with Ctrl+C. Each ring should
report 15,000 dispatches, with 30,000 in the aggregate. The example supports
up to 32 logical qubits; provide `QEC_DECODING_SERVER_PORT_<id>` for every
additional ring. This is the example's result-size limit, not a general
server limit.

**Host loops.** `--host-loop=threaded` (default) serves each host ring on its
own dispatcher thread; with UDP, the provider's RX/TX pump threads move frames
between the wire and the ring. `--host-loop=unified` runs each host ring's
dispatcher as CUDA-Q realtime's single-thread unified loop: it pulls requests
through `rx_poll`, decodes inline, and pushes replies through `tx_publish`,
with no provider pump threads in between. It needs a provider with a CPU
data plane (`udp` supports this; `gpu_roce` does not). The server passes
`--unified` to the provider itself; do not also add it to the provider args.
Mixed configurations leave device-graph rings unaffected; an all-device-graph
configuration rejects `--host-loop=unified`.

**Client setup.** `QEC_DECODING_SERVER_PORT`, the per-ring port variables and
`QEC_DECODING_SERVER_TRANSPORT` are conventions of `surface_code-1-cqr`.
The application translates them into channel arguments for
`cudaq::realtime::initialize`, such as `--cudaq-device-call=udp`,
`udp-host=127.0.0.1`, `udp-port=<port>` and `udp-port.1=<port>`. The default
is UDP; the example also supports `cpu_roce` for one logical qubit, using its
RDMA device and address settings. The kernel itself needs no change; your
application configures a matching client channel independently of the
server's provider. The example still constructs local decoder objects when
loading the YAML; the CQR channel routes its decoding RPCs to the standalone
server.

**Observability.** The host path reports per-ring and aggregate dispatch
counts at orderly shutdown. `QEC_DECODING_SERVER_STATS=1` adds per-decoder
`decodes`, `enqueues`, `corrections`, `resets` and `errors` counters, as shown
in the smoke test. `QEC_DECODING_SERVER_HOP_STATS=1` records host-handler
latency by RPC kind and stage, and prints `QEC_HOP_STATS` lines containing
min/avg/p50/p90/p99/max durations in microseconds. These probes measure
server handling, not transport latency. Set
`QEC_DECODING_SERVER_HOP_STATS_CSV=<path>` as well to write raw samples in
nanoseconds. Recording retains at most 262,144 samples and reports drops.

All-device-graph mode currently supports one decoder and prints
`QEC_DECODING_SERVER_READY device_graph`; it uses GPU diagnostics instead
of the host counters and latency probes above. In mixed configurations,
device-graph rings have dispatch counts, but their GPU-executed requests do
not populate host-handler session counters or latency samples. The aggregate
`QEC_DECODING_SERVER_DISPATCHED` line also counts only host-service requests
in this mixed mode.

On Linux, `QEC_DECODING_SERVER_CPU_DMA_LATENCY_US=<microseconds>` requests a
CPU idle-exit latency limit by holding `/dev/cpu_dma_latency` open. It needs
write access to that device, normally root; failure warns and continues.
This is a latency tuning option, not a correctness requirement.

## Decoding task graphs

Instead of calling a decoder directly, a decoder entry can name a **decoding
task graph** — a small DAG of decoding nodes that the server executes.
For a single window, replace the entry-level `type` with a `task_graph` block
naming the node kind and the decoder it wraps:

```yaml
task_graph:
  type: single_window
  inner_type: pymatching
```

The entry's matrices, dimensions, and `decoder_custom_args` still apply.
[`decoding_server_config_task_graph.yaml`](decoding_server_config_task_graph.yaml)
is a complete example. Pre-expanded H/O/D matrices describe one fixed
syndrome volume; a chunk-based `single_window` configuration also needs a
fixed round count.

`dispatch: host` is the default. A task graph can use `dispatch: device_graph`
when it contains one decoder-backed node that supports CUDA graph capture.
For a window node, the window must have equal size and stride and no guards;
its inner decoder must support graph dispatch and capture successfully.
Multi-node graphs and graphs with a CPU-only inner decoder require host
dispatch. Host-dispatched graphs also support parallel-window decoding,
sparse window models, and asynchronous streaming; see the
[task-graph documentation](../../../../docs/sphinx/examples_rst/qec/decoding_task_graph.rst).

Device dispatch requires a CUDA GPU with compute capability 9.0 or newer,
GPU-accessible provider rings, and a server linked with the device-graph component,
`cudaq-realtime-dispatch`, and the proprietary cudevice archive supplied through
`CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE`. CMake can build the server
without that archive; such a binary cannot run `dispatch: device_graph`.

## Transport providers

The server never links a transport. A wire is a shared library loaded at
runtime, implementing the bridge interface in the installed CUDA-Q header
`include/cudaq/realtime/daemon/bridge/bridge_interface.h` and exporting
`cudaq_realtime_get_bridge_interface`.
Build it against the matching CUDA-Q realtime headers: the
[CUDA-Q revision pinned by this checkout](https://github.com/NVIDIA/cuda-quantum/blob/d39eeed57a7c160de0946de16e373d41a87034e2/realtime/include/cudaq/realtime/daemon/bridge/bridge_interface.h#L238-L250)
uses bridge interface version **4**. A different reported version is rejected
with `Bridge interface version mismatch` before the provider is created.

A provider value containing `/` is used as a library path. A bare name maps
underscores to hyphens: `cpu_roce` becomes
`libcudaq-realtime-bridge-cpu-roce.so`. The server searches the CUDA-Q realtime
library directory configured at build time, trying the hyphenated name and
then the literal spelling; it falls back to the hyphenated soname through the
dynamic loader's search path.

CUDA-Q realtime's providers include `udp`, `cpu_roce`, and `gpu_roce`.
The default development-image build packages `udp` and `cpu_roce`; both support
`--host-loop=unified`. `gpu_roce` has no CPU unified data plane, so its
host-dispatch rings require `--host-loop=threaded`.

Select `gpu_roce`, and any provider used with `dispatch: device_graph`, in
the YAML `transport:` section, with all provider arguments there too.
For these configurations, select the GPU with decoder `cuda_device_id`;
provider arguments must not set `--gpu`, `--unified`, or `--forward`.
For host unified mode, the server adds `--unified` to host-ring provider
arguments automatically. Other host-only configurations can select a
provider with `--transport`; an explicit CLI selection conflicts with a
provider named in YAML.

An out-of-tree provider built on your own NIC stack or controller
interconnect uses the same interface; install it and name its library
in the configuration:

```yaml
transport:
  provider: /opt/my-transport/libmy-bridge.so
  args: [--lane=3]  # arguments understood by this example provider
```

Implement the bridge lifecycle and ring geometry queries, and the data plane
required by the selected dispatch mode. Keep transport-specific framing and
conversion in the provider: publish canonical decoder RPC requests into the
receive ring and transmit the responses from the send ring. A QPU controller
can construct those requests directly or have its provider translate native
packets. The pinned
[UDP bridge](https://github.com/NVIDIA/cuda-quantum/blob/d39eeed57a7c160de0946de16e373d41a87034e2/realtime/lib/daemon/bridge/udp/bridge_impl.cpp)
provides a reference implementation.

The three data-plane RPCs are `enqueue_syndromes`, `get_corrections`, and
`reset_decoder`. Their function IDs and tightly packed payload structures are
defined in
[`decoder_rpc_wire_format.h`](../../include/cudaq/qec/realtime/decoder_rpc_wire_format.h).
Use little-endian scalar fields and LSB-first packed bits. Requests have a
24-byte `RPCHeader`; responses have a 24-byte `RPCResponse`, followed by any
result bytes. Transport frames may include padding beyond the logical message;
`arg_len` and `result_len` identify the payload lengths.

The enqueue payload carries `decoder_id`, `counter`, `syndrome_mapping_id`,
and the syndrome bit count as four 64-bit fields, followed by the packed bits.
Mapping ID `0` is always available and defaults to a pass-through bulk round;
a YAML `syndrome_mapping` row can override it or add other IDs, including a
`final` round. Send a declared ID; an unknown ID is rejected. The ordinary
`enqueue_syndromes` client API sends `0`; `enqueue_syndromes_mapped` selects
an explicit row. For parser behavior and response construction, see
[`RpcSlot.h`](../../lib/realtime/decoding-server-cqr/RpcSlot.h) and
[`DecodingSession.cpp`](../../lib/realtime/decoding-server-cqr/DecodingSession.cpp).

## Troubleshooting

- **App exits with `Error: no realtime channel configured`** —
  configure `QEC_DECODING_SERVER_PORT` in the app shell as shown above.
  This error occurs when neither a server port nor
  `CUDAQ_DEVICE_CALL_CHANNEL` is configured.
- **Server exits with a bind error** — check whether another process holds
  the port. Pick a different `--port`, or use `--port=0` and read it from
  the `QEC_DECODING_SERVER_READY` line.
- **Multi-decoder server reports a bind failure** — the provider prints
  `bind(port=...) failed`; the server then identifies the decoder whose
  provider could not be created. A fixed UDP port cannot be shared by two
  rings; use `--port=0`.
- **Server refuses to start with `--transport=... conflicts with the
  transport section`** — the YAML names a provider *and* the command line
  passes `--transport`; drop one of the two. Keep the provider and its
  arguments in YAML for `device_graph` or `gpu_roce`.
- **`Failed to load plugin ...` for a decoder type** — set
  `CUDAQ_LOG_LEVEL=info` in the server's environment to see the exact
  `dlopen` error. Check dependencies with `ldd` and build the plugin
  against the target environment. Missing libraries or symbols can make
  a decoder unavailable; load failures are logged at INFO.
- **Server stops while you are preparing the app** — increase `--timeout`;
  its default is 60 seconds.
- **Scripted runs can't find `decoding_server`** — a non-interactive shell
  does not source `~/.bashrc`; export `PATH`, `CUDAQX_INSTALL_PREFIX`, and
  `CUDAQ_DEFAULT_SIMULATOR` explicitly in automation.

## Further reading

- [Getting started with realtime decoding](../../../../docs/sphinx/examples_rst/qec/getting_started_realtime_decoding.rst) — the in-process API viewpoint: DEM generation, in-app decoder configuration, kernel usage
- [Realtime decoding demo](../../../../docs/sphinx/examples_rst/qec/realtime_decoding_demo.rst) — the container viewpoint: the delivered server, the FPGA playback tool, and the `cpu_roce`/FPGA syndrome sources
- [Realtime decoding components](../../../../docs/sphinx/components/qec/realtime_decoding.rst) — workflow, terminology, and data flow
- [Playback emulator](../../../../docs/sphinx/examples_rst/qec/playback_emulator.rst) — replaying recorded syndrome streams into a decoding server without hardware
- [Decoding task graphs](../../../../docs/sphinx/examples_rst/qec/decoding_task_graph.rst) — window models and parallel-window streaming
- [`MESSAGE_FORMAT.md`](MESSAGE_FORMAT.md) — companion wire-format notes; use the canonical definitions under [Transport providers](#transport-providers)
- [`Building.md`](../../../../Building.md) — build options, including the
  decoders-only configuration (`CUDAQ_QEC_DECODERS_ONLY`) that builds the
  decoder stack and this server without the full CUDA-Q toolchain
- [Configuration validator](../../tools/config-validator/README.md) — offline schema validation and its limits

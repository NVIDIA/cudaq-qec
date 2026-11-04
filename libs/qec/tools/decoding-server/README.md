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
shared library loaded at runtime — so the same server binary, configuration
format, and decoder setup carry over unchanged when the in-tree UDP provider
is replaced by a hardware transport (see [Transport providers](#transport-providers)).

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
| `MESSAGE_FORMAT.md` | Byte-level wire-format reference for the three RPCs |
| `CMakeLists.txt` | Build rules |

## Building from source

The full build runs in the dev container, which carries a prebuilt CUDA-Q
**with realtime support** at `/usr/local/cudaq` — see the root
[`Building.md`](../../../../Building.md) for the complete story. In short:

```bash
docker run -it --gpus all --name cudaq-qec-dev ghcr.io/nvidia/cudaq-qec-dev:latest-amd64-cu13.0
# inside the container:
cd /workspaces
git clone https://github.com/NVIDIA/cudaq-qec.git && cd cudaq-qec
cat >> ~/.bashrc << 'EOF'
export CUDAQX_INSTALL_PREFIX=~/.cudaq-qec
export PATH="${CUDAQX_INSTALL_PREFIX}/bin:${PATH}"
export CUDAQ_DEFAULT_SIMULATOR=stim
EOF
source ~/.bashrc
mkdir build && cd build
cmake -G Ninja -S .. -DCUDAQ_INSTALL_DIR=/usr/local/cudaq \
  -DCMAKE_INSTALL_PREFIX=$CUDAQX_INSTALL_PREFIX -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq
ninja install
```

The environment block goes into `~/.bashrc` on purpose: every shell opened
in the container afterwards (`docker exec -it cudaq-qec-dev bash`) picks it
up, including the `CUDAQ_DEFAULT_SIMULATOR=stim` setting the example below
needs.

`ninja install` produces:

- `${CUDAQX_INSTALL_PREFIX}/bin/decoding_server` — the installed server
- `${CUDAQX_INSTALL_PREFIX}/lib/libcudaq-qec-decoders.so` — the decoder runtime
- `${CUDAQX_INSTALL_PREFIX}/lib/decoder-plugins/` — the decoder plugins
  (PyMatching, `multi_error_lut`, Chromobius, and the `sliding_window`
  wrapper), each a shared library the runtime `dlopen`s by the `type` name
  in the configuration

**Build without the CUDA-Q toolchain.** If you only need the decoder stack
and the server — not the compiler — configure with
`-DCUDAQ_QEC_DECODERS_ONLY=ON` (see [`Building.md`](../../../../Building.md)).
CMake will report a few *skipped* test targets (TensorRT and realtime-pipeline
tests whose components are not part of the from-source build); that is
expected and does not affect anything on this page.

**Pairing rules.** The CUDA-Q prebuilt in the dev image and this repository
move together: the `latest-*` images are rebuilt as the `.cudaq_version`
pin advances, and a repository checkout may stop compiling against an image
whose prebuilt CUDA-Q is too old (missing APIs) or too new (removed fields).
If you pin a repository commit, pair it with an image built around the same
date, and re-run the smoke test below to confirm. The same applies to
out-of-tree decoder plugins and transport providers: build them against the
headers of the environment you will run them in (see
[Transport providers](#transport-providers) for the version check).

**The release image.** A dedicated decoding container — server, decoder
plugins, the FPGA playback tool, and the examples — is built from
[`docker/release/Dockerfile.decoding-server`](../../../../docker/release/Dockerfile.decoding-server)
and published on the NGC catalog; `scripts/validation/decoding-server/validate_decoding_server.sh`
runs an end-to-end validation of it. The
[demo page](../../../../docs/sphinx/examples_rst/qec/realtime_decoding_demo.rst)
documents that path; the rest of this page stays on the from-source flow.

## Verify your build: the two-process smoke test

The fastest check that a from-source build works end to end is the two-process
flow, using the in-tree UDP transport over loopback with both processes in
one container, so it runs anywhere with no special hardware. The application
(`surface_code-1`) simulates the QPU with Stim, streams its syndrome data
to the server, and applies the corrections it gets back. (The automated
version of the same flow is in the `two-process` tests:
`ctest -R two-process` from the build directory.)

**1. Generate the decoder configuration** (the app characterizes the
detector error model and writes the YAML; no server needed for this pass):

```bash
cd /workspaces/cudaq-qec/build/libs/qec/unittests/realtime/app_examples
./surface_code-1-cqr --distance 3 --num_rounds 12 --num_shots 1000 \
    --decoder_type pymatching --save_dem config-sc1-pymatching.yml
```

**2. In a second shell, start the server on that configuration.** On the
host, `docker exec -it cudaq-qec-dev bash` opens it, and it picks up the
environment from `~/.bashrc`:

```bash
cd /workspaces/cudaq-qec/build/libs/qec/unittests/realtime/app_examples
decoding_server --config=config-sc1-pymatching.yml --transport=udp --port=52100 --timeout=3600
# wait for: QEC_DECODING_SERVER_READY ... ring0=52100
```

**3. Run the experiment against the server** (back in the first shell):

```bash
QEC_DECODING_SERVER_PORT=52100 ./surface_code-1-cqr --distance 3 --num_rounds 12 \
    --num_shots 1000 --load_dem config-sc1-pymatching.yml
```

The application reports `CQR service dispatch count: 0` — nothing was
decoded in-process; every decode crossed the wire. At shutdown the server
prints one `QEC_DECODING_SERVER_RING decoder=<id> dispatched=<N>` line per
ring plus the aggregate `QEC_DECODING_SERVER_DISPATCHED count=<N>`, proving
the traffic crossed the process boundary.

What a correct run prints, for `--distance 3 --num_rounds 12 --num_shots
1000` with the default seed — use it as your build's checksum:

| Decoder type at generation | Non-zero values measured | Corrections found | Server dispatched |
|---|---:|---:|---:|
| `pymatching` | 1 | 76 | 15000 |

(15 calls per shot: 1 `reset_decoder`, 13 `enqueue_syndromes`, 1
`get_corrections`. Your exact counts can differ on a different machine or
seed; the structure of the output should not.)

## Configuration

A configuration names one **decoder entry** per decoder instance. Each entry
carries everything the server needs to reconstruct the decoder in another
process — there is no decoder code on the application side:

```yaml
decoders:
  - id:              0
    type:            pymatching      # decoder plugin to instantiate
    block_size:      363             # error mechanisms in the DEM
    syndrome_size:   96              # detectors
    H_sparse:        [...]           # detector-to-mechanism matrix (-1 terminated groups)
    O_sparse:        [...]           # observables matrix
    D_sparse:        [...]           # measurement-to-detector map
    decoder_custom_args:            # decoder-specific parameters,
      error_rate_vec: [...]         # validated against the plugin's schema
      merge_strategy: smallest_weight
```

- `H_sparse`, `O_sparse`, `D_sparse` describe the detector error model; the
  server turns the raw measurement stream into detector values using
  `D_sparse` before decoding.
- The `type` names a decoder plugin loaded from the `decoder-plugins/`
  directory next to `libcudaq-qec-decoders.so`. Out-of-the-box plugins
  include `pymatching`, `multi_error_lut`, and `chromobius`, plus the
  `sliding_window` wrapper around any of them. NVIDIA ships two additional
  closed-source decoders — `nv-fusion-decoder` (CPU MWPM for low-latency
  streaming) and `nv-qldpc-decoder` (GPU BP/OSD, Relay-BP) — in the
  official release images, under the NVIDIA Software License Agreement.
  A plugin is just a shared library: an out-of-tree decoder drops into
  `decoder-plugins/` the same way. Two rules govern plugins: every
  `decoder_custom_args` entry is validated against the parameter schema
  the plugin registers (the schema is exactly what `--export-schema`
  prints for that decoder type), and a plugin built against a different
  decoder ABI or CUDA major version than the runtime will fail to load —
  the loader skips it *silently*, so set `CUDAQ_LOG_LEVEL=info` on the
  server and check with `ldd` when a decoder type is unexpectedly
  "unknown".
- A top-level `transport:` section names the wire in the file itself
  (instead of `--transport` on the command line); the two are mutually
  exclusive — the server refuses to start if both name a provider.
- A per-decoder `dispatch:` key (default `host`) selects how that decoder's
  RPCs execute; `device_graph` is the GPU scheduler path for graph-capable
  decoders.

**Checking a configuration.** `decoding_server --export-schema` prints the
full JSON Schema (draft 2020-12) of the server's YAML, and
[`libs/qec/tools/config-validator/validate_decoder_config.py`](../../tools/config-validator/validate_decoder_config.py)
validates a YAML offline (pyyaml + jsonschema) on a machine with no CUDA-Q —
handy for CI checks of deployment configs. (Task-graph configurations
require `--allow-task-graph` there.)

## Running the server

```text
decoding_server --config=<file> [--transport=<name|/path/to/lib.so>] [--timeout=N]
                [--host-loop=threaded|unified] [--export-schema[=FILE]]
                [--port=N] [provider args, forwarded verbatim...]
```

- The server opens **one ring buffer per configured decoder**, each with its
  own endpoint and its own dispatcher, and prints a
  `QEC_DECODING_SERVER_READY` line with one `ring<id>=<port>` token per
  decoder once every ring is listening. With several decoders use
  `--port=0` (an explicit port cannot be shared by two rings) and pass each
  ring's port to the application as `QEC_DECODING_SERVER_PORT_<id>`:

  ```bash
  # config with one decoder entry per logical qubit; the server picks ports:
  decoding_server --config=config-multi.yml --transport=udp --port=0 --timeout=3600
  # QEC_DECODING_SERVER_READY port=<P0> transport=udp ring0=<P0> ring1=<P1>

  # the application, with one endpoint per ring from the READY line:
  QEC_DECODING_SERVER_PORT=<P0> QEC_DECODING_SERVER_PORT_1=<P1> \
      ./surface_code-1-cqr --distance 3 --num_rounds 12 --num_shots 1000 \
      --num_logical 2 --load_dem config-multi.yml
  ```

  Up to 32 decoders work the same way (`QEC_DECODING_SERVER_PORT_2`, ...).
- `--host-loop=threaded` (default) serves each ring on its own dispatcher
  thread, with the provider's own RX/TX pump threads moving frames between
  the wire and the ring. `--host-loop=unified` runs each ring's dispatcher
  as CUDA-Q realtime's single-thread unified loop: the dispatcher itself
  pulls each request through the provider's `rx_poll` hook, decodes inline,
  and pushes the reply through `tx_publish`, with no provider pump threads
  in between. It needs a provider that serves a CPU data plane (the in-tree
  `udp` provider does; `gpu_roce` does not), and the server passes
  `--unified` to the provider itself (do not also add it to the provider
  args).
- On the client, the example application (`surface_code-1`) reads
  `QEC_DECODING_SERVER_PORT` (and `QEC_DECODING_SERVER_PORT_<id>` for extra
  rings) and `QEC_DECODING_SERVER_TRANSPORT` (default `udp`; also accepts
  `cpu_roce`) and turns them into CUDA-Q device-call channel arguments for
  `cudaq::realtime::initialize` (`--cudaq-device-call=udp
  udp-host=127.0.0.1 udp-port=<P>`). The kernel itself needs no change;
  your own application sets up its channel the same way, and the client
  channel is separate from the server's transport provider.

**Observability.** At shutdown the server prints per-ring dispatch counts,
and with `QEC_DECODING_SERVER_STATS=1` one
`QEC_DECODING_SERVER_DECODER_STATS id=<id> decodes=<n> ...` line per decoder.
`QEC_DECODING_SERVER_HOP_STATS=1` adds per-request latency probes
(min/avg/p50/p90/p99/max per RPC kind and internal stage), which work
unchanged over any transport provider because they bracket only the
server-side handling; `QEC_DECODING_SERVER_HOP_STATS_CSV=<path>` writes the
raw samples to a file. `QEC_DECODING_SERVER_CPU_DMA_LATENCY_US` holds
`/dev/cpu_dma_latency` at a target to keep CPUs out of deep idle states
(root; a latency QoS, not a correctness requirement).

## Decoding task graphs

Instead of calling a decoder directly, a decoder entry can name a **decoding
task graph** — a small DAG of decoding nodes that the server executes.
`decoding_server_config_task_graph.yaml` in this directory shows the
single-window shape: the entry-level `type` becomes a `task_graph` block
naming the node kind and the decoder it wraps, and everything else is what
the plain config says:

```yaml
decoders:
  - id: 0
    # ... H/O/D and sizes exactly as before ...
    task_graph:
      type: single_window      # one window over the whole syndrome volume
      inner_type: pymatching   # the decoder the window node calls
```

`single_window` is the fixed-round shape that pre-expanded H/O/D matrices
describe. Task graphs are CPU-hosted (`dispatch: host`; `device_graph` is
rejected at load). Beyond single-window, the task-graph executor also
supports parallel-window streaming decoding — window models, sparse window
decoding, and async streaming — documented in
[the decoding task graph documentation](../../../../docs/sphinx/examples_rst/qec/decoding_task_graph.rst).

## Transport providers

The server never links a transport. A wire is a shared library implementing
the bridge interface defined in the installed CUDA-Q header
`include/cudaq/realtime/daemon/bridge/bridge_interface.h` (interface
version 4 at the current CUDA-Q pin; a provider built against a different
version is rejected at load with `Bridge interface version mismatch`). A
`--transport` value containing a `/` is loaded verbatim as a library path;
a bare name resolves
to `libcudaq-realtime-bridge-<name>.so` next to the installed CUDA-Q
realtime libraries. The in-tree providers are `udp`, `cpu_roce`, and
`gpu_roce` (the dev images ship `udp` and `cpu_roce`; `gpu_roce` must be
named in the YAML `transport:` section, not via `--transport`) — an
out-of-tree provider built on your own NIC stack or
controller interconnect drops in the same way with zero changes to this
server or to the QEC libraries:

```yaml
transport:
  provider: /opt/my-transport/libmy-bridge.so
  args: [--lane=3]
```

What travels on the wire is the decoder RPC format — the three data-plane
RPCs with tightly packed, little-endian payloads — documented byte by byte
in [`MESSAGE_FORMAT.md`](MESSAGE_FORMAT.md). A QPU control system can
produce these frames natively, or the provider translates its native
packets into them as it fills the receive ring; the translation lives
entirely inside your library. The in-tree UDP provider
(`bridge/udp/bridge_impl.cpp` in the CUDA-Q repository) is a compact
reference implementation.

## Troubleshooting

- **App exits with `Error: no realtime channel configured`** —
  `QEC_DECODING_SERVER_PORT` was not set in the shell running the app.
- **Server exits with a bind error** — another process holds the port; pick
  a different `--port`, or use `--port=0` and read the assigned port from
  the `QEC_DECODING_SERVER_READY` line.
- **A multi-decoder YAML fails with `bind(port=...) failed ... for decoder
  1`** — an explicit `--port=N` cannot be shared by two rings; use
  `--port=0`.
- **Server refuses to start with `--transport=... conflicts with the
  transport section`** — the YAML names a provider *and* the command line
  passes `--transport`; drop one of the two.
- **`Failed to load plugin ...` for a decoder type** — set
  `CUDAQ_LOG_LEVEL=info` in the server's environment to see the exact
  `dlopen` error; a plugin built against a different decoder ABI or CUDA
  major version is the usual cause, and the plugin loader skips
  unloadable plugins silently otherwise.
- **Scripted runs can't find `decoding_server`** — a non-interactive shell
  does not source `~/.bashrc`; export `PATH`, `CUDAQX_INSTALL_PREFIX`, and
  `CUDAQ_DEFAULT_SIMULATOR` explicitly in automation.

## Further reading

- [Getting started with realtime decoding](../../../../docs/sphinx/examples_rst/qec/getting_started_realtime_decoding.rst) — the in-process API viewpoint: DEM generation, in-app decoder configuration, kernel usage
- [Realtime decoding demo](../../../../docs/sphinx/examples_rst/qec/realtime_decoding_demo.rst) — the container viewpoint: the delivered server, the FPGA playback tool, and the `cpu_roce`/FPGA syndrome sources
- [Realtime decoding components](../../../../docs/sphinx/components/qec/realtime_decoding.rst) — workflow, terminology, and data flow
- [Playback emulator](../../../../docs/sphinx/examples_rst/qec/playback_emulator.rst) — replaying recorded syndrome streams into a decoding server without hardware
- [Decoding task graphs](../../../../docs/sphinx/examples_rst/qec/decoding_task_graph.rst) — window models and parallel-window streaming
- [`MESSAGE_FORMAT.md`](MESSAGE_FORMAT.md) — the byte-level wire format this server speaks
- [`Building.md`](../../../../Building.md) — build options, including the
  decoders-only configuration (`CUDAQ_QEC_DECODERS_ONLY`) that builds the
  decoder stack and this server without the full CUDA-Q toolchain

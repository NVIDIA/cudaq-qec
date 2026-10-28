# Memory-experiment generator

This public tool prepares a bounded Surface Code model and writes one
YAML file containing the framework's public `parallel_window`
configuration. Experiment, decoder and pattern parameters are command-line
inputs. The supported scope is Surface Code memory experiments with a requested
finite horizon. Executable pattern support and server integration are tracked
in [PR #885](https://github.com/NVIDIA/cudaq-qec/pull/885).
The framework owns A/B construction, fault ownership, routing and scheduling.

A QEC build providing the public parallel-window configuration, named model
artifacts and terminal-readout metadata is required. The generator uses those
APIs directly. **Parsing and configuration validation do not establish that
the target runtime can execute the pattern.**

## Run in the decoding-server image

The decoding-server Docker image includes the compiled generator and its
runtime dependencies. No source checkout, compiler or Python installation is
needed to generate YAML. Inside the image:

```sh
memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
  --output=memory.yaml
```

To save the file on the host, bind an existing output directory into the
container. Set `DECODING_IMAGE` to the decoding-server image tag for your CUDA
version, then run from that directory:

```sh
docker run --rm --volume "$PWD:/configs" \
  --entrypoint memory-experiment-generator "$DECODING_IMAGE" \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
  --output=/configs/memory.yaml
```

Both executables are installed under `/opt/nvidia/cudaq-qec/bin`, on the image's
`PATH`. The generator does not contact or start the server. After successful
generation, give the same file to `decoding_server --config=PATH` in an image
with executable parallel-window support. Official release archives omit both
the standalone server and generator; use the decoding-server image for this
workflow.

### Building for development

For library development, a full QEC build configured with CUDA-Q, Stim and
realtime configuration also provides the tool:

```sh
cmake --build build --target memory-experiment-generator
build/bin/memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
  --output=memory.yaml
```

This is the distance-three reference. A/B commit and buffer widths
default to distance, so the emitted pattern contains `commit_rounds: 3`
and `buffer_rounds: 3`. The gap is omitted unless explicitly supplied,
leaving its rule to the framework. The public pattern documents that rule as
`commit_rounds + 2 * buffer_rounds`, giving 9 for this profile.

For a longer experiment with two independent patches:

```sh
memory-experiment-generator \
  --distance=5 --rounds=1000 --basis=X --logical-qubits=2 \
  --decoder=pymatching --output=long-memory.yaml
```

For PyMatching, this A/B profile writes `merge_strategy: smallest_weight` when
no merge policy is supplied. This handles parallel boundary edges in A windows
without changing PyMatching's own decoder defaults. An explicit
`--decoder-arg=merge_strategy=...` is preserved, including `independent` or
`disallow`; the selected policy still has to support the concrete window model.

To request four pool workers for each of those two patch graphs, add
`--workers-per-graph=4`. The generated configuration sets `async_streaming: true`
and the framework's per-graph `async_num_workers` count to four. It also sets the
pattern node's `max_in_flight` field to four, which applies to both A and B layers.
With no worker option, the pool policy remains framework-selected and the node
retains its default capacity of one.

`max_in_flight` bounds how many windows in each layer may decode concurrently;
`async_num_workers` bounds pool work across the graph. Dependencies and available
work can further limit concurrency. Under the node-copy execution model, the
N-per-layer default permits 2N decode-window node copies sharing N workers;
they cannot all run together. For N > 1, a runtime warning about that
oversubscription is expected. Each copy can own variant-specific decoders and
buffers, so increasing the layer capacity also increases memory use; it does not
add workers.

The command reports eight configured workers for two four-worker graphs,
excluding the additional coordinator thread for each graph. This is a per-graph
setting, not a shared process-wide core budget or CPU reservation.

Start with the synchronous default for inexpensive windows, such as small-distance
PyMatching: worker handoff overhead can outweigh the decoding work. Workers can
help when windows are more expensive or have variable decoding times. Measure the
chosen decoder, distance and arrival workload rather than assuming more workers
are faster.

These examples are separate configurations. Retain or move an existing
configuration before generating another; the tool never replaces it.

## Output and server handoff

This is a standalone YAML-generation tool. `--output=PATH` is required;
there is no default filename or implicit output location. Relative paths
resolve from the generator's working directory. The tool does not read
the decoding service's `CUDAQ_QEC_DECODER_CONFIG` environment variable or
change how the server finds its configuration.

Once the target runtime supports the generated pattern, pass the file to the
server explicitly. For example, with an existing destination directory:

```sh
memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
  --output=/path/to/configs/memory.yaml
decoding_server --config=/path/to/configs/memory.yaml
```

The generator neither starts the server nor configures its transport. The
server's existing required `--config=PATH` interface is unchanged.

The only generated file contains its shared model and all independent patch
entries. A separate model file, manifest, measurement sample or JSON file is
not needed. The destination parent must exist. Existing files, directories and
symlinks are never overwritten. Model and exact YAML round-trip validation
finish before the output is created exclusively. Concurrent generators cannot
overwrite each other's result. A detected write or close failure removes the
file created by that invocation.
New files use standard permissions (`0666` restricted by the deployment's
umask); the tool does not change the process umask. Wait for successful
completion before reading or passing the file to the server: a reader may
otherwise see an incomplete write. A crash or forced termination can leave a
partial file; inspect and remove that file before retrying.

## Inputs

Arguments use `--name=value` form; there is no JSON or YAML input format.
`--help` does not prepare a model.

| Input | Meaning / default |
|---|---|
| `--code` | `surface_code`; other code families are not currently supported |
| `--distance` | Required odd Surface Code distance, at least 3 |
| `--rounds` | Required extraction-cycle horizon T, at least 4, excluding terminal data readout; records `num_rounds: T+1` |
| `--basis` | Required `X` or `Z` preparation/readout basis; case-insensitive |
| `--logical-qubits` | Independent patches sharing one model; default 1 |
| `--error-rate` | CNOT two-qubit depolarization probability; default 0.001, strictly between 0 and 1 |
| `--decoder` | Required installed inner decoder name |
| `--commit-rounds` | Optional positive A-layer commit width; defaults to distance |
| `--buffer-rounds` | Optional positive A-layer buffer width; defaults to distance |
| `--gap-rounds` | Optional positive B-layer gap override; omitted otherwise |
| `--decoder-arg` | Repeatable `PATH=VALUE`, typed by the registered decoder schema |
| `--pattern-arg` | Repeatable `PATH=VALUE`, typed by the registered `parallel_window` schema |
| `--workers-per-graph` | Optional positive pool-worker count per patch; also sets the same per-layer window capacity; excludes the coordinator |
| `--output` | Required YAML destination; parent must exist; existing entries are never replaced |

`--rounds=T` records the requested horizon as `dem_chunks.num_rounds=T+1`,
including the terminal readout layer. It is not the characterization depth:
the tool establishes the repeating phase recipe using bounded circuits whose
internal depths are independent of T. The parallel-window ingest contract uses
FINAL to discover each shot's actual length and accepts valid lengths at or
above the repeating model's minimum, rather than requiring the recorded horizon.
For these Surface Code models the minimum is four extraction cycles plus readout.
Executing that contract requires the pattern runtime described above.

Window geometry belongs to the A/B scheduling scheme, not to an individual
decoder's parameters. Every `parallel_window` configuration requires commit
and buffer widths irrespective of its inner algorithm. The Surface Code
generator supplies those fields; users need not enter them for each decoder.
Geometry overrides tune the same A/B scheme, not a different algorithm.
The dedicated geometry options are aliases for `commit_rounds`, `buffer_rounds`
and `gap_rounds` in `--pattern-arg`. A pattern argument alone can override the
default geometry, for example `--pattern-arg=commit_rounds=5`. Do not also supply
`--commit-rounds` for that field: using both spellings is rejected, even when
their values agree. The same rule applies to buffer and gap widths.

Decoder arguments are validated through the selected decoder's registered
schema. Dotted paths select nested leaves,
for example `--decoder-arg=global_decoder_params.max_errors=2`. Booleans are
`true` or `false`; numeric values must be finite and fully consumed. Double
vectors use commas (`0.1,0.2`); matrices use semicolon-separated rows
(`0.1,0.2;0.3,0.4`). Quote shell-sensitive values. Strings retain their literal
value. Unknown or duplicate arguments and missing required parameters fail
before preparation.

This Surface Code A/B model profile emits YAML for PyMatching, NV-Fusion,
Relay BP (`nv-qldpc-decoder`), and the single- and multi-error LUT decoders.
Matching backends use graphlike decomposition; Relay BP and LUT retain joint
hyperedge faults. The existing monolithic Ising/TRT profile requires its own
model artifacts and is not generated by this tool. YAML generation does not
qualify backend execution.

Pattern arguments use the same schema-typed conversion rules as decoder
arguments, but are validated against the linked framework's registered
`parallel_window` schema. For example:

```sh
build/bin/memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
  --output=pattern-memory.yaml \
  --pattern-arg=output=corrected_logical \
  --pattern-arg=boundary_prior=0.01 \
  --pattern-arg=verify_residual=report \
  --pattern-arg=gap_rounds=9 \
  --pattern-arg=merge_identical=true \
  --pattern-arg=conflicting_parallel=most_likely
```

The boundary-column prior must be strictly between zero and one. Alternatively,
use `--pattern-arg=input_boundary=truncate` without `boundary_prior`; truncation
keeps the crossing faults' own priors, so combining those options is rejected.
Boolean column options use `true` or `false`, for example
`--pattern-arg=merge_identical=false`. Unknown names, duplicate arguments and
invalid combinations fail before model preparation.

The same configuration structure is used for each decoder; its name and
schema-typed parameters differ. The selected backend must support fault-column
output for the concrete A/B windows. Parameter validation does not establish
that compatibility or execute the generated graph.

## Server configuration and deployment

The YAML embeds one compact model in `data_artifacts`; artifact names are
references within that document, not file paths. Each patch has a distinct
decoder ID and one `parallel_window` pattern, with the same recipe and
geometry. Output defaults to `corrections`, which existing consumers may XOR
with their own raw readout. `--pattern-arg=output=corrected_logical` instead
requests that the framework combine the raw logical readout with its correction
frame. The model already embeds the terminal `m2o_sparse` readout map; this
selection does not require a separate readout file or change the experiment.

The mappings remain distinct:

- D / m2d maps chronological measurements to detectors.
- H maps faults to detector effects.
- Phase O maps faults to logical flips.
- Terminal `m2o_sparse` maps final data measurements to the raw logical value.

The public serializer and parser must preserve the model, geometry, decoder
parameters and pattern parameters exactly, including floating-point vectors and
matrices. Bounded validation checks seams, detector-map rows and indices, widths,
probabilities, horizons, code geometry and raw-logical support before writing.

Pass-through syndrome mapping ID 0 is BULK; ID 1 is FINAL. A producer must select
the terminal mapping correctly. Packet batching and transport-control schedules
are not model geometry.

The generator leaves transport and CPU placement unset. Configure these for
the server deployment and ensure the producer follows the generated record
layout and BULK/FINAL mapping IDs. The CLI does not sample measurements, create
replay schedules or adapt the FPGA producer protocol.

## Counts and bounded preparation

T counts extraction cycles, excluding the distinct terminal readout input
round. There are T+1 raw rounds. Actual D determines detector rows and lookback;
endpoint counts, including the basis-dependent readout layer, need not equal
regular counts.

The provider characterizes bounded representative circuits using the public
`decoder_context_from_memory_circuit` API. It compares retained phases,
probabilities, measurement maps and readout support before retargeting.
Preparation and stored phases do not grow with requested T; code construction
grows with distance. This relies on the known repeating Surface Code circuit,
not inferred periodicity for arbitrary circuits.

The public `memory_dem_chunks_from_decoder_inputs` helper is promoted from
the Surface Code example. It requires round-major detectors, D lookback of at
most one raw round, faults touching at most two adjacent detector rounds,
independent fault IDs, at most 64 observables and a repeating bulk. It retains
transient phases and merges identical H/O mechanisms with independent XOR
probabilities. Unsupported faults fail instead of being dropped or replaced.

At least four extraction cycles are required; shorter shots fail without
rounding. Generated `beforeN`, `bulk` and `afterN` phase names are preserved,
not hardcoded executor bindings. Multiple logical qubits mean independent
patches, not lattice surgery.

## Build scope and verification

The executable installs through `qec-tools`, with existing QEC and CUDA-Q
runtime dependencies. Official release archives currently omit both this tool
and the standalone decoding server; build them from a full QEC checkout for
development. Circuit
characterization requires full QEC, CUDA-Q and Stim, so the target is skipped
in decoders-only builds. Building the server does not acquire those
dependencies. Tool-local interfaces are not new installed SDK APIs.

```sh
python3 -m pip install -r libs/qec/unittests/memory-experiment-generator/requirements.txt
cmake --build build --target memory-experiment-generator test_memory_dem_chunks test_memory_model_provider
ctest --test-dir build \
  -R 'MemoryDemChunks|MemoryExperiment|MemoryRecord|memory_experiment_generator_cli' \
  --output-on-failure
```

PyYAML is a black-box test dependency only; the installed C++ tool has no
Python dependency. Install it before running the CLI tests.

Tests cover model agreement with complete circuits, input validation, large
horizons, schema-typed parameters, round trips, worker settings and safe output
creation. These checks do not establish full graph execution, decoder accuracy,
transport compatibility or real-time performance.

## Generated configuration

The following abbreviated output shows the configuration for the distance-three
example. The actual file contains the complete model, including every phase's
sparse matrices, priors and terminal readout map; the comments below stand in
for that content and are not a runnable replacement.

```yaml
data_artifacts:
  memory_model:
    dem_chunks:
      num_rounds: 27       # 26 extraction cycles plus terminal readout
      measurements_per_round: 8
      # seam, connections and complete phase specifications omitted
decoders:
  - id: 0
    dem_chunks: memory_model
    task_graph:
      nodes:
        - id: parallel_memory
          type: parallel_window
          commit_rounds: 3
          buffer_rounds: 3
          output: corrections
          inner: {type: pymatching, merge_strategy: smallest_weight}
      edges:
        - {from: measurements, to: parallel_memory}
        - {from: parallel_memory, to: output}
syndrome_mapping:
  - {id: 0}
  - {id: 1, type: final}
```

## Exit status and troubleshooting

The generator returns 0 after publishing a complete configuration, or for
`--help` / `-h` anywhere in the command. It returns 1 for invalid arguments,
preparation or serialization errors, or file-publication failures, with a
diagnostic on standard error. Failures do not replace an existing destination.
A generated configuration is not evidence that the server can execute it.

- **Server reports `the 'parallel_window' pattern is not available yet`:**
  the linked framework recognizes the configuration but does not implement
  its runtime expansion yet. The standalone server exits with status 1.
  Executable parallel-window support is required; changing decoder arguments
  does not enable that runtime.
- **`decoder is not registered in this QEC build: pymatching`:** PyMatching
  must be enabled in the build. When its target exists, building the generator
  also builds the plugin; run from the matching build or installation so the
  runtime loader can discover `lib/decoder-plugins`.
- **`configuration already exists`:** retain or move the old configuration
  before generating another. The tool intentionally never overwrites it.
- **`configuration parent must exist`:** create the destination directory or
  choose `--output=PATH` inside an existing directory.
- **`cannot create server configuration` / `cannot write server configuration`:**
  check directory permissions, free space and filesystem write errors. Do not
  start the server until generation finishes successfully.

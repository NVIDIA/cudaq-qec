# Memory-experiment generator

This public tool prepares a bounded Surface Code model and writes one
`server.yaml` containing the framework's public `parallel_window`
configuration. Experiment, decoder and pattern parameters are command-line
inputs. The supported scope is finite-length Surface Code experiments;
execution requires the pending parallel-window runtime.
The framework owns A/B construction, fault ownership, routing and scheduling;
the tool does not implement a second scheduler or substitute a global or
sliding-window algorithm.

A QEC build providing the public parallel-window configuration, named model
artifacts and terminal-readout metadata is required. The generator uses those
APIs directly. **Parsing and configuration validation do not establish that
the target runtime can execute the pattern.**

## Build and run

In a full QEC build configured with CUDA-Q, Stim and realtime configuration:

```sh
cmake --build build --target memory-experiment-generator
build/bin/memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching
```

This is the distance-three reference. A/B commit and buffer widths
default to distance, so the emitted pattern contains `commit_rounds: 3`
and `buffer_rounds: 3`. The gap is omitted unless explicitly supplied,
leaving its rule to the framework. The public pattern documents that rule as
`commit_rounds + 2 * buffer_rounds`, giving 9 for this profile. Executing this
configuration requires the parallel-window runtime expansion.

For a longer experiment with two independent patches:

```sh
build/bin/memory-experiment-generator \
  --distance=5 --rounds=1000 --basis=X --logical-qubits=2 \
  --decoder=pymatching
```

For PyMatching, this A/B profile writes `merge_strategy: smallest_weight` when
no merge policy is supplied. This handles parallel boundary edges in A windows
without changing PyMatching's own decoder defaults. An explicit
`--decoder-arg=merge_strategy=...` is preserved, including `independent` or
`disallow`; the selected policy still has to support the concrete window model.
Window-level tests do not establish the accuracy or performance of the complete
decoding graph.

To request four pool workers for each of those two patch graphs, add
`--workers-per-graph=4`. The generated configuration sets `async_streaming: true`
and the framework's per-graph `async_num_workers` count to four. The command
reports eight configured workers in total, excluding the additional coordinator
thread for each graph.
This is a per-graph setting, not a shared process-wide core budget or CPU
reservation. The framework must support the worker-pool runtime to execute it.

These examples are separate configurations. Retain or move an existing
configuration before generating another; the tool never replaces it.

## Shared configuration location

There is no generator output-path argument. The generator and standalone
`decoding_server` use the same tools-private path resolver:

1. A nonempty `CUDAQ_QEC_DECODER_CONFIG` selects the deployment's configuration
   file, using the environment setting already recognized by the server service.
2. Otherwise, both use `server.yaml` in their working directory.

The default therefore follows the container image's `WORKDIR` and also works
outside Docker without requiring write access to an installation or system
directory. Deployments can set the existing environment variable once for
both processes, for example when mounting a configuration volume; it is not a
per-experiment CLI input. Relative environment paths resolve from the process
working directory. Both processes must use the same working directory or
deployment setting.

The server no longer requires `--config`. Its existing explicit
`--config=PATH` remains available and takes precedence for compatibility with
other configurations. The in-process service's configuration precedence is
unchanged; it does not acquire implicit file loading.

Once the target runtime supports the generated pattern, the matching launch
is simply:

```sh
build/bin/decoding_server
```

The only generated file contains its shared model and all independent patch
entries. A separate model file, manifest, measurement sample or JSON file is
not needed. The destination parent must exist. Existing files, directories and
symlinks are never overwritten. Model and exact YAML round-trip validation
finish before publication; a uniquely reserved sibling holds the incomplete
write, then the completed file is published atomically without replacement.
Failures remove only that invocation's temporary file. Concurrent generators
cannot overwrite each other's result.
New files use standard permissions (`0666` restricted by the deployment's
umask); the tool does not change the process umask. The destination filesystem
must support hard links; unsupported publication fails and removes the staging
file without falling back to an overwrite.

## Inputs

Arguments use `--name=value` form; there is no JSON or YAML input format.
`--help` does not prepare a model.

| Input | Meaning / default |
|---|---|
| `--code` | `surface_code`; other code families are not currently supported |
| `--distance` | Required odd Surface Code distance, at least 3 |
| `--rounds` | Required finite extraction-cycle count, excluding terminal data readout |
| `--basis` | Required `X` or `Z` preparation/readout basis |
| `--logical-qubits` | Independent patches sharing one model; default 1 |
| `--error-rate` | CNOT two-qubit depolarization probability; default 0.001, strictly between 0 and 1 |
| `--decoder` | Required installed inner decoder name |
| `--commit-rounds` | Optional positive A-layer commit width; defaults to distance |
| `--buffer-rounds` | Optional positive A-layer buffer width; defaults to distance |
| `--gap-rounds` | Optional positive B-layer gap override; omitted otherwise |
| `--decoder-arg` | Repeatable `PATH=VALUE`, typed by the registered decoder schema |
| `--pattern-arg` | Repeatable `PATH=VALUE`, typed by the registered `parallel_window` schema |
| `--workers-per-graph` | Optional positive pool-worker count for each patch's graph; excludes its coordinator |

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
schema, not a tool-owned decoder whitelist. Dotted paths select nested leaves,
for example `--decoder-arg=global_decoder_params.max_errors=2`. Booleans are
`true` or `false`; numeric values must be finite and fully consumed. Double
vectors use commas (`0.1,0.2`); matrices use semicolon-separated rows
(`0.1,0.2;0.3,0.4`). Quote shell-sensitive values. Strings retain their literal
value. Unknown or duplicate arguments and missing required parameters fail
before preparation.

Pattern arguments use the same schema-typed conversion rules as decoder
arguments, but are validated against the linked framework's registered
`parallel_window` schema. For example:

```sh
build/bin/memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
  --pattern-arg=output=corrected_logical \
  --pattern-arg=boundary_prior=0.01 \
  --pattern-arg=verify_residual=report \
  --pattern-arg=max_in_flight=4 \
  --pattern-arg=merge_identical=true \
  --pattern-arg=conflicting_parallel=most_likely
```

The boundary-column prior must be strictly between zero and one. Alternatively,
use `--pattern-arg=input_boundary=truncate` without `boundary_prior`; truncation
keeps the crossing faults' own priors, so combining those options is rejected.
Boolean column options use `true` or `false`, for example
`--pattern-arg=merge_identical=false`. Unknown names, duplicate arguments and
invalid combinations fail before model preparation. New parameters using
existing schema types need no generator key-list change; semantic/default
changes still need validation and may require adaptation.

The same configuration structure is used for each decoder; its name and
schema-typed parameters differ. Configuration emission does not qualify a
decoder's fault-output capability, A/B accuracy or parallel execution.
The CLI does not construct inner decoders against the actual A/B window models.
That validation must use the framework's authoritative pattern lowering when
available, requesting fault-column outputs and checking the concrete model and
parameters. Component tests use the framework's window construction and decode
helpers; they do not execute the generated pattern. A tiny or whole-shot model
is not a substitute for those windows. Parameter validation uses registered
decoder schemas, not a backend-name whitelist. PyMatching has the one
profile-specific merge-policy default described above.
When a worker count is supplied, every independent patch receives that same
per-graph count. The total is checked for overflow before model preparation and
reported explicitly; no allocation across graphs is invented. Omitting the
option preserves framework execution defaults and reports that the worker total
is unspecified. The count is an execution option, not an inner-decoder argument.
CPU affinity remains unset, allowing the framework to inherit process affinity;
the generator neither assigns cores nor duplicates cross-graph overlap checks.
Configuring workers does not establish runtime availability or measured overlap.
`--pattern-arg=max_in_flight=4` is a different setting: it requests concurrent
window instances/decoder replicas from the pattern, not four worker threads or
four dedicated cores. It does not replace `--workers-per-graph`.

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

The generator leaves transport and CPU placement unset. It does not render a
qualified CPU RoCE/GPU RoCE deployment or adapt the FPGA producer protocol.
GPU RoCE transport and GPU decoder dispatch are different choices; host A/B
decoding must not be switched to device dispatch merely to select a wire.

Raw samples, seeds, replay schedules and result-oracle records belong to the
qualification workflow. The production CLI does not sample measurements.
Tests retain the public `sample_memory_circuit_raw` sampler and record/batch
helpers. Raw logical values alone are not expected corrections or an accuracy
oracle.

Internal UDP assessment is not customer RoCE/FPGA qualification. Installed
two-process replay needs an executable A/B runtime, a transport-specific
schedule, correctly tagged FINAL input and independent expected results.
Real FPGA qualification also needs producer framing, mapping IDs, patch routing
and lifecycle. A successful YAML parse is not evidence that these work.

## Counts and bounded preparation

T counts extraction cycles, excluding the distinct terminal readout input
round. There are T+1 raw rounds. Actual D determines detector rows and lookback;
endpoint counts, including the basis-dependent readout layer, need not equal
regular counts.

The reference has `26 * 8 + 9 = 217` measurements. The layout test groups
these as eight 24-bit batches and a terminal batch of `2 * 8 + 9 = 25` bits.
B0 needs raw rounds `[2,12)` (80 bits); A2 needs `[20,27)` (57 bits). The
fixture tiles exactly and exercises both experiment edges, but not a terminal
commit remainder or actual A/B execution.

The provider characterizes 12 and 13 extraction cycles using the public
`decoder_context_from_memory_circuit` API. It compares retained phases,
probabilities, measurement maps and readout support before retargeting.
These fixed depths leave margin beyond transient endpoints and check even and
odd horizons. They are provenance, not schema constraints. Preparation and
stored phases do not grow with requested T; code construction grows with
distance. No process address-space cap is imposed. This relies on the known
repeating Surface Code circuit, not inferred periodicity for arbitrary circuits.

The public `memory_dem_chunks_from_decoder_inputs` helper is promoted from
the Surface Code example. It requires round-major detectors, D lookback of at
most one raw round, faults touching at most two adjacent detector rounds,
independent fault IDs, at most 64 observables and a repeating bulk. It retains
transient phases and merges identical H/O mechanisms with independent XOR
probabilities. Unsupported faults fail instead of being dropped or replaced.

At least four extraction cycles are required; shorter shots fail without
rounding. Zero-noise/zero-fault phases await public framework
dimension-preservation support; probability 1 is rejected before circuit
analysis. Generated `beforeN`, `bulk` and `afterN` phase names are preserved,
not hardcoded executor bindings. Multiple logical qubits mean independent
patches, not lattice surgery.

This CLI takes a finite cycle count. An open model alone does not implement an
unknown-duration producer, terminal flush or admission policy.

## Build scope and verification

The executable installs through `qec-tools`, with existing QEC and CUDA-Q
runtime dependencies, and is included in full-QEC release archives. Circuit
characterization requires full QEC, CUDA-Q and Stim, so the target is skipped
in decoders-only builds. Building the server does not acquire those
dependencies. Tool-local interfaces are not new installed SDK APIs.

```sh
python3 -m pip install -r libs/qec/unittests/memory-experiment-generator/requirements.txt
cmake --build build --target memory-experiment-generator test_memory_dem_chunks test_memory_model_provider decoding_server
ctest --test-dir build \
  -R 'MemoryDemChunks|MemoryExperiment|MemoryRecord|memory_experiment_generator_cli' \
  --output-on-failure
```

PyYAML is a black-box test dependency only; the installed C++ tool has no
Python dependency. CTest reports missing PyYAML as a skip with installation
instructions; CI installs it before testing. CLI tests have a five-minute
timeout, with measured runtime recorded separately.

Tests cover bounded/full-circuit agreement, the reference layout, awkward batches,
overflow and invalid inputs, exact round trips, huge horizons, independent
patches, typed decoder arguments, malformed detector maps, default geometry,
shared generator/server configuration lookup, safe publication and rejection of
removed output/replay/bundle flags. Per-graph worker tests check exact policy
round trips, independent patches, unchanged defaults and checked worker totals.
PyMatching tests check explicit merge-policy preservation and, when the public
window model is available, fault estimates and committed effects on the
X/Z reference windows. With the sparse window implementation, they exercise
the framework's preallocated decode workspace and canonical solve/commit
operations. They use the framework's existing models and helpers, not a
tool-owned scheduler.
Configuration tests use the framework's registered public pattern schema.

Windowed-graph wiring validation applies to concrete `ingest` and
`decode_window` nodes. It cannot validate the generated `parallel_window`
pattern's internal wiring until the framework expands that pattern. The
framework also owns the resulting XOR node and its input count; no additional
generator argument is needed.

Actual graph construction, same-plan serial/parallel correctness, seam faults,
sustained overlap, sampled logical accuracy, installed replay and
cores-versus-capacity/tail-latency measurements remain runtime gates. Model and
configuration tests do not establish those results.

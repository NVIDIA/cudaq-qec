# Memory-experiment generator (draft)

This public tool prepares a reusable Surface Code memory model and optional
chronological raw replay inputs. **Executable A/B task-graph YAML generation
is pending the public parallel-window and executor interfaces.** No global or
sliding-window substitute is emitted.

Experiment parameters are command-line inputs. Both the generated model
(`model.yaml`) and its metadata (`manifest.yaml`) use YAML; optional raw
measurement records are plain-text files. No JSON output is generated.

## Build and run

In a full QEC build configured with CUDA-Q, Stim and realtime configuration:

```sh
cmake --build build --target memory-experiment-generator
build/bin/memory-experiment-generator \
  --distance=3 --rounds=26 --basis=Z --logical-qubits=1 \
  --batch-rounds=3 --output=/tmp/memory-d3-r26
build/bin/memory-experiment-generator --validate-bundle=/tmp/memory-d3-r26
```

The output directory must not exist, and its parent must exist. Existing
files, directories and symlinks are never overwritten. Failures return a
diagnostic and nonzero exit code; cleanup removes only the directory created
by that invocation. The manifest is written last as the completion marker.
Do not modify a bundle concurrently with generation or validation.

With `--replay-shots=N`, the tool simulates N noisy experiment shots per
logical qubit and writes their measurement records: extraction-round bits
followed by the final data-qubit readout. These provide repeatable inputs for
offline decoder testing without quantum hardware. No measurements are
simulated by default. Sampling uses the public `sample_memory_circuit_raw`
API with the Stim backend, not a second circuit implementation:

```sh
build/bin/memory-experiment-generator \
  --distance=3 --rounds=26 --basis=X --logical-qubits=2 \
  --batch-rounds=3 --replay-shots=4 --seed=100 \
  --output=/tmp/memory-d3-r26-inputs
```

Experiment inputs are command-line arguments in `--name=value` form; no
YAML or JSON input file is required or accepted. `--help` requires no model
preparation. The current preparation policy decomposes circuit errors into
independent mechanisms.

| Input | Meaning / default |
|---|---|
| `--code` | `surface_code`; other families are rejected in this draft |
| `--distance` | Required odd Surface Code distance, at least 3 |
| `--rounds` | Required extraction-cycle count, excluding data readout |
| `--basis` | Required `X` or `Z` preparation/readout basis |
| `--logical-qubits` | Independent patches sharing one model; default 1 |
| `--error-rate` | CNOT two-qubit depolarization probability; default 0.001, strictly between 0 and 1 |
| `--batch-rounds` | Extraction cycles per regular replay record; default 1 |
| `--replay-shots` | Simulated experiment shots per logical-qubit patch, saved as measurement records for offline testing; default 0 |
| `--seed` | Positive base CUDA-Q sampling seed; default 1 (zero means unseeded in CUDA-Q and is rejected when sampling) |

This is not a general circuit/noise-model parser. Additional providers can
reuse the specification, record layout and bundle interfaces. The tool-local
interfaces are not installed as new SDK APIs.

The executable installs through `qec-tools` and needs the existing QEC and
CUDA-Q runtime dependencies. Circuit characterization needs full QEC, CUDA-Q
and Stim; this target is skipped in decoders-only builds. Pure layout code
does not need a CUDA-Q runtime or GPU. Building the server alone does not
acquire circuit-generation dependencies.

## Bundle contract

`manifest.yaml` is a versioned tool-owned contract (`schema_version: 1`), not
a server configuration. It records experiment/layout/batching metadata,
noise, normalization, QEC build identity and model SHA-256. `model.yaml` uses
the existing typed `dem_chunks` mapping in standalone form, not a fabricated
decoder entry. Phase probabilities survive serialization exactly.

The four mappings remain distinct:

- D / m2d maps chronological measurements to detectors.
- H maps independent faults to detector effects.
- Phase O maps faults to logical flips.
- `final_m2o` maps final data-readout measurements to the raw logical value.

Validation checks hashes, code geometry/readout support, widths, probabilities,
seams, horizons and delivery counts. Seam validation expands only a bounded
phase chain, not the long requested duration. Hashes detect alteration; they
are not signatures or proof of circuit origin. Provenance identifies the
linked QEC build; its repository revision does not describe uncommitted edits.
Version-one manifests are single YAML mapping documents with required fields;
unknown, duplicate or missing fields are rejected. Numeric and boolean scalars
follow the existing QEC YAML parser's conventions. Before publication and
sampling, the prepared model is checked
against its preparation identity, including code, basis, duration, noise and
normalization. Independent patch counts may reuse the same model. This guards
against accidental metadata/model mismatches; it is not authentication of an
externally supplied bundle.

Optional `raw-PATCH-SHOT.txt` files use the existing playback static-source
format: binary regular batches, then one terminal batch. Metadata records
patch, shot, SHA-256, seed and raw logical value. Seeds are assigned as
`base_seed + patch * shots_per_patch + shot`, with overflow checks. Sampling
is serialized around CUDA-Q global random state. Reproducibility is for the
same backend/build, not guaranteed across versions. Validation checks record
lengths/content, terminal presence and raw-logical parity.
The optional `replay` metadata section is omitted when no shots are sampled.

**Raw logical values are not expected corrections or an accuracy oracle.**
The bundle reports `graph_generated: false`, `graph_constructed: false`,
`replayed: false`, and raw inputs `decoded: false`. It contains no server YAML,
playback control schedule or decoder-result oracle. Decoder recipes, A/B
geometry, output composition and core-budget fields belong to the forthcoming
framework adapter, not guessed fields in this model-only format.

## Counts and coordinates

T counts extraction cycles, excluding the distinct terminal readout input
round. There are T+1 raw rounds. Actual D determines detector rows and
lookback; endpoint detector counts, including the basis-dependent readout
layer, need not equal regular detector counts.

Justin's distance-three reference has `26 * 8 + 9 = 217` measurements.
Three-cycle batching yields eight 24-bit batches and a terminal batch of
`2 * 8 + 9 = 25` bits. With its D mapping, B0 needs raw rounds `[2,12)`
(80 bits) and A2 needs `[20,27)` (57 bits). Its A/B commits tile the detector
horizon exactly; it does not exercise a terminal commit remainder or prove
parallel execution.

Record arithmetic also handles unaligned lengths, exact batches and batches
larger than the shot. Packet granularity never rounds duration or changes the
model. Provider minimum-length restrictions are checked separately. Multiple
logical qubits are independent patches, not lattice surgery, and do not imply
an execution pool per patch.

## Bounded preparation and supported scope

The current provider characterizes 12 and 13 extraction cycles through
`decoder_context_from_memory_circuit`, comparing prefix, bulk, suffix,
probabilities, measurement maps and raw-logical support before retargeting.
These bounded depths leave margin beyond the transient endpoints and include
both even and odd horizons. They are preparation-policy choices recorded as
provenance, not schema-version constraints: version-one bundles accept other
distinct, representable depths with enough rounds to establish a repeating bulk.
Stored phases and model preparation do not grow with requested T. This uses
the known repeating public Surface Code circuit; comparing two depths is not
a proof of periodicity for arbitrary circuits.

The public `memory_dem_chunks_from_decoder_inputs` helper is promoted from
the Surface Code example. Its documented restrictions are round-major
detectors; D lookback of at most one raw round; faults touching at most two
adjacent detector rounds; independent fault IDs; at most 64 observables; and
a verified repeating bulk. Prefix/suffix transients are retained. Identical
H/O mechanisms merge with independent XOR probabilities. Unsupported faults
fail rather than being dropped or replaced by fictitious faults.

The current provider requires at least four extraction cycles. Shorter shots
fail explicitly, without rounding. Zero-noise/zero-fault phases await public
framework dimension-preservation support. Probability 1 is rejected before
the current circuit-analysis backend, which cannot handle that boundary.
Dimensions and coordinate arithmetic are checked before allocation; code
preparation still grows with distance. No process address-space cap is imposed.

The promoted splitter reports invalid inputs with `std::invalid_argument` and
unrepresentable dimensions with `std::overflow_error`, rather than the example's
previous catch-all `std::runtime_error`. Generated `beforeN`, `bulk` and `afterN`
phase names describe retained model phases; they are not bindings to future
executor phase names. The framework adapter must resolve that correspondence.

Optional raw sampling executes and retains one whole shot, so its time and
transient memory grow with T. Shots stream to the writer one at a time;
patches/shots grow output size. Preparation time is not decoder tail latency.
Replay-file validation reads fixed 4 KiB chunks and checks framing, SHA-256 and
readout parity in one pass, without retaining a shot or a transport batch.
The manifest and model are still retained; manifest metadata grows with the
number of replay files. No artificial process address-space limit is imposed.

## Verification and remaining integration

```sh
python3 -m pip install -r libs/qec/unittests/memory-experiment-generator/requirements.txt
cmake --build build --target test_memory_dem_chunks test_memory_model_provider
ctest --test-dir build \
  -R 'MemoryDemChunks|MemoryExperiment|MemoryRecord|memory_experiment_generator_cli' \
  --output-on-failure
```

PyYAML is needed only by the black-box Python tests. The installed C++ tool
uses the library's existing LLVM YAML support and has no Python dependency.
If PyYAML is absent, CTest reports the CLI suite as skipped with installation
instructions. CI installs it in the workflow before running tests. The CLI
suite has a five-minute timeout; measured runtime belongs in the qualification
record, not a claim that every supported platform finishes at the same speed.

Tests cover Justin's raw layout, awkward batches, invalid/overflow inputs,
synthetic phase mappings, exact YAML round trips, independent full-circuit
comparisons, billion-round retargeting, multi-patch sampling, deterministic
seeds, artifact corruption and safe output handling. CLI tests cover unknown
and duplicate options, invalid values, mode separation and failure before
output publication. Model comparisons do not
establish A/B-versus-global decoder accuracy.

Remaining integration is A/B YAML emission through the public
node/ownership/coverage and executor interfaces, real construction on the
target install, an agreed correction-versus-logical output contract, a
playback schedule/oracle and installed two-process replay. Same-plan
serial/parallel correctness, seam faults, sustained worker overlap, logical
accuracy and cores-versus-capacity/tail-latency qualification are separate
gates; model/input tests cannot establish those results.

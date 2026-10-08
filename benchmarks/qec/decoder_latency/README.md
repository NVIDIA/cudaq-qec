# QEC decoder latency benchmark

`benchmark-qec-decoder` measures latency, throughput, and logical error rate
for registered CUDA-QX QEC decoders. It generates Stim rotated-surface-code
Z-memory circuits and can sweep code distance, round count, physical noise,
decoder parameters, and concurrent decoder instances.

The benchmark supports two execution modes:

- `batch`: times one `decode()` call per shot.
- `stream`: enqueues detector rounds individually and reports both total
  decoder work and the latency from the final round through observable
  correction readout.

## Build

Configure CUDA-QX from the repository root. The benchmark does not require the
unit-test suite:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAQ_INSTALL_DIR=/usr/local/cudaq \
  -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq \
  -DCUDAQX_ENABLE_LIBS=qec \
  -DCUDAQX_INCLUDE_TESTS=OFF \
  -DCUDAQX_BINDINGS_PYTHON=OFF

cmake --build build --target benchmark-qec-decoder -j$(nproc)
```

The executable is written to:

```text
build/benchmarks/qec/decoder_latency/benchmark-qec-decoder
```

NV-Fusion is enabled by default and requires access to its private dependency.
Pass `-DCUDAQ_QEC_BUILD_FUSION_DECODER=OFF` when it is not needed.

## Example

Run from the build directory:

```bash
./benchmarks/qec/decoder_latency/benchmark-qec-decoder \
  --decoders pymatching,nv-fusion-decoder \
  --distances 5,7,13 \
  --rounds 100,1000,10000 \
  --mode stream \
  --round_interval_us 1 \
  --shots 1000 \
  --csv
```

Use `--help` for all sweep, decoder-parameter, output, and CPU-affinity
options. Repeated `--param KEY=VALUE` arguments are converted according to the
selected decoder's registered parameter schema.

## Workload and sweep behavior

The benchmark generates a Stim `rotated_memory_z` circuit for every requested
distance, round count, and noise value. The noise value is applied to
after-Clifford depolarization, reset flips, measurement flips, and inter-round
data depolarization. Stim uses a fixed random seed so decoders at the same
sweep point receive the same sampled shots.

The sweep is the Cartesian product of:

- `--decoders NAME[,NAME,...]` (default: `pymatching`)
- `--distance N` or `--distances N[,N,...]` (default: 5)
- `--rounds N[,N,...]` (default: the distance)
- `--noise P` or `--noises P[,P,...]` (default: 0.001)
- `--decoder_instances N[,N,...]` (default: 1)

Option names accept either hyphens or underscores.

Distances must be odd and at least 3. Each circuit shot contains one more
detector round than the requested stabilizer-round count because final data
readout contributes a detector round.

`--warmup N` controls untimed shots (default: 20), and `--shots N` controls
timed shots **per decoder instance** (default: 500). Increasing the instance
count therefore increases the total samples and work; it does not divide a
fixed shot pool.

## Decoder options

Use a repeatable `--param KEY=VALUE` to pass decoder-specific settings:

```bash
--param num_threads=8 --param block_leaf_size=50
```

Values are converted using the selected decoder's registered schema. Boolean,
integer, floating-point, and string parameters are supported. A key absent
from a decoder's schema is skipped for that decoder, allowing one command to
compare decoders with different option sets.

The benchmark supplies `H`, `O`, the per-error priors, and (for streaming) the
measurement map together through `decoder_init`. It explicitly requests
observable output by default, so batch results can be compared directly with
the sampled logical outcomes; the presence of `O` does not select that output
basis. `--output errors` requests error-mechanism output instead: batch
`decode()` then returns one entry per `H` column, and the logical error rate is
scored from the observables those columns flip. Streaming reads observable
corrections in either case. The benchmark also supplies temporal detector-round
information to NV-Fusion.

`--source chunks` builds every decoder from the same model as per-round DEM
chunks instead, the form a realtime configuration's `dem_chunks` takes:
`decoder_init::from_dem_chunks` for a decoder that declared it reads chunks
natively (`register_decoder_native_model_source`), such as NV-Fusion, and
`decoder_init::from_dem_chunks_closed` for any other. Each column belongs to
the round of its lowest detector. Columns with the same detectors and
observables within a round are merged, because Stim's decomposition repeats
only every two rounds. The rounds that match the middle round form the
repeating phase. Every round is padded to the bulk's detector count, since
chunks have one width per round and Stim's first and last rounds are
narrower; the padding detectors never fire. NV-Fusion is given no
`detector_round` here, so it builds from the chunks. Use this to measure the
chunk path against the default `--source matrices`.

NV-Fusion's `rolling_error_output` and `rolling_error_margin` are not in its
realtime YAML schema, so the benchmark forwards them to NV-Fusion by name.
Compare rolling error output, a narrower cut margin, and finite error output
with:

```bash
--output errors
--output errors --param rolling_error_margin=1
--output errors --param rolling_error_output=false
```

`--block_leaf_size N` is a convenience option for decoders whose schema
supports that parameter. Omitting it lets the decoder select its own schedule.

When `--param num_threads=0` is used with multiple instances, the benchmark
replaces zero with a per-instance share of the available CPUs. It reserves
CPUs for the instance thread so the decoder's worker pool does not contend
with the thread driving its fork-join work. Explicit nonzero thread counts are
forwarded unchanged.

## What is timed

Select `--mode batch`, `--mode stream`, or `--mode both` (the default).

In batch mode:

- The stopwatch wraps only `decode()`.
- Observable comparison and logical-error bookkeeping are outside the timed
  region.
- Reported p50, p90, and p99 values are per-shot decode latencies.

In streaming mode:

- The syndrome is enqueued one detector round at a time.
- Each `enqueue_syndrome()` call is timed separately.
- Per-shot decoder work is the sum of those calls plus correction readout.
- Streaming tail latency covers the final round's enqueue and
  `get_obs_corrections()`. This is the response time after the final syndrome
  round arrives.
- Pacing waits are excluded from decoder-work and tail-latency samples.

`--round_interval_us T` paces detector rounds with a busy-wait. Pacing models a
real-time arrival rate, but it also occupies the instance's CPU between rounds.
Without this option, rounds are enqueued as quickly as possible.

The `rounds/s` value uses a separate wall-clock interval around each
instance's complete timed shot loop. It includes pacing, decoder reset, and
bookkeeping, while excluding decoder construction, warmup, and thread
creation. Rates are computed per instance and then summed. Consequently, a
paced run is capped near `1e6 / T` rounds/s per instance and measures whether
the decoders keep up, not their unconstrained peak throughput.

## Concurrent instances and CPU placement

Each value passed to `--decoder_instances` creates that many independent
decoder objects and drives them concurrently. A start gate ensures no
instance begins its timed region before all instances have completed decoder
construction and warmup. Latency samples are pooled across instances, while
throughput is the sum of their independently measured rates.

Each decoder is constructed and driven by the same instance thread. This
preserves decoder implementations that bind their construction thread to a
CUDA device.

Instances are unpinned by default. `--pin_instances` divides the CPUs in the
process's current affinity mask into non-overlapping, equal-sized blocks for
each row. Any remainder is left idle. The instance pins itself before decoder
construction, so worker threads created by the decoder inherit its CPU block.

Manual placement is also available:

- `--instance_core_base N`: first CPU assigned to instance zero.
- `--instance_core_width W`: consecutive CPUs available to each instance.
- `--instance_core_stride S`: distance between instance starting CPUs;
  defaults to the width.

Do not combine automatic and manual placement. Ensure the width can
accommodate the decoder's internal worker count, and inspect the host's CPU,
core, SMT, and NUMA topology before assuming adjacent CPU IDs are independent
physical cores.

## Output

The human-readable report includes decoder, distance, rounds, instances,
shots, noise, latency percentiles, detector rounds per second, and logical
error rate. Streaming output has a separate table for final-round tail
latency. Tables are printed after the requested point list finishes. While the
sweep runs, progress for each point is printed immediately to standard error,
including its ordinal, configuration, and elapsed wall time, so table and CSV
output on standard output stays machine-readable.

`--csv` additionally emits rows prefixed with `csv,` for machine parsing. The
CSV columns are:

```text
mode,id,decoder,d,rounds,det_rounds,instances,shots,total_shots,noise,
latency_min,p50,p90,p99,latency_max,tail_min,tail_p50,tail_p99,tail_max,
rounds_s,ler
```

Percentile resolution is limited by the sample count. With 1,000 timed shots
per instance, p99 is approximately the tenth-slowest sample; use more shots
or repeated runs for stable extreme-tail claims.

## Supported decoders

The benchmark accepts any name resolved by the C++ decoder registry, but a
registered decoder is not necessarily compatible with the benchmark's current
surface-code matrix input and flat parameter interface.

The following decoders are supported directly:

- `pymatching`: the default CPU comparison decoder.
- `nv-fusion-decoder`: supports batch and native real-time streaming
  measurements. The CMake option `CUDAQ_QEC_BUILD_FUSION_DECODER` must be
  enabled.

The CMake target explicitly builds these decoder plugins when they are
available. Additional registered plugins must be built separately or added as
target dependencies before the benchmark can discover them.

## Decoders requiring additional work

The following registered decoders are not currently supported as turnkey
benchmark selections:

- `chromobius`: requires a color-code detector error model. Supporting it
  requires a color-code workload and a benchmark construction path that passes
  a serialized DEM instead of the generated surface-code parity-check matrix.
- `trt_decoder`: requires an ONNX model or TensorRT engine whose input and
  output dimensions match the workload. It also needs an explicit CMake target
  dependency and a documented model-specific benchmark configuration.
- `sliding_window`: requires an inner decoder and nested
  `inner_decoder_params`. The current `--param KEY=VALUE` interface cannot
  construct that discriminated nested parameter map; add structured
  configuration-file or nested-parameter support.
- `single_error_lut` and `multi_error_lut`: registry construction is possible,
  but lookup-table growth makes the generated sweep impractical beyond very
  small codes. Supporting meaningful comparisons requires constrained
  distance/round validation and decoder-appropriate defaults.
- `single_error_lut_example`: this is an example plugin rather than a
  production performance target. It would need an explicit build dependency
  and the same small-workload restrictions as the LUT decoders.

When adding another decoder, verify all of the following:

1. Its plugin target is built before `benchmark-qec-decoder`.
2. It accepts the generated `H`, `O`, and error-rate vector, or the benchmark
   provides the decoder's required DEM/model input.
3. Required parameters can be represented by the CLI.
4. Its result represents the same observables used for logical-error checks.
5. Streaming mode either has a meaningful real-time implementation or is
   clearly documented as using the base-class batch-at-boundary behavior.

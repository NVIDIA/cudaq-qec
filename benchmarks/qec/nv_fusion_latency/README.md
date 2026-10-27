# NV-Fusion latency study

This directory contains the plotting script for the NV-Fusion performance page
at `docs/sphinx/performance/nv_fusion_latency_user_guide.rst`.

The published measurements were collected on an NVIDIA Vera CPU system with
`benchmark-qec-decoder`.
Commit `3539fecbb1c6e0ca4331b2980a4312d24e54d8b1` is the benchmark-source
revision used to interpret and plot the supplied logs; the logs themselves do
not contain a source revision or complete software manifest. The benchmark
generates rotated-surface-code Z-memory circuits with Stim and writes
machine-readable rows prefixed by `csv,`.

## Collect results

Build the benchmark as described in
`benchmarks/qec/decoder_latency/README.md`, then run from the build directory:

```bash
mkdir -p benchmark-logs

for d in 5 7 13; do
  numactl --cpunodebind=0 --membind=0 \
    ./benchmarks/qec/decoder_latency/benchmark-qec-decoder \
      --decoders pymatching,nv-fusion-decoder \
      --decoder_instances 1,2,4,8 \
      --pin-instances \
      --distances "$d" \
      --rounds 10,100,500,1000,2000,5000,10000 \
      --param num_threads=0 \
      --mode stream \
      --round_interval_us 1 \
      --shots 1000 \
      --csv 2>&1 |
    tee "benchmark-logs/log-py-nvfusion-stream-d${d}.txt"
done
```

For a machine with fewer available cores, reduce the streaming instance list.
The published Vera data contains one, two, four, and eight instances for all
three distances.

## Generate figures

From the repository root:

```bash
python benchmarks/qec/nv_fusion_latency/plot_results.py \
  /path/to/benchmark-logs assets/docs
```

The script generates the three PNG files referenced by the performance page,
including the conceptual fusion-block diagram.
It reads CSV records from both per-run `.txt` logs and consolidated
`log*.out` files; consolidated output supplements records missing from the
per-run logs.

# Tesseract long beam vs TCBP-JOBS on the Tesseract SI1000 benchmark

One figure: logical error rate per round against decoding time per round for the circuits of the
Tesseract benchmark at `p = 0.002` (Z basis), with the unsparsified long-beam Tesseract results from the
tesseract-decoder repository on one side and `nv-qldpc-decoder` TCBP-JOBS on the other, decoding the same number of shots per
circuit. The LER on both sides and the compute panel come from throughput-style runs; the x position of each
TCBP-JOBS point is its serial latency (one shot per `decode()` call), while Tesseract's x position is its
per-thread time from the multi-threaded runs recorded in that repository. The footnote inside the figure states this.

## Steps

```
python3 tradeoff.py fetch                 # circuits + aggregated_results.jsonl from quantumlib/tesseract-decoder
python3 tradeoff.py run --gpus 0,1,3      # TCBP-JOBS batched on the 14 circuits; ~20 min on three GB200 GPUs
python3 tradeoff.py serial --gpu 1        # serial decode() latency, 4,096 shots per circuit; a few minutes, idle machine
python3 tradeoff.py plot                  # plots/tradeoff_longbeam_vs_tcbp_jobs_p0.002_Z.{pdf,png,csv}
```

`fetch` downloads the 14 `.stim` files and the results file, and checks each circuit's
sha256 against the value recorded in the repository's rows. `run` spreads the circuits over the given GPUs by
expected cost, writes one JSON record and a per-shot `.npz` per circuit under `results/`, and can be
restarted (finished circuits are skipped). `serial` adds a `serial` block to each record; if it has not been
run, `plot` falls back to the batched time for x and says so in the footnote. `--basis X` and `--p` select other rows of the sweep. With a
decoders-only build that has no installed `cudaq_qec` package, add `--build <build dir> --src <checkout>`
to `run`.

The 14 records behind the published figure are tracked in `results/*.json`. With them, `fetch` followed by
`plot` regenerates the figure without a GPU. `run` and `serial` skip every circuit that already has a
record: delete `results/*.json` to repeat the batched run, or pass `--force` to `serial` to repeat only the
latency measurement.

## What is compared

| | Tesseract | TCBP-JOBS |
|---|---|---|
| source | `benchmarking/sparsify_errors/aggregated_results.jsonl`, rows with `sparsify_errors=false` (quantumlib/tesseract-decoder PR 254) | this script |
| settings | `--beam 20 --beam-climbing --num-det-orders 21 --pqlimit 1000000 --no-revisit-dets --det-order-index`, no sparsification | preset `tcbp-jobs-sc` (CUDA-Q QEC 0.9.0): sparse sum-product BP, 10 iterations, clip 15, fp32; JOBS with the TCBP ordering, 16 draws, pad 8, beam 32/40/8; every shot searched, including BP-converged ones |
| error model | circuit DEM (Tesseract's own construction) | circuit DEM, `flatten_loops=True, approximate_disjoint_errors=True`, undecomposed, one column per error instruction; DEM rates as priors |
| TCBP model | n/a | `init_tcbp_model(H, seed=1)`, fitted once per circuit |
| shots | `num_shots` of the row | the same number, sampled here with Stim (seed 2026): same count and distribution, not the same shots |
| failure | any wrong observable, or low confidence | any wrong observable (invalid corrections counted separately; none occur) |
| time per shot (x axis) | `total_time_seconds / num_shots`: Tesseract's per-shot decode timer summed over worker threads; CPU model not recorded | mean of `decode()` on 4,096 shots of a second seeded stream, one shot per call, `bp_batch_size=1`, one GPU, machine otherwise idle (`serial` step) |
| GPU-hours (panel) | n/a | wall time of `decode_batch` calls of 16,384 shots on one GPU, summed over the full shot count; excludes sampling, DEM construction and model fitting |

The plotted y value is the shot failure probability divided by the number of rounds (= distance), with the
95% Clopper-Pearson interval, as in the upstream `plot.py`; the x value is time per shot divided by rounds.
Each arrow label gives the per-shot time ratio and an approximate 95% interval on the LER ratio. The panel
in the upper right sums the compute behind the points: Tesseract CPU core-hours from the rows, TCBP-JOBS
GPU-hours from the decode wall time.

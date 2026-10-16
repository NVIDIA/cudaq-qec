Reproduces the data and figures in the min-LLR OSD user guide
(`docs/sphinx/performance/nv_qldpc_minllr_osd_user_guide.rst`): logical error
rate versus physical error rate for BP+OSD with the default final-LLR OSD column
ordering versus `osd_init_method="min_llr"`, on bivariate-bicycle codes.

**Circuits.** Two families are used:

- `assets/benchmarks/bb{72,144}_memory_Z.stim` -- "split X/Z" circuits: only
  the detectors of the basis being decoded are kept, from Relay-BP (<https://github.com/trmue/relay>,
  Apache-2.0). Every noise argument in these files is the literal `(0.002)`;
  `run_sweep.py` rescales to other `p` by substituting that literal.
- Relay-BP's own `tests/testdata/bicycle_bivariate/*.stim` -- "joint XYZ"
  `memory_Z` circuits with both X and Z check detectors, one file per error rate. Clone the Relay-BP repo and pass the per-rate glob (with
  `{p}` in place of the rate) as `--circuit`.

**Requirements:** a GPU; `cudaq-qec` with the `nv-qldpc-decoder` plugin
(`osd_init_method` needs CUDA-Q QEC >= 0.8), `stim`, `numpy`, `matplotlib`.

**Usage** (one process per DEM; all arms decode the same sampled syndromes):
```
export PYTHONPATH=<cuda-qx build>/python:/usr/local/cudaq
R=<relay checkout>/tests/testdata/bicycle_bivariate
CUDA_VISIBLE_DEVICES=0 python3 -u run_sweep.py --label bb72_fullZ \
  --circuit "$R/circuit=bicycle_bivariate_72_12_6_memory_Z,*error_rate={p},*" \
  --rates 0.001,0.0015,0.002,0.003,0.004,0.005 --max-shots 200000 --out report_data
CUDA_VISIBLE_DEVICES=0 python3 -u run_sweep.py --label bb72_Zonly \
  --circuit ../../../assets/benchmarks/bb72_memory_Z.stim \
  --rates 0.001,0.0015,0.002,0.003,0.004,0.005 --max-shots 200000 --out report_data
python3 plot_sweep.py report_data
```

**Arms** -- the decoder configurations under test (`--arms`, default all six; `bp_method` is the default sum-product,
`use_sparsity=True`, `bp_batch_size=2048`, priors from the DEM, `O=L`, and
explicit `output="observables"`):

| Arm | `max_iterations` | `osd_method` | `osd_order` | `osd_init_method` |
|---|---|---|---|---|
| `BP60+OSD-CS10` | 60 | 3 (combination sweep) | 10 | `final_llr` |
| `BP10-minLLR+OSD-CS10` | 10 | 3 | 10 | `min_llr` |
| `BP60+OSD-0` | 60 | 1 (OSD-0) | -- | `final_llr` |
| `BP10-minLLR+OSD-0` | 10 | 1 | -- | `min_llr` |
| `BP10+OSD-CS10` (control) | 10 | 3 | 10 | `final_llr` |
| `BP60-minLLR+OSD-CS10` (control) | 60 | 3 | 10 | `min_llr` |

The two controls change only one factor each relative to the first two arms, so the
effect of `osd_init_method` can be separated from that of `max_iterations`.

**Stopping rule.** Syndromes are drawn in `--batch` chunks (default 10,000) from
one `stim` sampler (`--seed`). An arm stops once it has `--stop-fails` (default
100) logical failures or `--max-shots` shots. Because the baseline arms fail far
more often they stop early; only the `min_llr` arms run to the shot cap at low
`p`. Points with zero failures are plotted as their 95% Wilson upper bound.

**Outputs** in `--out`: `<label>.jsonl` (one record per arm and rate: shots,
fails, LER, BP-converged count, decode seconds, DEM sizes) and
`<label>_p<p>_<arm>_seed<seed>.npz` (per-shot `fail` and `bp_converged` flags).
Running the same label again with a different `--seed` appends an independent
sample; `plot_sweep.py` pools all records for the same (label, arm, p). The shipped
`report_data` does this twice: the `[[72,12,6]]` DEMs were re-run with
`--seed 24680 --stop-fails 1000` (cheap, tightens every point), and the
`[[144,12,12]]` joint-XYZ `BP10-minLLR+OSD-CS10` and `BP60-minLLR+OSD-CS10` points at
`p = 0.002` were extended with `--seed 67890` and `--seed 13579`, `--max-shots 1000000`
each.
`plot_sweep.py` prints the table (fails/shots, LER, Wilson 95% interval, ratio to
the `min_llr` CS10 arm) and writes `report_data/figures/minllr_osd_joint_vs_split.png` (the
main figure of the guide: joint XYZ vs split X/Z DEM of the same memory-Z circuit,
OSD-CS10 arms, one column per code), `minllr_osd_controls.png` (the matched-iteration
controls, one panel per DEM), `minllr_osd_ler.png` (one panel per DEM, OSD-CS10 arms) and
`minllr_osd_0_ler.png` (the OSD-0 arms); the latter two are kept here for reference and
are not shown on the docs page.

Reproduces the data and figures in the gamma-ensemble Relay-BP user guide
(`docs/sphinx/examples_rst/qec/nv_qldpc_gamma_ensemble_user_guide.rst`).

The Z-only stim circuits (`assets/benchmarks/`) come from Relay-BP
(<https://github.com/trmue/relay>, Apache-2.0); detectors insensitive to
Z-stabilizer errors have been removed. They are tracked via Git LFS -- run
`git lfs pull` if they appear as pointer stubs.

**Requirements:** a GPU; `cudaq-qec`, the `nv-qldpc-decoder` plugin
(a build from public source does not include it; ships in the released wheel),
`stim`, `numpy`, `matplotlib`.

**Usage:** pin `run_sweep.py` to an idle GPU and run:
```
export QEC_DATA_ROOT=<output directory>
CUDA_VISIBLE_DEVICES=<idle gpu> python3 -u run_sweep.py
python3 plot_sweep.py
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `QEC_DATA_ROOT` | `report_data` | Output root; both scripts must agree |
| `SHOTS` | `150000` | Shots per configuration (~30 min on a GB200) |
| `CIRCUIT_DIR` | `../../../assets/benchmarks` | Directory of Z-only stim circuits |

`run_sweep.py` writes `$QEC_DATA_ROOT/report_data.npz` (~23 MB at 150,000
shots), holding per-shot latency, iteration count, and logical-error flag for
every code and ensemble size.

`plot_sweep.py` reads `report_data.npz` and prints the numeric values in 
`nv_qldpc_gamma_ensemble_user_guide.rst` as well as writing the four figures under
`$QEC_DATA_ROOT/figures/`:
- `relaybp_gamma_ensemble_perf.png`
- `relaybp_latency_percentiles.png`
- `relaybp_hard_deadline_ler.png`
- `relaybp_ler_multiplier.png`

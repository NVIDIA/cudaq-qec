# Validating a decoder configuration before the server loads it

`validate_decoder_config.py` checks decoding-server YAML files
(`multi_decoder_config` documents) against the JSON Schema that CUDA-Q QEC
exports, on any machine with Python. No GPU, no CUDA-Q and no `cudaq_qec`
install are needed to run it; `cudaq_qec` is needed once, to produce the
schema file.

## Produce the schema (machine with `cudaq_qec`)

```bash
python3 validate_decoder_config.py --export-schema decoder_config.schema.json
```

The schema reflects the decoder plugins loaded in that installation, so
export it from the same CUDA-Q QEC release the server runs; a decoder type
whose plugin was missing at export time is reported as "has no parameter
schema in this schema file" when a configuration uses it with arguments.
`task_graph` entries are removed from the exported schema unless
`--allow-task-graph` is given (a schema exported that way cannot check task
graphs later, and the script says so).

## Validate (any machine)

```bash
pip install pyyaml jsonschema
python3 validate_decoder_config.py --schema decoder_config.schema.json config.yml more.yml
```

```
config.yml: OK
more.yml: FAIL
  decoders/0: Additional properties are not allowed ('D_sparse_matrix' was unexpected)
  decoders/0/decoder_custom_args: Additional properties are not allowed ('merge_stratgey' was unexpected)
```

Exit status is 0 when every file validates, 1 when at least one does not, 2
for a usage or schema error. Each problem names the location in the document
(`decoders/<index>/...`) and the rule it breaks.

## What is and is not checked

Checked: key names and placement, value types, which model form a decoder
entry uses (`H_sparse`/`O_sparse`/`D_sparse`, `dem_chunks`, or
`stim_dem_path`), and the parameter names of each decoder type.

Not checked: cross-field rules (matrix row counts against `syndrome_size`,
seam widths between DEM chunks) and per-decoder validation hooks. Those run
only when the server loads the configuration, so a file that passes here may
still be rejected at load. Task-graph configurations are reported as invalid
unless `--allow-task-graph` is given.

## Numbers

The script reads numbers with YAML 1.2 rules, matching the server's LLVM
YAML parser: `6e-05` and `1e3` are floats. A plain `yaml.safe_load` follows
YAML 1.1 and reads those spellings as strings, which the schema then rejects
as "not of type 'number'". Use the script, or its `Yaml12Loader`, rather than
`yaml.safe_load` when feeding these files to other tooling.
`check-jsonschema` (which parses YAML with ruamel.yaml, a YAML 1.2 parser)
is also fine.

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
`stim_dem_path`), the parameter names of each decoder type, and the
cross-field rules the server applies while parsing: the five flat-form
fields must all be present, `H_sparse` must not be empty and `block_size`
not zero, `H_sparse` and `D_sparse` rows (counted by `-1` terminators) must
equal `syndrome_size`, `H_sparse`/`O_sparse` entries must be below
`block_size`, no `D_sparse` row may be empty, and a form that derives a
field (`dem_chunks`, `stim_dem_path`) rejects that field when it is also
given. For the `dem_chunks` form the shape of `seam`, `connections` and
`phases` (names, `spec` with `num_faults` and matching `error_rates` in
[0, 1], every connection naming a phase that has a spec) and the
`syndrome_mapping` rules (unique ids below 16, `bulk` or `final`) are
checked as the parser checks them.

Not checked: per-decoder validation hooks (for example `sliding_window`'s
`step_size` against `window_size`), DEM-chunk seam consistency and the
`phases_supply_D_sparse` rule, and the `D_sparse` row count of a
`stim_dem_path` decoder. Those run only when the server loads the
configuration, so a file that passes here may still be rejected at load.
Task-graph configurations are reported as invalid unless
`--allow-task-graph` is given.

Known parser-level differences, all recorded as expected failures in the
test suite: PyYAML rejects a tab after `:`, before a `#` comment or inside a
flow collection, an empty flow-sequence entry (`[a,, b]`) and a repeated
`%YAML` directive, all of which LLVM YAML accepts; LLVM YAML misreads a
block scalar with an explicit indentation indicator (`|2`), which PyYAML
parses. The test suite also runs a mutation fuzzer (random edits of valid
documents, both parsers must agree); `CONFIG_VALIDATOR_FUZZ_N` sets how many
documents it tries.

## Scalars: the server's rules, not PyYAML's

The server parses with LLVM YAML, which takes every scalar as raw text and
converts it by the type of the field it lands in. The script does the same:
it loads every scalar as text and applies LLVM's conversion rules while
checking the schema. In practice:

- Quoting does not matter: `'3'` is a valid integer, `'true'` a valid
  boolean, and `yes`, `~`, `123` in a string field are the strings `yes`,
  `~`, `123`.
- Integers: decimal, `0x`/`0X` hex, `0b` binary, `0o` octal (lowercase only),
  a leading `0` means octal (`010` is 8). A `-` sign is accepted where the
  schema allows negatives, `+` never is. No `_`, no exponent, no decimal
  point, at most 64 bits.
- Floats: whatever `strtod` accepts in full: `6e-05`, `1.`, `1.e3`, `.5`,
  hex floats, `inf`, `nan`. Not `1_000.5`, not YAML's `.inf`/`.nan`.
- Booleans: exactly `y yes true on` / `n no false off` in their usual
  capitalisations (`Yes`, `TRUE`, `Off`, ...). Not `1`/`0`, not `t`/`f`.
- A plain empty value (`key:`) or a plain `~`/`null` is an empty sequence
  for a list field and an error for a number or boolean; in a string field
  `~` is the text `~` and an empty value is an error.
- Tags (`!!int`, `!foo`) are ignored; anchors are fine, aliases (`*a`) are
  rejected; duplicate mapping keys are rejected; only the first document of
  a stream is read; an unknown directive (`%FOO`) is an error.

A plain `yaml.safe_load` follows YAML 1.1 instead: `6e-05` becomes a
string, `on` a boolean, `010` an int but `0o10` a string, duplicates are
silently merged. Do not feed these files to the schema through
`yaml.safe_load`. `check-jsonschema` (ruamel.yaml, YAML 1.2) gets the
common numeric spellings right but still differs from the server on
quoting, booleans and duplicates.

The test suite (`libs/qec/python/tests/test_config_validator.py`) runs a
corpus of such documents through both this script and the server's parser
(`multi_decoder_config.from_yaml_str`) and requires the verdicts to agree.

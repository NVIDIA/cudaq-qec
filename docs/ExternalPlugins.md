# External binary plugins

Decoder modules are installed in `<libdir>/decoder-plugins`; task-graph modules
are installed in `<libdir>/task-graph-plugins`. The existing loader registers
decoders before loading task-graph modules. Optional modules do not change the
interfaces or requirements of built-in decoders.

## Import an archive

The binary producer supplies a `.tar.gz` archive and a basename-only
`<archive>.sha256` sidecar. Its schema-1 `manifest.json` lists each module's
provider ID, `decoder` or `task_graph` kind, destination, build requirements and
SHA-256. Native and wheel archives are separate deliveries. Model bundles are
not included in these ELF-only archives; the companion runtime-data delivery
below keeps code and data validation separate.

```bash
cmake -S libs/qec -B build \
  -DQEC_EXTERNAL_PLUGIN_ARCHIVE=/artifacts/plugins.tar.gz \
  -DQEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT=<reviewed-producer-commit> \
  -DQEC_REQUIRED_EXTERNAL_PLUGINS=<required-provider-id>
cmake --build build
cmake --install build --prefix /path/to/install
```

The producer commit is an exact, reviewed 40-character Git commit identifying
the delivered binaries, not an automatic public/private source mapping. Each
task-graph module also declares `framework_abi_version` in its manifest. Decoder
modules use `null` for this task-graph-specific field.

## Task-graph plugin ABI gate

Task-graph modules must implement the C gateway in
`cudaq/qec/task_graph_plugin.h`: `cudaq_qec_task_graph_plugin_describe` and
`cudaq_qec_task_graph_plugin_register`. The description is immutable and reports
the protocol/framework ABI versions, compiled C++ layouts and toolchain ABI,
framework source identity and build identity. Neither entry point may throw.

The loader resolves both entry points from the module itself, checks the
description before invoking registration, and refuses missing/incompatible
modules with a warning. Versioned C++ registration symbols prevent pre-gate
task-graph binaries from resolving the old by-value interface before their
constructors run. Registration from a new module's static initializer is also
refused: there is no task-graph legacy grace path. Initialize registrations only
inside the explicit entry point. Existing ordinary decoder plugins retain their
current interface and load before task-graph modules.

Registration cannot replace an existing kind or alias. Failed registration is
rolled back while module-owned callbacks remain mapped, then the handle is
closed. Successful task-graph modules remain mapped for the process lifetime.
The gate protects compatibility, not execution of untrusted code: native plugins
remain trusted inputs. Bump the framework ABI version and registration symbols
together when the factory, metadata or node C++ interfaces change, and rebuild
all affected modules. These checks run at loading, not in the decode hot path.

`QEC_EXTERNAL_PLUGIN_DELIVERY` defaults to `native`, or `wheel` under
scikit-build. A native build rejects a wheel delivery and vice versa. The target
architecture, CUDA major/minor, required providers and selected producer commit
are checked before extraction. Generic builds need no archive; specifying
required providers without an archive fails rather than silently dropping them.

The importer validates the archive checksum before decompressing it, permits only
the manifest and its named regular ELF modules, and checks module hashes and
architecture. It does not execute the modules. Reconfiguration revalidates an
existing content-addressed staging directory without changing it.

The build patches **copies** of the modules, never the original archive or
validated extraction. Final RPATH includes `$ORIGIN/..` and the manifest's
relative runtime paths, including wheel TensorRT paths when supplied. The
original manifest is installed under
`<libdir>/external-plugins/<archive-sha256>/source-manifest.json`; its module
hashes describe the original exported bytes, not the RPATH-patched install.
Existing `QEC_EXTERNAL_DECODERS` inputs remain supported and are also copied
before patching. If both deliveries contain the same destination and identical
original bytes, the module is installed once with the combined relative RPATHs.
Conflicting binaries for one destination are rejected.

The native and wheel release workflows select the matching
`qec-private-plugins-<platform>_ubuntu24.04_cuda<version>_release[_py<version>].tar.gz`
from their downloaded assets. Set `plugin_framework_commit` to the reviewed
producer commit and `required_plugin_ids` when the delivery must contain a
particular provider. An absent optional archive leaves generic builds unchanged;
a requested provider makes absence an error. The existing decoder-archive path
is retained independently. Producer/provider requirements require a fresh wheel
build; combining them with `artifacts_from_run` is rejected rather than silently
claiming that previously built wheels meet new requirements.

## Validate the installed consumer

An archive-import success is not runtime or model qualification. Check the final
installed files after copying, RPATH patching, stripping and wheel repair:

```bash
patchelf --print-rpath /path/to/install/lib/task-graph-plugins/<module>.so
readelf -d /path/to/install/lib/task-graph-plugins/<module>.so
ldd /path/to/install/lib/task-graph-plugins/<module>.so
c++ -std=c++20 scripts/ci/plugin_registration_probe.cpp \
  -I/path/to/install/include -I/path/to/cuda/include -L/path/to/install/lib \
  -Wl,-rpath,/path/to/install/lib -lcudaq-qec-decoders \
  -o plugin-registration-probe
./plugin-registration-probe <expected-node-kind>
./plugin-registration-probe --decoder <expected-decoder>
./plugin-registration-probe --native-streaming-decoder <expected-decoder>
```

The probe uses public installed headers and checks registration and the parameter
schema without constructing a decoder or GPU engine. The native-streaming check
also requires the decoder's bounded DEM-chunk source declaration, without a
matrix-materialization requirement. Positional node-kind arguments remain
supported; checks can be combined. Also run missing-name negative cases,
ordinary-decoder regressions without external artifacts, and
real configuration/replay tests with the intended runtime libraries and model
bundle. TensorRT's SONAME alone does not establish engine compatibility.

For decoding-server containers, use the native delivery through the release
Dockerfile's artifacts context. Compile the server and any private provider in
builder stages against matching public interfaces, then test the installed
runtime with no access to those source/build trees. A builder-only success is
not validation of the downloaded release image. Pin the image digest and record
the framework/provider commits, CUDA and TensorRT versions, bundle hashes and
replay results before making a support statement.

The decoding-server release workflow extends the existing release image; it
does not define a second server image. Its optional `plugin_framework_commit`,
`required_plugin_ids` and `required_task_graph_kinds` inputs select and require
the native archive in the existing artifacts context. Use `cuda_version=13.0`
for a TensorRT delivery requiring both amd64 and arm64; unsupported required
providers fail rather than disappearing from a release matrix.

## Runtime model and example delivery

The same native release can supply a separate companion archive:

```
qec-runtime-assets-<asset-id>-<platform>_ubuntu24.04_cuda<version>_release.tar.gz
qec-runtime-assets-<asset-id>-<platform>_ubuntu24.04_cuda<version>_release.tar.gz.sha256
```

The schema-1 `manifest.json` declares `asset_id`, `provider_id`, the exact
`provider_commit`, a nullable `framework_abi_version`, `bundle_root`, typed
`files` with SHA-256 and byte counts, and `examples` with config, schedule and
expected-correction paths. `compatibility` declares CPU architecture, CUDA and
TensorRT versions and explicit supported GPU compute capabilities. `model`
records profile, variant, producer commit/version and checkpoint SHA-256. These
are provenance and compatibility constraints, not an approval or qualification
label. Each archive supplies one model profile.

`scripts/ci/runtime_assets.py` is the shared standard-library-only packer and
validator. A private producer first validates its own model format, then passes
an explicit file list to this helper. The archive permits only named engine,
binary-constant, JSON, YAML, schedule, text-record and expected-CSV data. It
rejects source, ONNX, checkpoints, executables, symlinks, special files,
traversal, undeclared members, extended tar headers and hash mismatches. Archive
and decompressed size bounds apply before extraction. No payload is executed
during validation, and neither the archive nor an existing install is replaced.

```bash
python3 scripts/ci/runtime_assets.py verify \
  --archive=/artifacts/runtime-data.tar.gz \
  --architecture=x86_64 --cuda-version=13.0 --tensorrt-version=10.14.1.48 \
  --require-provider=<provider-id> --provider-commit=<reviewed-producer-commit> \
  --require-asset=<asset-id> --gpu-compute-capability=12.0
```

The importer compares CUDA major/minor and the exact full TensorRT version,
including its build component; the manifest retains complete producer versions.
GPU capability support is an
explicit model property, not inferred from the CPU architecture or TensorRT
SONAME. The final image validation stage rehashes the copied data and compares
the actual runtime's architecture and TensorRT package version. A real GPU
example run on each intended target remains a separate gate.

For the decoding-server workflow, `delivery_profile=generic` remains the
default and needs no runtime archive. An explicit `runtime_asset_id` requires
its matching per-platform archive. `delivery_profile=ising` additionally
requires `cuda_version=13.0`, an asset ID and the reviewed ordinary provider
`cudaq-qec-ising-decoder`. The installed no-GPU check requires decoder name
`ising`, its parameter schema and native DEM-chunk streaming declaration, not a
private task-graph kind. Its module and runtime-data manifests use a null
task-graph ABI. Missing binary, model or example
artifacts fail the build; the profile must not silently fall back to a generic
image. The current matrix includes amd64 and arm64, so an amd64-only archive
does not establish an arm64 delivery.

Direct Docker builds use `QEC_DELIVERY_PROFILE`, `QEC_RUNTIME_ASSET_ID` and
`QEC_RUNTIME_ASSET_ARCHIVE` alongside the existing plugin arguments. The archive
and checksum enter through the same read-only `artifacts` context, but are
validated by their separate data contract. Data installs under
`/opt/nvidia/cudaq-qec/share/cudaq-qec/runtime-assets/<asset-id>/`. Example configs
use relative bundle paths, so the image-local shell runner changes to this
directory before invoking the installed server and playback client. Run
`bash /root/examples/run-ising-example.sh x-global` (or `z-global`,
`x-partitioned`, `z-partitioned`) in a compatible GPU-enabled Ising image.
No private source, training framework or external fixture generation is needed.

The `runtime` image target depends on a separate `validation` stage that checks
the copied install without a GPU: manifests, RPATHs, dependencies,
positive/negative requested decoder or node-kind registration and eight ordinary PyMatching shots
over UDP. Validation failure prevents publication. Python, patchelf, the probe,
fixtures and reports stay in that stage, not the shipped runtime. CI can collect
`/tmp/qec-plugin-validation/results` from the `validation` target. This packaging
check does not replace model or GPU qualification.

Optimized Ising examples select the ordinary `ising` decoder directly or use the
built-in `streaming` node with `inner.type: ising`. No private Ising task-graph
module is required. The generic task-graph ABI gate above remains applicable to
other optional task-graph providers; it is not an Ising delivery requirement.

## Send requests through the installed server

Registration without a GPU is a packaging check, not an end-to-end decode.
Run the installed server and playback client with external configuration,
schedule, measurement fixtures and a completed-shot correction oracle:

```bash
python3 scripts/ci/replay_decoding_server.py \
  --server=/path/to/install/bin/decoding_server \
  --playback=/path/to/install/bin/playback-emulator \
  --config=/fixtures/server.yaml \
  --schedule=/fixtures/playback.schedule \
  --expected=/fixtures/expected.csv \
  --output=/results/new-replay \
  --transport=/path/to/runtime/lib/libcudaq-realtime-bridge-udp.so \
  --udp-timeout=30s \
  --require-final \
  --server-arg=--slot-size=8192 \
  --server-arg=--num-slots=8 \
  --server-arg=--host-loop=threaded
```

The output directory must be new. `--transport` is optional when the installed
runtime already selects the intended bridge. Use `--require-final` for an
unknown-duration stream; omit it for finite decoders that do not require FINAL.
The oracle CSV requires an `expected` header containing exact correction bit
strings in completed-shot order, and optionally `decoder_id` (default `0`).
This is a generic completed-shot smoke profile: it validates every raw RPC
status, complete reads, exact corrections and the server's dispatched counts.
Cancellation and intentionally failing requests need their own test profile.
The public harness itself generates neither models nor fixtures; an
Ising-enabled release image supplies its separately validated companion data.
Pin compatible framework, provider, runtime and model versions, and expose their installed
dependencies rather than loading libraries from source/build trees. A passing
smoke run alone does not establish performance, logical-error accuracy,
concurrent-request behavior or release qualification.

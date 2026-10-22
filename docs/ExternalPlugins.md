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
not included.

```bash
cmake -S libs/qec -B build \
  -DQEC_EXTERNAL_PLUGIN_ARCHIVE=/artifacts/plugins.tar.gz \
  -DQEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT=<reviewed-producer-commit> \
  -DQEC_REQUIRED_EXTERNAL_PLUGINS=<required-provider-id>
cmake --build build
cmake --install build --prefix /path/to/install
```

The producer commit is an exact, reviewed 40-character Git commit identifying
the delivered binaries. It is **not** an automatic public/private source mapping
or a runtime ABI handshake. Before importing a task-graph binary, confirm that
its build interfaces match the target framework and rebuild it when those
interfaces change. Current modules may register from static initializers during
`dlopen`; a post-load metadata check cannot make an incompatible binary safe.

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
```

The probe uses public installed headers and checks both registration and the
parameter schema without constructing a GPU engine. Also run a missing-kind
negative case, ordinary-decoder regressions without external artifacts, and
real configuration/replay tests with the intended runtime libraries and model
bundle. TensorRT's SONAME alone does not establish engine compatibility.

For decoding-server containers, use the native delivery through the release
Dockerfile's artifacts context. Compile the server and any private provider in
builder stages against matching public interfaces, then test the installed
runtime with no access to those source/build trees. A builder-only success is
not validation of the downloaded release image. Pin the image digest and record
the framework/provider commits, CUDA and TensorRT versions, bundle hashes and
replay results before making a support statement.

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
Models and fixtures are not bundled with the public harness. Pin compatible
framework, provider, runtime and model versions, and expose their installed
dependencies rather than loading libraries from source/build trees. A passing
smoke run alone does not establish performance, logical-error accuracy,
concurrent-request behavior or release qualification.

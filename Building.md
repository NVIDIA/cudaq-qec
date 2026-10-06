# Building CUDA-Q QEC from Source

This document is intended for anyone who wants to develop their own
modifications of, or contributions to, this code base. This document may change
over time, so be sure to always refer to the latest version of this document.

Using the latest version of CUDA-Q QEC often requires using a recent version of
CUDA-Q. The instructions below refer to a public dev container that is made
available on this repository. It will always contain a recent version of CUDA-Q
(currently updated approximately weekly).

The instructions below provide a complete set of commands to get you up and
running. There are images available called

- `ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6` for AMD64 platforms with CUDA >= 12.6, < 13
- `ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu13.0` for AMD64 platforms with CUDA >= 13.0
- `ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu12.6` for ARM64 platforms with CUDA >= 12.6, < 13
- `ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu13.0` for ARM64 platforms with CUDA >= 13.0

With the image appropriate for your system, run

```bash
docker pull <image-name>
docker run -it --gpus all --name cudaqx-dev <image-name>
```

If your system does not have local GPUs (eg. a Macbook), omit the `--gpus all`
argument.

Then inside the container...

```bash
# Then inside the container
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
export CUDAQX_INSTALL_PREFIX=~/.cudaqx
cd /workspaces

# Get latest source code
git clone https://github.com/NVIDIA/cudaq-qec.git
cd cudaq-qec
mkdir build && cd build

# Configure your build (adjust as necessary)
cmake -G Ninja -S .. \
  -DCUDAQ_INSTALL_DIR=$CUDAQ_INSTALL_PREFIX \
  -DCMAKE_INSTALL_PREFIX=${CUDAQX_INSTALL_PREFIX} \
  -DCUDAQ_DIR=${CUDAQ_INSTALL_PREFIX}/lib/cmake/cudaq \
  -DCMAKE_BUILD_TYPE=Release

# Install your build
ninja install

# Perform tests just to prove that it is running
export PYTHONPATH=${CUDAQ_INSTALL_PREFIX}:${CUDAQX_INSTALL_PREFIX}
export PATH="${CUDAQ_INSTALL_PREFIX}/bin:${CUDAQX_INSTALL_PREFIX}/bin:${PATH}"
ctest
# Run the python tests
# The --ignore option is to bypass tests that require additional packages not contained in
# the standard docker container
cd ..
python3 -m pytest -v libs/qec/python/tests --ignore libs/qec/python/tests/test_tensor_network_decoder.py
```

Additionally, the following CMake options can be configured:

- `CUDAQX_ENABLE_LIBS`: Specify which libraries to build (`all`, `qec`)
- `CUDAQX_INCLUDE_TESTS`: Enable building of tests
- `CUDAQX_BINDINGS_PYTHON`: Enable Python bindings
- `CUDAQ_QEC_DECODERS_ONLY`: Build only the QEC decoder stack, with no CUDA-Q
  install required (see [Decoders-only Build](#decoders-only-build-no-cuda-q-required))

If you want to change which version of CUDA-Q that CUDA-Q QEC is paired with, you
will need to rebuild CUDA-Q from source. This is achievable by going to the
`/workspaces/cudaq` directory in that image and using the appropriate `git`
commands to switch to whichever version you need. You can then use
[these instructions](https://github.com/NVIDIA/cuda-quantum/blob/main/Building.md)
to re-build CUDA-Q. Alternatively, you can use the `scripts/install_cudaq_with_realtime.sh`
script to install CUDA-Q (from the pinned SHA in `.cudaq_version`) with realtime
support. This script rebuilds CUDA-Q from source using the same recipe as the
CUDA-Q QEC CI.

The above instructions provide a fully open-source way of building and
contributing to CUDA-Q QEC, but it should be noted that while this environment
will have many GPU-accelerated simulators installed in it, it won't contain the
*highest* performing CUDA-Q simulators. See [this note](https://nvidia.github.io/cuda-quantum/latest/using/install/data_center_install.html)
for more details.

## Decoders-only Build (no CUDA-Q required)

If you only need the decoders -- for example to run a decoder as a standalone
process, or to develop a decoder plugin -- you can build the decoder stack on
its own, without a CUDA-Q install. Configure `libs/qec` as the top-level
project with `CUDAQ_QEC_DECODERS_ONLY=ON`:

```bash
cmake -S libs/qec -B build_decoders_only \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAQ_QEC_DECODERS_ONLY=ON \
  -DCUDAQX_QEC_BINDINGS_PYTHON=ON
cmake --build build_decoders_only -j$(nproc)
```

This builds the decoder interface (`libcudaq-qec-decoders.so`), the built-in
decoders, and the decoder plugins. Neither a CUDA-Q install nor LLVM/MLIR is
required; a CUDA toolkit still is, since the decoder API links `cudart`. The
following are skipped because they are the only parts that need CUDA-Q:
`libcudaq-qec.so` (codes, experiments, DEM sampling), cuStabilizer, and the
unit tests. The realtime libraries and the decoding server are built too if a
cudaq-realtime install is found -- see [The decoding server](#the-decoding-server)
below.

The option must be set on a build configured with `libs/qec` as the top-level
directory (`-S libs/qec`); configuring the whole repository with it set is an
error.

With `CUDAQX_QEC_BINDINGS_PYTHON=ON` you also get a decoders-only Python
module, with the bindings under `module.qecrt`:

```bash
PYTHONPATH=build_decoders_only/python python3 -c '
import numpy as np, _qec_decoders_standalone
qec = _qec_decoders_standalone.qecrt
H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
print(qec.get_decoder("pymatching", H).decode([1, 1, 0]).result)
'
```

Note that this module is intentionally not packaged or installed: it is only
importable as the bare `_qec_decoders_standalone` module from
`<build>/python`, not as `cudaq_qec`. Use the normal build above if you want
the installable `cudaq_qec` package.

### The decoding server

The decoding server is driven by **cudaq-realtime**, which is a separate
product from CUDA-Q: `libcudaq-realtime.so` links nothing but libc, and the
server uses none of the CUDA-Q runtime, the simulators, or the CUDA-Q
compiler. So a decoders-only build can produce it too. Point the build at a
cudaq-realtime install:

```bash
cmake -S libs/qec -B build_decoders_only \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAQ_QEC_DECODERS_ONLY=ON \
  -DCUDAQ_REALTIME_ROOT=/path/to/cudaq-realtime
cmake --build build_decoders_only -j$(nproc)
```

This is detected, not requested: when cudaq-realtime is found, the decoding
server and the realtime decoding libraries are added to the build; when it is
not, CMake prints a warning and builds the decoders alone. `CUDAQ_REALTIME_ROOT`
is only a hint -- `CUDAQ_INSTALL_DIR` and `CUDAQ_INSTALL_PREFIX` are searched
too, so a full CUDA-Q install works as the source of cudaq-realtime without
any of the rest of it being used.

The server needs `libcudaq-realtime.so` and the cudaq-realtime headers, and
nothing else from that prefix. It does add one build-time dependency the
decoders alone do not have: LLVM. The realtime decoder config is parsed with
`llvm::yaml` and its JSON schema emitted with `llvm::json`, so `LLVMSupport`
is required -- but still no MLIR and no Clang.

You can confirm the resulting binary is free of CUDA-Q:

```bash
readelf -d build_decoders_only/bin/decoding_server | grep NEEDED
```

The only `libcudaq*` entries should be `libcudaq-realtime.so` and the QEC
libraries built here. To run it:

```bash
cd build_decoders_only/bin
./decoding_server --config=decoding_server_config.yaml --transport=udp --timeout=60
```

It prints `QEC_DECODING_SERVER_READY` with its bound port once the decoders are
constructed and the transport is up.

Two pieces are deliberately left out of a decoders-only build, because both
need the CUDA-Q compiler and runtime:

- The **in-process host_dispatch** path, where CUDA-Q discovers the QEC service
  in its own process rather than talking to an external server. Its shim lives
  in `decoding_server_cqr_device_call_service.cpp` and is compiled only with a
  full CUDA-Q install; the external server reaches the same function table
  through a plain-C accessor.
- The **QPU-side** wrappers in `lib/realtime/quantinuum`, `lib/realtime/simulation`
  and `lib/realtime/simulation-cqr`, which compile with `nvq++`.

Consequently the two-process test (`test_decoding_server`) is a full-build
test: it compiles the caller side, a quantum kernel, with `nvq++`. A
decoders-only build verifies the server by building and starting it.

## Building CUDA-Q QEC Documentation from Source

If you want to build and render our documentation from source, you can do this
with the same environment as above. In particular, after running `ninja install`,
you can run `ninja docs`. This places the documentation into the
`/workspaces/cudaq-qec/build/docs/build/` directory. From there, you can open
the `index.html` file in your browser, or if you are using VSCode or Cursor, you
can simply browse to the `index.html` file in the Explorer panel, right click on
the file, and select "Open with Live Server", and that will open your browser
with the main docs page loaded automatically.

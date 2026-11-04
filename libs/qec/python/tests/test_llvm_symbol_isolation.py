# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for NVIDIA/cudaq-qec#883.

`import cudaq_qec` must not put CUDA-Q's bundled LLVM/MLIR into the
process-wide symbol scope. When it does, any other LLVM user loaded later in
the same process (Numba/llvmlite, Triton) can bind to CUDA-Q's copy and abort.
CUDA-Q keeps those libraries local since cuda-quantum#5804/#5811, and the
realtime decoding libraries that `cudaq_qec` dlopen()s must not undo that.

Each check runs in a fresh interpreter: the pytest process has already
imported cudaq_qec, so its scope says nothing about a clean import.
"""
import os
import subprocess
import sys
import textwrap

import pytest

# Exported by CUDA-Q's bundled LLVM/MLIR (same probes as the upstream
# cuda-quantum regression test). None may resolve through the global scope.
_LLVM_PROBES = (
    "_ZN4llvm2cl6Option11addArgumentEv",
    "_ZTVN4llvm21BranchProbabilityInfo20BasicBlockCallbackVHE",
    "_ZN4mlir11MLIRContextC1ENS0_9ThreadingE",
)

_CHILD = textwrap.dedent(f"""
    import ctypes, os, sys
    import cudaq

    probes = {_LLVM_PROBES!r}
    process = ctypes.CDLL(None)

    def mapped(fragment):
        with open("/proc/self/maps") as maps:
            return any(fragment in line for line in maps)

    def check_scope(stage):
        leaked = [s for s in probes if hasattr(process, s)]
        assert not leaked, f"LLVM/MLIR symbols are process-global after {{stage}}: {{leaked}}"

    # Precondition: CUDA-Q itself keeps LLVM local. A failure here is an
    # upstream regression, not a cudaq_qec one.
    if not mapped("libcudaq-kernel-runtime"):
        print("SKIP: CUDA-Q predates cuda-quantum#5811 (no libcudaq-kernel-runtime)")
        sys.exit(75)
    check_scope("import cudaq")

    # The set-target callback registered at import dlopen()s the realtime
    # decoding library, which carries nvq++-compiled kernels and must resolve
    # them without CUDA-Q being made global.
    import cudaq_qec
    assert mapped("libcudaq-qec-realtime-decoding-simulation"), \\
        "realtime decoding library was not loaded at import"
    check_scope("import cudaq_qec")

    # Every target change unloads and reloads the realtime library.
    for target in ("qpp-cpu", "density-matrix-cpu"):
        cudaq.set_target(target)
        assert mapped("libcudaq-qec-realtime-decoding-simulation"), target
        check_scope(f"cudaq.set_target({{target!r}})")
    print("OK")
""")


@pytest.mark.skipif(sys.platform != "linux",
                    reason="uses /proc/self/maps and ELF scope semantics")
def test_import_does_not_expose_llvm_globally():
    result = subprocess.run([sys.executable, "-c", _CHILD],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env=os.environ.copy())
    if result.returncode == 75:
        # Old CUDA-Q on a developer machine: nothing to verify. In CI the pin
        # includes #5811, so a missing facade is a packaging bug, not a skip.
        assert not os.environ.get("CI"), result.stdout.strip()
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout

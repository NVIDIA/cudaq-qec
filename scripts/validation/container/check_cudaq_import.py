#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ``import cudaq`` for pre-test checks.

If the import fails, we print hints. One CUDA-Q failure mode mentions
``get_gpu_compatibility`` and ``not yet initialized`` (e.g. MLIR ``%system``);
we detect that with two substrings so a small wording change upstream only
requires updating this file, not shell scripts.
"""

from __future__ import annotations

import os
import sys
import traceback

# Both must match for the specialized hint (reduces false positives).
_GPU_COMPAT_MARKERS = ("get_gpu_compatibility", "not yet initialized")


def _err(msg: str) -> None:
    prefix = "::error:: " if os.environ.get("GITHUB_ACTIONS") else "ERROR: "
    print(prefix + msg, file=sys.stderr)


def main() -> int:
    try:
        import cudaq
    except ImportError:
        err = sys.exc_info()[1]
        msg = str(err) if err else ""
        traceback.print_exc()
        _err("import cudaq failed.")
        if all(s in msg for s in _GPU_COMPAT_MARKERS):
            _err(
                "Likely CUDA-Q init-order around get_gpu_compatibility (see %system / "
                "not yet initialized in the traceback).")
            _err("Check: nvidia-smi; driver and CUDA vs your CUDA-Q build. "
                 "NVSwitch hosts: systemctl status nvidia-fabricmanager.")
        else:
            _err(
                "Check: nvidia-smi; driver/CUDA vs CUDA-Q (see traceback). "
                "NVSwitch: systemctl status nvidia-fabricmanager if applicable."
            )
        return 1

    print("OK: cudaq", cudaq.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

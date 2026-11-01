# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if `auditwheel repair` grafted any library into the wheel.

auditwheel copies external dependencies into `<pkg>.libs/` under a
hash-suffixed SONAME (e.g. `libfoo-1a2b3c4d.so`) and rewrites the dependents to
point at that copy. The cudaq-qec wheels ship nothing that way: every runtime
dependency either comes from a sibling wheel (cuda-quantum, cuquantum,
tensorrt, the CUDA runtime) or is linked statically, so any `.libs/` content
means a dependency was not excluded. Two cases are called out specially
because they are actively harmful rather than merely bloat: a grafted copy of
a library the wheel already ships un-hashed (two SONAMEs, two copies of the
global state such as the decoder registry), and any CUDA-Q library.

The fix in every case is an `--exclude` in the `auditwheel repair` call.

Usage: python check_grafted_libs.py <wheel> [<wheel> ...]
"""

import os
import re
import sys
import zipfile

_HASH_SUFFIX = re.compile(r"-[0-9a-f]{8}(?=\.so)")


def check_wheel(path):
    errors = []
    with zipfile.ZipFile(path) as whl:
        names = [n for n in whl.namelist() if ".so" in os.path.basename(n)]
    grafted = [n for n in names if n.split("/")[0].endswith(".libs")]
    shipped = {os.path.basename(n) for n in names if n not in grafted}
    for lib in grafted:
        base = os.path.basename(lib)
        unhashed = _HASH_SUFFIX.sub("", base)
        if unhashed in shipped:
            errors.append(f"{lib} duplicates {unhashed} already in the wheel")
        elif unhashed.startswith("libcudaq"):
            errors.append(f"{lib} is a CUDA-Q library and must not be grafted")
        else:
            errors.append(f"{lib} was grafted; the wheel must not carry a "
                          "`.libs/` directory")
    return errors


def main(wheels):
    failed = False
    for whl in wheels:
        for err in check_wheel(whl):
            print(f"::error::{os.path.basename(whl)}: {err} "
                  "(add an --exclude for it to `auditwheel repair`)")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

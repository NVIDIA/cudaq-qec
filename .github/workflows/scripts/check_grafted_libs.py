# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if `auditwheel repair` grafted a library it should not have.

auditwheel copies external dependencies into `<pkg>.libs/` under a
hash-suffixed SONAME (e.g. `libfoo-1a2b3c4d.so`) and rewrites the dependents to
point at that copy. If the library also ships un-hashed elsewhere in the wheel,
the two copies have different SONAMEs, so both can be loaded into the same
process, each with its own global state (e.g. the decoder registry). CUDA-Q
libraries must never be grafted either; they come from the cuda-quantum wheel.

The fix for either case is an `--exclude` in the `auditwheel repair` call.

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

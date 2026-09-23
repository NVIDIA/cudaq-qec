# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
import importlib
import importlib.machinery
import importlib.util
import pkgutil
import sys
import types

import pytest

import cudaq_qec as qec

_NATIVE_EXT = "_pycudaqx_qec_the_suffix_matters_cudaq_qec"
_PROBE_PKG = "_qec_init_native_import_probe"


class _RaisingLoader:

    def __init__(self, message):
        self.message = message

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(self.message)


class _RaisingFinder:

    def __init__(self, fullname, message):
        self.fullname = fullname
        self.message = message

    def find_spec(self, fullname, path, target=None):
        if fullname != self.fullname:
            return None
        return importlib.machinery.ModuleSpec(fullname,
                                              _RaisingLoader(self.message))


def _load_init_with_native_import_error(message):
    # Load the real package __init__ under a temporary name so the native
    # extension import can fail without replacing the live cudaq_qec module.
    ext_name = f"{_PROBE_PKG}.{_NATIVE_EXT}"
    finder = _RaisingFinder(ext_name, message)
    added = [_PROBE_PKG, f"{_PROBE_PKG}.patch", ext_name]
    sys.meta_path.insert(0, finder)
    try:
        patch_mod = types.ModuleType(f"{_PROBE_PKG}.patch")
        patch_mod.patch = object()
        sys.modules[f"{_PROBE_PKG}.patch"] = patch_mod
        spec = importlib.util.spec_from_file_location(
            _PROBE_PKG, qec.__file__, submodule_search_locations=[])
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PROBE_PKG] = module
        spec.loader.exec_module(module)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in added:
            sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "native_msg, match, wrapped",
    [
        ("libcustabilizer.so: cannot open shared object file", "cuStabilizer",
         True),
        ("libcudart.so.12: cannot open shared object file",
         "nvidia-cuda-runtime", True),
        ("unrelated native loader failure", "unrelated native loader failure",
         False),
    ],
)
def test_native_extension_import_diagnostics(native_msg, match, wrapped):
    # Missing cuStabilizer / CUDA runtime get install guidance; any other
    # ImportError is re-raised unchanged and keeps the original object.
    with pytest.raises(ImportError, match=match) as ei:
        _load_init_with_native_import_error(native_msg)
    if wrapped:
        assert isinstance(ei.value.__cause__, ImportError)
        assert native_msg in str(ei.value.__cause__)
    else:
        assert ei.value.__cause__ is None
        assert str(ei.value) == native_msg


def test_package_import_tolerates_missing_cudaq_and_plugins(monkeypatch):
    # Optional cudaq and plugin imports are swallowed; public helpers such as
    # get_code must still be available after the package finishes importing.
    real_import = builtins.__import__
    real_import_module = importlib.import_module
    real_iter_modules = pkgutil.iter_modules

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cudaq" and level == 0:
            raise ImportError("cudaq is not installed")
        return real_import(name, globals, locals, fromlist, level)

    def fake_iter_modules(path=None, prefix=""):
        if prefix.endswith("decoders."):
            return [pkgutil.ModuleInfo(None, prefix + "missing_decoder", False)]
        if prefix.endswith("codes."):
            return [pkgutil.ModuleInfo(None, prefix + "missing_code", False)]
        return real_iter_modules(path, prefix)

    def fake_import_module(name, package=None):
        if name.endswith(("missing_decoder", "missing_code")):
            raise ImportError(f"cannot import {name}")
        return real_import_module(name, package)

    try:
        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", fake_import)
            mp.setattr(pkgutil, "iter_modules", fake_iter_modules)
            mp.setattr(importlib, "import_module", fake_import_module)
            importlib.reload(qec)
            assert callable(qec.get_code)
            assert qec.get_code("steane") is not None
    finally:
        importlib.reload(qec)


if __name__ == "__main__":
    pytest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudaq_qec.plugins.nv_qldpc._check``: rc -1 is a rejected argument (``ValueError``), any other
non-zero rc a failure while running, e.g. a CUDA 'out of memory' (``RuntimeError``). Plugin-free:
the loader is stubbed."""
import pytest

# Python-only import (numpy): the plugin library is loaded lazily by ``_load()``,
# which the test replaces below, so no plugin library and no GPU are needed.
nv_qldpc = pytest.importorskip("cudaq_qec.plugins.nv_qldpc")


class _StubLib:

    def nvqldpc_last_error(self):
        return b"plugin error message"


def test_check_maps_return_codes(monkeypatch):
    monkeypatch.setattr(nv_qldpc, "_load", lambda: _StubLib())
    nv_qldpc._check(0)
    with pytest.raises(ValueError, match="plugin error message"):
        nv_qldpc._check(-1)
    with pytest.raises(RuntimeError, match="plugin error message"):
        nv_qldpc._check(-2)

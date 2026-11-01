# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cudaq_qec without CUDA-Q.

CUDA-Q is an optional extra of the wheel. These tests run the package in a
subprocess with CUDAQ_QEC_DISABLE_CUDAQ=1, which makes it behave as if the
`cudaq` package were not installed: decoders work, CUDA-Q-only names raise an
ImportError that names the extra to install, and cudaq is never imported.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    import cudaq_qec as qec

    assert not qec.has_cudaq()
    assert "cudaq" not in sys.modules, "cudaq must not be imported"

    # The decoder stack works: a repetition-code parity check matrix through
    # a built-in decoder, plus the DEM / PCM helpers from the core module.
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    dec = qec.get_decoder("single_error_lut", H)
    res = dec.decode(np.array([1, 0], dtype=np.uint8))
    assert list(np.asarray(res.result, dtype=np.uint8)) == [1, 0, 0], res.result
    assert qec.pcm_is_sorted(H)
    dem = qec.dem_from_stim_text("error(0.1) D0 L0\\nerror(0.2) D0 D1")
    assert dem.num_detectors() == 2
    cfg = qec.decoder_config()
    assert qec.registered_decoder_schemas()

    # CUDA-Q-only names point at the extra to install.
    for name in ("get_code", "sample_memory_circuit", "stabilizer_grid",
                 "patch", "dem_sampling"):
        try:
            getattr(qec, name)
        except ImportError as e:
            msg = str(e)
            assert name in msg and "[cudaq]" in msg and "pip install" in msg, msg
        else:
            raise AssertionError(f"{name} did not raise")
    try:
        from cudaq_qec import patch  # noqa: F401
    except ImportError:
        pass
    else:
        raise AssertionError("from-import of patch did not raise")
    # Unknown names are still AttributeErrors, not ImportErrors.
    try:
        qec.definitely_not_a_real_name
    except AttributeError:
        pass
    assert "cudaq" not in sys.modules
    print("OK")
""")


def test_package_works_without_cudaq():
    env = dict(os.environ, CUDAQ_QEC_DISABLE_CUDAQ="1")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT],
                          env=env,
                          capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


if __name__ == "__main__":
    pytest.main([__file__])

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudaq
from ._pycudaqx_qec_the_suffix_matters_cudaq_qec import *


def qec_set_target_callback(target):
    if target.name == "quantinuum":
        qecrt.load_quantinuum_realtime_decoding()
    else:
        qecrt.load_simulation_realtime_decoding()
    qecrt.load_device_kernels()

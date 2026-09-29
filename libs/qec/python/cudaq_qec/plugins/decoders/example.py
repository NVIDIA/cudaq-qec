# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cudaq_qec as qec
import numpy as np


@qec.decoder("example_byod")
class ExampleDecoder:

    def __init__(self, H, **kwargs):
        qec.Decoder.__init__(self, H)
        self.H = H
        if 'weights' in kwargs:
            print(kwargs['weights'])

    def decode(self, syndrome):
        res = qec.DecoderResult()
        res.converged = True
        res.result = np.random.random(len(syndrome)).tolist()
        res.opt_results = None
        return res

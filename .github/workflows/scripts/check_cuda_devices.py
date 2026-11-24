# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail a GPU CI job if its required CUDA devices cannot allocate memory."""

import argparse
import ctypes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("device_count", type=int, choices=(1, 2))
required = parser.parse_args().device_count

cuda = ctypes.CDLL("libcudart.so")
cuda.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cudaSetDevice.argtypes = [ctypes.c_int]
cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda.cudaFree.argtypes = [ctypes.c_void_p]
cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
cuda.cudaGetErrorString.restype = ctypes.c_char_p


def check(status, operation):
    if status:
        raise SystemExit(
            f"{operation}: {cuda.cudaGetErrorString(status).decode()}")


count = ctypes.c_int()
check(cuda.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
if count.value < required:
    raise SystemExit(
        f"Expected at least {required} CUDA devices, found {count.value}")

for device in range(required):
    check(cuda.cudaSetDevice(device), f"cudaSetDevice({device})")
    buffer = ctypes.c_void_p()
    check(cuda.cudaMalloc(ctypes.byref(buffer), 1),
          f"cudaMalloc on device {device}")
    check(cuda.cudaFree(buffer), f"cudaFree on device {device}")
print(f"Verified {required} usable CUDA devices ({count.value} visible)")

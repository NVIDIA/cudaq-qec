/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Decoders-only python module: binds the decoder API from py_decoder.cpp
// without importing cudaq and without the code/experiment/realtime bindings.

#include "py_decoder.h"

#include <nanobind/nanobind.h>

NB_MODULE(_qec_decoders_standalone, mod) {
  mod.doc() = "Decoders-only python bindings for CUDA-Q QEC (no CUDA-Q "
              "install required).";
  cudaq::qec::bindDecoder(mod);
  nanobind::set_leak_warnings(false);
}

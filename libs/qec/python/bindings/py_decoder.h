/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cudaq::qec {
void bindDecoder(nb::module_ &mod);
} // namespace cudaq::qec

/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;
namespace cudaq::qec {
void bindExtendedDem(nb::module_ &mod);
} // namespace cudaq::qec

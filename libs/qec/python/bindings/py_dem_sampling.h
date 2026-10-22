/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cudaq::qec::dem_sampler {
void bindDemSampling(nb::module_ &mod);
} // namespace cudaq::qec::dem_sampler

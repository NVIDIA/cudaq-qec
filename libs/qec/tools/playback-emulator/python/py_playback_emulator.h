/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace cudaq::qec::playback {
void bindPlaybackEmulator(nanobind::module_ &mod);
} // namespace cudaq::qec::playback

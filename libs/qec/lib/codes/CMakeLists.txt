# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cudaqx_add_device_code(cudaq-qec
  SOURCES
    steane_device.cpp
    surface_code_device.cpp
    repetition_device.cpp
)

target_sources(cudaq-qec PRIVATE steane.cpp repetition.cpp surface_code.cpp)

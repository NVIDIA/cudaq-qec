# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cudaqx-core STATIC
  tear_down.cpp
  graph.cpp
  tensor_impls/xtensor_impl.cpp
)

target_include_directories(cudaqx-core
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_CORE_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaqx-core
  PRIVATE
    xtensor
    xtensor-blas
    fmt::fmt-header-only
)

install(DIRECTORY ${CUDAQX_CORE_INCLUDE_DIR}/cuda-qx
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT cudaqx-core
  FILES_MATCHING PATTERN "*.h"
)

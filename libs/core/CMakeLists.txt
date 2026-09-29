# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# We need 3.28 because of the `EXCLUDE_FROM_ALL` in FetchContent_Declare
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Project setup
# ==============================================================================

# Check if core is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cudaqx-core)
  set(CUDAQX_CORE_STANDALONE_BUILD TRUE)
endif()

# The following must go after `project(...)` 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDAQX_CORE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQX_CORE_INCLUDE_DIR ${CUDAQX_CORE_SOURCE_DIR}/include)

# Options
# ==============================================================================

option(CUDAQX_CORE_INCLUDE_TESTS
       "Generate build targets for the CUDA-QX Core unit tests."
       ${CUDAQX_INCLUDE_TESTS})

# External Dependencies 
# ==============================================================================

include(FetchContent)
include(GNUInstallDirs)

FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl
  GIT_TAG 0.7.7
  EXCLUDE_FROM_ALL
)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
  GIT_TAG 0.25.0
  EXCLUDE_FROM_ALL
)

FetchContent_Declare(
  xtensor-blas
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
  GIT_TAG 0.21.0
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(xtl xtensor xtensor-blas)

# Directory setup
# ==============================================================================

add_subdirectory(lib)

if (CUDAQX_CORE_INCLUDE_TESTS)
  add_custom_target(CUDAQXCoreUnitTests)
  add_subdirectory(unittests)
endif()

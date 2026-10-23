# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# External Dependencies
# ==============================================================================

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:--param=evrp-mode=legacy>)
endif()
include(GoogleTest)

# ==============================================================================
add_compile_options(-Wno-attributes)

add_executable(test_chromobius test_chromobius.cpp)
target_link_libraries(test_chromobius PRIVATE GTest::gtest_main cudaq-qec-decoders)
add_dependencies(test_chromobius cudaq-qec-chromobius)
add_dependencies(CUDAQXQECUnitTests test_chromobius)
gtest_discover_tests(test_chromobius)

# Correctness test using reference data copied from upstream Chromobius.
add_executable(test_chromobius_correctness test_chromobius_correctness.cpp)
target_compile_definitions(test_chromobius_correctness PRIVATE
  CHROMOBIUS_REFERENCE_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/../../../test_data/chromobius")
target_link_libraries(test_chromobius_correctness PRIVATE
  GTest::gtest_main cudaq-qec-decoders)
add_dependencies(test_chromobius_correctness cudaq-qec-chromobius)
add_dependencies(CUDAQXQECUnitTests test_chromobius_correctness)
gtest_discover_tests(test_chromobius_correctness)

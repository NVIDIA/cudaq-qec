# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

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

add_executable(test_tear_down_helper test_tear_down_helper.cpp)
target_link_libraries(test_tear_down_helper PRIVATE cudaqx-core)

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE GTest::gtest_main cudaqx-core)
target_compile_definitions(test_core PRIVATE
  CUDAQX_TEAR_DOWN_HELPER="$<TARGET_FILE:test_tear_down_helper>")
add_dependencies(test_core test_tear_down_helper)
add_dependencies(CUDAQXCoreUnitTests test_core test_tear_down_helper)
gtest_discover_tests(test_core)


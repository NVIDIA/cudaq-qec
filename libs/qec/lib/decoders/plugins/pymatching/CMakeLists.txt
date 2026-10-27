# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-pymatching")

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  pymatching
  GIT_REPOSITORY https://github.com/oscarhiggott/PyMatching
  GIT_TAG e27b8c6a5f5ba10fb74d4ebb29822f2df5e12bcd # v2.3.1
  EXCLUDE_FROM_ALL
  # Disable test targets because they cause wheel build issues and GTest issues.
  PATCH_COMMAND sed -i
    -e "/^add_executable(pymatching_tests /i if(FALSE)  # skip tests when used as FetchContent"
    -e "/^target_link_libraries(pymatching_tests /a endif()"
    -e "s/^gtest_discover_tests/#gtest_discover_tests/g"
    CMakeLists.txt
)
FetchContent_MakeAvailable(pymatching)

project(${MODULE_NAME})

# Specify the source file for the plugin
set(PLUGIN_SRC
  pymatching.cpp
)

# Create the shared library
add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

# Don't export any symbols (specifically from static libs like PyMatching)
# to avoid conflicts with other Python packages.
target_link_options(${MODULE_NAME} PRIVATE "-Wl,--exclude-libs,ALL")

# Set the include directories for dependencies
target_include_directories(${MODULE_NAME}
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
)

# Link with required libraries 
target_link_libraries(${MODULE_NAME}
  PRIVATE
    cudaq-qec-decoders
    libpymatching
)

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
)

# RPATH configuration 
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib"
  )
endif()

# Install
# ==============================================================================

install(TARGETS ${MODULE_NAME}
  COMPONENT qec-lib-plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
)

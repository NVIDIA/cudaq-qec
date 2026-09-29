# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-chromobius")

# External Dependencies
# ==============================================================================

include(FetchContent)

if(NOT TARGET libpymatching)
  message(FATAL_ERROR
    "Chromobius decoder plugin requires the libpymatching target. "
    "Ensure decoders/plugins/pymatching is added before chromobius.")
endif()

if(NOT TARGET libstim)
  message(FATAL_ERROR
    "Chromobius decoder plugin requires the libstim target provided by PyMatching.")
endif()

FetchContent_Declare(
  chromobius
  GIT_REPOSITORY https://github.com/quantumlib/chromobius.git
  GIT_TAG efae0325e4a8fc13be981434d38e499aa06b989f # v1.1.1
  EXCLUDE_FROM_ALL  # don't add chromobius's own executables/tests to ALL
)
FetchContent_MakeAvailable(chromobius)

project(${MODULE_NAME})

# Specify the source file for the plugin
set(PLUGIN_SRC
  chromobius.cpp
)

# Create the shared library
add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

# Don't export any symbols (specifically from static libs like Chromobius,
# Stim, and PyMatching) to avoid conflicts with other Python packages.
target_link_options(${MODULE_NAME} PRIVATE "-Wl,--exclude-libs,ALL")

# Set the include directories for dependencies
target_include_directories(${MODULE_NAME}
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${chromobius_SOURCE_DIR}/src
)

# Link with required libraries
target_link_libraries(${MODULE_NAME}
  PRIVATE
    cudaq-qec-decoders
    libchromobius
    libstim
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

# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-example")

project(${MODULE_NAME})

# Specify the source file for the plugin
set(PLUGIN_SRC
  single_error_lut_example.cpp
  # single_error_lut_example2.cpp // add other decoder source files here
)

# Create the shared library
add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

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

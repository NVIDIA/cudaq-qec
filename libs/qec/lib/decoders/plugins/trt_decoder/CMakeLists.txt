# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-trt-decoder")

project(${MODULE_NAME})

# Set default TensorRT root if not provided
if(DEFINED TENSORRT_ROOT)
    message(STATUS "Using TENSORRT_ROOT: ${TENSORRT_ROOT}")
endif()

# Check for TensorRT availability first
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
        ${TENSORRT_ROOT}/include
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu/
        /usr/local/cuda/include
        /usr/local/tensorrt/include
        /opt/tensorrt/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIBRARY nvinfer
    PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    NO_DEFAULT_PATH
)

find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    NO_DEFAULT_PATH
)

# Resolve the tri-state `CUDAQ_QEC_BUILD_TRT_DECODER` option (AUTO / ON / OFF)
# against the dependency-detection results above:
#   AUTO -> enable the plugin iff TensorRT was found (silent skip otherwise).
#   ON   -> enable the plugin and hard-fail if TensorRT was not found.
#   OFF  -> the parent CMakeLists.txt does not descend here, but handle it
#           defensively in case this subdirectory is added directly.
# The outcome is published via the `CUDAQ_QEC_TRT_DECODER_ENABLED` cache
# variable so that other parts of the build (e.g. the unit-tests) can react.
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_LIBRARY)
    set(_trt_found TRUE)
else()
    set(_trt_found FALSE)
endif()

if(CUDAQ_QEC_BUILD_TRT_DECODER STREQUAL "AUTO")
    set(_trt_enabled ${_trt_found})
    if(NOT _trt_found)
        message(STATUS
            "TensorRT not found; auto-disabling the ${MODULE_NAME} plugin.\n"
            "  TENSORRT_INCLUDE_DIR : ${TENSORRT_INCLUDE_DIR}\n"
            "  TENSORRT_LIBRARY     : ${TENSORRT_LIBRARY}\n"
            "  TENSORRT_ONNX_LIBRARY: ${TENSORRT_ONNX_LIBRARY}\n"
            "Set -DTENSORRT_ROOT=<path> to enable it, or pass "
            "-DCUDAQ_QEC_BUILD_TRT_DECODER=OFF to silence this message.")
    endif()
elseif(CUDAQ_QEC_BUILD_TRT_DECODER STREQUAL "ON")
    if(NOT _trt_found)
        message(FATAL_ERROR
            "CUDAQ_QEC_BUILD_TRT_DECODER=ON but TensorRT could not be found.\n"
            "  TENSORRT_INCLUDE_DIR : ${TENSORRT_INCLUDE_DIR}\n"
            "  TENSORRT_LIBRARY     : ${TENSORRT_LIBRARY}\n"
            "  TENSORRT_ONNX_LIBRARY: ${TENSORRT_ONNX_LIBRARY}\n"
            "Set -DTENSORRT_ROOT=<path> to point at a TensorRT install, "
            "or pass -DCUDAQ_QEC_BUILD_TRT_DECODER=AUTO to fall back to "
            "auto-detection (default), or =OFF to disable the plugin.")
    endif()
    set(_trt_enabled TRUE)
elseif(CUDAQ_QEC_BUILD_TRT_DECODER STREQUAL "OFF")
    set(_trt_enabled FALSE)
else()
    message(FATAL_ERROR
        "Invalid value for CUDAQ_QEC_BUILD_TRT_DECODER: "
        "'${CUDAQ_QEC_BUILD_TRT_DECODER}'. Expected one of: AUTO, ON, OFF.")
endif()

set(CUDAQ_QEC_TRT_DECODER_ENABLED ${_trt_enabled} CACHE INTERNAL
    "Whether the TRT decoder plugin is being built")

if(CUDAQ_QEC_TRT_DECODER_ENABLED)
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT ONNX parser: ${TENSORRT_ONNX_LIBRARY}")

    # Specify the source file for the plugin
    set(PLUGIN_SRC
      trt_decoder.cpp
      # Add additional source files here as needed
    )

    find_package(CUDAToolkit REQUIRED)

    # Create the shared library
    add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

    # Set compile definitions
    target_compile_definitions(${MODULE_NAME} PRIVATE TENSORRT_AVAILABLE)

    # Set the include directories for dependencies
    target_include_directories(${MODULE_NAME}
      PUBLIC
        ${CMAKE_SOURCE_DIR}/libs/qec/include
        ${CMAKE_SOURCE_DIR}/libs/core/include
      PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # Link with required libraries 
    target_link_libraries(${MODULE_NAME}
      PRIVATE
        cudaq-qec-decoders
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_LIBRARY}
        CUDA::cudart
    )

    set_target_properties(${MODULE_NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
    )

    # RPATH configuration 
    # ==============================================================================

    if (NOT SKBUILD)
      if(DEFINED TENSORRT_ROOT)
        set(TENSORRT_BUILD_RPATH   "$ORIGIN:${TENSORRT_ROOT}/lib")
        set(TENSORRT_INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:${TENSORRT_ROOT}/lib")
      else()
        set(TENSORRT_BUILD_RPATH   "$ORIGIN")
        set(TENSORRT_INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../../tensorrt_libs")
      endif()
      set_target_properties(${MODULE_NAME} PROPERTIES
        BUILD_RPATH "${TENSORRT_BUILD_RPATH}"
        INSTALL_RPATH "${TENSORRT_INSTALL_RPATH}"
      )

      # Let CMake automatically add paths of linked libraries to the RPATH:
      set_target_properties(${MODULE_NAME} PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE)
    else()
      # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
      # Thus, we need the $ORIGIN/../lib
      set_target_properties(${MODULE_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../../tensorrt_libs"
      )
    endif()

    # Install
    # ==============================================================================

    install(TARGETS ${MODULE_NAME}
      COMPONENT qec-lib-plugins
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
    )
endif()




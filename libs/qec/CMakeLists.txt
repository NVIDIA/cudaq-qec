# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Requiring the same version as the others.
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Set a default build type if none was specified. Must set this before
# project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")

# Set a default install prefix if none was specified.
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaqx" CACHE STRING
    "Install path prefix, prepended onto install directories")

# Project setup
# ==============================================================================

# Check if core is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cudaqx-qec)
  set(CUDAQX_QEC_STANDALONE_BUILD TRUE)

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Add our Modules to the path
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

  # Include custom CUDA-QX modules
  include(CUDA-QX)

  # Helper target to collect python modules
  add_custom_target(cudaqx-pymodules)
endif()

# The following must go after `project(...)` 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDAQX_QEC_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQX_QEC_INCLUDE_DIR ${CUDAQX_QEC_SOURCE_DIR}/include)

# Options
# ==============================================================================

option(CUDAQX_QEC_INCLUDE_TESTS
       "Generate build targets for the CUDA-QX QEC unit tests."
       ${CUDAQX_INCLUDE_TESTS})

option(CUDAQX_QEC_BINDINGS_PYTHON
       "Generate build targets for python bindings."
       ${CUDAQX_BINDINGS_PYTHON})

option(CUDAQX_QEC_INSTALL_PYTHON
       "Install python files alongside the library."
       ${CUDAQX_INSTALL_PYTHON})

# TRT decoder plugin: tri-state (ON, OFF, AUTO).
#   AUTO (default): build the plugin if TensorRT is detected; otherwise skip
#                   it silently (with a STATUS message).
#   ON           : require TensorRT; configure fails if it cannot be found.
#   OFF          : do not build the plugin and do not search for TensorRT.
set(CUDAQ_QEC_BUILD_TRT_DECODER "AUTO" CACHE STRING
    "Build the TensorRT decoder plugin (ON, OFF, AUTO)")
set_property(CACHE CUDAQ_QEC_BUILD_TRT_DECODER
             PROPERTY STRINGS "AUTO" "ON" "OFF")

# Decoders-only build: the decoder interface (libcudaq-qec-decoders.so), the
# built-in decoders, and the decoder plugins, with no CUDA-Q install and no
# LLVM/MLIR required. Skips libcudaq-qec.so (codes, experiments, DEM
# sampling), tools, and unit tests; with CUDAQX_QEC_BINDINGS_PYTHON=ON it
# builds a decoders-only python module instead of the full cudaq_qec one.
# A CUDA toolkit is still required (the decoder API links cudart).
option(CUDAQ_QEC_DECODERS_ONLY
       "Build only the decoder stack; no CUDA-Q install required."
       OFF)
if (CUDAQ_QEC_DECODERS_ONLY AND NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "CUDAQ_QEC_DECODERS_ONLY requires configuring libs/qec as the top-level "
    "project: cmake -S libs/qec -B <build-dir> -DCUDAQ_QEC_DECODERS_ONLY=ON")
endif()

# Decoding server in decoders-only mode
# ==============================================================================
# The decoding server is driven by cudaq-realtime, which is a separate product
# from CUDA-Q: libcudaq-realtime.so itself needs nothing but libc, and the
# server uses none of the CUDA-Q runtime, the simulators, or the caller-side
# libcudaq-device-call-runtime.  So the server can be built in decoders-only
# mode whenever cudaq-realtime is present -- gate on finding it, and warn (do
# not fail) when it is absent, since the decoder stack itself is unaffected.
if (CUDAQ_QEC_DECODERS_ONLY)
  set(CUDAQ_REALTIME_ROOT "" CACHE PATH "Prefix to a cudaq-realtime install")
  find_path(CUDAQ_QEC_REALTIME_PROBE_INCLUDE_DIR
    NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR} ${CUDAQ_INSTALL_PREFIX}
    PATH_SUFFIXES include)
  find_library(CUDAQ_QEC_REALTIME_PROBE_LIBRARY
    NAMES cudaq-realtime
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR} ${CUDAQ_INSTALL_PREFIX}
    PATH_SUFFIXES lib)
  if (CUDAQ_QEC_REALTIME_PROBE_INCLUDE_DIR AND CUDAQ_QEC_REALTIME_PROBE_LIBRARY)
    set(CUDAQ_QEC_BUILD_DECODING_SERVER ON)
    message(STATUS
      "decoders-only: cudaq-realtime found (${CUDAQ_QEC_REALTIME_PROBE_LIBRARY}); "
      "the decoding server will be built")
  else()
    set(CUDAQ_QEC_BUILD_DECODING_SERVER OFF)
    message(WARNING
      "decoders-only: cudaq-realtime not found, so the decoding server and the "
      "realtime decoding libraries will NOT be built (the decoders themselves "
      "are unaffected).  Set -DCUDAQ_REALTIME_ROOT=/path/to/cudaq-realtime to "
      "enable them.")
  endif()
endif()

# Find and configure LLVM, Clang and MLIR
# ==============================================================================
# The user can specify the path to LLVM cmake directory using
# `-DLLVM_DIR=path/to/cmake/llvm`.  If this definition is not provided, we look
# for `llvm-config` tool.  The user can also provide a LLVM version to look for
# using `LLVM_VERSION_MAJOR`, e.g. "-LLVM_VERSION_MAJOR=16".  Note that this
# version variable is set to the latest LLVM version by default, and setting it
# to an older version might break the project.
# Skipped in decoders-only mode: nothing on the decoder path needs LLVM, MLIR,
# or Clang.  The exception is the LLVM half when the decoding server is being
# built -- the realtime decoding config parser reads its YAML with llvm::yaml
# and emits JSON schema with llvm::json, so it needs LLVMSupport (but still no
# MLIR and no Clang).
if(NOT CUDAQ_QEC_DECODERS_ONLY OR CUDAQ_QEC_BUILD_DECODING_SERVER)
if(NOT LLVM_VERSION_MAJOR)
  set(LLVM_VERSION_MAJOR 22)
endif()
if(NOT LLVM_VERSION_MINOR)
  set(LLVM_VERSION_MINOR 1)
endif()
find_package(LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} CONFIG QUIET)

if(NOT LLVM_DIR)
  message(STATUS "LLVM_DIR not found, will try with llvm-config executable.")

  macro(find_llvm_config name version_major)
    set(extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0)
      list(GET extra_args 0 path)
      find_program(LLVM_CONFIG NAMES ${name} PATHS ${path} NO_DEFAULT_PATH
        DOC "Path to llvm-config tool")
    else()
      find_program(LLVM_CONFIG NAMES ${name} DOC "Path to llvm-config tool")
    endif()

    if(LLVM_CONFIG)
      execute_process(
        COMMAND ${LLVM_CONFIG} --version
        RESULT_VARIABLE LLVM_CONFIG_RESULT
        OUTPUT_VARIABLE LLVM_CONFIG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if(NOT LLVM_CONFIG_VERSION MATCHES "^${version_major}[.][0-9]+[.][0-9]+")
        unset(LLVM_CONFIG CACHE)
      endif()
    endif()
  endmacro()

  # First we try to find llvm-config in the llvm submodule.
  find_llvm_config(llvm-config ${LLVM_VERSION_MAJOR}
    "${CMAKE_CURRENT_SOURCE_DIR}/tpls/llvm/build/bin")

  # Try to find a system llvm-config and make sure it is the correct version.
  if(NOT LLVM_CONFIG)
    find_llvm_config(llvm-config ${LLVM_VERSION_MAJOR})
  endif()

  # If it is not the correct version, try finding llvm-config-VERSION
  if(NOT LLVM_CONFIG)
    find_llvm_config(llvm-config-${LLVM_VERSION_MAJOR} ${LLVM_VERSION_MAJOR})
    if (LLVM_CONFIG)
      set(NVQPP_LLVM_EXECUTABLE_SUFFIX -${LLVM_VERSION_MAJOR})
    endif()
  endif()

  execute_process(
    COMMAND ${LLVM_CONFIG} --cmakedir
    RESULT_VARIABLE LLVM_CONFIG_RESULT
    OUTPUT_VARIABLE LLVM_CONFIG_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(LLVM_CONFIG_RESULT)
    message(FATAL_ERROR
      "Could not find suitable llvm-config(-${LLVM_VERSION_MAJOR}).\
      \nTry providing valid -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm.")
  else()
    find_package(LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} REQUIRED CONFIG
      HINTS ${LLVM_CONFIG_CMAKE_DIR} NO_DEFAULT_PATH)
  endif()
endif()

endif() # NOT CUDAQ_QEC_DECODERS_ONLY OR CUDAQ_QEC_BUILD_DECODING_SERVER

# MLIR and Clang are only ever needed by the full build.
if(NOT CUDAQ_QEC_DECODERS_ONLY)
if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_BINARY_DIR}/lib/cmake/mlir)
endif()

find_package(MLIR REQUIRED CONFIG)

if(NOT Clang_DIR)
  set(Clang_DIR ${LLVM_BINARY_DIR}/lib/cmake/clang)
endif()

find_package(Clang CONFIG)
if (NOT Clang_FOUND) 
  message(STATUS "Clang not found, turning off C++ Frontend.")
  set (CUDAQ_DISABLE_CPP_FRONTEND ON)
endif()

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Add LLVM, Clang and MLIR headers to the include path
include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
endif() # NOT CUDAQ_QEC_DECODERS_ONLY

# Check for CUDA Support (ref: cuda-quantum/CMakeLists.txt)
# ==============================================================================
include(CheckLanguage)
check_language(CUDA)
set(CUDA_FOUND FALSE)
# Generate -gencode arch=compute_XX,code=sm_XX for list of supported
# arch values.
# List should be sorted in increasing order.
function(CUDA_get_gencode_args out_args_string arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
    set(out "${out} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)

  # Repeat the last one as to ensure the generation of PTX for most
  # recent virtual architecture for forward compatibility
  list(GET arch_list -1 last_arch)
  set(out "${out} -gencode arch=compute_${last_arch},code=compute_${last_arch}")
  set(${out_args_string} ${out} PARENT_SCOPE)
endfunction()

if(CMAKE_CUDA_COMPILER)
  if (NOT CUDA_TARGET_ARCHS)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

    if (CUDAToolkit_VERSION VERSION_LESS 13.0)
      # Ampere, Hopper
      set(CUDA_TARGET_ARCHS  "75;80;90")
    else()
      # Ampere, Hopper, Blackwell
      set(CUDA_TARGET_ARCHS  "75;80;90;100")
    endif()
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_TARGET_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++17 ${CUDA_gencode_flags} --compiler-options -fPIC")

  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  message(STATUS "Cuda language found.")
endif()

# External Dependencies 
# ==============================================================================

if (CUDAQX_QEC_STANDALONE_BUILD)
  if (CUDAQ_QEC_DECODERS_ONLY)
    # The only CUDA-Q-provided target the decoder stack consumes is fmt
    # (normally imported via NVQIRConfig.cmake); fetch the same pinned commit
    # directly instead of importing the CUDA-Q targets.
    include(FetchContent)
    set(FMT_INSTALL OFF CACHE BOOL "Generate the fmt install target." FORCE)
    FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt
      GIT_TAG fc8d07cfe54ba9f5019453dfdb112491246ee017
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(fmt)
  else()
    cudaqx_import_cudaq_targets()

    if (CUDAQX_QEC_BINDINGS_PYTHON)
      find_package(CUDAQPythonInterop CONFIG)
    endif()
  endif()
  # FIXME
  add_subdirectory(../core core_build)
endif()

# Wheel building setup
# ==============================================================================

if (SKBUILD)
  # When building with scikit, i.e., building wheels, we want all the install
  # to be on the package directory.
  set(CMAKE_INSTALL_BINDIR cudaq_qec/bin)
  set(CMAKE_INSTALL_INCLUDEDIR cudaq_qec/include)
  set(CMAKE_INSTALL_LIBDIR cudaq_qec/lib)
endif()

# Version
# ==============================================================================
if (DEFINED ENV{CUDAQX_QEC_VERSION})
  # The version was defined by the user (likely a bot performing the build), so
  # use the value provided as is.
  set(CUDAQX_QEC_VERSION "$ENV{CUDAQX_QEC_VERSION}")
  message(STATUS "CUDAQX_QEC_VERSION is set to ${CUDAQX_QEC_VERSION}")
else()
  # Otherwise, create a 0.0.0 version string.
  set(CUDAQX_QEC_VERSION "0.0.0")
endif()

# Retrieve the commit SHA for full revision description
execute_process(COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDAQX_QEC_COMMIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lib/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/lib/version.cpp" @ONLY)

# Directory setup
# ==============================================================================

add_subdirectory(lib)
add_subdirectory(
  "${CUDAQX_QEC_SOURCE_DIR}/../../benchmarks/qec"
  "${CMAKE_CURRENT_BINARY_DIR}/benchmarks")
if (NOT CUDAQ_QEC_DECODERS_ONLY OR CUDAQ_QEC_BUILD_DECODING_SERVER)
  add_subdirectory(tools)
endif()

if (CUDAQX_QEC_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if (CUDAQX_QEC_INCLUDE_TESTS AND NOT CUDAQ_QEC_DECODERS_ONLY)
  add_custom_target(CUDAQXQECUnitTests)
  if (CUDAQX_QEC_STANDALONE_BUILD)
    include(CTest)

    add_custom_target(run_tests
      COMMAND ${CMAKE_COMMAND} -E env 
              PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
              ${CMAKE_CTEST_COMMAND} --output-on-failure
      DEPENDS CUDAQXQECUnitTests
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    if (CUDAQX_QEC_BINDINGS_PYTHON)
      add_custom_target(run_python_tests
        COMMAND ${CMAKE_COMMAND} -E env 
                PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
                pytest -v ${CUDAQX_QEC_SOURCE_DIR}/python/tests
        DEPENDS cudaqx-pymodules
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
    endif()
  else()
    add_dependencies(CUDAQXUnitTests CUDAQXQECUnitTests)
  endif()
  add_subdirectory(unittests)
endif()

# External decoder setup
# ==============================================================================
# Make an external decoder available inside wheel. This will modify the rpath of
# any .so files provided.
function(add_target_libs_to_wheel decoder_lib)
  if (NOT EXISTS "${decoder_lib}")
    message(FATAL_ERROR "Invalid file path to external decoder: ${decoder_lib}.")
  endif()
  message(STATUS "Installing external decoder '${decoder_lib}'")
  # Patch the file's rpath during configuration time
  execute_process(
    COMMAND patchelf --set-rpath $ORIGIN/.. "${decoder_lib}"
    RESULT_VARIABLE patchelf_result
    OUTPUT_VARIABLE patchelf_output
    ERROR_VARIABLE patchelf_error
  )
  if (NOT patchelf_result EQUAL 0)
    message(FATAL_ERROR "Failed to patch rpath on ${decoder_lib}: ${patchelf_error}")
  endif()
  message(STATUS "Successfully patched rpath on ${decoder_lib}")
  install(FILES ${decoder_lib}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
          COMPONENT qec-lib-plugins)
  # Mirror the (rpath-patched) plugin into the build tree so build-tree ctests
  # can load it -- the runtime scans <libdir>/decoder-plugins next to
  # libcudaq-qec.so. configure_file re-runs cmake when the source .so changes.
  get_filename_component(_decoder_lib_name "${decoder_lib}" NAME)
  configure_file("${decoder_lib}"
    "${CMAKE_BINARY_DIR}/lib/decoder-plugins/${_decoder_lib_name}" COPYONLY)
endfunction()

if(NOT QEC_EXTERNAL_DECODERS)
  SET(QEC_EXTERNAL_DECODERS $ENV{QEC_EXTERNAL_DECODERS})
endif()

if (QEC_EXTERNAL_DECODERS)
  while(QEC_EXTERNAL_DECODERS)
    list(POP_FRONT QEC_EXTERNAL_DECODERS LIB_FILE)
    add_target_libs_to_wheel(${LIB_FILE})
  endwhile()
endif()

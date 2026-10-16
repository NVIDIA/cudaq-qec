# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# We need 3.28 because of the `EXCLUDE_FROM_ALL` in FetchContent_Declare
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Set a default build type if none was specified. Must set this before
# project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")

# Set a default install prefix if none was specified.
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaqx" CACHE STRING
    "Install path prefix, prepended onto install directories")

project(CUDAQX
  VERSION 0.0.0
  LANGUAGES C CXX)

# Do not let vendored dependencies add their own install rules to CUDA-QX
# packages. CUDA-QX installs only its public artifacts.
set(FMT_INSTALL OFF CACHE BOOL "Generate the fmt install target." FORCE)

set(CUDAQX_ALL_LIBS "qec")
set(CUDAQX_ENABLE_LIBS "all" CACHE STRING
  "Semicolon-separated list of libs to build (${CUDAQX_ALL_LIBS}), or \"all\".")

# We don't want to handle "all" later, thus expand it here.
if(CUDAQX_ENABLE_LIBS STREQUAL "all" )
  set(CUDAQX_ENABLE_LIBS ${CUDAQX_ALL_LIBS})
endif()

# Sanity check.
foreach(lib ${CUDAQX_ENABLE_LIBS})
  if (NOT "${lib}" IN_LIST CUDAQX_ALL_LIBS)
    message(FATAL_ERROR "${lib} isn't a known library: ${CUDAQX_ALL_LIBS}.")
  endif()
endforeach()

# Project setup
# ==============================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add our Modules to the path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Include custom CUDA-QX modules
include(CUDA-QX)

# Options
# ==============================================================================

option(CUDAQX_INCLUDE_TESTS "Generate build targets for unit tests." ON)
option(CUDAQX_INCLUDE_DOCS "Generate build targets for the docs." ON)
option(CUDAQX_BINDINGS_PYTHON "Generate build targets for python bindings." ON)

# Top-level External Dependencies
# ==============================================================================

# Find and configure LLVM, Clang and MLIR
# ==============================================================================
# The user can specify the path to LLVM cmake directory using
# `-DLLVM_DIR=path/to/cmake/llvm`.  If this definition is not provided, we look
# for `llvm-config` tool.  The user can also provide a LLVM version to look for
# using `LLVM_VERSION_MAJOR`, e.g. "-LLVM_VERSION_MAJOR=16".  Note that this
# version variable is set to the latest LLVM version by default, and setting it
# to an older version might break the project.
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

cudaqx_import_cudaq_targets()

# Top-level targets
# ==============================================================================

# Helper targets to collect libraries and python modules
add_custom_target(cudaqx-pymodules)

# Top-level testing
# ==============================================================================

if (CUDAQX_INCLUDE_TESTS)
  include(CTest)
  add_custom_target(CUDAQXUnitTests)
  add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
            ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS CUDAQXUnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  if (CUDAQX_BINDINGS_PYTHON)
    set(PYTHON_TEST_DIRS "")
    foreach(lib ${CUDAQX_ENABLE_LIBS})
      list(APPEND PYTHON_TEST_DIRS ../libs/${lib}/python/tests)
    endforeach()
    add_custom_target(run_python_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
              pytest -v ${PYTHON_TEST_DIRS}
      DEPENDS cudaqx-pymodules
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  endif()
endif()

# Hooks setup. If the repo contains a custom pre-push hook, attempt to install
# it. If the user has a different one installed, then warn them but do not fail
# configuration.
# ==============================================================================

# Define the directory where your hooks are stored
set(SRC_HOOK_PRE_PUSH "${CMAKE_SOURCE_DIR}/.githooks/pre-push")

if(EXISTS "${SRC_HOOK_PRE_PUSH}")
  # Get the Git hooks directory from the Git configuration
  execute_process(
    COMMAND git config --get core.hooksPath
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_HOOKS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  
  # Determine the target hooks directory
  if(GIT_HOOKS_DIR)
    set(TARGET_HOOKS_DIR "${GIT_HOOKS_DIR}")
  else()
    set(TARGET_HOOKS_DIR "${CMAKE_SOURCE_DIR}/.git/hooks")
  endif()
  set(DST_HOOK_PRE_PUSH "${TARGET_HOOKS_DIR}/pre-push")
  
  if(EXISTS "${DST_HOOK_PRE_PUSH}")
    # Compare the contents of the src and dst hook.
    execute_process(
      COMMAND git hash-object "${DST_HOOK_PRE_PUSH}"
      OUTPUT_VARIABLE SHA_DST
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND git hash-object "${SRC_HOOK_PRE_PUSH}"
      OUTPUT_VARIABLE SHA_SRC
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT SHA_SRC STREQUAL SHA_DST)
      message(WARNING
        "You already have a ${DST_HOOK_PRE_PUSH} script installed. "
        "This configuration script will not overwrite it despite the fact that "
        "it is strongly recommended to use ${SRC_HOOK_PRE_PUSH} in your environment."
        "\nProceed with caution!")
    endif()
  else()
    if(EXISTS "${TARGET_HOOKS_DIR}")
      file(COPY "${SRC_HOOK_PRE_PUSH}"
           DESTINATION "${TARGET_HOOKS_DIR}"
           FILE_PERMISSIONS
             OWNER_READ OWNER_WRITE OWNER_EXECUTE
             GROUP_READ GROUP_EXECUTE
             WORLD_READ WORLD_EXECUTE)
      message(STATUS "Git pre-push hook installed to ${TARGET_HOOKS_DIR}")
    else()
      message(WARNING
        "The Git hooks directory does not exist: ${TARGET_HOOKS_DIR}\n"
        "Are you sure this is a Git repository? Hook cannot be installed."
      )
    endif()
  endif()
endif()

# Directory setup
# ==============================================================================

add_subdirectory(libs/core)

if (CUDAQX_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

foreach(lib ${CUDAQX_ENABLE_LIBS})
  add_subdirectory(libs/${lib})
endforeach()


# Custom targets
# ==============================================================================
set(ZIP_OUTPUT "${CMAKE_BINARY_DIR}/installed_files.zip")

# Custom target to zip up installed files. The resulting zip file can be used to
# overlay CUDA-QX files into a separate (already existing) CUDA-Q install
# directory.
add_custom_target(zip_installed_files
    COMMAND bash -c "sed 's#^${CMAKE_INSTALL_PREFIX}/##' ${CMAKE_BINARY_DIR}/install_manifest.txt | zip -@ '${ZIP_OUTPUT}'"
    WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}"
    COMMENT "Creating zip archive of installed files"
    VERBATIM
)

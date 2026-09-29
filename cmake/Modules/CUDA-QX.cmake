# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#[=======================================================================[.rst:
cudaqx_add_device_code
----------------------------

Add NVQ++ custom-compiled quantum device source files to a target library using nvq++.

This function compiles specified source files using the nvq++ compiler
from the CUDAQ installation and adds the resulting object files to the
given library target.

.. command:: add_custom_compiled_sources

  .. code-block:: cmake

    cudaqx_add_device_code(
      <library_name>
      SOURCES <source1> [<source2> ...]
      [COMPILER_FLAGS <flag1> [<flag2> ...]]
    )

  ``<library_name>``
    The name of the existing library target to which the compiled
    sources will be added.

  ``SOURCES <source1> [<source2> ...]``
    A list of source files to be compiled.

  ``COMPILER_FLAGS <flag1> [<flag2> ...]``
    Optional. A list of compiler flags to be passed to nvq++.

This function creates custom commands to compile each source file using
nvq++, generates custom targets for each compilation, and adds the
resulting object files to the specified library target.

Note: This function assumes that the CUDAQ_INSTALL_DIR variable is set
to the CUDAQ installation directory.

Note: You can use DEPENDS_ON if you want to delay compilation until some other
target has been built.

Example usage:
  cudaqx_add_device_code(
    my_library
    SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/file1.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/file2.cpp
    COMPILER_FLAGS
      -v
    DEPENDS_ON
      SomeOtherTarget
  )

#]=======================================================================]
function(cudaqx_add_device_code LIBRARY_NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES COMPILER_FLAGS DEPENDS_ON)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT DEFINED CUDAQ_INSTALL_DIR)
    message(FATAL_ERROR "CUDAQ_INSTALL_DIR must be defined")
  endif()

  if(NOT ARGS_SOURCES)
    message(FATAL_ERROR "At least one SOURCE file is required")
  endif()

  set(COMPILER ${CUDAQ_INSTALL_DIR}/bin/nvq++)

  # It might be that our CXX toolchain is installed in non-standard path and
  # `cudaq-quake`, being a clang-based compiler, won't be able to find it. In
  # such cases, setting CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN will allows us
  # to tell `cudaq-quake` where to look for the toolchain. (This happens when
  # building wheels inside the manylinux container, for example.)
  if (CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
    set(ARGS_COMPILER_FLAGS "${ARGS_COMPILER_FLAGS} --gcc-install-dir=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
  endif()

  set(prop "$<TARGET_PROPERTY:${LIBRARY_NAME},INCLUDE_DIRECTORIES>")
  foreach(source ${ARGS_SOURCES})
    get_filename_component(filename ${source} NAME_WE)
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_${filename}.o")

    # nvq++ drops intermediates named after the *source* (<src>.o, <src>.qke.o,
    # <src>.classic.o, ...) into its working directory, so two targets
    # compiling the same source file in the same directory race and corrupt
    # each other's objects under parallel builds. Isolate each object's
    # compile in its own working directory.
    set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_${filename}.nvqpp")
    file(MAKE_DIRECTORY ${work_dir})

    # TODO: this custom command only depends on the source file, not on the
    # headers it includes (nvq++ emits no depfile here), so header changes do
    # NOT trigger recompilation -- a stale object silently survives ninja
    # after e.g. a config-struct layout change (ABI-mismatch segfaults).
    # Until nvq++ depfile output is wired up, `rm` the affected
    # <target>_<source>.o under the build tree after header changes.
    add_custom_command(
      OUTPUT ${output_file}
      COMMAND ${COMPILER}
        ${ARGS_COMPILER_FLAGS} -c -fPIC
        ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${output_file}
        "$<$<BOOL:${prop}>:-I $<JOIN:${prop}, -I >>"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source} ${ARGS_DEPENDS_ON}
      WORKING_DIRECTORY ${work_dir}
      COMMENT "Compiling ${source} with nvq++"
      VERBATIM
    )

    list(APPEND object_files ${output_file})
    list(APPEND custom_targets ${LIBRARY_NAME}_${filename}_target)

    add_custom_target(${LIBRARY_NAME}_${filename}_target DEPENDS ${output_file})
  endforeach()

  add_dependencies(${LIBRARY_NAME} ${custom_targets})
  target_sources(${LIBRARY_NAME} PRIVATE ${object_files})
endfunction()

#[=======================================================================[.rst:
_cudaqx_import_nvqir_target
---------------------------

Private helper used by ``cudaqx_import_cudaq_targets`` to recreate CUDA-Q's
NVQIR backend imported targets without loading ``CUDAQConfig.cmake``.

#]=======================================================================]
function(_cudaqx_import_nvqir_target target_name library_name)
  if(NOT TARGET ${target_name})
    add_library(${target_name} SHARED IMPORTED)
    set_target_properties(${target_name} PROPERTIES
      IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/${library_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
      IMPORTED_SONAME "${library_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default;cudaq::cudaq-mlir-runtime")
  endif()
endfunction()

#[=======================================================================[.rst:
cudaqx_import_cudaq_targets
---------------------------

Import the CUDA-Q CMake targets used by CUDA-QX without loading
``CUDAQConfig.cmake``. This avoids enabling the custom CUDAQ CMake language
while still providing imported targets such as ``cudaq::cudaq``,
``cudaq::cudaq-common``, and ``cudaq::cudaq-stim-target``.

The caller must provide ``CUDAQ_DIR`` pointing to the CUDA-Q CMake package
directory, e.g. ``<cudaq-prefix>/lib/cmake/cudaq``.

Example usage:
  .. code-block:: cmake

    # Configure with:
    #   -DCUDAQ_DIR=<cudaq-prefix>/lib/cmake/cudaq
    cudaqx_import_cudaq_targets()

#]=======================================================================]
function(cudaqx_import_cudaq_targets)
  if (NOT CUDAQ_DIR)
    message(FATAL_ERROR
      "CUDAQ_DIR must point to the CUDA-Q CMake package directory, e.g. "
      "<cudaq-prefix>/lib/cmake/cudaq.")
  endif()

  set(CUDAQ_CMAKE_DIR "${CUDAQ_DIR}")
  get_filename_component(_cudaq_parent_dir "${CUDAQ_CMAKE_DIR}" DIRECTORY)
  get_filename_component(CUDAQ_LIBRARY_DIR "${_cudaq_parent_dir}" DIRECTORY)
  get_filename_component(CUDAQ_INSTALL_DIR "${CUDAQ_LIBRARY_DIR}" DIRECTORY)
  set(CUDAQ_INCLUDE_DIR "${CUDAQ_INSTALL_DIR}/include")

  include(CMakeFindDependencyMacro)
  set(NVQIR_DIR "${_cudaq_parent_dir}/nvqir")

  foreach(_cudaq_pkg IN ITEMS
      CUDAQCommon
      CUDAQEmDefault
      CUDAQEnsmallen
      CUDAQLogger
      CUDAQMlirRuntime
      CUDAQNlopt
      CUDAQOperator
      CUDAQPlatformDefault
      CUDAQPythonInterop)
    set(${_cudaq_pkg}_DIR "${CUDAQ_CMAKE_DIR}")
  endforeach()

  find_package(NVQIR REQUIRED CONFIG)
  find_package(CUDAQOperator REQUIRED CONFIG)
  find_package(CUDAQCommon REQUIRED CONFIG)
  find_package(CUDAQNlopt REQUIRED CONFIG)
  find_package(CUDAQEnsmallen REQUIRED CONFIG)
  find_package(CUDAQEmDefault REQUIRED CONFIG)
  find_package(CUDAQPlatformDefault REQUIRED CONFIG)
  find_package(CUDAQPythonInterop CONFIG)

  # Import the CUDA-Q library target without loading CUDAQConfig.cmake, which
  # enables the CUDAQ CMake language in current CUDA-Q installs.
  if(NOT TARGET cudaq::cudaq)
    include("${CUDAQ_CMAKE_DIR}/CUDAQTargets.cmake")
  endif()

  set(__base_nvtarget_name "custatevec")
  find_library(CUDAQ_CUSVSIM_PATH NAMES cusvsim-fp32 HINTS ${CUDAQ_LIBRARY_DIR})
  if (CUDAQ_CUSVSIM_PATH)
    set(__base_nvtarget_name "cusvsim")
  endif()

  _cudaqx_import_nvqir_target(cudaq::cudaq-default-target "libnvqir-${__base_nvtarget_name}-fp64")
  _cudaqx_import_nvqir_target(cudaq::cudaq-nvidia-target "libnvqir-${__base_nvtarget_name}-fp32")
  _cudaqx_import_nvqir_target(cudaq::cudaq-nvidia-fp64-target "libnvqir-${__base_nvtarget_name}-fp64")
  _cudaqx_import_nvqir_target(cudaq::cudaq-nvidia-mgpu-target "libnvqir-mgpu-fp32")
  _cudaqx_import_nvqir_target(cudaq::cudaq-nvidia-mgpu-fp64-target "libnvqir-mgpu-fp64")
  _cudaqx_import_nvqir_target(cudaq::cudaq-qpp-cpu-target "libnvqir-qpp")
  _cudaqx_import_nvqir_target(cudaq::cudaq-qpp-density-matrix-cpu-target "libnvqir-dm")
  _cudaqx_import_nvqir_target(cudaq::cudaq-stim-target "libnvqir-stim")

  if(NOT COMMAND cudaq_set_target)
    function(cudaq_set_target TARGETNAME)
      message(STATUS "CUDA Quantum Target = ${TARGETNAME}")
      target_link_libraries(cudaq::cudaq INTERFACE cudaq::cudaq-${TARGETNAME}-target)
    endfunction()
  endif()

  set(__tmp_cudaq_target "qpp-cpu")
  find_program(NVIDIA_SMI "nvidia-smi")
  if(NVIDIA_SMI)
    execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l"
                    OUTPUT_VARIABLE NGPUS OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (${NGPUS} GREATER_EQUAL 1)
      message(STATUS "Number of NVIDIA GPUs detected: ${NGPUS}")
      set(__tmp_cudaq_target "nvidia")
    endif()
  endif()

  set(CUDAQ_TARGET ${__tmp_cudaq_target} CACHE STRING
      "The CUDA Quantum target to compile for and execute on. Defaults to `${__tmp_cudaq_target}`")
  cudaq_set_target(${CUDAQ_TARGET})

  foreach(_cudaq_var IN ITEMS
      CUDAQ_CMAKE_DIR
      CUDAQ_INCLUDE_DIR
      CUDAQ_INSTALL_DIR
      CUDAQ_LIBRARY_DIR
      CUDAQPythonInterop_DIR)
    set(${_cudaq_var} "${${_cudaq_var}}" PARENT_SCOPE)
  endforeach()
endfunction()

#[=======================================================================[.rst:
cudaqx_set_target
-------------------------
   Set up a CUDA-QX target with the specified name.

   This function creates an interface library for the given CUDA-QX
   target, links it to the main CUDAQ library, and adds target-specific libraries.

   :param TARGETNAME: The name of the CUDA-QX target to set up.

   .. note::
      This function will create an interface library
      named ``cudaq_${TARGETNAME}`` and an alias target ``cudaq::cudaq_${TARGETNAME}``.

   **Example:**

   .. code-block:: cmake

      cudaqx_set_target(my_target)

   This will:

   1. Create an interface library ``cudaq_my_target``
   2. Link it to ``cudaq::cudaq``
   3. Link it to ``cudaq::cudaq-my_target-target``
   4. Create an alias target ``cudaq::cudaq_my_target``

   This function simplifies the process of setting up CUDA-QX targets by
   automating the creation of interface libraries and establishing the necessary linkages.
#]=======================================================================]
function(cudaqx_set_target TARGETNAME)
  message(STATUS "Setting CUDA-QX Target = ${TARGETNAME}")

  # Create a new interface target
  add_library(cudaq_${TARGETNAME} INTERFACE)

  # Link to the original cudaq target
  target_link_libraries(cudaq_${TARGETNAME} INTERFACE cudaq::cudaq)

  # Add the additional target-specific library
  target_link_libraries(cudaq_${TARGETNAME} INTERFACE cudaq::cudaq-${TARGETNAME}-target)

  # Create an alias target to make it easier to use
  add_library(cudaq::cudaq_${TARGETNAME} ALIAS cudaq_${TARGETNAME})
endfunction()

#[=======================================================================[.rst:
cudaqx_add_pymodule
-------------------------
    This is a helper function to add CUDAQ-QX libraries' python modules. It's
    main purpose is to create a custom target, cudaqx-pymodules, which depends
    on all libraries' python modules.

#]=======================================================================]
function(cudaqx_add_pymodule module)
  nanobind_add_module(${module} NB_DOMAIN cudaq ${ARGN})
  add_dependencies(cudaqx-pymodules ${module})
endfunction()

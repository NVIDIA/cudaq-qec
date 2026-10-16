# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# External Dependencies
# ==============================================================================

# Find Python before nanobind so Python_INCLUDE_DIRS is available when nanobind
# builds its nanobind-static library target.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT COMMAND nanobind_add_module)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.12.0
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(nanobind)
endif()

# ==============================================================================

# Decoders-only module: binds the decoder API (py_decoder.cpp is cudaq-free;
# the CUDA-Q result-type casters live in cudaq_type_casters.h, which only the
# full module below compiles). Bindings land under `module.qecrt`.
if (CUDAQ_QEC_DECODERS_ONLY)
  set(MODULE_NAME _qec_decoders_standalone)

  cudaqx_add_pymodule(${MODULE_NAME}
    bindings/cudaqx_qec_decoders_only.cpp
    bindings/py_decoder.cpp
  )

  target_link_libraries(${MODULE_NAME} PRIVATE cudaq-qec-decoders)

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
  target_include_directories(${MODULE_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

  set_target_properties(${MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    BUILD_RPATH "$ORIGIN/../lib")

  return()
endif()

set(MODULE_NAME _pycudaqx_qec_the_suffix_matters_cudaq_qec)

cudaqx_add_pymodule(${MODULE_NAME}
  bindings/cudaqx_qec.cpp
  bindings/py_code.cpp
  bindings/py_decoder.cpp
  bindings/py_decoding_config.cpp
  bindings/py_decoding.cpp
  bindings/py_dem_construction.cpp
  bindings/py_dem_sampling.cpp
  bindings/py_extended_dem.cpp
  bindings/py_surface_code.cpp
  bindings/py_patch.cpp
)

target_link_libraries(${MODULE_NAME}
  PRIVATE
    cudaq-qec
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
    cudaq::cudaq-python-interop
    cudaq::cudaq-mlir-runtime
)

# cudaq-qec-playback-emulator only exist when cudaq-realtime was found;
# CUDAQX_QEC_HAS_CUDAQ_REALTIME lets the bindings reflect that at compile
# time.
if(CUDAQ_REALTIME_INCLUDE_DIR)
  target_sources(${MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools/playback-emulator/python/py_playback_emulator.cpp
  )
  target_include_directories(${MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools/playback-emulator/python
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
  )
  target_link_libraries(${MODULE_NAME} PRIVATE cudaq-qec-playback-emulator)
  target_compile_definitions(${MODULE_NAME} PRIVATE CUDAQX_QEC_HAS_CUDAQ_REALTIME)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
target_include_directories(${MODULE_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_qec")

if (NOT SKBUILD)
  add_custom_target(
    copy_qec_python_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_qec
            ${CMAKE_BINARY_DIR}/python/cudaq_qec
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_qec
    COMMENT "Copying Python files to binary directory"
  )

  add_dependencies(${MODULE_NAME} copy_qec_python_files)
endif()

# RPATH configuration
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/../lib"
  )
endif()

# Install
# ==============================================================================

install(DIRECTORY cudaq_qec
  COMPONENT qec-python
  DESTINATION .
)

install(TARGETS ${MODULE_NAME}
  COMPONENT qec-python
  DESTINATION cudaq_qec/
)

# Undocumented: also ship the cudaq-free decoders module (_qec_decoders_standalone,
# the same one the decoders-only build produces) inside the wheel, so the
# decoders can be used without importing cudaq: `import _qec_decoders_standalone`.
# It links only cudaq-qec-decoders + cudart. It is installed at the wheel root,
# NOT under cudaq_qec/: importing anything under cudaq_qec runs cudaq_qec/__init__
# -> patch.py -> `from cudaq import qvector`, which would pull in cudaq. Wheel
# (SKBUILD) only; $ORIGIN/${CMAKE_INSTALL_LIBDIR} resolves the in-package
# libcudaq-qec-decoders.so (with its decoder-plugins/ sibling) in cudaq_qec/lib.
if (SKBUILD)
  cudaqx_add_pymodule(_qec_decoders_standalone
    bindings/cudaqx_qec_decoders_only.cpp
    bindings/py_decoder.cpp
  )
  target_link_libraries(_qec_decoders_standalone PRIVATE
    cudaq-qec-decoders CUDA::cudart)
  target_include_directories(_qec_decoders_standalone PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(_qec_decoders_standalone PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
  install(TARGETS _qec_decoders_standalone
    COMPONENT qec-python
    DESTINATION .)
endif()

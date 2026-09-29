# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LIBRARY_NAME cudaq-qec)
set(DECODERS_LIBRARY_NAME cudaq-qec-decoders)
# Build-only PIC archive of DEM / sparse-PCM math. Linked whole into
# cudaq-qec-decoders.so; not installed as its own artifact.
set(DEM_LIBRARY_NAME cudaq-qec-dem)

include(FetchContent)

add_compile_options(-Wno-attributes) 

find_package(CUDAToolkit REQUIRED)

# cuStabilizer is shipped inside the cuquantum-python-cuXX pip wheels
# (FindcuStabilizer auto-discovers it from the active Python environment).
# If it is missing, surface a clear, actionable error rather than CMake's
# default "Could NOT find cuStabilizer" message.
# Decoders-only builds skip it: only libcudaq-qec.so (DEM sampling) needs it.
if(NOT CUDAQ_QEC_DECODERS_ONLY)
find_package(cuStabilizer QUIET)
if(NOT cuStabilizer_FOUND)
  set(_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
  if(NOT _cuda_major)
    set(_cuda_major "<CUDA major version, e.g. 12 or 13>")
  endif()
  message(FATAL_ERROR
    "cuStabilizer was not found.\n"
    "It is required to build cudaq-qec and is distributed as part of the "
    "cuQuantum SDK pip wheels. Install it into the Python environment used "
    "for this build with:\n"
    "    pip install cuquantum-python-cu${_cuda_major}\n"
    "If the wheel is installed but discovery still fails (e.g. because the "
    "build runs in an isolated venv such as `python -m build`), point CMake "
    "at the install prefix explicitly via either of:\n"
    "    -DCUQUANTUM_ROOT=<site-packages>/cuquantum\n"
    "    -DCUSTABILIZER_ROOT=/path/to/custabilizer\n"
    "(both are also honored as environment variables).")
endif()
endif() # NOT CUDAQ_QEC_DECODERS_ONLY

set(DEM_SOURCES
  dem_chunks_memory.cpp
  dem_construction.cpp
  dem_construction_utils.cpp
  detector_error_model.cpp
  extended_dem.cpp
  pcm_utils.cpp
  sparse_binary_matrix.cpp
)

set(DECODERS_SOURCES
  decoder.cpp
  decoder_init.cpp
  decoder_config_payload.cpp
  decoder_config_schema.cpp
  decoder_stats.cpp
  logger.cpp
  logger_forwarder.cpp
  plugin_loader.cpp
  decoders/lut.cpp
  decoders/sliding_window.cpp
  version.cpp
)

set(QEC_SOURCES
  code.cpp
  decoder_config_payload_provider.cpp
  dem_construction_code.cpp
  experiments.cpp
  stabilizer_utils.cpp
)

list(APPEND QEC_SOURCES
  dem_sampling/dem_sampling_utils.cu
  dem_sampling/dem_sampling_gpu.cpp
)

add_library(${DEM_LIBRARY_NAME} STATIC ${DEM_SOURCES})
set_target_properties(${DEM_LIBRARY_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  VISIBILITY_INLINES_HIDDEN ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_library(${DECODERS_LIBRARY_NAME} SHARED ${DECODERS_SOURCES})
set_target_properties(${DECODERS_LIBRARY_NAME} PROPERTIES
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT CUDAQ_QEC_DECODERS_ONLY)
  # FIXME?: This must be a shared library. Trying to build a static one will fail.
  add_library(${LIBRARY_NAME} SHARED ${QEC_SOURCES})
endif()

if(NOT TARGET libstim)
  FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG v1.15.0
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(stim)
endif()

if(NOT TARGET libstim)
  message(FATAL_ERROR
    "Stim FetchContent did not provide the libstim target.")
endif()
if(TARGET ${LIBRARY_NAME})
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

target_link_libraries(${DEM_LIBRARY_NAME} PRIVATE libstim)
target_link_options(${DECODERS_LIBRARY_NAME} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wl,--exclude-libs,libstim.a>
  # cudaqx-core is linked into this DSO only; keep its symbols local so they
  # do not preempt libcudaq-xxx' tensor_impl registry in combined builds.
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wl,--exclude-libs,libcudaqx-core.a>
)

add_subdirectory(decoders/plugins/example)
add_subdirectory(decoders/plugins/pymatching)
add_subdirectory(decoders/plugins/chromobius)

# The TRT decoder plugin honors the tri-state `CUDAQ_QEC_BUILD_TRT_DECODER`
# cache variable (AUTO/ON/OFF) declared in the parent CMakeLists.txt.  Skip
# descending entirely when the user explicitly opted out; otherwise let the
# subdirectory handle dependency detection and publish the result via the
# `CUDAQ_QEC_TRT_DECODER_ENABLED` cache variable.
if(NOT CUDAQ_QEC_BUILD_TRT_DECODER STREQUAL "OFF")
  add_subdirectory(decoders/plugins/trt_decoder)
else()
  set(CUDAQ_QEC_TRT_DECODER_ENABLED FALSE CACHE INTERNAL
      "Whether the TRT decoder plugin is being built")
endif()

if(NOT CUDAQ_QEC_DECODERS_ONLY)
  add_subdirectory(codes)
  add_subdirectory(device)
endif()

if (CUDAQX_QEC_USE_FLOAT)
   target_compile_definitions(${DEM_LIBRARY_NAME} PUBLIC -DCUDAQX_QEC_FLOAT_TYPE=float)
   target_compile_definitions(${DECODERS_LIBRARY_NAME} PUBLIC -DCUDAQX_QEC_FLOAT_TYPE=float)
   if(TARGET ${LIBRARY_NAME})
     target_compile_definitions(${LIBRARY_NAME} PUBLIC -DCUDAQX_QEC_FLOAT_TYPE=float)
   endif()
endif()

target_include_directories(${DEM_LIBRARY_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:cudaqx-core,INTERFACE_INCLUDE_DIRECTORIES>>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:fmt::fmt-header-only,INTERFACE_INCLUDE_DIRECTORIES>>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${DECODERS_LIBRARY_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:fmt::fmt-header-only,INTERFACE_INCLUDE_DIRECTORIES>>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(${DECODERS_LIBRARY_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>
)

target_link_libraries(${DECODERS_LIBRARY_NAME}
  PUBLIC
    $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaqx-core>
    fmt::fmt-header-only
  PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,${DEM_LIBRARY_NAME}>
    # decoder::get() consumes the "cuda_device_id" construction parameter
    # (validation via cudaGetDeviceCount + persistent cudaSetDevice pin).
    CUDA::cudart
)

if(TARGET ${LIBRARY_NAME})
  target_include_directories(${LIBRARY_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_options(${LIBRARY_NAME} PUBLIC
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>
  )

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC
      ${DECODERS_LIBRARY_NAME}
      cudaq::cudaq-operator
    PRIVATE
      cudaq::cudaq-common
  )

  target_link_libraries(${LIBRARY_NAME} PRIVATE
    cuStabilizer::cuStabilizer
    CUDA::cudart
  )
  set_source_files_properties(
    dem_sampling/dem_sampling_utils.cu
    PROPERTIES LANGUAGE CUDA
  )
endif()

# RPATH configuration 
# ==============================================================================

if (NOT TARGET ${LIBRARY_NAME})
  # Decoders-only build: no libcudaq-qec.so to configure RPATHs for.
elseif (NOT SKBUILD)
  # Append the cuStabilizer library directory (typically inside a
  # cuquantum-python pip wheel) so the loader can find libcustabilizer.so.0
  # at run-time without requiring LD_LIBRARY_PATH.
  set(_qec_build_rpath "$ORIGIN")
  if(cuStabilizer_LIBRARY_DIR)
    list(APPEND _qec_build_rpath "${cuStabilizer_LIBRARY_DIR}")
  endif()

  set_target_properties(${LIBRARY_NAME} PROPERTIES
    BUILD_RPATH "${_qec_build_rpath}"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH
  # (this picks up cuStabilizer_LIBRARY_DIR for INSTALL_RPATH as well):
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib.
  #
  # cuStabilizer ships in the sibling cuquantum-python-cuXX pip wheel (declared
  # as a runtime dependency in pyproject.toml).  Its layout in site-packages is
  #   <sp>/cudaq_qec/lib/libcudaq-qec.so   (this target -- $ORIGIN)
  #   <sp>/cuquantum/lib/libcustabilizer.so.0
  # so we add a relative RPATH entry pointing into that sibling wheel.  Using a
  # relative path (rather than the absolute cuStabilizer_LIBRARY_DIR resolved
  # at build time) keeps the wheel portable across environments.
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib:$ORIGIN/../../cuquantum/lib"
  )
endif()

# Install 
# ==============================================================================
# cudaq-qec-dem.a is intentionally omitted: it is a private build artifact
# folded into cudaq-qec-decoders.so.

set(_qec_install_targets ${DECODERS_LIBRARY_NAME})
if(TARGET ${LIBRARY_NAME})
  list(APPEND _qec_install_targets ${LIBRARY_NAME})
endif()
install(TARGETS ${_qec_install_targets}
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
    
install(DIRECTORY ${CUDAQX_QEC_INCLUDE_DIR}/cudaq
  COMPONENT qec-headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)

# realtime/ carries the decoding server's libraries. In decoders-only mode it
# is built only when cudaq-realtime was found (CUDAQ_QEC_BUILD_DECODING_SERVER).
if(NOT CUDAQ_QEC_DECODERS_ONLY OR CUDAQ_QEC_BUILD_DECODING_SERVER)
  add_subdirectory(realtime)
endif()

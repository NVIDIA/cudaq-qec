# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The realtime decoding config parser (config.cpp) uses only llvm::yaml,
# llvm::json and llvm::raw_ostream from LLVMSupport, whose API is stable across
# LLVM versions.  The top-level qec project already locates LLVM (honoring
# LLVM_DIR / LLVM_VERSION_MAJOR / LLVM_VERSION_MINOR) before descending here, so
# reuse that result rather than pinning an exact version -- any LLVM providing
# the Support component works, e.g. a distro `llvm-dev`, with no NVIDIA LLVM
# toolchain required.  Only fall back to a fresh lookup if LLVM was not already
# found (e.g. this directory configured on its own).
if(NOT LLVM_FOUND)
  find_package(LLVM REQUIRED CONFIG)
endif()

# Enable CUDA for device code
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  # Find cuda-quantum realtime headers for device library
  set(CUDAQ_REALTIME_ROOT "" CACHE PATH "Prefix to cuda-quantum install")
  set(_cudaq_realtime_prefixes "")
  if(CUDAQ_REALTIME_ROOT)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
  endif()
  if(CUDAQ_INSTALL_DIR)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_DIR}")
  endif()
  if(CUDAQ_INSTALL_PREFIX)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
  endif()
  
  macro(cudaq_qec_find_realtime_path output_var)
    cmake_parse_arguments(ARG "" "" "NAMES;PATH_SUFFIXES" ${ARGN})
    find_path(${output_var}
      NAMES ${ARG_NAMES}
      HINTS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES ${ARG_PATH_SUFFIXES}
      NO_DEFAULT_PATH
    )
    if(NOT DEFINED ${output_var} OR NOT "${${output_var}}")
      find_path(${output_var}
        NAMES ${ARG_NAMES}
        PATHS ${_cudaq_realtime_prefixes}
        PATH_SUFFIXES ${ARG_PATH_SUFFIXES}
      )
    endif()
  endmacro()

  macro(cudaq_qec_find_realtime_library output_var)
    cmake_parse_arguments(ARG "" "" "NAMES;PATH_SUFFIXES" ${ARGN})
    find_library(${output_var}
      NAMES ${ARG_NAMES}
      HINTS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES ${ARG_PATH_SUFFIXES}
      NO_DEFAULT_PATH
    )
    if(NOT DEFINED ${output_var} OR NOT "${${output_var}}")
      find_library(${output_var}
        NAMES ${ARG_NAMES}
        PATHS ${_cudaq_realtime_prefixes}
        PATH_SUFFIXES ${ARG_PATH_SUFFIXES}
      )
    endif()
  endmacro()

  cudaq_qec_find_realtime_path(CUDAQ_REALTIME_INCLUDE_DIR
    NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
    PATH_SUFFIXES include
  )
  if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
    cudaq_qec_find_realtime_path(CUDAQ_REALTIME_INCLUDE_DIR
      NAMES cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h
      PATH_SUFFIXES include ../include
    )
  endif()
  
  if(CUDAQ_REALTIME_INCLUDE_DIR)
    message(STATUS "Found cuda-quantum realtime headers at ${CUDAQ_REALTIME_INCLUDE_DIR}")
    cudaq_qec_find_realtime_library(CUDAQ_REALTIME_LIBRARY
      NAMES cudaq-realtime
      PATH_SUFFIXES lib
    )
    if(CUDAQ_REALTIME_LIBRARY)
      message(STATUS "Found cuda-quantum realtime library at ${CUDAQ_REALTIME_LIBRARY}")
      get_filename_component(_CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
    endif()

    # The device-graph dispatch API (cudaq_create_dispatch_graph_regular,
    # cudaq_launch_dispatch_graph, cudaq_destroy_dispatch_graph) lives in
    # libcudaq-realtime-dispatch, separate from libcudaq-realtime.  It is
    # relocatable device code, resolved at runtime via dlsym(RTLD_DEFAULT, ...)
    # from the host executable's absorbed copy --
    # it must NOT be linked into cudaq-qec-realtime-decoding.so: a second
    # device-linked copy of the dispatch kernel would live in a different CUDA
    # module than the decoder's DEVICE_CALL handlers (captured from the
    # executable's module), and cross-module device function pointer calls trap
    # with cudaErrorIllegalInstruction.  The find result is used by device-mode
    # test executables, which absorb the archive themselves.
    cudaq_qec_find_realtime_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
      NAMES cudaq-realtime-dispatch
      PATH_SUFFIXES lib
    )
    if(CUDAQ_REALTIME_DISPATCH_LIBRARY)
      message(STATUS "Found cuda-quantum realtime dispatch library at ${CUDAQ_REALTIME_DISPATCH_LIBRARY}")
    endif()

    # CUDA-Q lowered `cudaq::device_call` host-dispatch integration needs the
    # DeviceCallService provider API from a realtime/device-call-enabled CUDA-Q
    # build.
    cudaq_qec_find_realtime_path(CUDAQ_DEVICE_CALL_INCLUDE_DIR
      NAMES cudaq/realtime/device_call_service.h
      PATH_SUFFIXES include
    )
    if(CUDAQ_DEVICE_CALL_INCLUDE_DIR)
      message(STATUS "Found cuda-quantum device_call headers at ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}")
    else()
      message(STATUS "cuda-quantum device_call headers not found. "
                     "QEC lowered device_call host-dispatch service will not be built.")
    endif()

    cudaq_qec_find_realtime_library(CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY
      NAMES cudaq-device-call-runtime
      PATH_SUFFIXES lib
    )
    if(CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
      message(STATUS "Found cuda-quantum device_call runtime library at ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}")
      get_filename_component(_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR
        "${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}" DIRECTORY)
    else()
      message(STATUS "cuda-quantum device_call runtime library not found. "
                     "QEC lowered device_call host-dispatch service will not be built.")
    endif()

    # Gates the CUDA-Q-facing device_call pieces: the in-process
    # DeviceCallService shim in decoding-server-cqr and the caller-side
    # simulation-cqr wrappers.  The decoding server proper needs neither --
    # it reaches the HOST_CALL table through a plain-C accessor.
    set(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE FALSE)
    if(CUDAQ_DEVICE_CALL_INCLUDE_DIR AND CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
      set(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE TRUE)
    endif()
    
    # Create static device library for device code
    add_library(cudaq-qec-realtime-cudevice STATIC
      gpu_kernels.cu
    )
    
    set_target_properties(cudaq-qec-realtime-cudevice PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      POSITION_INDEPENDENT_CODE ON
    )
    
    # Handle both standalone (libs/qec) and full repo builds
    if(CUDAQX_QEC_STANDALONE_BUILD)
      set(_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
    else()
      set(_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
    endif()
    
    target_include_directories(cudaq-qec-realtime-cudevice PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${_core_include_dir}>
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    )
    
    target_link_libraries(cudaq-qec-realtime-cudevice PUBLIC CUDA::cudart)
    
    set_target_properties(cudaq-qec-realtime-cudevice PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    
    # NOTE: cudaq-qec-realtime-cudevice is a STATIC archive consumed only at
    # link time -- the decoding_server executable (and the realtime tests)
    # absorb its objects WHOLE_ARCHIVE during their own link.  It is not needed
    # at runtime, so it is deliberately NOT installed.
  else()
    message(WARNING "cuda-quantum realtime headers not found. "
                    "Device library cudaq-qec-realtime-device will not be built. "
                    "Set CUDAQ_REALTIME_ROOT to enable it.")
  endif()
endif()

set(CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE "" CACHE FILEPATH
    "Path to prebuilt proprietary realtime cudevice archive")
if(CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE)
  if(NOT EXISTS "${CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE}")
    message(FATAL_ERROR
      "CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE does not exist: "
      "${CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE}")
  endif()
  add_library(cudaq-qec-realtime-cudevice-proprietary STATIC IMPORTED GLOBAL)
  set_target_properties(cudaq-qec-realtime-cudevice-proprietary PROPERTIES
    IMPORTED_LOCATION "${CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE}")
endif()

# Shared library for host-side API. The cqr host-dispatch service (server) and
# the simulated-QPU client wrappers live in their own sibling libraries built
# from ./decoding-server-cqr and ./simulation-cqr (added below), not here.
add_library(cudaq-qec-realtime-decoding SHARED
  realtime_decoding.cpp
  config.cpp
)

target_compile_options(cudaq-qec-realtime-decoding 
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

target_include_directories(cudaq-qec-realtime-decoding
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<$<BOOL:${CUDAQ_REALTIME_INCLUDE_DIR}>:${CUDAQ_REALTIME_INCLUDE_DIR}>
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(CUDAQ_REALTIME_INCLUDE_DIR)
  target_compile_definitions(cudaq-qec-realtime-decoding PRIVATE
    CUDAQ_REALTIME_ROOT)
endif()

target_link_options(cudaq-qec-realtime-decoding PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding
  PUBLIC 
    cudaq-qec-decoders
  PRIVATE
    LLVMSupport
    CUDA::cudart
    $<$<BOOL:${CUDAQ_REALTIME_LIBRARY}>:${CUDAQ_REALTIME_LIBRARY}>
)

set_target_properties(cudaq-qec-realtime-decoding PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(_CUDAQ_REALTIME_LIB_DIR)
  set_target_properties(cudaq-qec-realtime-decoding PROPERTIES
    BUILD_RPATH "${_CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${_CUDAQ_REALTIME_LIB_DIR};${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )
endif()


install(TARGETS cudaq-qec-realtime-decoding
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------
# RealtimePipeline shared library
# Requires pre-installed cudaq-realtime (set CUDAQ_REALTIME_ROOT)
#
# EXPERIMENTAL / temporarily disabled: realtime_pipeline.cu relies on host
# dispatch API surface (skip_stream_sweep, caller-owned worker streams, and
# per-worker pre/post-launch hooks) that was removed upstream in CUDA-Q PR4770
# ("Split graph launch dispatch from the host ring loop").  Until it is ported
# to the new graph_launch_engine API, keep this target off by default so CI
# builds green.  Dependent unit tests (test_realtime_pipeline,
# test_realtime_predecoder_w_pymatching, gpu_roce bridge) auto-skip when this
# target does not exist.
# ---------------------------------------------------------------------------
option(CUDAQX_QEC_ENABLE_REALTIME_PIPELINE
  "Build the experimental cudaq-realtime-pipeline library (needs porting to \
the post-PR4770 CUDA-Q graph_launch_engine API)" OFF)

if(NOT CUDAQX_QEC_ENABLE_REALTIME_PIPELINE)
  message(STATUS "RealtimePipeline: skipping (experimental; disabled by "
                 "default -- set CUDAQX_QEC_ENABLE_REALTIME_PIPELINE=ON to build)")
elseif(CMAKE_CUDA_COMPILER AND CUDAQ_REALTIME_INCLUDE_DIR)
    cudaq_qec_find_realtime_library(_CUDAQ_RT_LIB
      NAMES cudaq-realtime
      PATH_SUFFIXES lib
    )
    cudaq_qec_find_realtime_library(_CUDAQ_RT_HD_LIB
      NAMES cudaq-realtime-host-dispatch
      PATH_SUFFIXES lib
    )

    if(_CUDAQ_RT_LIB AND _CUDAQ_RT_HD_LIB)
      message(STATUS "RealtimePipeline: building with CUDAQ_REALTIME_INCLUDE_DIR=${CUDAQ_REALTIME_INCLUDE_DIR}")

      add_library(cudaq-realtime-pipeline SHARED
        realtime_pipeline.cu
      )

      get_filename_component(_cuda_bin_pl "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_cuda_root_pl "${_cuda_bin_pl}" DIRECTORY)
      set(_cuda_cccl_include_pl "${_cuda_root_pl}/include/cccl")

      target_include_directories(cudaq-realtime-pipeline
        PUBLIC
          $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
          $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
          $<BUILD_INTERFACE:${_cuda_cccl_include_pl}>
          $<INSTALL_INTERFACE:include>
      )

      target_link_libraries(cudaq-realtime-pipeline
        PUBLIC CUDA::cudart_static
        PRIVATE ${_CUDAQ_RT_LIB} ${_CUDAQ_RT_HD_LIB}
      )

      option(ENABLE_NVTX "Enable NVTX profiling ranges" OFF)
      if(ENABLE_NVTX)
        target_compile_definitions(cudaq-realtime-pipeline PRIVATE ENABLE_NVTX)
        message(STATUS "NVTX profiling enabled for cudaq-realtime-pipeline")
      endif()


      get_filename_component(_CUDAQ_RT_LIB_DIR "${_CUDAQ_RT_LIB}" DIRECTORY)
      set_target_properties(cudaq-realtime-pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        BUILD_RPATH "${_CUDAQ_RT_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "${_CUDAQ_RT_LIB_DIR};$ORIGIN")

      install(TARGETS cudaq-realtime-pipeline
        COMPONENT qec-lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )
    else()
      message(STATUS "RealtimePipeline: skipping (cudaq-realtime or cudaq-realtime-host-dispatch not found)")
    endif()
else()
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "RealtimePipeline: skipping (CUDAQ_REALTIME_INCLUDE_DIR not set)")
  endif()
endif()

add_subdirectory(decoding-server-cqr)
# quantinuum/ and simulation/ are QPU-side device wrappers: they compile with
# nvq++ (cudaqx_add_device_code), so they need a full CUDA-Q install and are
# not part of a decoders-only build.  The decoding server is the service side
# and needs neither.
if(NOT CUDAQ_QEC_DECODERS_ONLY)
  add_subdirectory(quantinuum)
  add_subdirectory(simulation)
endif()
# simulation-cqr is the caller (QPU) side of the device_call ring: it links
# cudaq::cudaq-common and the device-call caller runtime, both from CUDA-Q
# proper, so it is not available in decoders-only builds.
if(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE AND NOT CUDAQ_QEC_DECODERS_ONLY)
  # cudaq-realtime (cqr) host-dispatch: simulated-QPU client that drives the
  # decoding server over cudaq-realtime.
  add_subdirectory(simulation-cqr)
endif()

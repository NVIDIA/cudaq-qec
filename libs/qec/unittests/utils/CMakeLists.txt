# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HSB FPGA syndrome-playback tool
# ==============================================================================
# Gated by CUDAQX_QEC_ENABLE_HSB_TOOLS.  Builds hsb_fpga_syndrome_playback,
# which requires a holoscan-sensor-bridge build (hololink_core).  The
# remaining discovery below (gpu_roce_transceiver, DOCA, cuda-quantum
# realtime) is optional; nothing in this directory fails without it.
# These are NOT CI tests -- they need FPGA hardware or an FPGA emulator.

if (NOT HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR must be set when building HSB tools.")
endif()
if (NOT HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR must be set when building HSB tools.")
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# --------------------------------------------------------------------------- #
# Find HSB core library
# --------------------------------------------------------------------------- #

find_library(HSB_CORE_LIB
  NAMES hololink_core
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/core"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT HSB_CORE_LIB)
  message(FATAL_ERROR
    "Could not find hololink_core library under ${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}.")
endif()

# --------------------------------------------------------------------------- #
# Find GPU RoCE Transceiver library (static, from holoscan-sensor-bridge)
# --------------------------------------------------------------------------- #

find_library(GPU_ROCE_TRANSCEIVER_LIB
  NAMES gpu_roce_transceiver
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators/gpu_roce_transceiver"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT GPU_ROCE_TRANSCEIVER_LIB)
  message(WARNING
    "gpu_roce_transceiver library not found (optional for this directory).")
endif()

# --------------------------------------------------------------------------- #
# Find transitive HSB libraries (required by gpu_roce_transceiver)
# --------------------------------------------------------------------------- #

find_library(HSB_COMMON_LIB
  NAMES hololink
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/common"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(BASE_RECEIVER_OP_LIB
  NAMES base_receiver_op
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(IBVERBS_LIB NAMES ibverbs)

# --------------------------------------------------------------------------- #
# Find DOCA libraries (required by gpu_roce_transceiver)
# --------------------------------------------------------------------------- #

set(DOCA_PATH "/opt/mellanox/doca")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)")
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib/x86_64-linux-gnu")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu")
else()
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib")
endif()

find_path(DOCA_INCLUDE_DIR doca_verbs.h
  PATHS ${DOCA_PATH}/include
  NO_DEFAULT_PATH)

find_library(DOCA_VERBS_LIB doca_verbs
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH)

find_library(DOCA_GPUNETIO_LIB doca_gpunetio
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH)

find_library(DOCA_COMMON_LIB doca_common
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH)

# --------------------------------------------------------------------------- #
# Find Holoscan (required by gpu_roce_transceiver -> holoscan::core)
# --------------------------------------------------------------------------- #

find_package(holoscan QUIET)

# --------------------------------------------------------------------------- #
# Find cuda-quantum realtime libraries
# --------------------------------------------------------------------------- #

set(_cudaq_realtime_prefixes "")
if(CUDAQ_REALTIME_ROOT)
  list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
  list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
endif()
if(CUDAQ_INSTALL_PREFIX)
  list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
endif()

find_path(CUDAQ_REALTIME_INCLUDE_DIR
  NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES include ../include
)

find_library(CUDAQ_REALTIME_LIBRARY
  NAMES cudaq-realtime
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
  NAMES cudaq-realtime-dispatch
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

find_library(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY
  NAMES cudaq-realtime-host-dispatch
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

find_library(CUDAQ_REALTIME_BRIDGE_GPU_ROCE_LIBRARY
  NAMES cudaq-realtime-bridge-gpu-roce
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

# =========================================================================== #
# hsb_fpga_syndrome_playback
# =========================================================================== #

add_executable(hsb_fpga_syndrome_playback
  hsb_fpga_syndrome_playback.cpp)

target_include_directories(hsb_fpga_syndrome_playback
  PRIVATE "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src"
  # cudaq/qec/realtime/decoder_rpc_wire_format.h (per-round RPC wire format).
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  # decoding_config.h uses the core heterogeneous map API in both build modes.
  PRIVATE $<TARGET_PROPERTY:cudaqx-core,INTERFACE_INCLUDE_DIRECTORIES>)

if (DEFINED CUDAQ_INCLUDE_DIR AND CUDAQ_INCLUDE_DIR)
  target_include_directories(hsb_fpga_syndrome_playback
    PRIVATE "${CUDAQ_INCLUDE_DIR}")
endif()

if (DEFINED CUDAQ_INSTALL_DIR AND CUDAQ_INSTALL_DIR)
  target_include_directories(hsb_fpga_syndrome_playback
    PRIVATE "${CUDAQ_INSTALL_DIR}/include")
endif()

if (CUDAQ_REALTIME_INCLUDE_DIR)
  target_include_directories(hsb_fpga_syndrome_playback
    PRIVATE "${CUDAQ_REALTIME_INCLUDE_DIR}")
endif()

target_include_directories(hsb_fpga_syndrome_playback
  PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(hsb_fpga_syndrome_playback
  PRIVATE
    cudaq-qec-realtime-decoding
    ${HSB_CORE_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    Threads::Threads)

set_target_properties(hsb_fpga_syndrome_playback PROPERTIES
  BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
  INSTALL_RPATH "$ORIGIN/../lib")

# Ship the FPGA syndrome-playback tool as a deliverable alongside decoding_server
# (see tools/decoding-server/CMakeLists.txt). This whole file is already gated by
# the HSB tools, so this install is a no-op in the vanilla release and is
# present only in the HSB-enabled deliverable image. The
# realtime_decoding_demo example resolves this tool from the install prefix.
install(TARGETS hsb_fpga_syndrome_playback
  COMPONENT qec-tools
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# gpu_roce_predecoder_bridge is now in unittests/realtime/

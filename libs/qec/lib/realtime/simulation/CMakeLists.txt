# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cudaq-qec-realtime-decoding-simulation SHARED
  simulation_host.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-simulation
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

cudaqx_add_device_code(cudaq-qec-realtime-decoding-simulation
  SOURCES
    simulation_device.cpp
)

target_include_directories(cudaq-qec-realtime-decoding-simulation
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# PRIVATE, not PUBLIC: PUBLIC propagates --exclude-libs,ALL to every consumer,
# including test executables that absorb libcudaq-realtime-dispatch.a and
# dlsym() its CUDAQ_REALTIME_DISPATCH_API symbols at runtime.
target_link_options(cudaq-qec-realtime-decoding-simulation PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-simulation
  PUBLIC 
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
)

set_target_properties(cudaq-qec-realtime-decoding-simulation PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")


install(TARGETS cudaq-qec-realtime-decoding-simulation
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


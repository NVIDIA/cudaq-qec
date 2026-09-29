# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cudaq-qec-realtime-decoding-quantinuum SHARED
  quantinuum_realtime_decoding.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-quantinuum 
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

cudaqx_add_device_code(cudaq-qec-realtime-decoding-quantinuum
  SOURCES
    quantinuum_device.cpp
)

target_include_directories(cudaq-qec-realtime-decoding-quantinuum
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# PRIVATE, not PUBLIC: PUBLIC propagates --exclude-libs,ALL to every consumer,
# including test executables that absorb libcudaq-realtime-dispatch.a and
# dlsym() its CUDAQ_REALTIME_DISPATCH_API symbols at runtime.
target_link_options(cudaq-qec-realtime-decoding-quantinuum PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-quantinuum
  PUBLIC 
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
  PRIVATE
    LLVMSupport
)

set_target_properties(cudaq-qec-realtime-decoding-quantinuum PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Allow the -quantinuum library to find -quantinuum-private library via dlopen
# if it exists in the same directory.
set_target_properties(cudaq-qec-realtime-decoding-quantinuum PROPERTIES
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS cudaq-qec-realtime-decoding-quantinuum
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Create a cudaq-stubs library for use in the Quantinuum GPU Server. This allows
# a decoupling of some CUDA-Q symbols that are not needed for the GPU Server.
add_library(cudaq-qec-realtime-decoding-cudaq-stubs SHARED
  cudaq_stubs.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-cudaq-stubs
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

set_target_properties(cudaq-qec-realtime-decoding-cudaq-stubs PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

install(TARGETS cudaq-qec-realtime-decoding-cudaq-stubs
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


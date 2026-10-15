# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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

# Fail the build if the Quantinuum library needs a CUDA-Q symbol the stubs do
# not export. Do not link libcudaq. --no-as-needed keeps both libraries in
# DT_NEEDED; --no-allow-shlib-undefined reports a missing stub. Not installed.
# ctest runs it so library startup executes the stubs it uses.
add_executable(cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck
  quantinuum_stubs_linkcheck.cpp)
target_link_libraries(cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck
  PRIVATE
    "-Wl,--no-as-needed"
    cudaq-qec-realtime-decoding-quantinuum
    cudaq-qec-realtime-decoding-cudaq-stubs
    "-Wl,--as-needed")
target_link_options(cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck
  PRIVATE
    "LINKER:--no-allow-shlib-undefined")
set_target_properties(
  cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
add_test(NAME cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck
  COMMAND cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck)
set_tests_properties(cudaq-qec-realtime-decoding-quantinuum-stubs-linkcheck
  PROPERTIES LABELS "qec")


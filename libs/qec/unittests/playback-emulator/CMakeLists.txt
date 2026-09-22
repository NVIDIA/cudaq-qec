# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT TARGET cudaq-qec-playback-emulator)
  return()
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:--param=evrp-mode=legacy>)
endif()
include(GoogleTest)

add_executable(test_playback_emulator
  test_parsing.cpp
  test_emulator.cpp
  test_sources.cpp
  test_sessions.cpp
)

target_include_directories(test_playback_emulator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/decoding-server-cqr
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/playback-emulator/lib
  # device/memory_circuit.h: test_sources.cpp's CudaqMemorySource tests call
  # the kernel directly to build their own reference trajectory (same
  # in-tree-only access test_qec_stim.cpp uses).
  ${CUDAQX_QEC_SOURCE_DIR}/lib
)

# test_sources.cpp's CudaqMemorySource tests run the actual memory_circuit
# kernel and need a concrete NVQIR backend statically linked in

target_link_directories(test_playback_emulator PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(test_playback_emulator PRIVATE
  GTest::gtest_main
  cudaq-qec
  cudaq::cudaq
  cudaq::cudaq-common
  cudaq-mlir-runtime
  nvqir
  nvqir-stim
  cudaq-qec-playback-emulator
  cudaq-qec-decoding-server
  cudaq-qec-decoders
  libstim
)

add_dependencies(CUDAQXQECUnitTests test_playback_emulator)
gtest_discover_tests(test_playback_emulator)

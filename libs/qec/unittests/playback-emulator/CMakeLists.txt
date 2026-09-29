# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

# The CpuRoceBackend tests need a Soft RoCE (or real RDMA) device to run, so
# they are only compiled when asked for with the QEC_TEST_PLAYBACK_CPU_ROCE
# environment variable at configure time.
# TODO: integrate the cpu_roce tests into CI.
if(DEFINED ENV{QEC_TEST_PLAYBACK_CPU_ROCE})
  get_target_property(_playback_defs cudaq-qec-playback-emulator
    INTERFACE_COMPILE_DEFINITIONS)
  if("CUDAQ_QEC_PLAYBACK_CPU_ROCE" IN_LIST _playback_defs)
    target_compile_definitions(test_playback_emulator PRIVATE
      QEC_TEST_PLAYBACK_CPU_ROCE)
  else()
    message(STATUS "test_playback_emulator: QEC_TEST_PLAYBACK_CPU_ROCE is set "
                   "but the cpu_roce backend was not built; skipping its tests.")
  endif()
endif()

add_dependencies(CUDAQXQECUnitTests test_playback_emulator)
gtest_discover_tests(test_playback_emulator)

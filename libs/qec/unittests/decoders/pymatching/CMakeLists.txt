# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

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

# ==============================================================================
add_compile_options(-Wno-attributes) 

add_executable(test_pymatching test_pymatching.cpp)
target_link_libraries(test_pymatching PRIVATE GTest::gtest_main cudaq-qec-decoders)
add_dependencies(CUDAQXQECUnitTests test_pymatching)
gtest_discover_tests(test_pymatching)

if(CUDAQ_REALTIME_INCLUDE_DIR AND TARGET cudaq-qec-realtime-decoding-server-cqr)
  add_executable(test_pymatching_device_call_realtime)
  set_target_properties(test_pymatching_device_call_realtime PROPERTIES
    LINKER_LANGUAGE CXX)

  target_include_directories(test_pymatching_device_call_realtime PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  cudaqx_add_device_code(test_pymatching_device_call_realtime
    SOURCES
      test_pymatching_device_call_realtime.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
  )

  target_link_directories(test_pymatching_device_call_realtime
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  # This test wires up both ends of the cqr host-dispatch path:
  #   - simulation-cqr: the simulated-QPU client whose device_calls are lowered
  #     (-frealtime-lowering) to the realtime frame ABI, rather than resolving to
  #     the plain -simulation library's direct trampolines (which would bypass
  #     the server).
  #   - decoding-server-cqr: the host-dispatch service that receives them (and
  #     provides the force-link + self-check hooks the test asserts on).
  # The cudaq runtime resolves the quantum platform at runtime via
  # dlopen(NULL) + dlsym("getQuantumPlatform") (common/PluginUtils.h), so
  # libcudaq-platform-default.so must already be loaded even though nothing
  # references its symbols at link time. GNU's default --as-needed would drop
  # it (and libcudaq-em-default.so) from DT_NEEDED, and the test then fails
  # with "undefined symbol: getQuantumPlatform"; keep them.
  target_link_options(test_pymatching_device_call_realtime PRIVATE
   $<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>)

  target_link_libraries(test_pymatching_device_call_realtime PRIVATE
    GTest::gtest_main
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation-cqr
    cudaq-qec-realtime-decoding-server-cqr
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    CUDA::cudart
    nvqir nvqir-stim
  )

  set_target_properties(test_pymatching_device_call_realtime PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )

  add_dependencies(test_pymatching_device_call_realtime cudaq-qec-pymatching)
  add_dependencies(CUDAQXQECUnitTests test_pymatching_device_call_realtime)
  # CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch routes the device_call over the
  # cudaq-realtime host-dispatch ring to the decoding server, which decodes on
  # the CPU host.
  gtest_discover_tests(test_pymatching_device_call_realtime
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES ENVIRONMENT "CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch")
endif()

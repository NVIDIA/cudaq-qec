# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Threads REQUIRED)

# General decoder benchmark: compares any registered plugin by name.
# EXCLUDE_FROM_ALL keeps it out of normal builds while leaving the target
# available independently of whether the unit-test suite is enabled.
add_executable(benchmark-qec-decoder EXCLUDE_FROM_ALL
    benchmark-qec-decoder.cpp
    benchmark_instance_pool.cpp)
target_include_directories(benchmark-qec-decoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark-qec-decoder PRIVATE
    cudaq-qec-decoders
    libstim
    Threads::Threads)
set_target_properties(benchmark-qec-decoder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/benchmarks/qec/decoder_latency"
    BUILD_RPATH
    "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins")

# Plugins are discovered through BUILD_RPATH. Build the directly supported
# comparison plugins before the benchmark.
add_dependencies(benchmark-qec-decoder cudaq-qec-pymatching)
if(TARGET cudaq-qec-nv-fusion-decoder)
    add_dependencies(benchmark-qec-decoder cudaq-qec-nv-fusion-decoder)
endif()

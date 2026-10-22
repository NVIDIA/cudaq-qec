# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The generator will use full-QEC circuit characterization. Do not add that
# dependency to the standalone decoders/server build. The first draft's pure
# specification/layout library is testable without a CUDA-Q backend.
if(CUDAQ_QEC_DECODERS_ONLY OR NOT TARGET cudaq-qec)
  message(STATUS "Skipping memory-experiment-generator: full QEC build required")
  return()
endif()

add_library(cudaq-qec-memory-experiment-spec STATIC experiment_spec.cpp)
target_compile_features(cudaq-qec-memory-experiment-spec PUBLIC cxx_std_20)
target_include_directories(cudaq-qec-memory-experiment-spec PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT TARGET cudaq-qec-realtime-decoding)
  message(STATUS "Skipping memory-experiment-generator CLI: config library unavailable")
  return()
endif()

add_library(cudaq-qec-memory-experiment-provider STATIC
  model_provider.cpp
  raw_replay.cpp
  bundle.cpp)
target_link_libraries(cudaq-qec-memory-experiment-provider PUBLIC
  cudaq-qec-memory-experiment-spec
  cudaq-qec
  cudaq-qec-realtime-decoding
  PRIVATE LLVMSupport)

add_executable(memory-experiment-generator memory_experiment_generator_main.cpp)
target_link_directories(memory-experiment-generator PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
target_link_libraries(memory-experiment-generator PRIVATE
  cudaq-qec-memory-experiment-provider
  cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime nvqir nvqir-stim)
set_target_properties(memory-experiment-generator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS memory-experiment-generator COMPONENT qec-tools
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

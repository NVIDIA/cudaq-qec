# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT TARGET cudaq-qec-memory-experiment-spec)
  return()
endif()

include(GoogleTest)
add_executable(test_memory_experiment_generator test_experiment_spec.cpp)
target_link_libraries(test_memory_experiment_generator PRIVATE
  GTest::gtest_main cudaq-qec-memory-experiment-spec)
add_dependencies(CUDAQXQECUnitTests test_memory_experiment_generator)
gtest_discover_tests(test_memory_experiment_generator PROPERTIES LABELS "qec")

if(TARGET memory-experiment-generator)
  add_executable(test_memory_model_provider test_model_provider.cpp test_raw_replay.cpp)
  target_link_directories(test_memory_model_provider PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
  target_link_libraries(test_memory_model_provider PRIVATE
    GTest::gtest_main cudaq-qec-memory-experiment-provider
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime nvqir nvqir-stim)
  add_dependencies(CUDAQXQECUnitTests test_memory_model_provider)
  gtest_discover_tests(test_memory_model_provider PROPERTIES LABELS "qec")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME memory_experiment_generator_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py
      --generator=$<TARGET_FILE:memory-experiment-generator>)
  set_tests_properties(memory_experiment_generator_cli PROPERTIES
    LABELS "qec"
    SKIP_RETURN_CODE 77
    TIMEOUT 300)
  add_dependencies(CUDAQXQECUnitTests memory-experiment-generator)
endif()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
  add_executable(test_memory_model_provider
    test_model_provider.cpp
    raw_replay.cpp
    test_raw_replay.cpp
    test_server_config.cpp)
  target_link_directories(test_memory_model_provider PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
  target_link_libraries(test_memory_model_provider PRIVATE
    GTest::gtest_main cudaq-qec-memory-experiment-provider
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime nvqir nvqir-stim)
  if(TARGET cudaq-qec-pymatching)
    add_dependencies(test_memory_model_provider cudaq-qec-pymatching)
  endif()
  add_dependencies(CUDAQXQECUnitTests test_memory_model_provider)
  gtest_discover_tests(test_memory_model_provider PROPERTIES LABELS "qec")
  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "import yaml"
      RESULT_VARIABLE memory_generator_yaml_status
      OUTPUT_QUIET ERROR_QUIET)
    if(memory_generator_yaml_status STREQUAL "0")
      message(STATUS "Generator CLI tests: PyYAML available in ${Python_EXECUTABLE}")
    else()
      # CI installs requirements after configuration; absence here is not fatal.
      message(STATUS "Generator CLI tests: PyYAML missing in ${Python_EXECUTABLE}; install test requirements before running")
    endif()
    set(memory_generator_cli_args
      --generator=$<TARGET_FILE:memory-experiment-generator>)
    if(TARGET decoding_server)
      list(APPEND memory_generator_cli_args
        --decoding-server=$<TARGET_FILE:decoding_server>)
      add_dependencies(CUDAQXQECUnitTests decoding_server)
    endif()
    add_test(NAME memory_experiment_generator_cli
      COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py" -v
        ${memory_generator_cli_args})
    set_tests_properties(memory_experiment_generator_cli PROPERTIES
      LABELS "qec"
      SKIP_RETURN_CODE 77
      TIMEOUT 300)
  endif()
  add_dependencies(CUDAQXQECUnitTests memory-experiment-generator)
endif()

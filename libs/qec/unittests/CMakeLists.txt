# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

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

find_package(CUDAToolkit REQUIRED)

# ==============================================================================
add_compile_options(-Wno-attributes) 

add_executable(test_decoders test_decoders.cpp decoders/sample_decoder.cpp)
target_include_directories(test_decoders PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
target_link_libraries(test_decoders PRIVATE GTest::gtest_main cudaq-qec-decoders libstim CUDA::cudart)
add_dependencies(CUDAQXQECUnitTests test_decoders)
gtest_discover_tests(test_decoders)

# These two tests need cudaq-qec-decoding-server, which is only built when a
# realtime-enabled CUDA-Q install is found (see
# lib/realtime/decoding-server-cqr/CMakeLists.txt).
if(TARGET cudaq-qec-decoding-server)
  add_executable(test_decoders_yaml test_decoders_yaml.cpp decoders/sample_decoder.cpp)
  target_include_directories(test_decoders_yaml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/realtime/decoding-server-cqr)
  target_link_libraries(test_decoders_yaml PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq-qec-decoding-server
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq)
  # Chromobius is the DEM-native decoder the raw-DEM model source exists for;
  # link it when this configuration built it so the acceptance tests can run,
  # and keep main's feature definition for its conditional cases.
  if(TARGET cudaq-qec-chromobius)
    add_dependencies(test_decoders_yaml cudaq-qec-chromobius)
    target_link_libraries(test_decoders_yaml PRIVATE cudaq-qec-chromobius)
    target_compile_definitions(test_decoders_yaml PRIVATE CUDAQX_QEC_HAS_CHROMOBIUS)
  endif()
  add_dependencies(CUDAQXQECUnitTests test_decoders_yaml)
  gtest_discover_tests(test_decoders_yaml)

  add_executable(test_decoding_server_core test_decoding_server_core.cpp)
  target_link_libraries(test_decoding_server_core PRIVATE
    GTest::gtest_main
    cudaq-qec-decoding-server
    cudaq::cudaq
    CUDA::cudart)
  add_dependencies(CUDAQXQECUnitTests test_decoding_server_core)
  gtest_discover_tests(test_decoding_server_core)
else()
  message(STATUS "Skipping test_decoders_yaml / test_decoding_server_core "
                 "(cudaq-qec-decoding-server not built)")
endif()

add_executable(test_qec test_qec.cpp)
target_link_libraries(test_qec PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq-stim-target)
add_dependencies(CUDAQXQECUnitTests test_qec)
gtest_discover_tests(test_qec)

add_executable(test_dem_sampling test_dem_sampling.cpp)
target_link_libraries(test_dem_sampling PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq)
target_link_libraries(test_dem_sampling PRIVATE CUDA::cudart)
target_include_directories(test_dem_sampling PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
add_dependencies(CUDAQXQECUnitTests test_dem_sampling)
gtest_discover_tests(test_dem_sampling)

add_executable(test_sparse_binary_matrix test_sparse_binary_matrix.cpp)
target_link_libraries(test_sparse_binary_matrix PRIVATE GTest::gtest_main
                                               cudaq-qec-decoders)
add_dependencies(CUDAQXQECUnitTests test_sparse_binary_matrix)
gtest_discover_tests(test_sparse_binary_matrix)

add_executable(test_extended_dem test_extended_dem.cpp)
target_link_libraries(test_extended_dem PRIVATE GTest::gtest_main
                                        cudaq-qec-decoders)
add_dependencies(CUDAQXQECUnitTests test_extended_dem)
gtest_discover_tests(test_extended_dem)

add_executable(test_dem_construction test_dem_construction.cpp)
target_link_libraries(test_dem_construction PRIVATE GTest::gtest_main
                                            cudaq-qec-decoders)
add_dependencies(CUDAQXQECUnitTests test_dem_construction)
gtest_discover_tests(test_dem_construction)

add_executable(test_dem_construction_code test_dem_construction_code.cpp)
target_link_libraries(test_dem_construction_code PRIVATE GTest::gtest_main
                                                  cudaq-qec
                                                  cudaq::cudaq-stim-target)
add_dependencies(CUDAQXQECUnitTests test_dem_construction_code)
gtest_discover_tests(test_dem_construction_code)

add_executable(test_logger test_logger.cpp)
target_link_libraries(test_logger PRIVATE GTest::gtest_main cudaq-qec-decoders)
add_dependencies(CUDAQXQECUnitTests test_logger)
gtest_discover_tests(test_logger)

# TensorRT decoder test is only built for x86 architectures
if(CUDAQ_QEC_TRT_DECODER_ENABLED AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
  add_executable(test_trt_decoder ./decoders/trt_decoder/test_trt_decoder.cpp)

  # Find TensorRT for the test
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS
          ${TENSORRT_ROOT}/include
          /usr/include/x86_64-linux-gnu
          /usr/local/cuda/include
          /usr/local/tensorrt/include
          /opt/tensorrt/include
      NO_DEFAULT_PATH
  )

  # test_trt_decoder.cpp calls TensorRT APIs directly (nvinfer1::createInferBuilder,
  # nvinfer1::ILogger), so the test must link libnvinfer itself. Relying on the
  # transitive dependency through cudaq-qec-trt-decoder fails at link time with
  # "DSO missing from command line".
  find_library(TENSORRT_LIBRARY nvinfer
      PATHS
          ${TENSORRT_ROOT}/lib
          /usr/lib/x86_64-linux-gnu
          /usr/lib/aarch64-linux-gnu
          /usr/local/cuda/lib64
          /usr/local/tensorrt/lib
          /opt/tensorrt/lib
      NO_DEFAULT_PATH
  )

  target_include_directories(test_trt_decoder PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
      ${TENSORRT_INCLUDE_DIR}
  )

  target_link_libraries(test_trt_decoder PRIVATE GTest::gtest_main cudaq-qec-decoders cudaq-qec-trt-decoder CUDA::cudart ${TENSORRT_LIBRARY})
  # Chromobius is the DEM-native child the raw-DEM nesting case needs.
  if(TARGET cudaq-qec-chromobius)
    target_link_libraries(test_trt_decoder PRIVATE cudaq-qec-chromobius)
  endif()

  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import onnx"
      RESULT_VARIABLE _trt_onnx_python_result
      OUTPUT_QUIET
      ERROR_QUIET)
    if(_trt_onnx_python_result EQUAL 0)
      set(_trt_generated_onnx_dir "${CMAKE_CURRENT_BINARY_DIR}/trt_decoder_onnx")
      set(_trt_dynamic_identity_onnx "${_trt_generated_onnx_dir}/trt_dynamic_identity.onnx")
      set(_trt_uint8_identity_onnx "${_trt_generated_onnx_dir}/trt_uint8_identity.onnx")
      set(_trt_uint8_to_float_onnx "${_trt_generated_onnx_dir}/trt_uint8_to_float.onnx")
      add_custom_command(
        OUTPUT
          "${_trt_dynamic_identity_onnx}"
          "${_trt_uint8_identity_onnx}"
          "${_trt_uint8_to_float_onnx}"
        COMMAND ${Python_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/decoders/trt_decoder/generate_trt_decoder_onnx.py"
          --out-dir "${_trt_generated_onnx_dir}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/decoders/trt_decoder/generate_trt_decoder_onnx.py"
        COMMENT "Generating TensorRT decoder ONNX test fixtures")
      add_custom_target(generate_trt_decoder_onnx
        DEPENDS
          "${_trt_dynamic_identity_onnx}"
          "${_trt_uint8_identity_onnx}"
          "${_trt_uint8_to_float_onnx}")
      add_dependencies(test_trt_decoder generate_trt_decoder_onnx)
      target_compile_definitions(test_trt_decoder PRIVATE
        TRT_TEST_DYNAMIC_ONNX_PATH="${_trt_dynamic_identity_onnx}"
        TRT_TEST_UINT8_ONNX_PATH="${_trt_uint8_identity_onnx}"
        TRT_TEST_UINT8_TO_FLOAT_ONNX_PATH="${_trt_uint8_to_float_onnx}")
    else()
      message(WARNING "Python package onnx not found. "
                      "Skipping generated TensorRT decoder ONNX fixtures.")
    endif()
  endif()

  # Path to ONNX test asset at repo root (absolute so test finds it regardless of CWD)
  target_compile_definitions(test_trt_decoder PRIVATE
    TRT_TEST_ONNX_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/tests/surface_code_decoder.onnx")
  add_dependencies(CUDAQXQECUnitTests test_trt_decoder)
  gtest_discover_tests(test_trt_decoder)
endif()

# ==============================================================================

# GPU kernel tests for realtime decoder infrastructure
if(CUDAQ_REALTIME_ROOT AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  add_executable(test_gpu_kernels 
    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/test_gpu_kernels.cu
  )
  
  set_target_properties(test_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
  )
  
  target_include_directories(test_gpu_kernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
  )
  
  # Define the test data directory path
  target_compile_definitions(test_gpu_kernels PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/data"
  )
  
  target_link_libraries(test_gpu_kernels PRIVATE 
    GTest::gtest_main 
    CUDA::cudart
    cudaq-qec-realtime-cudevice
  )
  
  add_dependencies(CUDAQXQECUnitTests test_gpu_kernels)
  gtest_discover_tests(test_gpu_kernels)
  
  # Realtime decoding end-to-end test with mock decoder
  # This test integrates with cuda-quantum's host API (libcudaq-realtime.so)

  # Optional hint for cuda-quantum install prefix
  set(CUDAQ_REALTIME_ROOT "" CACHE PATH "Prefix to cuda-quantum install")

  set(_cudaq_realtime_prefixes "")
  if(CUDAQ_REALTIME_ROOT)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
  endif()
  if(CUDAQ_INSTALL_PREFIX)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
  endif()

  # Realtime API lives under install prefix (CUDAQ_REALTIME_ROOT = install directory).
  # Header layout: include/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
  find_path(CUDAQ_REALTIME_INCLUDE_DIR
    NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
    HINTS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
  )
  if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
    find_path(CUDAQ_REALTIME_INCLUDE_DIR
      NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
      PATHS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES include
    )
  endif()
  if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
    find_path(CUDAQ_REALTIME_INCLUDE_DIR
      NAMES cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h
      HINTS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES include ../include
    )
  endif()

  find_library(CUDAQ_REALTIME_LIBRARY
    NAMES cudaq-realtime
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES lib
  )

  find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
    NAMES cudaq-realtime-dispatch
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES lib
  )

  # libcudaq-realtime.so has a DT_NEEDED for libcudaq-realtime-host-dispatch.so.
  # With DT_RUNPATH (CMake default), the executable's RUNPATH is not inherited
  # by transitive dependencies. Link it directly so the executable's own RUNPATH
  # resolves it at runtime.
  find_library(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY
    NAMES cudaq-realtime-host-dispatch
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES lib
  )

  set(_have_realtime_for_tests FALSE)
  if(CUDAQ_REALTIME_INCLUDE_DIR AND CUDAQ_REALTIME_LIBRARY AND CUDAQ_REALTIME_DISPATCH_LIBRARY)
    set(_have_realtime_for_tests TRUE)
    message(STATUS "Found cuda-quantum realtime headers at ${CUDAQ_REALTIME_INCLUDE_DIR}")
    message(STATUS "Found cuda-quantum realtime library at ${CUDAQ_REALTIME_LIBRARY}")
    message(STATUS "Found cuda-quantum realtime dispatch library at ${CUDAQ_REALTIME_DISPATCH_LIBRARY}")
  endif()
  if(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY)
    message(STATUS "Found cuda-quantum realtime host dispatch library at ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}")
  endif()

  if(_have_realtime_for_tests)
    add_executable(test_realtime_decoding 
      ${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/test_realtime_decoding.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/realtime/mock_decode_handler.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/realtime/mock_decoder_launch_params.cpp
    )

    # TODO: Re-enable once libcudaq-realtime-host-dispatch.so RPATH is resolved
    # add_executable(test_realtime_decoding 
    #   ${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/test_realtime_decoding.cu
    # )
    #
    set_target_properties(test_realtime_decoding PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_STANDARD 17
    )
    #
    # target_include_directories(test_realtime_decoding PRIVATE
    #   ${CUDAToolkit_INCLUDE_DIRS}
    #   ${CMAKE_CURRENT_SOURCE_DIR}/../include
    #   ${CMAKE_SOURCE_DIR}/libs/core/include
    #   ${CUDAQ_REALTIME_INCLUDE_DIR}
    # )
    #
    # target_compile_definitions(test_realtime_decoding PRIVATE
    #   TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/data"
    # )
    #
    # target_link_libraries(test_realtime_decoding PRIVATE 
    #   GTest::gtest_main 
    #   CUDA::cudart
    #   cudaq-qec-realtime-cudevice
    #   ${CUDAQ_REALTIME_LIBRARY}
    #   ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
    # )
    #
    # get_filename_component(CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
    # set_target_properties(test_realtime_decoding PROPERTIES
    #   BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR}"
    #   INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR}"
    # )
    #
    # add_dependencies(CUDAQXQECUnitTests test_realtime_decoding)
    # gtest_discover_tests(test_realtime_decoding
    #   TEST_PREFIX "test_realtime_decoding."
    # )

    target_include_directories(test_realtime_decoding PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
      ${CMAKE_CURRENT_SOURCE_DIR}/realtime
      ${CMAKE_SOURCE_DIR}/libs/core/include
      ${CUDAQ_REALTIME_INCLUDE_DIR}
    )

    target_compile_definitions(test_realtime_decoding PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/data"
    )

    target_link_libraries(test_realtime_decoding PRIVATE 
      GTest::gtest_main 
      CUDA::cudart
      cudaq-qec-realtime-cudevice
      ${CUDAQ_REALTIME_LIBRARY}
      ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
      $<$<BOOL:${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}>:${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}>
    )

    # Ensure runtime can locate libcudaq-realtime.so
    get_filename_component(CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)

    # ----------------------------------------------------------------
    # Realtime pipeline unit tests (SKIP_TRT passthrough at runtime;
    # still needs TRT headers+libs at compile/link time)
    # ----------------------------------------------------------------
    find_path(TENSORRT_INCLUDE_DIR_FOR_PIPELINE NvInfer.h
      PATHS
        ${TENSORRT_ROOT}/include
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu
        /usr/local/cuda/include
        /usr/local/tensorrt/include
        /opt/tensorrt/include
      NO_DEFAULT_PATH
    )
    find_library(TENSORRT_LIBRARY_FOR_PIPELINE nvinfer
      PATHS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/tensorrt/lib /opt/tensorrt/lib
    )
    find_library(TENSORRT_ONNX_PARSER_FOR_PIPELINE nvonnxparser
      PATHS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/tensorrt/lib /opt/tensorrt/lib
    )

    if(TENSORRT_INCLUDE_DIR_FOR_PIPELINE AND TENSORRT_LIBRARY_FOR_PIPELINE AND TENSORRT_ONNX_PARSER_FOR_PIPELINE
       AND TARGET cudaq-realtime-pipeline)
      get_filename_component(_cuda_bin_pipe "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_cuda_root_pipe "${_cuda_bin_pipe}" DIRECTORY)
      set(_cuda_cccl_include_pipe "${_cuda_root_pipe}/include/cccl")

      add_executable(test_realtime_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/realtime/ai_decoder_service.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/realtime/ai_predecoder_service.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/test_realtime_pipeline.cu
      )

      set_target_properties(test_realtime_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        LINKER_LANGUAGE CUDA
      )

      target_include_directories(test_realtime_pipeline PRIVATE
        ${_cuda_cccl_include_pipe}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR_FOR_PIPELINE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/libs/core/include
        ${CUDAQ_REALTIME_INCLUDE_DIR}
      )

      target_link_libraries(test_realtime_pipeline PRIVATE
        GTest::gtest_main
        CUDA::cudart
        ${TENSORRT_LIBRARY_FOR_PIPELINE}
        ${TENSORRT_ONNX_PARSER_FOR_PIPELINE}
        ${CUDAQ_REALTIME_LIBRARY}
        ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
        ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
        cudaq-realtime-pipeline
      )
      set_target_properties(test_realtime_pipeline PROPERTIES
        BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
      )

      if(ENABLE_NVTX)
        target_compile_definitions(test_realtime_pipeline PRIVATE ENABLE_NVTX)
      endif()

      add_dependencies(CUDAQXQECUnitTests test_realtime_pipeline)
      gtest_discover_tests(test_realtime_pipeline
        TEST_PREFIX "test_realtime_pipeline."
      )

      set(_ai_decoder_int8_onnx
        "${CMAKE_CURRENT_SOURCE_DIR}/realtime/data/qdq_int8.onnx")
      set(_ai_decoder_fp8_onnx
        "${CMAKE_CURRENT_SOURCE_DIR}/realtime/data/qdq_fp8.onnx")

      add_executable(test_ai_decoder_quantized_onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/realtime/ai_decoder_service.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/test_ai_decoder_quantized_onnx.cpp
      )

      set_target_properties(test_ai_decoder_quantized_onnx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        LINKER_LANGUAGE CUDA
      )

        target_compile_definitions(test_ai_decoder_quantized_onnx PRIVATE
          INT8_QDQ_ONNX_PATH="${_ai_decoder_int8_onnx}"
          FP8_QDQ_ONNX_PATH="${_ai_decoder_fp8_onnx}"
        )

        target_include_directories(test_ai_decoder_quantized_onnx PRIVATE
          ${_cuda_cccl_include_pipe}
          ${CUDAToolkit_INCLUDE_DIRS}
          ${TENSORRT_INCLUDE_DIR_FOR_PIPELINE}
          ${CMAKE_CURRENT_SOURCE_DIR}/../include
          ${CMAKE_SOURCE_DIR}/libs/core/include
          ${CUDAQ_REALTIME_INCLUDE_DIR}
        )

        target_link_libraries(test_ai_decoder_quantized_onnx PRIVATE
          GTest::gtest_main
          CUDA::cudart
          ${TENSORRT_LIBRARY_FOR_PIPELINE}
          ${TENSORRT_ONNX_PARSER_FOR_PIPELINE}
          ${CUDAQ_REALTIME_LIBRARY}
          ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
          ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
          cudaq-realtime-pipeline
        )
        set_target_properties(test_ai_decoder_quantized_onnx PROPERTIES
          BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
          INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
        )
        add_dependencies(CUDAQXQECUnitTests test_ai_decoder_quantized_onnx)
        gtest_discover_tests(test_ai_decoder_quantized_onnx
          TEST_PREFIX "test_ai_decoder_quantized_onnx.")
    else()
      message(WARNING "TensorRT not found. Skipping test_realtime_pipeline (needs NvInfer.h + TRT libs for compile/link).")
    endif()

    # Hybrid AI predecoder + PyMatching pipeline test
    # Requires TensorRT + ONNX parser for building engines from ONNX models
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS
        ${TENSORRT_ROOT}/include
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu
        /usr/local/cuda/include
        /usr/local/tensorrt/include
        /opt/tensorrt/include
      NO_DEFAULT_PATH
    )
    find_library(TENSORRT_LIBRARY nvinfer
      PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    )
    find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
      PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    )

    # test_realtime_predecoder_w_pymatching is now in unittests/realtime/

  else()
    message(WARNING "cuda-quantum realtime dependency not found. "
                    "Set CUDAQ_REALTIME_ROOT to enable "
                    "test_realtime_pipeline and test_realtime_predecoder_w_pymatching.")
  endif()
endif()

# ==============================================================================
# Dependency-closure guardrail
# ------------------------------------------------------------------------------
# The realtime decoding, decoder, decoder-plugin and realtime-pipeline shared
# libraries are deliberately kept free of the heavy CUDA-Q runtime libraries
# (libcudaq-common / libcudaq-operator / libcudaq-qec / libcudaq / libnvqir /
# libcudaq-mlir-runtime).  These CTest checks inspect the built ELF dependency
# closure (via readelf, no loader involved) and fail if any of those libraries
# leaks in.
# ==============================================================================
if(TARGET cudaq-qec-decoders AND TARGET cudaq-qec-realtime-decoding)
  set(_dep_closure_script
    "${CMAKE_CURRENT_SOURCE_DIR}/check_dependency_closure.cmake")
  # Heavy CUDA-Q libraries that must never appear in the realtime/decoder
  # dependency closure.  libcudaq (core runtime) and libnvqir (simulator
  # dispatch) are the roots of the quantum-runtime subtree, so blocking them
  # transitively covers their runtime-loaded plugins as well.  Matching is
  # anchored (^<stem>\.so), so e.g. "libcudaq" flags libcudaq.so but never
  # libcudaq-realtime.so / libcudaq-qec-decoders.so.
  set(_dep_closure_forbidden
    "libcudaq-common|libcudaq-operator|libcudaq-qec|libcudaq|libnvqir|libcudaq-mlir-runtime|libcustabilizer")

  # Always-present core libraries under guard, plus every decoder plugin and
  # the realtime pipeline that were actually built in this configuration.  The
  # plugins/pipeline are conditional (TensorRT / CUDA / realtime availability),
  # so they are only added when their target exists.
  set(_dep_closure_targets cudaq-qec-decoders cudaq-qec-realtime-decoding)
  foreach(_t IN ITEMS
      cudaq-qec-example
      cudaq-qec-pymatching
      cudaq-qec-chromobius
      cudaq-qec-trt-decoder
      cudaq-realtime-pipeline
      cudaq-qec-realtime-decoding
      cudaq-qec-realtime-decoding-simulation
      cudaq-qec-realtime-decoding-quantinuum)
    if(TARGET ${_t})
      list(APPEND _dep_closure_targets ${_t})
    endif()
  endforeach()

  # Build the "|"-separated $<TARGET_FILE:...> list consumed by the script.
  set(_dep_closure_libs "")
  foreach(_t IN LISTS _dep_closure_targets)
    if(_dep_closure_libs)
      string(APPEND _dep_closure_libs "|")
    endif()
    string(APPEND _dep_closure_libs "$<TARGET_FILE:${_t}>")
  endforeach()

  # Decoder plugins live in lib/decoder-plugins, everything else in lib; include
  # both so transitive sonames resolve during the closure walk.
  set(_dep_closure_search
    "${CMAKE_BINARY_DIR}/lib|${CMAKE_BINARY_DIR}/lib/decoder-plugins")

  # Make sure the libraries under test are built before CTest runs.
  add_dependencies(CUDAQXQECUnitTests ${_dep_closure_targets})

  add_test(NAME qec_dependency_closure
    COMMAND ${CMAKE_COMMAND}
      "-DLIBS=${_dep_closure_libs}"
      "-DFORBIDDEN=${_dep_closure_forbidden}"
      "-DTRANSITIVE=ON"
      "-DEXTRA_SEARCH_DIRS=${_dep_closure_search}"
      -P "${_dep_closure_script}")
  set_tests_properties(qec_dependency_closure PROPERTIES LABELS "qec")

  # Negative self-test: libcudaq-qec.so legitimately depends on the forbidden
  # libraries, so the guardrail MUST flag it.  This proves the check is not
  # silently passing due to a parsing or resolution bug.
  if(TARGET cudaq-qec)
    add_test(NAME qec_dependency_closure_selftest
      COMMAND ${CMAKE_COMMAND}
        "-DLIBS=$<TARGET_FILE:cudaq-qec>"
        "-DFORBIDDEN=${_dep_closure_forbidden}"
        "-DTRANSITIVE=ON"
        "-DEXTRA_SEARCH_DIRS=${_dep_closure_search}"
        "-DEXPECT_FAIL=ON"
        -P "${_dep_closure_script}")
    set_tests_properties(qec_dependency_closure_selftest PROPERTIES LABELS "qec")
    add_dependencies(CUDAQXQECUnitTests cudaq-qec)
  endif()
endif()

# ==============================================================================
# Architectural guardrail: the realtime decoding server deliverables must stay
# free of the *heavy* CUDA-Q runtime: the core runtime (libcudaq), the simulator
# dispatch (libnvqir), the QEC umbrella library (libcudaq-qec), and the MLIR JIT
# runtime (libcudaq-mlir-runtime).  Pulling any of those back in would drag the
# whole CUDA-Q execution stack (quantum_platform, execution manager, JIT
# compiler) into the server package.
#
# Covered here (each only when built):
#   - cudaq-qec-realtime-decoding-server-cqr / -simulation-cqr (the cudaq-realtime
#     host-dispatch server + simulated-QPU client; built only when the cudaq
#     device_call service is available)
#
# Unlike the decoder/realtime closure check above, the server libraries DO
# legitimately depend on libcudaq-common / libcudaq-operator (they provide the
# nvq++ kernel-registration symbols baked into the device code) and on the
# allowed libcudaq-realtime, so those are intentionally NOT forbidden here.  This
# guardrail is created whenever a server target is built.
# ==============================================================================
# Candidate server targets, each guarded so the check adapts to what was built.
set(_srv_closure_targets "")
foreach(_t IN ITEMS
    cudaq-qec-realtime-decoding-server-cqr
    cudaq-qec-realtime-decoding-simulation-cqr
    )
  if(TARGET ${_t})
    list(APPEND _srv_closure_targets ${_t})
  endif()
endforeach()

if(_srv_closure_targets)
  if(NOT DEFINED _dep_closure_script)
    set(_dep_closure_script
      "${CMAKE_CURRENT_SOURCE_DIR}/check_dependency_closure.cmake")
  endif()

  # Heavy CUDA-Q runtime roots the server must never acquire.  Matching is
  # anchored (^<stem>\.so), so "libcudaq" flags libcudaq.so but never
  # libcudaq-common.so / libcudaq-realtime.so / libcudaq-qec-decoders.so,
  # "libcudaq-qec" flags libcudaq-qec.so but never libcudaq-qec-decoders.so, and
  # "libcustabilizer" flags libcustabilizer.so.0 (cuStabilizer, pulled only by
  # libcudaq-qec).
  set(_srv_closure_forbidden
    "libcudaq-qec|libcudaq|libnvqir|libcudaq-mlir-runtime|libcustabilizer")

  set(_srv_closure_libs "")
  foreach(_t IN LISTS _srv_closure_targets)
    if(_srv_closure_libs)
      string(APPEND _srv_closure_libs "|")
    endif()
    string(APPEND _srv_closure_libs "$<TARGET_FILE:${_t}>")
  endforeach()

  # Search path for the transitive soname walk: build tree first, then the
  # CUDA-Q install and realtime prefixes so intentionally-allowed deps
  # (common/operator/realtime) resolve and their subtrees are inspected too.
  # The server libraries only DT_NEEDED libs in lib/ (decoder plugins are
  # dlopen'd, never linked), so lib/decoder-plugins is not needed here.
  set(_srv_closure_search "${CMAKE_BINARY_DIR}/lib")
  if(DEFINED CUDAQ_INSTALL_DIR AND CUDAQ_INSTALL_DIR)
    string(APPEND _srv_closure_search "|${CUDAQ_INSTALL_DIR}/lib")
  endif()
  if(DEFINED CUDAQ_REALTIME_ROOT AND CUDAQ_REALTIME_ROOT)
    string(APPEND _srv_closure_search "|${CUDAQ_REALTIME_ROOT}/lib")
  endif()

  add_dependencies(CUDAQXQECUnitTests ${_srv_closure_targets})

  add_test(NAME qec_realtime_server_dependency_closure
    COMMAND ${CMAKE_COMMAND}
      "-DLIBS=${_srv_closure_libs}"
      "-DFORBIDDEN=${_srv_closure_forbidden}"
      "-DTRANSITIVE=ON"
      "-DEXTRA_SEARCH_DIRS=${_srv_closure_search}"
      -P "${_dep_closure_script}")
  set_tests_properties(qec_realtime_server_dependency_closure
    PROPERTIES LABELS "qec")

  # Negative self-test: the server libraries DO legitimately depend on
  # libcudaq-common, so forbidding it MUST fire.  This proves the guardrail is
  # actually inspecting the server closure (and not silently passing).
  add_test(NAME qec_realtime_server_dependency_closure_selftest
    COMMAND ${CMAKE_COMMAND}
      "-DLIBS=${_srv_closure_libs}"
      "-DFORBIDDEN=libcudaq-common"
      "-DTRANSITIVE=ON"
      "-DEXTRA_SEARCH_DIRS=${_srv_closure_search}"
      "-DEXPECT_FAIL=ON"
      -P "${_dep_closure_script}")
  set_tests_properties(qec_realtime_server_dependency_closure_selftest
    PROPERTIES LABELS "qec")
endif()

# ==============================================================================

add_subdirectory(backend-specific)
add_subdirectory(realtime)
add_subdirectory(decoders/pymatching)
add_subdirectory(decoders/chromobius)
add_subdirectory(playback-emulator)

# HSB rig tools (FPGA playback + graph-launched QLDPC bridge)
if(CUDAQX_QEC_ENABLE_HSB_TOOLS)
  add_subdirectory(utils)
endif()

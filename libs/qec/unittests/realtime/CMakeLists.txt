# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(app_examples)

# =========================================================================== #
# test_realtime_predecoder_w_pymatching
# =========================================================================== #
# Software-only benchmark: AI predecoder (TRT) + PyMatching via
# realtime_pipeline with ring_buffer_injector.

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY
   AND TARGET cudaq-realtime-pipeline)

  get_filename_component(_cuda_bin_pred "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_cuda_root_pred "${_cuda_bin_pred}" DIRECTORY)
  set(_cuda_cccl_include_pred "${_cuda_root_pred}/include/cccl")

  add_executable(test_realtime_predecoder_w_pymatching
    test_realtime_predecoder_w_pymatching.cpp
    predecoder_pipeline_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_decoder_service.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_predecoder_service.cu
  )

  set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    LINKER_LANGUAGE CUDA
  )

  target_compile_definitions(test_realtime_predecoder_w_pymatching PRIVATE
    ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime"
  )

  target_include_directories(test_realtime_predecoder_w_pymatching PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_cuda_cccl_include_pred}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_link_libraries(test_realtime_predecoder_w_pymatching PRIVATE
    CUDA::cudart
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    ${CUDAQ_REALTIME_LIBRARY}
    ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
    ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
    cudaq-realtime-pipeline
    cudaq-qec
    cudaq::cudaq
  )

  get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${_cudaq_rt_lib_dir}"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib;${_cudaq_rt_lib_dir}"
  )

  if(ENABLE_NVTX)
    target_compile_definitions(test_realtime_predecoder_w_pymatching PRIVATE ENABLE_NVTX)
  endif()

else()
  message(WARNING "TensorRT or cudaq-realtime-pipeline not found. "
                  "Skipping test_realtime_predecoder_w_pymatching.")
endif()

# =========================================================================== #
# test_trt_decoder_composite
# =========================================================================== #
# Software-only demo: composite TensorRT decoder plugin with PyMatching as the
# global decoder.  Uses the same Stim-exported detector, observable, H, O, and
# priors files as test_realtime_predecoder_w_pymatching, but exercises the
# trt_decoder plugin's built-in [pre_L, residual] -> global_decoder composition.

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY
   AND CUDAQ_REALTIME_INCLUDE_DIR
   AND TARGET cudaq-qec-trt-decoder)

  get_filename_component(_cuda_bin_trt_comp "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_cuda_root_trt_comp "${_cuda_bin_trt_comp}" DIRECTORY)
  set(_cuda_cccl_include_trt_comp "${_cuda_root_trt_comp}/include/cccl")

  add_executable(test_trt_decoder_composite
    test_trt_decoder_composite.cpp
    predecoder_pipeline_common.cpp
  )

  set_target_properties(test_trt_decoder_composite PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  target_compile_definitions(test_trt_decoder_composite PRIVATE
    ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime"
  )

  target_include_directories(test_trt_decoder_composite PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_cuda_cccl_include_trt_comp}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_link_libraries(test_trt_decoder_composite PRIVATE
    CUDA::cudart
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-trt-decoder
    cudaq::cudaq
  )

  if(TARGET cudaq-qec-pymatching)
    target_link_libraries(test_trt_decoder_composite PRIVATE cudaq-qec-pymatching)
  endif()

  get_filename_component(_cudaq_rt_lib_dir_trt_comp "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(test_trt_decoder_composite PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${_cudaq_rt_lib_dir_trt_comp}"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${_cudaq_rt_lib_dir_trt_comp}"
  )

  if(ENABLE_NVTX)
    target_compile_definitions(test_trt_decoder_composite PRIVATE ENABLE_NVTX)
  endif()

else()
  message(WARNING "TensorRT or cudaq-qec-trt-decoder not found. "
                  "Skipping test_trt_decoder_composite.")
endif()

# =========================================================================== #
# gpu_roce_predecoder_bridge
# =========================================================================== #
# FPGA bridge: GPU RoCE RDMA -> realtime_pipeline with AI predecoder +
# PyMatching (HOST_LOOP, external ring buffer from the GpuRoceTransceiver).

if (GPU_ROCE_TRANSCEIVER_LIB AND CUDAQ_REALTIME_INCLUDE_DIR AND
    CUDAQ_REALTIME_LIBRARY AND CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY AND
    CUDAQ_REALTIME_BRIDGE_GPU_ROCE_LIBRARY AND
    DOCA_INCLUDE_DIR AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND
    DOCA_GPUNETIO_LIB AND
    TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY AND
    TARGET cudaq-realtime-pipeline)

  message(STATUS "Building gpu_roce_predecoder_bridge")

  get_filename_component(_cuda_bin_bridge "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_cuda_root_bridge "${_cuda_bin_bridge}" DIRECTORY)
  set(_cuda_cccl_include_bridge "${_cuda_root_bridge}/include/cccl")

  add_executable(gpu_roce_predecoder_bridge
    gpu_roce_predecoder_bridge.cpp
    predecoder_pipeline_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_predecoder_service.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_decoder_service.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_link_stub.cu
  )

  set_target_properties(gpu_roce_predecoder_bridge PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
  )

  target_compile_definitions(gpu_roce_predecoder_bridge PRIVATE
    ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime"
  )

  target_include_directories(gpu_roce_predecoder_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_cuda_cccl_include_bridge}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src
  )

  target_link_libraries(gpu_roce_predecoder_bridge PRIVATE
    cudaq-realtime-pipeline
    ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
    ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
    ${CUDAQ_REALTIME_BRIDGE_GPU_ROCE_LIBRARY}
    ${GPU_ROCE_TRANSCEIVER_LIB}
    ${BASE_RECEIVER_OP_LIB}
    ${HSB_CORE_LIB}
    ${HSB_COMMON_LIB}
    ${CUDAQ_REALTIME_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    cudaq-qec
    cudaq::cudaq
    $<$<BOOL:${_CUDAQ_LIBRARY}>:${_CUDAQ_LIBRARY}>
    $<$<BOOL:${_NVQIR_LIBRARY}>:${_NVQIR_LIBRARY}>
    CUDA::cudart
    CUDA::cuda_driver
    ${DOCA_VERBS_LIB}
    ${DOCA_GPUNETIO_LIB}
    ${DOCA_COMMON_LIB}
    ${IBVERBS_LIB}
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )

  target_link_options(gpu_roce_predecoder_bridge PRIVATE
    "LINKER:--allow-shlib-undefined"
  )

  if (holoscan_FOUND)
    target_link_libraries(gpu_roce_predecoder_bridge PRIVATE holoscan::core)
  endif()

  get_filename_component(_cudaq_rt_lib_dir_b "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(gpu_roce_predecoder_bridge PROPERTIES
    BUILD_RPATH "${_cudaq_rt_lib_dir_b};${DOCA_LIB_DIR};${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins"
    INSTALL_RPATH "${_cudaq_rt_lib_dir_b};${DOCA_LIB_DIR};${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins"
  )

  if(ENABLE_NVTX)
    target_compile_definitions(gpu_roce_predecoder_bridge PRIVATE ENABLE_NVTX)
  endif()

else()
  message(WARNING "gpu_roce_predecoder_bridge: skipped (missing deps). "
                  "Requires gpu_roce bridge libs + TensorRT + cudaq-realtime-pipeline.")
endif()

# =========================================================================== #
# Two-process decoding-server test (UDP transport)
# =========================================================================== #
# decoding_server is a decoder-agnostic decoding-server process: the
# service end of the CUDA-Q `udp` DeviceCallChannel (the loopback stand-in for
# the RoCE transport on hosts without an RDMA NIC), with its decoders selected
# entirely by a YAML config file (decoding_server_config.yaml -- currently
# pymatching; swap the decoder by editing the config, not the code).
# test_decoding_server runs the caller (simulated QPU) side and spawns the
# server as a subprocess.

# The decoding-server server itself is built from libs/qec/tools/decoding-server
# (this test spawns it as a subprocess and receives its path + config dir via
# compile definitions).
if(CUDAQ_REALTIME_INCLUDE_DIR AND TARGET decoding_server)
  include(GoogleTest)

  add_executable(test_decoding_server)
  set_target_properties(test_decoding_server PROPERTIES
    LINKER_LANGUAGE CXX)
  target_include_directories(test_decoding_server PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/decoding-server-cqr
  )
  # -DQEC_DECODING_SERVER_PATH tells the test where the server binary lives
  # (the example configs are placed next to it by tools/decoding-server). It
  # rides COMPILER_FLAGS because cudaqx_add_device_code compiles with nvq++,
  # which does not see target_compile_definitions.
  cudaqx_add_device_code(test_decoding_server
    SOURCES
      test_decoding_server.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
      -DQEC_DECODING_SERVER_PATH=\"$<TARGET_FILE:decoding_server>\"
  )

  target_link_directories(test_decoding_server
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  # Caller side only: the simulation-cqr device wrappers lower the kernel's
  # device_calls to the realtime frame ABI; deliberately NOT linking the
  # decoding-server-cqr service, so the only decoder is the server's.
  target_link_libraries(test_decoding_server PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation-cqr
    cudaq-device-call-runtime
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim
  )

  set_target_properties(test_decoding_server PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )

  add_dependencies(test_decoding_server decoding_server)
  add_dependencies(CUDAQXQECUnitTests test_decoding_server)
  gtest_discover_tests(test_decoding_server
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

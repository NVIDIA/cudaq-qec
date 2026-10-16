# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Handle both standalone (libs/qec) and full-repo builds.
if(CUDAQX_QEC_STANDALONE_BUILD)
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
else()
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
endif()

# ---------------------------------------------------------------------------
# Optional: device_graph dispatch via the CUDA-Q realtime bridge-provider
# interface.
#
# The GpuRoceTransceiver sits behind a runtime-loaded provider
# (libcudaq-realtime-bridge-gpu-roce.so, built in cuda-quantum where
# HSB + DOCA are provisioned), so compiling this path needs only the CUDA-Q
# realtime headers, libcudaq-realtime.so (the bridge loader + C API), and the
# CUDA toolkit -- NO GpuRoceTransceiver / DOCA / HSB build-time dependencies here.  The
# provider .so is resolved at runtime (dlopen), so a build without it still
# produces a server whose device_graph mode fails loudly at startup.
#
# cudaq-realtime-dispatch (the device-graph scheduler shims resolved via
# dlsym) is found here for the server target's link; see
# tools/decoding-server/CMakeLists.txt.
# ---------------------------------------------------------------------------
set(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE FALSE)

if(CUDAQ_REALTIME_INCLUDE_DIR)
  if(CUDAQ_REALTIME_LIBRARY)
    get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  else()
    get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_INCLUDE_DIR}/../lib" ABSOLUTE)
  endif()
  find_library(CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY
    NAMES cudaq-realtime
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  # cudaq-realtime-dispatch: declares cudaq_create_dispatch_graph_regular,
  # cudaq_launch_dispatch_graph, cudaq_destroy_dispatch_graph, cudaq_function_entry_t.
  find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
    NAMES cudaq-realtime-dispatch
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  if(CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY AND CUDAQ_REALTIME_DISPATCH_LIBRARY
      AND CUDAToolkit_FOUND)
    set(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE TRUE CACHE INTERNAL
      "device-graph dispatch component cudaq-qec-decoding-server-device-graph available")
    # Under CMP0126 (NEW) the CACHE set above does NOT update the normal
    # variable initialized FALSE at the top of this file, and the normal
    # variable shadows the cache in this scope -- so the component gate
    # below would never fire.  Set the normal variable explicitly.
    set(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE TRUE)
    message(STATUS
      "cudaq-qec-decoding-server-device-graph: device_graph dispatch ENABLED "
      "(bridge provider resolved at runtime; api=${CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY})")
  else()
    message(STATUS
      "cudaq-qec-decoding-server: device_graph dispatch DISABLED "
      "(missing one or more of: libcudaq-realtime "
      "cudaq-realtime-dispatch CUDA toolkit)")
  endif()
else()
  message(STATUS
    "cudaq-qec-decoding-server: device_graph dispatch DISABLED "
    "(CUDAQ_REALTIME_INCLUDE_DIR not set)")
endif()

# ---------------------------------------------------------------------------
# Core decoding server: transport abstraction, session registry, per-decoder
# worker threads, RPC dispatcher.  The wire format (function IDs, payload
# structs, RPCHeader/RPCResponse) comes from cudaq/qec/realtime/
# decoder_rpc_wire_format.h, which includes the cudaq realtime headers -- so
# the server (and everything below) is only built when a realtime-enabled
# CUDA-Q install is found.
#
# Deliberately excludes DeviceGraphTransceiver: consumers of this library (unit
# tests, the CQR plugin) must stay loadable on driverless machines, and the
# device-graph deps (DOCA, the gpu_roce bridge) require libcuda.so.1 at load
# time.  The device-graph transceiver lives in the
# cudaq-qec-decoding-server-device-graph component below, reached through a
# weak factory symbol (see DecodingServer.cpp).
# ---------------------------------------------------------------------------
if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
  message(STATUS
    "cudaq-qec-decoding-server: SKIPPED (cuda-quantum realtime headers not "
    "found; set CUDAQ_REALTIME_ROOT to enable the decoding server)")
  return()
endif()

add_library(cudaq-qec-decoding-server STATIC
  DecodingSession.cpp
  SessionRegistry.cpp
  DecodingServer.cpp
)

target_compile_features(cudaq-qec-decoding-server PUBLIC cxx_std_20)

target_include_directories(cudaq-qec-decoding-server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${_core_include_dir}>
    # decoder_rpc_wire_format.h pulls in the cudaq realtime dispatcher
    # headers; propagate the path so test targets linking this library
    # compile without their own realtime include setup.
    $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-qec-decoding-server
  PUBLIC
    # The decode-server core only needs the decoder interface + logger (in
    # cudaq-qec-decoders) and the realtime decoding config -- NOT the heavy
    # cudaq-qec (which pulls in libcustabilizer etc.). Keeping cudaq-qec here
    # violates the realtime-server dependency-closure contract.
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
  PRIVATE
    # DecodingSession worker threads pin themselves to their decoder's
    # cuda_device_id (cudaSetDevice).
    CUDA::cudart
)

set_target_properties(cudaq-qec-decoding-server PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

# NOTE: cudaq-qec-decoding-server is a STATIC archive consumed only at link
# time -- the decoding_server executable and the CQR plugin (.so) absorb its
# objects during their own link.  It is not needed at runtime and is not part
# of a public link-against-install SDK, so it is deliberately NOT installed.

# ---------------------------------------------------------------------------
# Optional device-graph dispatch component: DeviceGraphTransceiver plus the strong definition
# (DeviceGraphFactory.cpp) of the factory that DecodingServer.cpp references
# weakly.  Kept out of the core library so core consumers stay loadable on
# machines without the CUDA runtime.  The transport itself is a runtime-
# loaded bridge provider selected by the YAML transport section, so this
# component links only the CUDA-Q realtime bridge C API and the CUDA runtime --
# no GpuRoceTransceiver / DOCA / HSB link-time dependencies.
#
# Consumers must link this WHOLE_ARCHIVE: the only reference to the factory
# is weak, which does not pull archive members on its own.
# ---------------------------------------------------------------------------
if(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE)
  add_library(cudaq-qec-decoding-server-device-graph STATIC
    DeviceGraphRingConsumer.cpp
    DeviceGraphTransceiver.cpp
    DeviceGraphFactory.cpp
  )

  target_compile_definitions(cudaq-qec-decoding-server-device-graph
    PRIVATE
      CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
      QEC_BRIDGE_PROVIDER_DIR="${_cudaq_rt_lib_dir}")

  target_include_directories(cudaq-qec-decoding-server-device-graph PRIVATE
    "${CUDAQ_REALTIME_INCLUDE_DIR}"
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_libraries(cudaq-qec-decoding-server-device-graph
    PUBLIC
      cudaq-qec-decoding-server
      "${CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY}"
      CUDA::cudart
  )

  set_target_properties(cudaq-qec-decoding-server-device-graph PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
  )

  # NOTE: cudaq-qec-decoding-server-device-graph is a STATIC archive consumed
  # only at link time (absorbed WHOLE_ARCHIVE into the decoding_server
  # executable). It is not needed at runtime, so it is deliberately NOT
  # installed.

  # Link canary (never executed, not installed, not a test): a static
  # archive alone never resolves symbols, and the one real consumer
  # (decoding_server's device_graph block) is additionally gated on the
  # proprietary cudevice archive, which CI does not provision.  Without
  # this target, nothing would validate the component's link against the
  # CUDA-Q realtime bridge API.  The strong factory reference pulls the
  # component's members, and the component's PUBLIC deps complete the
  # resolution.
  add_executable(cudaq-qec-decoding-server-device-graph-linkcheck
    DeviceGraphLinkCheck.cpp
  )
  target_link_libraries(cudaq-qec-decoding-server-device-graph-linkcheck PRIVATE
    cudaq-qec-decoding-server-device-graph
  )
endif()

# ---------------------------------------------------------------------------
# CQR host-dispatch plugin (only when CUDAQ device_call headers are available):
# registers the HOST_CALL handlers, which serve each request inline on the
# CUDAQ dispatcher thread (DecodingSession::handle_*).
# ---------------------------------------------------------------------------
if(CUDAQ_REALTIME_INCLUDE_DIR)
  # decoding_server_cqr.cpp is the table + handlers: cudaq-realtime only, no
  # CUDA-Q types, reachable through the plain-C
  # cudaqx_qec_decoding_server_host_call_table() accessor.
  set(_server_cqr_sources decoding_server_cqr.cpp)

  # The in-process DeviceCallService shim is CUDA-Q's discovery ABI, and the
  # only part that needs libcudaq-device-call-runtime (it defines the two
  # service base classes' key functions).  Build it only with a full CUDA-Q
  # install: an external decoding server reaches the same table through the
  # accessor, and the in-process path needs nvq++-compiled callers anyway.
  set(_server_cqr_build_device_call_shim FALSE)
  if(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE AND NOT CUDAQ_QEC_DECODERS_ONLY)
    set(_server_cqr_build_device_call_shim TRUE)
    list(APPEND _server_cqr_sources decoding_server_cqr_device_call_service.cpp)
  endif()

  add_library(cudaq-qec-realtime-decoding-server-cqr SHARED
    ${_server_cqr_sources}
  )

  target_compile_options(cudaq-qec-realtime-decoding-server-cqr
    PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

  target_include_directories(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_options(cudaq-qec-realtime-decoding-server-cqr PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,libcudaq-qec-decoding-server.a>
  )

  # No CUDA-Q libraries on the base path: the service side is driven entirely
  # by libcudaq-realtime, a separate product from CUDA-Q, which
  # cudaq-qec-realtime-decoding already links.
  target_link_libraries(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      cudaq-qec-realtime-decoding
    PRIVATE
      cudaq-qec-decoding-server
  )

  # libcudaq-device-call-runtime defines DeviceCallService::~DeviceCallService
  # and DeviceCallServiceSession::~DeviceCallServiceSession, the key functions
  # of the two base classes the shim derives from.  Needed only when the shim
  # is compiled.
  if(_server_cqr_build_device_call_shim)
    target_link_libraries(cudaq-qec-realtime-decoding-server-cqr
      PRIVATE ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY})
  endif()

  set(_server_cqr_build_rpath "${CMAKE_BINARY_DIR}/lib")
  set(_server_cqr_install_rpath
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  if(_CUDAQ_REALTIME_LIB_DIR)
    list(APPEND _server_cqr_build_rpath "${_CUDAQ_REALTIME_LIB_DIR}")
    list(APPEND _server_cqr_install_rpath "${_CUDAQ_REALTIME_LIB_DIR}")
  endif()
  if(_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR)
    list(APPEND _server_cqr_build_rpath "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
    list(APPEND _server_cqr_install_rpath
      "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
  endif()

  set_target_properties(cudaq-qec-realtime-decoding-server-cqr PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    BUILD_RPATH "${_server_cqr_build_rpath}"
    INSTALL_RPATH "${_server_cqr_install_rpath}"
  )

  install(TARGETS cudaq-qec-realtime-decoding-server-cqr
    COMPONENT qec-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Simulated-QPU client for the cudaq-realtime (cqr) host-dispatch path. Sibling
# to ../simulation, but its __qpu__ wrappers are compiled with
# -frealtime-lowering, so each cudaq::device_call is rewritten to the realtime
# frame ABI and dispatched (by fnv1a(function_id)) through the host-dispatch
# channel to the decoding server (../decoding-server-cqr) -- rather than
# resolving to a direct in-process trampoline. Callees use the generic spec
# function_ids; the wrappers are target-agnostic (no --target needed).
add_library(cudaq-qec-realtime-decoding-simulation-cqr SHARED
  simulation_cqr_host.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-simulation-cqr
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

cudaqx_add_device_code(cudaq-qec-realtime-decoding-simulation-cqr
  SOURCES
    simulation_cqr_device.cpp
  COMPILER_FLAGS
    -frealtime-lowering
)

target_include_directories(cudaq-qec-realtime-decoding-simulation-cqr
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(cudaq-qec-realtime-decoding-simulation-cqr PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-simulation-cqr
  PUBLIC
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
  PRIVATE
    cudaq::cudaq-common
    # This is the caller (QPU) side: the nvq++-lowered device_calls here
    # reference __cudaq_device_call_{acquire,dispatch,safely_release}_
    # realtime_frame, which live in libcudaq-device-call-runtime.  Link it
    # directly rather than inheriting it from the decoding-server CQR plugin,
    # which is the service side and no longer depends on the caller runtime.
    ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}
)

set_target_properties(cudaq-qec-realtime-decoding-simulation-cqr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

install(TARGETS cudaq-qec-realtime-decoding-simulation-cqr
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

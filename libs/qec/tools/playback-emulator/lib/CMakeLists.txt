# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The playback emulator core: parser, timing loop, dispatch (polling streams),
# the three required session backends, syndrome sources, and the downstream
# CSV analyzer. This file is the one place that reaches outside this tool's
# own directory, for pieces it borrows rather than owns a copy of: the
# existing decoding-server (RpcSlot.h et al.) and core (heterogeneous_map).

if(CUDAQX_QEC_STANDALONE_BUILD)
  set(_playback_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
  set(_playback_decoding_server_dir "${CMAKE_SOURCE_DIR}/lib/realtime/decoding-server-cqr")
else()
  set(_playback_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
  set(_playback_decoding_server_dir "${CMAKE_SOURCE_DIR}/libs/qec/lib/realtime/decoding-server-cqr")
endif()

add_library(cudaq-qec-playback-emulator STATIC
  emulator.cpp
  parser.cpp
  session.cpp
  syndrome_source.cpp
)

target_compile_features(cudaq-qec-playback-emulator PUBLIC cxx_std_20)

target_include_directories(cudaq-qec-playback-emulator
  PUBLIC
    # No installed SDK to export: this tool's headers live right here in
    # lib/, next to the .cpp files that implement them.
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQX_QEC_INCLUDE_DIR}
    ${_playback_core_include_dir}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  PRIVATE
    # RpcSlot.h / SessionRegistry.h / DecodingSession.h: the in-process and
    # UDP backends dispatch straight to these payload-level cores rather
    # than duplicating wire parsing/decoder-set construction.
    ${_playback_decoding_server_dir}
    # device/memory_circuit.h: the kernel cudaq_memory_source streams from,
    # an internal (not installed) header -- same in-tree-only access
    # test_qec_stim.cpp uses.
    ${CUDAQX_QEC_SOURCE_DIR}/lib
)

find_package(Threads REQUIRED)

target_link_libraries(cudaq-qec-playback-emulator
  PUBLIC
    cudaq-qec-decoding-server
    cudaq-qec-realtime-decoding
    # cudaq_memory_source sits in syndrome_source.cpp alongside the other
    # two sources, so cudaq-qec is a dependency of this library as a whole --
    # and every executable linking this needs a concrete NVQIR backend too
    # (see the CLI's own link list).
    cudaq-qec
  PRIVATE
    cudaq-qec-decoders
    libstim
    Threads::Threads
)

set_target_properties(cudaq-qec-playback-emulator PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

# STATIC archive consumed only at link time (by the CLI tool, unit tests, and
# the Python bindings) -- not part of the installed SDK, so not installed.

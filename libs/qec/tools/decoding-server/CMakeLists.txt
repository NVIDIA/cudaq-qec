# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# decoding_server is a decoder- AND transport-agnostic decoding-server
# process: the service end of a CUDA-Q device_call transport, driving the
# libcudaq-realtime dispatcher object over a runtime-loaded transport
# provider's rings, and serving the decoding-server-cqr function table
# (enqueue_syndromes / get_corrections / reset_decoder).
#
# Its decoders are selected entirely by a YAML config file (see
# decoding_server_config.yaml) and its transport by --transport=<name|.so
# path>, resolved to a bridge provider shared library loaded via the CUDA-Q
# realtime transport-provider interface (bridge_interface.h).  udp and
# cpu_roce providers ship with the CUDA-Q realtime install; a partner
# transport is a path to their own provider .so, with no changes here.
#
# The two-process test (unittests/realtime/test_decoding_server.cpp) spawns
# this binary as a subprocess.

if(CUDAQ_REALTIME_INCLUDE_DIR AND TARGET cudaq-qec-realtime-decoding-server-cqr)
  # The server needs the DeviceCallService plugin headers and the
  # libcudaq-realtime shared lib (bridge loader + dispatcher object; the
  # HOST-path ring loop is linked into it).  Transports are runtime provider
  # .so's, NOT link dependencies.
  find_path(QEC_UDP_DEVICE_CALL_INCLUDE_DIR
    NAMES cudaq/realtime/device_call_service.h
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES include)
  find_library(QEC_UDP_REALTIME_LIBRARY cudaq-realtime
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES lib)

  if(QEC_UDP_DEVICE_CALL_INCLUDE_DIR AND QEC_UDP_REALTIME_LIBRARY)
    add_executable(decoding_server decoding_server.cpp)
    target_include_directories(decoding_server PRIVATE
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      ${QEC_UDP_DEVICE_CALL_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_directories(decoding_server
      PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
    # No CUDA-Q libraries: the server is a decoder process driven by
    # libcudaq-realtime, which is a separate product from CUDA-Q.  It uses only
    # the decoder interface (cudaq-qec-decoders, via the two targets below),
    # the realtime decoding config, and the CQR function table -- never the
    # heavy cudaq-qec (codes, experiments, DEM sampling), the CUDA-Q runtime,
    # or the simulators.  Those were link-only leftovers: none of their symbols
    # is referenced, and they were reaching DT_NEEDED through the
    # `-Wl,--no-as-needed` that cudaq-qec applies to its consumers.
    target_link_libraries(decoding_server PRIVATE
      cudaq-qec-realtime-decoding
      cudaq-qec-realtime-decoding-server-cqr
      ${QEC_UDP_REALTIME_LIBRARY}
      CUDA::cudart
    )
    # Where --transport=<name> resolves provider libraries
    # (libcudaq-realtime-bridge-<name>.so): next to libcudaq-realtime.so.
    get_filename_component(_decoding_server_realtime_lib_dir
      "${QEC_UDP_REALTIME_LIBRARY}" DIRECTORY)
    target_compile_definitions(decoding_server PRIVATE
      QEC_BRIDGE_PROVIDER_DIR="${_decoding_server_realtime_lib_dir}")
    # The device_graph dispatch branch ([2a] in decoding_server.cpp) references only
    # the core DecodingServer API, so compile it whenever the bridge API is
    # available -- machines without the proprietary archive (CI in
    # particular) still build the branch; at runtime it fails with a clear
    # "the device-graph component is not linked into this binary" error from
    # make_transport.
    if(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE)
      target_compile_definitions(decoding_server PRIVATE
        QEC_HAVE_DEVICE_GRAPH_DISPATCH)
      target_include_directories(decoding_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/decoding-server-cqr
      )
      target_link_libraries(decoding_server PRIVATE
        cudaq-qec-decoding-server)
    endif()
    # Actually LINKING device_graph dispatch needs BOTH the
    # cudaq-qec-decoding-server-device-graph component AND the proprietary
    # cudevice archive, which supplies the three populate-shim symbols
    # (cudaqx_qec_realtime_dispatch_populate_*_device_entry)
    # that DeviceGraphTransceiver::launch_scheduler resolves via dlsym. GpuRoceTransceiver
    # presence and the proprietary archive are provisioned independently, so
    # treat a missing archive as "device_graph dispatch not available" and fall
    # back to the provider transports rather than failing configuration.
    set(_decoding_server_linked_proprietary_cudevice FALSE)
    if(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE AND TARGET cudaq-qec-realtime-cudevice-proprietary)
      # Re-find in case the CQR subdir cache entry didn't propagate here.
      if(NOT CUDAQ_REALTIME_DISPATCH_LIBRARY)
        find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY cudaq-realtime-dispatch
          HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
          PATH_SUFFIXES lib)
      endif()
      if(NOT CUDAQ_REALTIME_DISPATCH_LIBRARY)
        message(FATAL_ERROR
          "CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE is set but cudaq-realtime-dispatch was "
          "not found (searched CUDAQ_REALTIME_ROOT=${CUDAQ_REALTIME_ROOT} and "
          "CUDAQ_INSTALL_DIR=${CUDAQ_INSTALL_DIR}).  "
          "DeviceGraphTransceiver::launch_scheduler resolves "
          "cudaq_create/launch/destroy_dispatch_graph_regular via dlsym; "
          "without this library in the server image those lookups will fail "
          "at runtime.  Set CUDAQ_REALTIME_ROOT to the CUDA-Q realtime "
          "install prefix and re-run CMake.")
      endif()
      target_link_libraries(decoding_server PRIVATE
        # WHOLE_ARCHIVE: DecodingServer reaches the device-graph transceiver
        # through a weak factory symbol, and a weak reference does not pull
        # archive members.
        $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-decoding-server-device-graph>
        ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
        $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>
      )
      set(_decoding_server_linked_proprietary_cudevice TRUE)
      target_link_options(decoding_server PRIVATE
        "LINKER:--export-dynamic"
      )
      message(STATUS "decoding_server: device_graph dispatch enabled")
    elseif(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE)
      message(WARNING
        "decoding_server: device_graph dispatch compiled but NOT linked "
        "(cudaq-qec-realtime-cudevice-proprietary target not defined; set "
        "CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE to enable). "
        "device_graph dispatch will fail at runtime with a not-linked error.")
    endif()
    # RelayBP/nv-qldpc captures a CUDA device graph during decoder
    # initialization even when the server transport is udp or cpu_roce. Export
    # the graph-dispatch shims whenever a cudevice archive is available so
    # plugin dlsym(RTLD_DEFAULT, ...) lookups succeed. The proprietary archive
    # also carries the public gpu_kernels.cu.o symbols, so do not whole-archive
    # both the proprietary and public cudevice archives into the same binary.
    if(TARGET cudaq-qec-realtime-cudevice OR TARGET cudaq-qec-realtime-cudevice-proprietary)
      target_link_libraries(decoding_server PRIVATE
        CUDA::cudart
      )
      if(TARGET cudaq-qec-realtime-cudevice-proprietary)
        if(NOT _decoding_server_linked_proprietary_cudevice)
          target_link_libraries(decoding_server PRIVATE
            $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>
          )
        endif()
      elseif(TARGET cudaq-qec-realtime-cudevice)
        target_link_libraries(decoding_server PRIVATE
          $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice>
        )
      endif()
      target_link_options(decoding_server PRIVATE
        "LINKER:--export-dynamic"
      )
      set_property(TARGET decoding_server PROPERTY CUDA_SEPARABLE_COMPILATION ON)
      set_property(TARGET decoding_server PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
      set_property(TARGET decoding_server PROPERTY CUDA_STANDARD 17)
      set_property(TARGET decoding_server PROPERTY LINKER_LANGUAGE CUDA)
    endif()
    # The installed binary lives in <prefix>/bin with the qec libraries and
    # decoder plugins in ../lib, so use $ORIGIN-relative rpaths for those;
    # the CUDA-Q realtime libs stay at their absolute install location.
    set_target_properties(decoding_server PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      BUILD_RPATH
        "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
      INSTALL_RPATH
        "$ORIGIN/../lib;$ORIGIN/../lib/decoder-plugins;${_decoding_server_realtime_lib_dir};${CUDAQ_INSTALL_DIR}/lib"
    )
    # Decoder plugins are loaded at runtime by decoder type name; the example
    # configs currently select pymatching / multi_error_lut.
    add_dependencies(decoding_server cudaq-qec-pymatching)

    # Place the example decoder configs next to the server binary.
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/decoding_server_config.yaml
      ${CMAKE_BINARY_DIR}/bin/decoding_server_config.yaml
      COPYONLY)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/decoding_server_config_multi_error_lut.yaml
      ${CMAKE_BINARY_DIR}/bin/decoding_server_config_multi_error_lut.yaml
      COPYONLY)

    # Install the server (and its example configs) alongside the libraries.
    # The qec-tools component is deliberately NOT in the Python wheel's
    # install.components (pyproject.toml): the server depends on the CUDA-Q
    # realtime libs, which the cudaq wheel does not ship.
    install(TARGETS decoding_server
      COMPONENT qec-tools
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/decoding_server_config.yaml
      ${CMAKE_CURRENT_SOURCE_DIR}/decoding_server_config_multi_error_lut.yaml
      COMPONENT qec-tools
      DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  else()
    message(STATUS "decoding_server: cudaq-realtime library not found; "
                   "decoding-server tool will not be built.")
  endif()
endif()

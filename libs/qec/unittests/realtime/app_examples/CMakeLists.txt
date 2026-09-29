# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------#
# Maintenance note: to generate QIR files and corresponding config files for
# Quantinuum, you may manually run tests like the following:
#    CUDAQ_DUMP_JIT_IR=1 KEEP_LOG_FILES=1 ctest -R quantinuum
# The resulting files will be in:
#    build/libs/qec/unittests/realtime/app_examples/config-*.yml
#    build/libs/qec/unittests/realtime/app_examples/qir-*.ll
#------------------------------------------------------------------------------#

# Resolve cudaq-realtime dispatch (DEVICE-mode graph launch) for app examples
# that link it.
set(_app_cudaq_realtime_prefixes "")
if(CUDAQ_REALTIME_ROOT)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
endif()
if(CUDAQ_INSTALL_PREFIX)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
endif()
find_library(_APP_CUDAQ_REALTIME_DISPATCH_LIB cudaq-realtime-dispatch
  PATHS ${_app_cudaq_realtime_prefixes} PATH_SUFFIXES lib)

function(qec_realtime_app_link_options _tgt)
  target_link_options(${_tgt} PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:--export-dynamic"
  )
  if(_APP_CUDAQ_REALTIME_DISPATCH_LIB)
    target_link_libraries(${_tgt} PRIVATE
      ${_APP_CUDAQ_REALTIME_DISPATCH_LIB})
  endif()
  if(TARGET cudaq-qec-realtime-cudevice-proprietary)
    target_link_libraries(${_tgt} PRIVATE
      CUDA::cudart
      $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>)
    set_property(TARGET ${_tgt} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${_tgt} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET ${_tgt} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${_tgt} PROPERTY LINKER_LANGUAGE CUDA)
  endif()
endfunction()

#------------------------------------------------------------------------------#
# Surface code 1, fully local, regular Stim simulator
add_executable(surface_code-1-local)
set_target_properties(surface_code-1-local PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-1-local
  SOURCES
    surface_code-1.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-1-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-1-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-1-local)

add_test(
  NAME app_examples.surface_code-1-local-test-distance-3
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      3 60 30 NULL 12
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test with sliding_window decoder
add_test(
  NAME app_examples.surface_code-1-local-test-distance-3-sliding-window
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      3 60 30 NULL 12
      ${CMAKE_BINARY_DIR}/lib
      sliding_window 6 1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# This must be disabled for now because the multi_error_lut decoder is not
# powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
# but that is not available on the GitHub repo.
# add_test(
#   NAME app_examples.surface_code-1-local-test-distance-5
#   COMMAND
#     bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#       5 40 40 NULL 20
#       ${CMAKE_BINARY_DIR}/lib
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# Surface code 1 Python test
add_test(
  NAME app_examples.py-surface_code-1-test
  COMMAND
    pytest ${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# --------------------------------------------------------------------------- #
# surface_code-1 negative / error-path tests
#
# These drive the three throw sites added to surface_code-1.cpp so that every
# error path is exercised by at least one ctest. Each test asserts BOTH a
# nonzero exit AND the expected diagnostic substring.
# --------------------------------------------------------------------------- #
set(_sc1_exe ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local)
set(_sc1_neg_dir ${CMAKE_CURRENT_BINARY_DIR}/sc1_negative)
file(MAKE_DIRECTORY ${_sc1_neg_dir})
set(_sc1_neg_env "CUDAQ_DEFAULT_SIMULATOR=stim")

# Helper: command must exit nonzero AND print the expected diagnostic.
function(_sc1_reject name diagnostic)
  add_test(NAME app_examples.surface_code-1-negative-${name}
    COMMAND bash -c [==[
diag=$1; shift
if out=$("$@" 2>&1); then
  printf '%s\n' "$out"
  echo "NEGATIVE TEST FAIL: command returned 0; expected a nonzero exit"
  exit 1
fi
printf '%s\n' "$out"
printf '%s' "$out" | grep -qF -- "$diag" || {
  echo "NEGATIVE TEST FAIL: missing expected diagnostic: $diag"
  exit 1
}
]==] _ "${diagnostic}" ${_sc1_exe} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(app_examples.surface_code-1-negative-${name}
    PROPERTIES ENVIRONMENT "${_sc1_neg_env}")
endfunction()

# 1a. --load_dem with a nonexistent file → setup_decoders throws.
_sc1_reject(load-dem-missing-file "Could not open dem config file"
  --load_dem ${_sc1_neg_dir}/nonexistent.yml
  --distance 3 --num_rounds 3)

# 1b. --load_dem with a mismatched geometry → syndrome_size check throws.
# Fixture: generate a d=3 config, then try to load it as d=5.
add_test(NAME app_examples.surface_code-1-negative-load-dem-fixtures
  COMMAND ${_sc1_exe} --distance 3 --num_rounds 3
    --save_dem ${_sc1_neg_dir}/d3.yml
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-1-negative-load-dem-fixtures
  PROPERTIES FIXTURES_SETUP sc1_dem ENVIRONMENT "${_sc1_neg_env}")
_sc1_reject(load-dem-geometry-mismatch "does not match current geometry"
  --distance 5 --num_rounds 5
  --load_dem ${_sc1_neg_dir}/d3.yml)
set_tests_properties(app_examples.surface_code-1-negative-load-dem-geometry-mismatch
  PROPERTIES FIXTURES_REQUIRED sc1_dem)

# 2. --save_syndrome to an unwriteable path → begin_syndrome_capture throws.
_sc1_reject(save-syndrome-bad-path "Could not open syndrome file for writing"
  --distance 3 --num_rounds 3 --num_shots 2
  --save_syndrome ${_sc1_neg_dir}/no_such_dir/out.txt)

# 3. --load_syndrome with a nonexistent file → replay_syndrome_file throws.
_sc1_reject(load-syndrome-missing-file "Could not open syndrome file"
  --distance 3 --num_rounds 3
  --load_syndrome ${_sc1_neg_dir}/nonexistent.txt)

# Fixture: capture a valid d=3 syndrome file, then derive a
# corrections-flipped copy (0↔1 in the CORRECTIONS block).
add_test(NAME app_examples.surface_code-1-negative-replay-fixtures
  COMMAND bash -c "set -e
'${_sc1_exe}' --distance 3 --num_rounds 3 --num_shots 5 \
  --save_syndrome '${_sc1_neg_dir}/cap.txt'
sed '/CORRECTIONS_START/,/CORRECTIONS_END/{s/^0$/X/;s/^1$/0/;s/^X$/1/}' \
  '${_sc1_neg_dir}/cap.txt' > '${_sc1_neg_dir}/cap_mismatch.txt'"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-1-negative-replay-fixtures
  PROPERTIES FIXTURES_SETUP sc1_replay ENVIRONMENT "${_sc1_neg_env}")

# 4. Replay d=3 capture against d=5 experiment → num_data mismatch (9 vs 25).
_sc1_reject(replay-geometry-mismatch "Syndrome file mismatch"
  --distance 5 --num_rounds 5
  --load_syndrome ${_sc1_neg_dir}/cap.txt)
set_tests_properties(app_examples.surface_code-1-negative-replay-geometry-mismatch
  PROPERTIES FIXTURES_REQUIRED sc1_replay)

# 5. Replay with flipped corrections → correction mismatch throw.
_sc1_reject(replay-correction-mismatch "Replay correction mismatch"
  --distance 3 --num_rounds 3
  --load_syndrome ${_sc1_neg_dir}/cap_mismatch.txt)
set_tests_properties(app_examples.surface_code-1-negative-replay-correction-mismatch
  PROPERTIES FIXTURES_REQUIRED sc1_replay)

# 6. Invalid --sw_inner_decoder is rejected before any circuit runs.
_sc1_reject(sw-inner-decoder-invalid "--sw_inner_decoder must be"
  --distance 3 --num_rounds 3
  --decoder_type sliding_window --sw_inner_decoder bogus)

# 7a. --param with an unknown key is rejected before any circuit runs.
_sc1_reject(param-unknown-key "Unknown parameter 'bogus'"
  --distance 3 --num_rounds 3
  --param bogus=1)

# 7b. --param threads overrides into the inner decoder: save the DEM and verify
# the YAML contains the overridden lut_error_depth value.
add_test(NAME app_examples.surface_code-1-negative-param-inner-decoder-fixture
  COMMAND ${_sc1_exe}
    --distance 3 --num_rounds 3
    --decoder_type sliding_window --sw_window_size 3
    --param lut_error_depth=1
    --save_dem ${_sc1_neg_dir}/param_inner.yml
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-1-negative-param-inner-decoder-fixture
  PROPERTIES FIXTURES_SETUP sc1_param_inner ENVIRONMENT "${_sc1_neg_env}")

add_test(NAME app_examples.surface_code-1-param-inner-decoder-threaded
  COMMAND bash -c "grep -q 'lut_error_depth: 1' '${_sc1_neg_dir}/param_inner.yml'"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-1-param-inner-decoder-threaded
  PROPERTIES FIXTURES_REQUIRED sc1_param_inner ENVIRONMENT "${_sc1_neg_env}")

# 7. sliding_window with a non-default inner decoder (pymatching) runs cleanly.
add_test(NAME app_examples.surface_code-1-sw-inner-pymatching
  COMMAND ${_sc1_exe}
    --distance 3 --num_rounds 3 --num_shots 50
    --decoder_type sliding_window --sw_window_size 3
    --sw_inner_decoder pymatching
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-1-sw-inner-pymatching
  PROPERTIES ENVIRONMENT "${_sc1_neg_env}")

# End of surface code 1, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 1, cudaq-realtime (cqr) host-dispatch wire
#
# Same application source, but the device wrappers are lowered with
# -frealtime-lowering and linked against the simulation-cqr client, so every
# cudaq::qec::decoding::* device_call crosses the cudaq-realtime ring to the
# decoding-server-cqr service (in-process here, selected via
# CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch) instead of resolving to the plain
# simulation trampolines. The test additionally asserts the service's
# dispatch count is non-zero, proving the syndromes went over the wire.
#------------------------------------------------------------------------------#
if(TARGET cudaq-qec-realtime-decoding-server-cqr)
  add_executable(surface_code-1-cqr)
  set_target_properties(surface_code-1-cqr PROPERTIES LINKER_LANGUAGE CXX)

  cudaqx_add_device_code(surface_code-1-cqr
    SOURCES
      surface_code-1.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
      -DQEC_APP_CQR
  )

  target_link_directories(surface_code-1-cqr
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  target_link_libraries(surface_code-1-cqr
    PRIVATE
      cudaq-qec
      cudaq-qec-realtime-decoding
      cudaq-qec-realtime-decoding-simulation-cqr
      cudaq-qec-realtime-decoding-server-cqr
      cudaq-device-call-runtime
      cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
      nvqir nvqir-stim)
  qec_realtime_app_link_options(surface_code-1-cqr)

  set(_surface_code_1_cqr_env CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch)

  add_test(
    NAME app_examples.surface_code-1-cqr-test-distance-3
    COMMAND
      ${CMAKE_COMMAND} -E env ${_surface_code_1_cqr_env}
      bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-cqr
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-cqr
        3 60 30 NULL 12
        ${CMAKE_BINARY_DIR}/lib
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Two-process: same application as the udp client, with the standard
  # decoding server (decoding_server, from tools/decoding-server) in a
  # separate process serving the decoders configured from the app's own
  # --save_dem YAML.
  if(TARGET decoding_server)
    add_test(
      NAME app_examples.surface_code-1-cqr-two-process-test-distance-3
      COMMAND
        # The app seeds the simulator (--seed, default 42), so the run is
        # deterministic: ~10 non-zero and ~67 corrections at 1000 shots. The
        # thresholds (non-zero <= 60, corrections >= 30) leave margin while
        # still catching a broken decode path (which applies ~0 corrections,
        # leaving all ~70 raw logical errors uncorrected).
        bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-cqr-two-process-test.sh"
          ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-cqr
          3 60 30
          $<TARGET_FILE:decoding_server>
          12
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Two-process with two logical qubits: the server opens one ring per
    # decoder and the client must wire each logical qubit's device_call
    # session to its own ring (QEC_DECODING_SERVER_PORT_<id>, exported by the
    # test script from the READY line's ring<id>= tokens). The script
    # additionally asserts per-ring traffic (a client that funnels every
    # decoder over ring 0 fails even if the totals look right) and exact
    # per-decoder session stats (resets == shots, errors == 0 — the guard
    # against concurrent rings losing RPCs to request-id collisions).
    # Thresholds are the single-logical values scaled by 2.
    add_test(
      NAME app_examples.surface_code-1-cqr-two-process-test-distance-3-num-logical-2
      COMMAND
        bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-cqr-two-process-test.sh"
          ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-cqr
          3 120 60
          $<TARGET_FILE:decoding_server>
          12
          multi_error_lut
          2
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endif()

# End of surface code 1, cqr host-dispatch wire
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 1, --target quantinuum --emulate
add_executable(surface_code-1-quantinuum-emulate)
set_target_properties(surface_code-1-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(surface_code-1-quantinuum-emulate surface_code-1-local)

cudaqx_add_device_code(surface_code-1-quantinuum-emulate
  SOURCES
    surface_code-1.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-1.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-1) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-1-local
)

target_link_directories(surface_code-1-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-1-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-1-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-1-quantinuum-emulate)

add_test(
  NAME app_examples.surface_code-1-quantinuum-emulate-test-distance-3-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-quantinuum-emulate
      # surface_code-1 attaches its noise via a host-side cudaq::noise_model
      # (not in-kernel cudaq::apply_noise, which the erase-noise pass strips).
      # Whether that model reaches the simulator under --emulate depends on the
      # CUDA-Q version, so the run finds either 0 errors / 0 corrections or a
      # realistic handful of both. Bound the residual logical errors the same
      # way the local tests do and leave the corrections bound at 0, so this
      # passes either way. This remains primarily a lowering + run smoke test.
      3 60 0 NULL 12
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME app_examples.surface_code-1-quantinuum-emulate-test-distance-5-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-quantinuum-emulate
      # See the distance-3 test above for why the non-zero bound is not 0.
      5 60 0 NULL 5
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# End of surface code 1, --target quantinuum --emulate
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Surface code 2, fully local, regular Stim simulator
add_executable(surface_code-2-local)
set_target_properties(surface_code-2-local PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-2-local
  SOURCES
    surface_code-2.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-2-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-2-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-2-local)

add_test(
  NAME app_examples.surface_code-2-local-test-distance-3
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      3 10 10 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# This must be disabled for now because the multi_error_lut decoder is not
# powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
# but that is not available on the GitHub repo.
# add_test(
#   NAME app_examples.surface_code-2-local-test-distance-5
#   COMMAND
#     bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#       5 10 10 NULL
#       ${CMAKE_BINARY_DIR}/lib
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# End of surface code 2, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 2, --target quantinuum --emulate
add_executable(surface_code-2-quantinuum-emulate)
set_target_properties(surface_code-2-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(surface_code-2-quantinuum-emulate surface_code-2-local)

cudaqx_add_device_code(surface_code-2-quantinuum-emulate
  SOURCES
    surface_code-2.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-2.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-2) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-2-local
)

target_link_directories(surface_code-2-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-2-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-2-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-2-quantinuum-emulate)

add_test(
  NAME app_examples.surface_code-2-quantinuum-emulate-test-distance-3-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-quantinuum-emulate
      3 0 1000 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME app_examples.surface_code-2-quantinuum-emulate-test-distance-5-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-quantinuum-emulate
      5 0 0 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# End of surface code 2, --target quantinuum --emulate
#------------------------------------------------------------------------------#


macro(add_surface_code_test base_name exe_local exe_remote dist server)
  # For now, disable 0.015 error rate because the multi_error_lut decoder is not
  # powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
  # but that is not available on the GitHub repo.
  foreach(err_rate IN ITEMS 0.010)
    foreach(prep IN ITEMS prep0 prepp)
      add_test(
        NAME ${base_name}-dist-${dist}-${prep}-err-${err_rate}
        COMMAND
          bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-3-test.sh"
            ${exe_local}
            ${exe_remote}
            ${dist} ${server} ${err_rate} "${prep}"
            ${CMAKE_BINARY_DIR}/quantinuum_decoding_server/lib
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
    endforeach()
  endforeach()
endmacro()

#------------------------------------------------------------------------------#
# Surface code 3, fully local, regular Stim simulator
add_executable(surface_code-3-local)
set_target_properties(surface_code-3-local PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-3-local
  SOURCES
    surface_code-3.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-3-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-3-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-3-local)

# --- Local tests ---
add_surface_code_test(
  app_examples.surface_code-3-local-test
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  3 NULL
)
add_surface_code_test(
  app_examples.surface_code-3-local-test
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  5 NULL
)

# End of surface code 3, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 3, --target quantinuum --emulate
add_executable(surface_code-3-quantinuum-emulate)
set_target_properties(surface_code-3-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(surface_code-3-quantinuum-emulate surface_code-3-local)

cudaqx_add_device_code(surface_code-3-quantinuum-emulate
  SOURCES
    surface_code-3.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-3.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-3) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-3-local
)

target_link_directories(surface_code-3-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-3-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-3-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-3-quantinuum-emulate)


    # --- Quantinuum emulate (in-process) ---
add_surface_code_test(
  app_examples.surface_code-3-quantinuum-emulate-test-in-process
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-quantinuum-emulate
  3 NULL
)
add_surface_code_test(
  app_examples.surface_code-3-quantinuum-emulate-test-in-process
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-quantinuum-emulate
  5 NULL
)

# End of surface code 3, --target quantinuum --emulate
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Surface code 4: YAML-switchable realtime decoders (pymatching / trt+pymatching
# / nv-qldpc relay-bp). The decoder is selected at realtime from the loaded YAML
# config (--yaml).
add_executable(surface_code-4-yaml)
set_target_properties(surface_code-4-yaml PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-4-yaml
  SOURCES
    surface_code-4-yaml.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-4-yaml
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-4-yaml
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-4-yaml)

if(TARGET cudaq-qec-pymatching)
  add_dependencies(surface_code-4-yaml cudaq-qec-pymatching)
endif()
if(TARGET cudaq-qec-trt-decoder)
  add_dependencies(surface_code-4-yaml cudaq-qec-trt-decoder)
endif()

# External decoding-server form of the same application. The quantum kernels and
# YAML generation/validation stay in surface_code-4-yaml.cpp; only the device-call
# lowering and client-side simulation library differ. Decoder construction lives
# exclusively in decoding_server for this target.
if(TARGET decoding_server
   AND TARGET cudaq-qec-realtime-decoding-simulation-cqr)
  add_executable(surface_code-4-yaml-cqr)
  set_target_properties(surface_code-4-yaml-cqr PROPERTIES LINKER_LANGUAGE CXX)

  cudaqx_add_device_code(surface_code-4-yaml-cqr
    SOURCES
      surface_code-4-yaml.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
      -DQEC_APP_EXTERNAL_DECODING_SERVER
  )

  target_link_directories(surface_code-4-yaml-cqr
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  target_link_libraries(surface_code-4-yaml-cqr
    PRIVATE
      cudaq-qec
      cudaq-qec-realtime-decoding
      cudaq-qec-realtime-decoding-simulation-cqr
      cudaq-device-call-runtime
      cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
      nvqir nvqir-stim)
  qec_realtime_app_link_options(surface_code-4-yaml-cqr)

  set_target_properties(surface_code-4-yaml-cqr PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )

  if(TARGET cudaq-qec-pymatching)
    add_dependencies(surface_code-4-yaml-cqr cudaq-qec-pymatching)
  endif()
  if(TARGET cudaq-qec-trt-decoder)
    add_dependencies(surface_code-4-yaml-cqr cudaq-qec-trt-decoder)
  endif()

  # Two-process MIXED-DISPATCH test (host LUT ring + device_graph nv-qldpc
  # ring in one server, wire named only by the YAML transport section).
  # Registered only when the server actually links the device-graph
  # component -- the same condition the server link uses in
  # tools/decoding-server/CMakeLists.txt (device-graph compiled AND the
  # proprietary cudevice archive present). Gating on the proprietary target
  # alone would register the test on a box where CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
  # is false, so the server cannot do device_graph and the test false-FAILs.
  # The script additionally skips (77) without a CUDA GPU or the nv-qldpc
  # plugin at runtime.
  if(CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE AND TARGET cudaq-qec-realtime-cudevice-proprietary)
    add_test(
      NAME app_examples.surface_code-4-yaml-mixed-dispatch
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/surface_code-4-yaml-mixed-dispatch-test.sh
              $<TARGET_FILE:surface_code-4-yaml-cqr>
              $<TARGET_FILE:decoding_server>)
    set_tests_properties(app_examples.surface_code-4-yaml-mixed-dispatch
      PROPERTIES LABELS "qec;gpu_required" SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()

  # Test-only plugin: its realtime decode calls rendezvous at a reusable barrier,
  # making three-worker overlap deterministic rather than scheduler-dependent.
  add_library(cudaq-qec-concurrency-test-decoder SHARED
    concurrency_test_decoder.cpp)
  target_compile_features(cudaq-qec-concurrency-test-decoder PUBLIC cxx_std_20)
  target_include_directories(cudaq-qec-concurrency-test-decoder
    PUBLIC
      ${CMAKE_SOURCE_DIR}/libs/qec/include
      ${CMAKE_SOURCE_DIR}/libs/core/include)
  # Like every other decoder plugin, this one needs only the decoder interface.
  # It must not link cudaq-qec or the CUDA-Q libraries: a decoder plugin is
  # dlopen'd by decoding_server, which no longer preloads CUDA-Q, so a plugin
  # that drags in libcudaq-qec.so fails to load with an undefined NVQIR symbol.
  target_link_libraries(cudaq-qec-concurrency-test-decoder
    PRIVATE
      cudaq-qec-decoders)
  set_target_properties(cudaq-qec-concurrency-test-decoder PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
    BUILD_RPATH "$ORIGIN"
  )

  add_dependencies(surface_code-4-yaml-cqr
    cudaq-qec-concurrency-test-decoder)
  add_dependencies(CUDAQXQECUnitTests
    surface_code-4-yaml-cqr
    cudaq-qec-concurrency-test-decoder)
endif()

# Surface-code-4 YAML tests: exercise both d=3 and d=5 for each decoder entry.
# Regular entries run through the direct-call decode path; the cqr variants
# exercise the external decoding server over UDP.
find_package(Python COMPONENTS Interpreter QUIET)
set(_SC4_TRY_TRT_TEST FALSE)
if(Python_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import onnx"
    RESULT_VARIABLE _sc4_onnx_python_result
    OUTPUT_QUIET
    ERROR_QUIET)
  if(_sc4_onnx_python_result EQUAL 0)
    set(_SC4_TRY_TRT_TEST TRUE)
  endif()
endif()
set(_SC4_HAVE_NV_QLDPC FALSE)
if(TARGET cudaq-qec-nv-qldpc-decoder
   OR QEC_EXTERNAL_DECODERS
   OR DEFINED ENV{QEC_EXTERNAL_DECODERS})
  set(_SC4_HAVE_NV_QLDPC TRUE)
endif()
# Same lesson as the trt gate below: name the gate that actually failed rather
# than letting the nv-qldpc cases evaporate silently.
if(NOT _SC4_HAVE_NV_QLDPC)
  message(WARNING "surface_code-4-yaml: nv-qldpc-decoder tests NOT registered "
    "(no cudaq-qec-nv-qldpc-decoder target and QEC_EXTERNAL_DECODERS unset); "
    "point QEC_EXTERNAL_DECODERS at the plugin to enable them.")
endif()
# A silent skip here once masked a dead test check for a whole release cycle:
# the trt surface-code tests register only when BOTH the trt plugin target and
# python-onnx are available, so name whichever gate actually failed.
if(NOT TARGET cudaq-qec-trt-decoder)
  message(WARNING "surface_code-4-yaml: trt_decoder tests NOT registered "
    "(cudaq-qec-trt-decoder target not built).")
elseif(NOT _SC4_TRY_TRT_TEST)
  if(Python_Interpreter_FOUND)
    message(WARNING "surface_code-4-yaml: trt_decoder tests NOT registered "
      "(python module 'onnx' not importable by ${Python_EXECUTABLE}); "
      "pip install onnx to enable them.")
  else()
    message(WARNING "surface_code-4-yaml: trt_decoder tests NOT registered "
      "(no python interpreter found).")
  endif()
endif()

function(add_surface_code_4_yaml_test test_suffix distance num_rounds
         decoder_type num_shots)
  set(_sc4_env "")
  if(Python_Interpreter_FOUND)
    set(_sc4_env "PYTHON=${Python_EXECUTABLE}")
  endif()

  add_test(
    NAME app_examples.surface_code-4-yaml-${test_suffix}
    COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-4-yaml-test.sh"
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-4-yaml
        ${distance} ${num_rounds}
        ${decoder_type} ${num_shots} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(
    app_examples.surface_code-4-yaml-${test_suffix}
    PROPERTIES ENVIRONMENT "${_sc4_env}")
endfunction()

function(add_surface_code_4_yaml_external_test test_suffix distance num_rounds
         decoder_type num_shots)
  set(_sc4_external_env
    "QEC_DECODING_SERVER=$<TARGET_FILE:decoding_server>")
  if(Python_Interpreter_FOUND)
    list(APPEND _sc4_external_env "PYTHON=${Python_EXECUTABLE}")
  endif()

  add_test(
    NAME app_examples.surface_code-4-yaml-cqr-${test_suffix}
    COMMAND
      ${CMAKE_COMMAND} -E env ${_sc4_external_env}
      bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-4-yaml-test.sh"
        $<TARGET_FILE:surface_code-4-yaml-cqr>
        ${distance} ${num_rounds} ${decoder_type} ${num_shots}
        --num_logical 3 ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(
    app_examples.surface_code-4-yaml-cqr-${test_suffix}
    PROPERTIES TIMEOUT 300)
endfunction()

# This example decodes ONE volume of num_rounds rounds (no sliding windows), so
# each case runs num_rounds == distance rounds.
function(add_surface_code_4_yaml_decoder_cases label decoder_type)
  add_surface_code_4_yaml_test(${label}-d3 3 3
    ${decoder_type} 200 ${ARGN})
  add_surface_code_4_yaml_test(${label}-d5 5 5
    ${decoder_type} 200 ${ARGN})
endfunction()

add_surface_code_4_yaml_decoder_cases(pymatching pymatching)

# TensorRT predecoder + PyMatching global decoder. The shell driver generates a
# tiny ONNX model sized for each test case: output [pre_L=0, residual], so TRT
# preserves the syndrome and PyMatching performs the correction.
if(TARGET cudaq-qec-trt-decoder AND _SC4_TRY_TRT_TEST)
  add_surface_code_4_yaml_decoder_cases(trt-pymatching trt_decoder
    --generate-identity-onnx)
endif()

# Relaxed round rule (T=6 not a multiple of d=5) and multi-logical patches, each
# through the full generate + reload + decode driver -- so no hand-written
# generate/reload duplication, and both get the driver's residual-LER check.
add_surface_code_4_yaml_test(d5t6 5 6 pymatching 20)
add_surface_code_4_yaml_test(multilogical 3 3 pymatching 20 --num_logical 2)

# Multi-decoder-type cases: --decoder_type lists one decoder per patch (patch i
# decodes with entry i through one realtime session). list-parse exercises the
# list machinery with no new dependencies (homogeneous pair); the trio cases
# bind three patches to heterogeneous decoder configs -- d5 exercises the
# vector result path beyond the old num_logical*d^2 < 64 packing ceiling.
add_surface_code_4_yaml_test(list-parse-d3 3 3 "pymatching,pymatching" 20
  --num_logical 2)
if(TARGET cudaq-qec-trt-decoder AND _SC4_TRY_TRT_TEST)
  add_surface_code_4_yaml_test(trio-trt-d3 3 3
    "pymatching,trt_decoder,pymatching" 200 --generate-identity-onnx
    --num_logical 3)
  add_surface_code_4_yaml_test(trio-trt-d5 5 5
    "pymatching,trt_decoder,pymatching" 200 --generate-identity-onnx
    --num_logical 3)
endif()

if(TARGET surface_code-4-yaml-cqr)
  add_surface_code_4_yaml_external_test(three-way-overlap 3 3
    "concurrency_test_decoder,concurrency_test_decoder,concurrency_test_decoder"
    2)
  set_property(TEST app_examples.surface_code-4-yaml-cqr-three-way-overlap
    APPEND PROPERTY ENVIRONMENT
      "QEC_CONCURRENCY_TEST_DECODERS=3;REQUIRE_DECODER_CONCURRENCY=3;EXPECTED_BARRIER_COMPLETIONS=2;EXPECTED_DECODER_CORRECTIONS=0,2,0;EXPECTED_SERVER_DECODER_CONSTRUCTIONS=3;CHECK_MISSING_SERVER_PORT=1;SKIP_LOGICAL_ERROR_CEILING=1")

  if(TARGET cudaq-qec-trt-decoder AND _SC4_TRY_TRT_TEST)
    add_surface_code_4_yaml_external_test(happy-path 3 3
      "pymatching,trt_decoder,pymatching" 200 --generate-identity-onnx)
    set_property(TEST app_examples.surface_code-4-yaml-cqr-happy-path
      APPEND PROPERTY ENVIRONMENT
        "REQUIRE_SERVER_DECODE_COUNTS=1")
  endif()

  # A real-workload skew case: all three patches use PyMatching, but patch 1
  # receives a much higher physical error rate and a correspondingly generated
  # decoder model. This exercises independent per-ring dispatchers without a
  # test-only delay in the decoder implementation.  Requests execute inline
  # on the dispatcher threads, so the >=2 concurrency evidence comes from
  # patch 1's slow decode overlapping a neighbor's — an opportunistic race
  # per shot; 500 shots make observing at least one overlap statistically
  # certain (the deterministic proof is the three-way-overlap barrier test).
  add_surface_code_4_yaml_external_test(hard-patch-pymatching 7 7
    "pymatching,pymatching,pymatching" 500
    --p_spam_per_patch 0.001,0.20,0.001)
  set_property(TEST app_examples.surface_code-4-yaml-cqr-hard-patch-pymatching
    APPEND PROPERTY ENVIRONMENT
      "CHECK_HARD_PATCH_MODELS=1;REQUIRE_DECODER_CONCURRENCY=2;SKIP_LOGICAL_ERROR_CEILING=1")

endif()

# --------------------------------------------------------------------------- #
# Negative / boundary tests for geometry, replay, and multi-logical reload.
# PASS_REGULAR_EXPRESSION alone ignores exit status, so _sc4_reject checks both.
# --------------------------------------------------------------------------- #
set(_sc4_exe ${CMAKE_CURRENT_BINARY_DIR}/surface_code-4-yaml)
set(_sc4_neg_dir ${CMAKE_CURRENT_BINARY_DIR}/negative)
file(MAKE_DIRECTORY ${_sc4_neg_dir})
set(_sc4_neg_env "CUDAQ_DEFAULT_SIMULATOR=stim")

# A rejection case must satisfy BOTH conditions: a nonzero exit AND the expected
# diagnostic. CTest's PASS_REGULAR_EXPRESSION ignores the exit code, so a wrapper
# enforces both -- a future regression that prints the message but returns 0
# then fails the test. The bash script is a literal bracket argument; the
# diagnostic, executable, and app args are passed as positionals so the
# diagnostic (which may contain quotes or leading dashes) needs no escaping.
function(_sc4_reject name diagnostic)
  add_test(NAME app_examples.surface_code-4-yaml-negative-${name}
    COMMAND bash -c [==[
diag=$1; shift
if out=$("$@" 2>&1); then
  printf '%s\n' "$out"
  echo "NEGATIVE TEST FAIL: command returned 0; expected a nonzero exit"
  exit 1
fi
printf '%s\n' "$out"
printf '%s' "$out" | grep -qF -- "$diag" || {
  echo "NEGATIVE TEST FAIL: missing expected diagnostic: $diag"
  exit 1
}
]==] _ "${diagnostic}" ${_sc4_exe} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(app_examples.surface_code-4-yaml-negative-${name}
    PROPERTIES ENVIRONMENT "${_sc4_neg_env}")
endfunction()

# Positive-path assertion: the command must exit 0 AND print the expected
# marker. Uses a wrapper for the same reason as _sc4_reject -- CTest's
# PASS_REGULAR_EXPRESSION ignores the exit code, and here we want both.
function(_sc4_accept name marker)
  add_test(NAME app_examples.surface_code-4-yaml-${name}
    COMMAND bash -c [==[
mark=$1; shift
if ! out=$("$@" 2>&1); then
  printf '%s\n' "$out"
  echo "POSITIVE TEST FAIL: command returned nonzero; expected success"
  exit 1
fi
printf '%s\n' "$out"
printf '%s' "$out" | grep -qF -- "$mark" || {
  echo "POSITIVE TEST FAIL: missing expected output: $mark"
  exit 1
}
]==] _ "${marker}" ${_sc4_exe} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(app_examples.surface_code-4-yaml-${name}
    PROPERTIES ENVIRONMENT "${_sc4_neg_env}")
endfunction()

# distance guard: an even or too-small distance is rejected (one guard, one
# diagnostic) before code construction; d=4 covers it.
_sc4_reject(distance-even "distance must be an odd integer"
  --distance 4 --num_rounds 4 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/even.yml)

# rounds guard: a single round has no cross-round detectors, so require >= 2.
_sc4_reject(rounds-too-few "must be >= 2"
  --distance 3 --num_rounds 1 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/t1.yml)

# replay requires --yaml.
_sc4_reject(replay-needs-yaml "load_syndrome requires --yaml"
  --load_syndrome ${_sc4_neg_dir}/none.txt --distance 3 --num_rounds 3)

# --decoder_type list guards: length must be 1 or num_logical, entries must
# name known decoders (a trailing comma is an empty entry), --use-relay-bp
# needs an nv-qldpc entry at generation, and the vector-result guards bound
# distance (numData < 64) and num_logical (correction bitmask width).
_sc4_reject(list-length-mismatch "expected 1 or num_logical"
  --distance 3 --num_rounds 3 --num_logical 3
  --decoder_type pymatching,pymatching
  --save_dem ${_sc4_neg_dir}/len.yml)
_sc4_reject(list-unknown-entry "--decoder_type entries must be"
  --distance 3 --num_rounds 3 --decoder_type pymatching,foo
  --save_dem ${_sc4_neg_dir}/unknown.yml)
_sc4_reject(list-trailing-comma "--decoder_type entries must be"
  --distance 3 --num_rounds 3 --decoder_type "pymatching,"
  --save_dem ${_sc4_neg_dir}/trail.yml)
_sc4_reject(relay-without-nv "--use-relay-bp requires"
  --distance 3 --num_rounds 3 --decoder_type pymatching --use-relay-bp
  --save_dem ${_sc4_neg_dir}/relaynonv.yml)
_sc4_reject(distance-too-large "only numData < 64"
  --distance 9 --num_rounds 9 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/d9.yml)
_sc4_reject(num-logical-too-large "--num_logical must be in [1, 64]"
  --distance 3 --num_rounds 3 --num_logical 65 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/nl65.yml)
# Pathological value: pins that the range guard fires BEFORE the decoder-type
# list is replicated num_logical times (no num_logical-sized allocation).
_sc4_reject(num-logical-pathological "--num_logical must be in [1, 64]"
  --distance 3 --num_rounds 3 --num_logical 2000000000
  --decoder_type pymatching --save_dem ${_sc4_neg_dir}/nlhuge.yml)
# Malformed numeric flags must error cleanly, not abort on an uncaught
# stoi/stod exception; > INT_MAX values take the same path.
_sc4_reject(distance-not-a-number "--distance expects an integer"
  --distance abc --num_rounds 3 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/nan.yml)
_sc4_reject(num-logical-overflows-int "--num_logical expects an integer"
  --distance 3 --num_rounds 3 --num_logical 5000000000
  --decoder_type pymatching --save_dem ${_sc4_neg_dir}/nlof.yml)
# Semantic ranges: lexically valid numbers outside the flag's domain must be
# rejected (a negative shot count would wrap to an enormous size_t; p_spam is
# advertised as [0, 1] and nan/inf must not pass).
_sc4_reject(num-shots-nonpositive "--num_shots must be >= 1"
  --distance 3 --num_rounds 3 --num_shots 0 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/ns0.yml)
_sc4_reject(num-shots-negative "--num_shots must be >= 1"
  --distance 3 --num_rounds 3 --num_shots -1 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/nsneg.yml)
_sc4_reject(p-spam-out-of-range "--p_spam must be a finite value in [0, 1]"
  --distance 3 --num_rounds 3 --p_spam 1.5 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/ps15.yml)
_sc4_reject(p-spam-nan "--p_spam must be a finite value in [0, 1]"
  --distance 3 --num_rounds 3 --p_spam nan --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/psnan.yml)
_sc4_reject(p-spam-per-patch-length "expected 1 or num_logical"
  --distance 3 --num_rounds 3 --num_logical 3
  --p_spam_per_patch 0.01,0.20 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/psplen.yml)
_sc4_reject(p-spam-per-patch-malformed "comma-separated list of numbers"
  --distance 3 --num_rounds 3 --num_logical 3
  --p_spam_per_patch 0.01,nope,0.01 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/pspbad.yml)
_sc4_reject(p-spam-per-patch-out-of-range "must be a finite value in [0, 1]"
  --distance 3 --num_rounds 3 --num_logical 3
  --p_spam_per_patch 0.01,1.5,0.01 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/psprange.yml)
# p_spam = 0 is in [0, 1] (valid for a --yaml run) but cannot generate a DEM;
# generation must fail with a nonzero status, not exit 0 with no config file.
_sc4_reject(p-spam-zero-generation "--p_spam must be > 0 to generate"
  --distance 3 --num_rounds 3 --p_spam 0 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/ps0.yml)

# Dual-parse pin: generation instantiates no decoder, but YAML emission
# requires the nv-qldpc-decoder parameter schema (registered by the plugin).
# Structural check: the BP entry carries the undecomposed hyperedge H --
# strictly fewer columns than the matching entry's decomposed H.
# Full check (python3 available): the mixed-list BP entry's column multiset
# of (detector support, observable support, probability) is IDENTICAL to a
# directly generated undecomposed reference config at the same geometry.
function(_sc4_dual_parse_test distance num_rounds)
  add_test(NAME app_examples.surface_code-4-yaml-dual-parse-gen-d${distance}
    COMMAND bash -c [==[
set -e
exe=$1; mixed=$2; pure=$3; d=$4; t=$5
"$exe" --distance $d --num_rounds $t --num_logical 2 \
  --decoder_type pymatching,nv-qldpc-decoder --p_spam 0.01 --save_dem "$mixed"
"$exe" --distance $d --num_rounds $t \
  --decoder_type nv-qldpc-decoder --p_spam 0.01 --save_dem "$pure"
read -r nv match < <(awk '/- id:/{n++} /type:/{t[n]=$2} /block_size:/{b[n]=$2}
  END{for(i=1;i<=n;i++){if(t[i]=="nv-qldpc-decoder")nv=b[i];else m=b[i]}
  print nv, m}' "$mixed")
echo "nv block_size=$nv matching block_size=$match"
test -n "$nv" && test -n "$match" && test "$nv" -lt "$match"
if command -v python3 >/dev/null 2>&1; then
python3 - "$mixed" "$pure" << 'PY'
import re, sys

def nv_columns(path):
    txt = open(path).read()
    for chunk in re.split(r"(?m)^\s*- id:", txt)[1:]:
        typ = re.search(r"type:\s*(\S+)", chunk).group(1)
        if typ != "nv-qldpc-decoder":
            continue
        block_size = int(re.search(r"block_size:\s*(\d+)", chunk).group(1))

        def sparse_rows(name):
            m = re.search(name + r":\s*\[([^\]]*)\]", chunk, re.S)
            rows, cur = [], []
            for tok in m.group(1).replace("\n", " ").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                v = int(tok)
                if v == -1:
                    rows.append(cur)
                    cur = []
                else:
                    cur.append(v)
            return rows

        def col_map(rows):
            cols = {}
            for r, row in enumerate(rows):
                for c in row:
                    cols.setdefault(c, []).append(r)
            return cols

        hcol = col_map(sparse_rows("H_sparse"))
        ocol = col_map(sparse_rows("O_sparse"))
        pm = re.search(r"error_rate_vec:\s*\[([^\]]*)\]", chunk, re.S)
        probs = [round(float(t), 12)
                 for t in pm.group(1).replace("\n", " ").split(",")
                 if t.strip()]
        if len(probs) != block_size:
            raise SystemExit("prob count %d != block_size %d in %s" %
                             (len(probs), block_size, path))
        return sorted((tuple(sorted(hcol.get(c, []))),
                       tuple(sorted(ocol.get(c, []))), probs[c])
                      for c in range(block_size))
    raise SystemExit("no nv-qldpc-decoder entry in " + path)

mixed = nv_columns(sys.argv[1])
pure = nv_columns(sys.argv[2])
if mixed != pure:
    print("FAIL: mixed-list BP entry differs from undecomposed reference "
          "(%d vs %d columns)" % (len(mixed), len(pure)))
    for a, b in zip(mixed, pure):
        if a != b:
            print("first differing column:", a, "vs", b)
            break
    raise SystemExit(1)
print("full column multiset (dets, obs, prob) identical: %d columns" %
      len(mixed))
PY
else
  echo "python3 not found; full multiset check skipped (structural check only)"
fi
]==] _ ${_sc4_exe} ${_sc4_neg_dir}/dualparse-mixed-d${distance}.yml
     ${_sc4_neg_dir}/dualparse-pure-d${distance}.yml ${distance} ${num_rounds}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(app_examples.surface_code-4-yaml-dual-parse-gen-d${distance}
    PROPERTIES ENVIRONMENT "${_sc4_neg_env}")
endfunction()
if(_SC4_HAVE_NV_QLDPC)
  _sc4_dual_parse_test(3 3)
  _sc4_dual_parse_test(5 5)
endif()

# Geometry binding: a d5/T5 config rejected when loaded as d3/T15 (same 120
# detectors, different measurement-buffer span).
add_test(NAME app_examples.surface_code-4-yaml-negative-d5t5-generate
  COMMAND ${_sc4_exe} --distance 5 --num_rounds 5 --num_shots 20 --p_spam 0.01
    --decoder_type pymatching --save_dem ${_sc4_neg_dir}/d5t5.yml
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-4-yaml-negative-d5t5-generate
  PROPERTIES FIXTURES_SETUP sc4_d5t5 ENVIRONMENT "${_sc4_neg_env}")
_sc4_reject(geometry-mismatch "measurement-buffer span"
  --yaml ${_sc4_neg_dir}/d5t5.yml --distance 3 --num_rounds 15
  --num_shots 20 --p_spam 0.01)
set_tests_properties(app_examples.surface_code-4-yaml-negative-geometry-mismatch
  PROPERTIES FIXTURES_REQUIRED sc4_d5t5)

# Replay structural integrity: build a valid d3 capture, then derive a truncated
# (footer dropped) and a corrections-flipped copy. Replaying a missing,
# truncated, or correction-mismatched capture must fail with a diagnostic.
add_test(NAME app_examples.surface_code-4-yaml-negative-replay-fixtures
  COMMAND bash -c "set -e
'${_sc4_exe}' --distance 3 --num_rounds 3 --num_shots 5 --p_spam 0.01 --decoder_type pymatching --save_dem '${_sc4_neg_dir}/d3.yml'
'${_sc4_exe}' --yaml '${_sc4_neg_dir}/d3.yml' --distance 3 --num_rounds 3 --num_shots 5 --p_spam 0.01 --save_syndrome '${_sc4_neg_dir}/cap.txt'
sed '/^CORRECTIONS_START/,$d' '${_sc4_neg_dir}/cap.txt' > '${_sc4_neg_dir}/cap_trunc.txt'
sed '/CORRECTIONS_START/,/CORRECTIONS_END/{s/^0$/X/;s/^1$/0/;s/^X$/1/}' '${_sc4_neg_dir}/cap.txt' > '${_sc4_neg_dir}/cap_mismatch.txt'"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-fixtures
  PROPERTIES FIXTURES_SETUP sc4_replay ENVIRONMENT "${_sc4_neg_env}")

_sc4_reject(replay-missing-file "Could not open syndrome file"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/missing.txt
  --distance 3 --num_rounds 3)
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-missing-file
  PROPERTIES FIXTURES_REQUIRED sc4_replay)
_sc4_reject(replay-truncated "missing the CORRECTIONS_START/CORRECTIONS_END"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/cap_trunc.txt
  --distance 3 --num_rounds 3 --num_shots 5)
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-truncated
  PROPERTIES FIXTURES_REQUIRED sc4_replay)
_sc4_reject(replay-correction-mismatch "replay correction mismatch"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/cap_mismatch.txt
  --distance 3 --num_rounds 3 --num_shots 5)
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-correction-mismatch
  PROPERTIES FIXTURES_REQUIRED sc4_replay)

# Positive replay: the untampered capture must round-trip -- every replayed
# correction matches the captured run. Proves the load/replay path works on good
# input, not just that it rejects tampered captures.
_sc4_accept(replay-roundtrip "SUCCESS: All corrections match"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/cap.txt
  --distance 3 --num_rounds 3 --num_shots 5)
set_tests_properties(app_examples.surface_code-4-yaml-replay-roundtrip
  PROPERTIES FIXTURES_REQUIRED sc4_replay)

# ---------------------------------------------------------------------------
# Surface code 5: one ring buffer (and one dispatcher) per decoder, via the
# cudaq::device_call device-id overload (device_id == decoder_id).  Runs over
# the in-process host_dispatch device_call channel; the simulation-cqr client
# lowers the decoding RPCs (-frealtime-lowering) and decoding-server-cqr
# serves them.  See surface_code-5-per-decoder-rings.cpp.
# ---------------------------------------------------------------------------
if(TARGET cudaq-qec-realtime-decoding-simulation-cqr
   AND TARGET cudaq-qec-realtime-decoding-server-cqr)
  add_executable(surface_code-5-per-decoder-rings)
  set_target_properties(surface_code-5-per-decoder-rings PROPERTIES
    LINKER_LANGUAGE CXX)

  target_include_directories(surface_code-5-per-decoder-rings PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  cudaqx_add_device_code(surface_code-5-per-decoder-rings
    SOURCES
      surface_code-5-per-decoder-rings.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
  )

  target_link_directories(surface_code-5-per-decoder-rings
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  # See test_pymatching_device_call_realtime for why --no-as-needed is needed
  # (dlopen(NULL)-resolved quantum platform symbols).
  target_link_options(surface_code-5-per-decoder-rings PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>)

  target_link_libraries(surface_code-5-per-decoder-rings PRIVATE
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation-cqr
    cudaq-qec-realtime-decoding-server-cqr
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    CUDA::cudart
    nvqir nvqir-stim
  )

  set_target_properties(surface_code-5-per-decoder-rings PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )

  add_dependencies(surface_code-5-per-decoder-rings cudaq-qec-pymatching)
endif()

if(TARGET surface_code-5-per-decoder-rings)
  add_test(NAME app_examples.surface_code-5-per-decoder-rings
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/surface_code-5-per-decoder-rings-test.sh
            $<TARGET_FILE:surface_code-5-per-decoder-rings>)
  set_tests_properties(app_examples.surface_code-5-per-decoder-rings
    PROPERTIES LABELS "qec")
endif()

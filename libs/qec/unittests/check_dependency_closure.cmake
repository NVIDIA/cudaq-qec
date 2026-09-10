# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ---------------------------------------------------------------------------
# check_dependency_closure.cmake
#
# Architectural guardrail run via `cmake -P`.  Verifies that a set of shared
# objects (the "clean" QEC realtime / decoder libraries) do NOT pull the heavy
# CUDA-Q libraries (libcudaq-common, libcudaq-operator, libcudaq-qec) into
# their ELF dependency closure.
#
# The closure is computed purely from the DT_NEEDED / DT_RUNPATH / DT_RPATH
# entries reported by `readelf -d`; the loader is never invoked (unlike `ldd`),
# so the check is hermetic and deterministic in the build tree.
#
# Inputs (passed with -D on the command line):
#   LIBS       : "|"-separated absolute paths of shared objects to inspect.
#   FORBIDDEN  : "|"-separated soname stems that must not appear in the
#                closure, e.g. "libcudaq-common|libcudaq-operator|libcudaq-qec".
#                A stem matches a soname only when it is followed immediately by
#                ".so" (so "libcudaq-qec" matches libcudaq-qec.so.1 but NOT
#                libcudaq-qec-decoders.so).
# Optional:
#   TRANSITIVE : ON (default) walks the full DT_NEEDED closure; OFF checks only
#                the direct DT_NEEDED entries of each input library.
#   EXTRA_SEARCH_DIRS : "|"-separated extra directories used to resolve sonames
#                during the transitive walk.
#   EXPECT_FAIL: ON inverts the result -- the script succeeds only if at least
#                one forbidden dependency is found.  Used by the negative
#                self-test to prove the check actually fires.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED LIBS)
  message(FATAL_ERROR "check_dependency_closure: LIBS not provided")
endif()
if(NOT DEFINED FORBIDDEN)
  message(FATAL_ERROR "check_dependency_closure: FORBIDDEN not provided")
endif()
if(NOT DEFINED TRANSITIVE)
  set(TRANSITIVE ON)
endif()

find_program(READELF_EXECUTABLE NAMES readelf llvm-readelf)
if(NOT READELF_EXECUTABLE)
  message(FATAL_ERROR
    "check_dependency_closure: neither readelf nor llvm-readelf was found")
endif()

# Custom "|" separators are used instead of ";" because add_test() builds its
# COMMAND as a CMake list, and ";" inside a -D value would be split into
# separate arguments.
string(REPLACE "|" ";" _input_libs "${LIBS}")
string(REPLACE "|" ";" _forbidden_stems "${FORBIDDEN}")

set(_extra_search_dirs "")
if(DEFINED EXTRA_SEARCH_DIRS)
  string(REPLACE "|" ";" _extra_search_dirs "${EXTRA_SEARCH_DIRS}")
endif()

# Turn each forbidden stem into an anchored regex "<stem>\.so" so that only the
# exact library (with any version suffix) matches -- never a longer sibling
# name that merely shares the prefix.
set(_forbidden_regexes "")
foreach(_stem IN LISTS _forbidden_stems)
  string(REGEX REPLACE "([][+.*()^$?|\\])" "\\\\\\1" _escaped "${_stem}")
  list(APPEND _forbidden_regexes "^${_escaped}\\.so")
endforeach()

# Parse `readelf -d <lib>` output into the list of DT_NEEDED sonames and the
# list of DT_RUNPATH / DT_RPATH search directories.
function(read_elf_dynamic lib out_needed out_rpath)
  set(_needed "")
  set(_rpath "")
  execute_process(
    COMMAND "${READELF_EXECUTABLE}" -d "${lib}"
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "check_dependency_closure: readelf failed on ${lib}: ${_err}")
    set(${out_needed} "" PARENT_SCOPE)
    set(${out_rpath} "" PARENT_SCOPE)
    return()
  endif()

  string(REGEX MATCHALL "[^\n]+" _lines "${_out}")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "\\(NEEDED\\).*\\[(.+)\\]")
      list(APPEND _needed "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "\\((RUNPATH|RPATH)\\).*\\[(.+)\\]")
      string(REPLACE ":" ";" _dirs "${CMAKE_MATCH_2}")
      list(APPEND _rpath ${_dirs})
    endif()
  endforeach()

  set(${out_needed} "${_needed}" PARENT_SCOPE)
  set(${out_rpath} "${_rpath}" PARENT_SCOPE)
endfunction() # end - read_elf_dynamic()

# Resolve a soname to a concrete file by scanning the provided search dirs in
# order.  Returns an empty string when the soname cannot be located.
function(resolve_soname soname search_dirs out_path)
  foreach(_dir IN LISTS search_dirs)
    if(EXISTS "${_dir}/${soname}")
      # Canonicalize so the same file reached via different path spellings
      # (e.g. an $ORIGIN/.. rpath vs an absolute dir) dedupes in `_visited`.
      get_filename_component(_real "${_dir}/${soname}" REALPATH)
      set(${out_path} "${_real}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_path} "" PARENT_SCOPE)
endfunction() # end - resolve_soname()

# Breadth-first walk over the dependency graph.  `_queue` holds the libraries
# still to inspect and `_chains` holds, in lock-step, the human-readable path
# that led to each queued library (used for diagnostics).
set(_visited "")
set(_queue "")
set(_chains "")
set(_input_names "")
foreach(_lib IN LISTS _input_libs)
  if(NOT EXISTS "${_lib}")
    message(FATAL_ERROR "check_dependency_closure: library not found: ${_lib}")
  endif()
  get_filename_component(_lib "${_lib}" REALPATH)
  list(APPEND _queue "${_lib}")
  get_filename_component(_libname "${_lib}" NAME)
  list(APPEND _chains "${_libname}")
  list(APPEND _input_names "${_libname}")
endforeach()

list(LENGTH _input_libs _input_count)
string(REPLACE ";" ", " _forbidden_display "${_forbidden_stems}")
message(STATUS
  "check_dependency_closure: inspecting ${_input_count} library(ies) for "
  "forbidden dependencies [${_forbidden_display}]:")
foreach(_name IN LISTS _input_names)
  message(STATUS "    - ${_name}")
endforeach()

set(_violations "")

while(_queue)
  list(POP_FRONT _queue _lib)
  list(POP_FRONT _chains _chain)

  if(_lib IN_LIST _visited)
    continue()
  endif()
  list(APPEND _visited "${_lib}")

  read_elf_dynamic("${_lib}" _needed _rpath)
  get_filename_component(_libdir "${_lib}" DIRECTORY)

  # Assemble the soname search path for this library: its own directory first,
  # then its (expanded) RUNPATH/RPATH entries, then any caller-provided extras.
  set(_search "${_libdir}")
  foreach(_r IN LISTS _rpath)
    string(REPLACE "$ORIGIN" "${_libdir}" _r "${_r}")
    string(REPLACE "\${ORIGIN}" "${_libdir}" _r "${_r}")
    list(APPEND _search "${_r}")
  endforeach()
  list(APPEND _search ${_extra_search_dirs})

  foreach(_soname IN LISTS _needed)
    foreach(_re IN LISTS _forbidden_regexes)
      if(_soname MATCHES "${_re}")
        list(APPEND _violations "${_chain} -> ${_soname}")
      endif()
    endforeach()

    if(TRANSITIVE)
      resolve_soname("${_soname}" "${_search}" _resolved)
      if(_resolved AND NOT _resolved IN_LIST _visited)
        list(APPEND _queue "${_resolved}")
        list(APPEND _chains "${_chain} -> ${_soname}")
      endif()
    endif()
  endforeach()
endwhile()

if(_violations)
  list(REMOVE_DUPLICATES _violations)
endif()

# Report.  In EXPECT_FAIL mode the presence of violations is the success
# condition (negative self-test); otherwise any violation is a hard failure.
if(EXPECT_FAIL)
  if(_violations)
    message(STATUS
      "check_dependency_closure: expected forbidden dependency was found "
      "(negative self-test passed).")
    return()
  endif()
  message(FATAL_ERROR
    "check_dependency_closure: EXPECT_FAIL was set but no forbidden "
    "dependency was found in the closure of: ${_input_libs}")
endif()

if(_violations)
  set(_msg "check_dependency_closure: forbidden CUDA-Q libraries found in the dependency closure:\n")
  foreach(_v IN LISTS _violations)
    string(APPEND _msg "    ${_v}\n")
  endforeach()
  message(FATAL_ERROR "${_msg}")
endif()

# Success: summarize the libraries that were checked and the full closure that
# was walked to reach that verdict.
string(REPLACE ";" ", " _input_display "${_input_names}")
list(LENGTH _visited _visited_count)
message(STATUS
  "check_dependency_closure: OK -- no forbidden CUDA-Q libraries "
  "[${_forbidden_display}] in the closure of: ${_input_display}")
message(STATUS
  "check_dependency_closure: ${_visited_count} library(ies) walked in the "
  "closure:")
set(_closure_names "")
foreach(_lib IN LISTS _visited)
  get_filename_component(_name "${_lib}" NAME)
  list(APPEND _closure_names "${_name}")
endforeach()
list(SORT _closure_names)
foreach(_name IN LISTS _closure_names)
  message(STATUS "    - ${_name}")
endforeach()

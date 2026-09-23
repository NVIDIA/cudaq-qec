# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#[=======================================================================[.rst:
FindcuStabilizer
----------------

Find the cuStabilizer library (shipped as a pip wheel: custabilizer-cu12 / cu13).

Imported Targets
^^^^^^^^^^^^^^^^

``cuStabilizer::cuStabilizer``
  The cuStabilizer library.

Result Variables
^^^^^^^^^^^^^^^^

``cuStabilizer_FOUND``
``cuStabilizer_INCLUDE_DIR``
``cuStabilizer_LIBRARY``
``cuStabilizer_LIBRARY_DIR``
  Directory containing the cuStabilizer shared library.  Useful for adding to
  ``BUILD_RPATH`` / ``INSTALL_RPATH`` on consuming targets so the loader can
  find ``libcustabilizer.so`` at runtime.

Hints
^^^^^

``CUSTABILIZER_ROOT``
  Preferred search prefix.  Accepted as a CMake variable (``-DCUSTABILIZER_ROOT=...``)
  or an environment variable.

``CUQUANTUM_ROOT``
  Standard cuQuantum SDK install prefix (CMake variable or environment
  variable).  Used as a fallback when ``CUSTABILIZER_ROOT`` is not set.  The
  cuquantum-python pip wheels expose this naturally as
  ``<site-packages>/cuquantum``.

If neither variable is provided, this module will additionally probe the
active Python environment for the ``custabilizer-cuXX`` / ``cuquantum-cuXX``
pip wheels, which ship the headers and library under
``<site-packages>/cuquantum/{include,lib}``.

#]=======================================================================]

cmake_policy(SET CMP0144 NEW)

if(NOT CUSTABILIZER_ROOT AND DEFINED ENV{CUSTABILIZER_ROOT})
  set(CUSTABILIZER_ROOT "$ENV{CUSTABILIZER_ROOT}")
endif()

# Fall back to the standard cuQuantum SDK env var.  This handles the common
# case of a system-wide cuQuantum install (CUQUANTUM_ROOT=/opt/nvidia/cuquantum)
# as well as the cuquantum-python pip wheel layout, where downstream tooling
# already exports CUQUANTUM_ROOT=<site-packages>/cuquantum.  In particular, this
# is what lets us discover the library when scikit-build runs the configure in
# an isolated build venv whose Python_EXECUTABLE has no cuquantum-python wheel
# installed (e.g. `python -m build`), but the outer environment does.
if(NOT CUSTABILIZER_ROOT)
  if(CUQUANTUM_ROOT)
    set(CUSTABILIZER_ROOT "${CUQUANTUM_ROOT}")
  elseif(DEFINED ENV{CUQUANTUM_ROOT})
    set(CUSTABILIZER_ROOT "$ENV{CUQUANTUM_ROOT}")
  endif()
  if(CUSTABILIZER_ROOT AND NOT cuStabilizer_FIND_QUIETLY)
    message(STATUS "FindcuStabilizer: using CUQUANTUM_ROOT=${CUSTABILIZER_ROOT}")
  endif()
endif()

# If still no root was found, try to discover the wheel layout from the active
# Python interpreter.  The custabilizer-cuXX / cuquantum-cuXX wheels install
# headers at  <site-packages>/cuquantum/include/custabilizer.h and the shared
# lib at <site-packages>/cuquantum/lib/libcustabilizer.so.0.
if(NOT CUSTABILIZER_ROOT)
  # Use whatever Python the user has on their PATH (or the one already located
  # by the parent project) without forcing a hard dependency.
  if(NOT Python3_EXECUTABLE AND NOT Python_EXECUTABLE)
    find_package(Python3 QUIET COMPONENTS Interpreter)
  endif()
  set(_custab_py "${Python3_EXECUTABLE}")
  if(NOT _custab_py)
    set(_custab_py "${Python_EXECUTABLE}")
  endif()

  if(_custab_py)
    execute_process(
      COMMAND "${_custab_py}" -c [==[
import importlib.util, pathlib, sys
for pkg in ('custabilizer', 'cuquantum'):
    spec = importlib.util.find_spec(pkg)
    if not spec or not spec.origin:
        continue
    root = pathlib.Path(spec.origin).parent
    if (root / 'include' / 'custabilizer.h').exists():
        sys.stdout.write(str(root))
        sys.exit(0)
sys.exit(1)
]==]
      OUTPUT_VARIABLE _custab_py_root
      ERROR_QUIET
      RESULT_VARIABLE _custab_py_rc
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_custab_py_rc EQUAL 0 AND _custab_py_root)
      set(CUSTABILIZER_ROOT "${_custab_py_root}")
      if(NOT cuStabilizer_FIND_QUIETLY)
        message(STATUS "FindcuStabilizer: discovered Python wheel at ${CUSTABILIZER_ROOT}")
      endif()
    endif()
  endif()
endif()

find_path(cuStabilizer_INCLUDE_DIR
  NAMES custabilizer.h
  HINTS
    ${CUSTABILIZER_ROOT}/include
)

find_library(cuStabilizer_LIBRARY
  NAMES custabilizer libcustabilizer.so.0
  HINTS
    ${CUSTABILIZER_ROOT}/lib64
    ${CUSTABILIZER_ROOT}/lib
)

set(cuStabilizer_VERSION "")
if(cuStabilizer_INCLUDE_DIR AND EXISTS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h")
  file(STRINGS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h"
       _custab_major_line REGEX "^#define[ \t]+CUSTABILIZER_MAJOR[ \t]+[0-9]+")
  file(STRINGS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h"
       _custab_minor_line REGEX "^#define[ \t]+CUSTABILIZER_MINOR[ \t]+[0-9]+")
  file(STRINGS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h"
       _custab_patch_line REGEX "^#define[ \t]+CUSTABILIZER_PATCH[ \t]+[0-9]+")
  if(_custab_major_line AND _custab_minor_line AND _custab_patch_line)
    string(REGEX REPLACE "^#define[ \t]+CUSTABILIZER_MAJOR[ \t]+([0-9]+).*$" "\\1"
           _custab_major "${_custab_major_line}")
    string(REGEX REPLACE "^#define[ \t]+CUSTABILIZER_MINOR[ \t]+([0-9]+).*$" "\\1"
           _custab_minor "${_custab_minor_line}")
    string(REGEX REPLACE "^#define[ \t]+CUSTABILIZER_PATCH[ \t]+([0-9]+).*$" "\\1"
           _custab_patch "${_custab_patch_line}")
    set(cuStabilizer_VERSION "${_custab_major}.${_custab_minor}.${_custab_patch}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuStabilizer
  REQUIRED_VARS cuStabilizer_INCLUDE_DIR cuStabilizer_LIBRARY
  VERSION_VAR cuStabilizer_VERSION
)

if(cuStabilizer_LIBRARY)
  get_filename_component(cuStabilizer_LIBRARY_DIR "${cuStabilizer_LIBRARY}" DIRECTORY)
endif()

if(cuStabilizer_FOUND AND NOT TARGET cuStabilizer::cuStabilizer)
  add_library(cuStabilizer::cuStabilizer INTERFACE IMPORTED GLOBAL)
  set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${cuStabilizer_INCLUDE_DIR}"
  )
  if(cuStabilizer_LIBRARY)
    set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
      INTERFACE_LINK_LIBRARIES "${cuStabilizer_LIBRARY}"
    )
  endif()
  if(cuStabilizer_LIBRARY_DIR)
    set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
      INTERFACE_LINK_DIRECTORIES "${cuStabilizer_LIBRARY_DIR}"
    )
  endif()
endif()

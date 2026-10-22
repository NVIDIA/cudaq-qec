# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #

# Prefer a native installation or an explicit cudaq-edem_DIR. The Python wheel
# also ships the native package, under cudaq_edem/lib/cmake/cudaq-edem.
find_package(cudaq-edem CONFIG QUIET)
if(NOT cudaq-edem_FOUND)
  if(Python_EXECUTABLE)
    set(_cudaq_edem_python "${Python_EXECUTABLE}")
  elseif(Python3_EXECUTABLE)
    set(_cudaq_edem_python "${Python3_EXECUTABLE}")
  else()
    find_package(Python3 QUIET COMPONENTS Interpreter)
    set(_cudaq_edem_python "${Python3_EXECUTABLE}")
  endif()

  set(_cudaq_edem_cmake_dir "")
  if(_cudaq_edem_python)
    execute_process(
      COMMAND "${_cudaq_edem_python}" -c
        "import cudaq_edem; print(cudaq_edem.get_cmake_dir())"
      OUTPUT_VARIABLE _cudaq_edem_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _cudaq_edem_result)
    if(NOT _cudaq_edem_result EQUAL 0)
      set(_cudaq_edem_cmake_dir "")
    endif()
  endif()

  find_package(cudaq-edem CONFIG REQUIRED HINTS "${_cudaq_edem_cmake_dir}")
endif()

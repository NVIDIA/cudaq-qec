# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Find all the public headers
set(CUDAQX_PUBLIC_HEADERS)
file(GLOB_RECURSE LIB_PUBLIC_HEADERS "../libs/core/include/*.h")
list(APPEND CUDAQX_PUBLIC_HEADERS ${LIB_PUBLIC_HEADERS})
foreach(lib ${CUDAQX_ALL_LIBS})
  file(GLOB_RECURSE LIB_PUBLIC_HEADERS "../libs/${lib}/include/*.h")
  list(APPEND CUDAQX_PUBLIC_HEADERS ${LIB_PUBLIC_HEADERS})
endforeach()

# Find all the public Python files
set(CUDAQX_PUBLIC_PYTHON)
foreach(lib ${CUDAQX_ALL_LIBS})
  file(GLOB_RECURSE LIB_PUBLIC_PYTHON "../libs/${lib}/python/*.py")
  list(APPEND CUDAQX_PUBLIC_PYTHON ${LIB_PUBLIC_PYTHON})
endforeach()

# Find all the rst files
set(CUDAQX_PUBLIC_RST)
foreach(lib ${CUDAQX_ALL_LIBS})
  file(GLOB_RECURSE RST_FILES "*.rst")
  list(APPEND CUDAQX_PUBLIC_RST ${RST_FILES})
endforeach()

# ==============================================================================
# Doxygen
# ==============================================================================

# We need to give doxygen a space separated list of files
list(JOIN CUDAQX_PUBLIC_HEADERS " " HEADERS_LIST)

set(DOXYGEN_INPUT ${HEADERS_LIST})
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Only regenerate Doxygen when the Doxyfile or public headers change
add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  DEPENDS ${CUDAQX_PUBLIC_HEADERS}
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
  MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
  COMMENT "Generating docs"
)

add_custom_target(doxygen_docs DEPENDS ${DOXYGEN_INDEX_FILE})

# ==============================================================================
# Sphinx
# ==============================================================================

find_package(Python COMPONENTS Interpreter REQUIRED)

function(require_python_module module_name)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import ${module_name}"
    RESULT_VARIABLE result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Python module '${module_name}' not found")
  endif()
endfunction()

require_python_module(IPython)
require_python_module(breathe)
require_python_module(enum_tools)
require_python_module(myst_parser)
require_python_module(nbsphinx)
require_python_module(sphinx_copybutton)
require_python_module(sphinx_inline_tabs)
require_python_module(sphinx_gallery)
require_python_module(sphinx_rtd_theme)
require_python_module(sphinx_reredirects)
require_python_module(sphinx_toolbox)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/build)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

set(SPHINX_CONF_IN ${SPHINX_SOURCE}/conf.py.in)
set(SPHINX_CONF ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

configure_file(${SPHINX_CONF_IN} ${SPHINX_CONF} @ONLY)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(
  OUTPUT ${SPHINX_INDEX_FILE}
  COMMAND ${CMAKE_COMMAND} -E env
          LD_LIBRARY_PATH=${CUDAQ_LIBRARY_DIR}:$ENV{LD_LIBRARY_PATH}
          ${SPHINX_EXECUTABLE} -v -n -W --keep-going -b html
          -c ${CMAKE_CURRENT_BINARY_DIR}
          -Dbreathe_projects.cudaqx=${DOXYGEN_OUTPUT_DIR}/xml
          ${SPHINX_SOURCE} ${SPHINX_BUILD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${SPHINX_SOURCE}/index.rst
    ${DOXYGEN_INDEX_FILE}
    cudaqx-pymodules
    cudaq-qec-realtime-decoding-simulation
    ${CUDAQX_PUBLIC_PYTHON}
    ${CUDAQX_PUBLIC_RST}
  MAIN_DEPENDENCY ${SPHINX_CONF_IN} ${SPHINX_CONF}
  COMMENT "Generating documentation with Sphinx"
)

add_custom_target(sphinx_docs DEPENDS ${SPHINX_INDEX_FILE})

# ==============================================================================

add_custom_target(docs DEPENDS doxygen_docs sphinx_docs)

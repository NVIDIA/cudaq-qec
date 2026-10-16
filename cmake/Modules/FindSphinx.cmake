# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Unfortunately, there is no "standard way to find sphinx"

find_program(SPHINX_EXECUTABLE
  NAMES sphinx-build
  DOC "Path to sphinx-build executable"
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  Sphinx "Failed to find sphinx-build executable" SPHINX_EXECUTABLE
)

mark_as_advanced(SPHINX_EXECUTABLE)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

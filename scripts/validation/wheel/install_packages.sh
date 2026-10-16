#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

PYTHON_WHEEL_VER=$1
PYTHON_VERSION=$2

# Install packages
PYENV_VERSION=${PYTHON_VERSION} pyenv exec pip install --user \
    custatevec_cu12-1.7.0-py3-none-manylinux2014_x86_64.whl \
    cutensornet_cu12-2.6.0-py3-none-manylinux2014_x86_64.whl \
    cudensitymat_cu12-0.0.5-py3-none-manylinux2014_x86_64.whl \
    cuquantum_python_cu12-24.11.0-77-${PYTHON_WHEEL_VER}-${PYTHON_WHEEL_VER}-linux_x86_64.whl

# Install CUDA-Q packages
PYENV_VERSION=${PYTHON_VERSION} pyenv exec pip install --user matplotlib \
    wheelhouse/cuda_quantum_cu12-0.9.0-${PYTHON_WHEEL_VER}-${PYTHON_WHEEL_VER}-manylinux_2_28_x86_64.whl \
    wheels-py$(echo ${PYTHON_VERSION} | cut -d'.' -f1,2)-$(dpkg --print-architecture)/cudaq_qec-0.1.0-${PYTHON_WHEEL_VER}-${PYTHON_WHEEL_VER}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# Install test dependencies
PYENV_VERSION=${PYTHON_VERSION} pyenv exec pip install pytest networkx --user
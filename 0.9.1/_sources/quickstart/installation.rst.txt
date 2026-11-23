Installation Guide
==================

Install the Python package with pip, use a Docker container with dependencies
included, or build CUDA-Q QEC from source. Some features also require PyTorch,
as described below.

pip install
-----------

The simplest way to install CUDA-Q QEC is via pip. (If you're on Mac, your only
option is to use the Docker container as described below.)

.. code-block:: bash

    # Install QEC library
    pip install cudaq-qec

CUDA-Q QEC provides optional pip-installable components. For the Tensor
Network Decoder on CUDA 12, first read the
:ref:`PyTorch compatibility note <pytorch-cuda12-compatibility>` below.

.. code-block:: bash

    # Install the Tensor Network Decoder from the QEC library
    pip install cudaq-qec[tensor-network-decoder]

.. note::

    Looking for CUDA-Q Solvers? It has been removed from this repository and is
    superseded by **CUDA-Q Algorithms**, which is where development continues:

    .. code-block:: bash

        pip install cudaq-algorithms

    See the `CUDA-Q Algorithms documentation
    <https://nvidia.github.io/cudaq-algorithms/>`__ for installation
    instructions, tutorials, and examples, `cudaq-algorithms on PyPI
    <https://pypi.org/project/cudaq-algorithms/>`__ for the package, and
    `NVIDIA/cudaq-algorithms on GitHub
    <https://github.com/NVIDIA/cudaq-algorithms>`__ for the source code.

.. _qec-docker-container:

Docker Container
----------------

CUDA-Q QEC is available as a Docker container with all dependencies pre-installed:

1. Pull the container:

.. code-block:: bash

    docker pull ghcr.io/nvidia/cudaq-qec

2. Run the container:

.. code-block:: bash

    docker run --gpus all -it ghcr.io/nvidia/cudaq-qec

.. note::

    If your system does not have local GPUs (eg. a MacBook), omit the `--gpus all`
    argument.

The container includes:
    * CUDA-Q compiler and runtime
    * CUDA-Q QEC library
    * All required dependencies
    * Example notebooks and tutorials

Building from Source
--------------------

The instructions for building CUDA-Q QEC from source are maintained on our GitHub
repository: `Building CUDA-Q QEC from Source <https://github.com/NVIDIA/cudaq-qec/blob/main/Building.md>`__.

.. _installing-pytorch:

Installing PyTorch
------------------

PyTorch (``torch``) is required for several CUDA-Q QEC features:

* **Tensor Network Decoder**: Used by the QEC library for tensor network-based decoding (CPU version of PyTorch is sufficient)
* **Training AI decoders**: Optionally used for training custom neural network decoders (see :ref:`Deploying AI decoders with TensorRT <deploying-ai-decoders>`)

PyTorch is automatically installed when you install the optional components:

.. code-block:: bash

    # Installs PyTorch as a dependency
    pip install cudaq-qec[tensor-network-decoder]

Alternatively, you can install PyTorch directly. For CUDA 12 installations,
use the version combination below. For other configurations, visit the
`PyTorch installation page <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

    pip install torch

.. note::
    Users with NVIDIA Blackwell architecture GPUs require PyTorch with CUDA 12.8 or later support. 
    When installing PyTorch, make sure to select the appropriate CUDA version for your system.

.. _cuda-12-pytorch-and-cuda-core-compatibility:
.. _pytorch-cuda12-compatibility:

.. note::

    If you use the Tensor Network Decoder with a CUDA 12 installation of CUDA-Q QEC
    `0.9.0 <https://github.com/NVIDIA/cudaq-qec/releases/tag/0.9.0>`_ or
    `0.9.1 <https://github.com/NVIDIA/cudaq-qec/releases/tag/0.9.1>`_, you may
    encounter a PyTorch dependency conflict. We recommend using PyTorch 2.9.0
    with compatible CUDA 12 bindings:

    .. code-block:: bash

        python3 -m pip install "torch==2.9.0" \
            --index-url https://download.pytorch.org/whl/cu128
        python3 -m pip install --upgrade \
            "cuda-bindings>=12.9.8,<13"

    Retain the PyTorch version pin when updating dependencies. See the
    `CUDA Python compatibility requirements
    <https://nvidia.github.io/cuda-python/cuda-core/1.3.0/release/1.3.0-notes.html>`_
    for details.

Once CUDA-Q QEC is installed, run the :doc:`Quick Start <quick_start>` to check
your installation with a small offline decoding experiment.

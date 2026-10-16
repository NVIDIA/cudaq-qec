Installation Guide
==================

Installation Methods
--------------------

CUDA-Q QEC provides multiple installation methods to suit your needs:

pip install
^^^^^^^^^^^^

The simplest way to install CUDA-Q QEC is via pip. (If you're on Mac, your only
option is to use the Docker container as described below.)

.. code-block:: bash

    # Install QEC library
    pip install cudaq-qec

CUDA-Q QEC provides optional pip-installable components:

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

Docker Container
^^^^^^^^^^^^^^^^

CUDA-Q QEC is available as a Docker container with all dependencies pre-installed:

1. Pull the container:

.. code-block:: bash

    docker pull ghcr.io/nvidia/cudaqx

2. Run the container:

.. code-block:: bash

    docker run --gpus all -it ghcr.io/nvidia/cudaqx

.. note::

    If your system does not have local GPUs (eg. a MacBook), omit the `--gpus all`
    argument.

The container includes:
    * CUDA-Q compiler and runtime
    * CUDA-Q QEC library
    * All required dependencies
    * Example notebooks and tutorials

Building from Source
^^^^^^^^^^^^^^^^^^^^

The instructions for building CUDA-Q QEC from source are maintained on our GitHub
repository: `Building CUDA-Q QEC from Source <https://github.com/NVIDIA/cudaq-qec/blob/main/Building.md>`__.

.. _installing-pytorch:

Installing PyTorch
------------------

PyTorch (``torch``) is required for several CUDA-Q QEC features:

* **Tensor Network Decoder**: Used by the QEC library for tensor network-based decoding (CPU version of PyTorch is sufficient)
* **Training AI Decoders**: Optionally used for training custom neural network decoders (see :ref:`Deploying AI Decoders with TensorRT <deploying-ai-decoders>`)

PyTorch is automatically installed when you install the optional components:

.. code-block:: bash

    # Installs PyTorch as a dependency
    pip install cudaq-qec[tensor-network-decoder]

Alternatively, you can install PyTorch directly. For detailed installation instructions, visit the 
`PyTorch installation page <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

    pip install torch

.. note::
    Users with NVIDIA Blackwell architecture GPUs require PyTorch with CUDA 12.8 or later support. 
    When installing PyTorch, make sure to select the appropriate CUDA version for your system.
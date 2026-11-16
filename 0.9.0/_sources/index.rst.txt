CUDA-Q QEC - The CUDA-Q Quantum Error Correction Library
=========================================================

CUDA-Q QEC is a library that builds upon the CUDA-Q programming model
to enable the rapid development of hybrid quantum-classical application code leveraging
state-of-the-art CPUs, GPUs, and QPUs. It provides a C++ library and Python
package that enable research, development, and application creation for use
cases in quantum error correction.

.. note::

   **CUDA-Q QEC is actively developed and fully supported.** The deprecation
   notice below applies *only* to the CUDA-Q Solvers library; it does not
   affect CUDA-Q QEC, which continues to receive new features, performance
   improvements, and releases.

.. attention::

   **CUDA-Q Solvers is deprecated (this does not affect CUDA-Q QEC).**
   Version 0.6.0 is the final planned
   release of the CUDA-Q Solvers library. Development continues in **CUDA-Q
   Algorithms**, which supersedes CUDA-Q Solvers and expands on it. Install it
   with :code:`pip install cudaq-algorithms`
   (`cudaq-algorithms on PyPI <https://pypi.org/project/cudaq-algorithms/>`__),
   read the `CUDA-Q Algorithms documentation
   <https://nvidia.github.io/cudaq-algorithms/>`__, and find the source code at
   `NVIDIA/cudaq-algorithms on GitHub <https://github.com/NVIDIA/cudaq-algorithms>`__.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   components/qec/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples_rst/qec/examples

.. toctree::
   :maxdepth: 1
   :caption: Performance Studies

   performance/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/core/cpp_api
   api/qec/cpp_api
   api/qec/python_api

Key Features
-------------

CUDA-Q QEC (``cudaq-qec``) is a library enabling performant research workflows for
quantum error correction, built upon the CUDA-Q programming model.

* **cudaq-qec** (actively developed and supported): Quantum Error Correction Library
    * Extensible framework describing quantum error correcting codes as a collection of CUDA-Q kernels.
    * Extensible framework for describing syndrome decoders
    * State-of-the-art, performant decoder implementations on NVIDIA GPUs
    * Real-time decoding for active error correction on quantum hardware
    * Pre-built numerical experiment APIs

Indices
-------

* :ref:`genindex`
* :ref:`search`

.. _dem_sampling_example:

DEM sampling — Monte Carlo sampling from detector error models
==============================================================

This example samples synthetic error and syndrome data from a detector error model, then walks through the GPU-accelerated and CPU paths and the supported input types. For the sampling model itself, see :ref:`DEM sampling <dem_sampling>`.

Example
~~~~~~~

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/dem_sampling.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/dem_sampling.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec dem_sampling.cpp
      ./a.out

GPU Acceleration
~~~~~~~~~~~~~~~~

When a CUDA-capable GPU is available, ``dem_sampling`` keeps the sampling and
syndrome computation on-device, which is significantly faster than per-shot CPU
sampling, especially for large numbers of shots and sparse error models (low
probabilities):

1. **Sparse Bernoulli sampling** — Errors are generated directly in compressed
   sparse row (CSR) format. For low error probabilities the CSR representation
   is compact, and the sampler skips mechanisms with zero probability entirely
   rather than evaluating a Bernoulli trial for every mechanism in every shot.

2. **GF(2) sparse-dense matrix multiply** — Syndromes are computed as
   :math:`\text{errors} \times H^T \pmod{2}` using a sparse-dense multiply
   over GF(2). The check matrix :math:`H^T` is stored in a bitpacked layout,
   reducing memory bandwidth by 8x compared to one byte per entry.

3. **On-device packing and unpacking** — :math:`H` is transposed and bitpacked
   on the GPU in a single kernel. Syndromes are unpacked from the bitpacked
   result, and the dense error matrix is produced from the CSR representation
   via a fused zero-and-scatter kernel.

The CPU path uses ``std::bernoulli_distribution`` per mechanism per shot
followed by a dense dot product for the syndrome.

Input Types and Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``backend`` parameter controls where sampling runs:

- ``"auto"`` (default) — try GPU first, fall back to CPU.
- ``"gpu"`` — require GPU; raise ``RuntimeError`` if unavailable.
- ``"cpu"`` — always use the CPU path.

The Python binding accepts several input types, each routed through a different
code path:

1. **NumPy arrays** (most common) — When the GPU is available the bindings
   automatically allocate device memory, copy inputs host-to-device, run
   cuStabilizer, and copy results back as NumPy ``uint8`` arrays. With
   ``backend="cpu"`` the GPU path is skipped entirely. No user action is
   required beyond passing standard ``uint8`` and ``float64`` arrays.

2. **PyTorch CUDA tensors** — The GPU path reads input device pointers directly
   via ``data_ptr()`` and writes outputs into ``torch.empty`` tensors on the
   same device, avoiding any host-device copies. This is the fastest path when
   inputs are already on the GPU. PyTorch is an optional dependency; install
   with ``pip install torch``.

3. **PyTorch CPU tensors** — With ``backend="gpu"`` the tensors are
   automatically moved to CUDA (via ``.to(device)``) before sampling. With
   ``backend="auto"`` CPU tensors are rejected with an error; convert them to
   NumPy with ``.numpy()`` first.

The C++ API exposes two namespaces:

- ``cudaq::qec::dem_sampler::cpu::sample_dem`` — takes a ``cudaqx::tensor``
  check matrix and a ``std::vector<double>`` of probabilities; returns
  ``(syndromes, errors)`` as tensors.
- ``cudaq::qec::dem_sampler::gpu::sample_dem`` — takes raw device pointers and
  writes results into caller-provided device buffers; returns ``false`` if
  cuStabilizer is not available at runtime.

The ``gpu`` overload works with device pointers that you allocate, populate,
and free yourself. Guard the call behind a device-count check and fall back to
the ``cpu`` overload when it returns ``false``:

.. code-block:: cpp

   #include "cudaq/qec/dem_sampling.h"
   #include <cuda_runtime.h>

   // H: [num_checks x num_mechanisms] uint8, probs: [num_mechanisms] double.
   uint8_t *d_H, *d_syndromes, *d_errors;
   double *d_probs;
   cudaMalloc(&d_H, num_checks * num_mechanisms);
   cudaMalloc(&d_probs, num_mechanisms * sizeof(double));
   cudaMalloc(&d_syndromes, num_shots * num_checks);
   cudaMalloc(&d_errors, num_shots * num_mechanisms);
   cudaMemcpy(d_H, h_data, num_checks * num_mechanisms, cudaMemcpyHostToDevice);
   cudaMemcpy(d_probs, prob_data, num_mechanisms * sizeof(double),
              cudaMemcpyHostToDevice);

   bool ok = cudaq::qec::dem_sampler::gpu::sample_dem(
       d_H, num_checks, num_mechanisms, d_probs, num_shots, /*seed=*/42,
       d_syndromes, d_errors);
   if (!ok) {
     // cuStabilizer unavailable at runtime — use the cpu overload instead.
   }
   // Copy d_syndromes / d_errors back to host, then cudaFree each buffer.

See Also
~~~~~~~~

- ``dem_sampling`` — :ref:`Python API reference <dem_sampling_python_api>`
- ``dem_sampler`` — :ref:`C++ API reference <dem_sampling_cpp_api>`

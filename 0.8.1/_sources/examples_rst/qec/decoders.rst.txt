Decoders
========

In quantum error correction, decoders are responsible for interpreting measurement outcomes (syndromes) to identify and correct quantum errors. 
We measure a set of stabilizers that give us information about what errors might have happened. The pattern of these measurements is called a syndrome, 
and the decoder's task is to determine what errors most likely caused that syndrome.

The relationship between errors and syndromes is captured mathematically by the parity check matrix. Each row of this matrix represents a 
stabilizer measurement, while each column represents a possible error. When we multiply an error pattern by this matrix, we get the syndrome 
that would result from those errors.

A detector error model (DEM) describes how the errors in a QEC circuit produce the syndrome bits that detect them. The examples below work with DEMs in three ways: the first constructs a decoder directly from raw Stim ``.dem`` text; the second expands a DEM into a multi-round parity check matrix; and the third samples synthetic error and syndrome data from a DEM to exercise a decoder. See :ref:`Detector Error Model <detector_error_model>` for more details.

.. _stim_dem_text_example:

Decoding From Stim DEM Text
+++++++++++++++++++++++++++

This example constructs a decoder from raw Stim ``.dem`` text and uses the matching parsed matrix for observable predictions. For what a detector error model is and how the text is parsed, see :ref:`Decoding from Stim DEM Text <decoding_from_stim_dem_text>`.

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/stim_dem_decoder.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/stim_dem_decoder.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec -lcudaq-qec-decoders stim_dem_decoder.cpp -o stim_dem_decoder
      ./stim_dem_decoder

Generating a Multi-Round Parity Check Matrix
++++++++++++++++++++++++++++++++++++++++++++

A single-round DEM captures one measurement cycle. Under circuit-level noise, errors accumulate across many rounds, and the DEM expands into a multi-round parity check matrix. The following example constructs one for an error correction code in Python:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/repetition_code_pcm.py
      :language: python
      :start-after: [Begin Documentation]

This example illustrates how to:

* Retrieve and configure an error correction code  
  Load a repetition code using ``qec.get_code(...)`` from the CUDA-Q QEC library, and define a custom circuit-level noise model using ``.add_all_qubit_channel(...)``.

* Generate a multi-round parity check matrix  
  Extend a single-round detector error model (DEM) across multiple rounds using ``qec.dem_from_memory_circuit(...)``. This captures syndrome evolution over time, including measurement noise, and provides:
  
  * ``detector_error_matrix`` – the multi-round parity check matrix
  * ``observables_flips_matrix`` – used to identify logical flips due to physical errors

* Simulate circuit-level noise and collect data  
  Run multiple shots of the memory experiment using ``qec.sample_memory_circuit(...)`` to sample both the data and syndrome measurements from noisy executions. The resulting bitstrings can be used for decoding and performance evaluation of the error correction scheme.

.. _dem_sampling_example:

DEM Sampling — Monte-Carlo Sampling from Detector Error Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example samples synthetic error and syndrome data from a detector error model, then walks through the GPU-accelerated and CPU paths and the supported input types. For the sampling model itself, see :ref:`DEM Sampling <dem_sampling>`.

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

- :doc:`/api/qec/python_api` — ``dem_sampling`` Python API reference
- :doc:`/api/qec/cpp_api` — ``dem_sampler`` C++ API reference

.. _qldpc_decoder_example:

Getting Started with the NVIDIA QLDPC Decoder
+++++++++++++++++++++++++++++++++++++++++++++

The remaining sections describe the built-in decoders that consume the parity check matrices and detector error models above. Each is selected by name through :func:`cudaq_qec.get_decoder` and targets a different regime, trading off speed, accuracy, and the class of codes it supports. We begin with the most general.

Starting with CUDA-Q QEC v0.2, a GPU-accelerated decoder is included with the
CUDA-Q QEC library. The library follows the CUDA-Q decoder Python and C++ interfaces
(namely :class:`cudaq_qec.Decoder` for Python and
:cpp:class:`cudaq::qec::decoder` for C++), but as documented in the API sections
(:ref:`nv_qldpc_decoder_api_python` for Python and
:ref:`nv_qldpc_decoder_api_cpp` for C++), there are many configuration options
that can be passed to the constructor.

Belief Propagation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nv-qldpc-decoder`` supports several belief-propagation algorithms -- sum-product, min-sum, and memory-based variants, plus Sequential Relay BP -- selected via ``bp_method`` and ``composition``, with optional BP+OSD post-processing. For the complete list of methods, parameters, and defaults, see the ``nv-qldpc-decoder`` entries in the :ref:`C++ <nv_qldpc_decoder_api_cpp>` and :ref:`Python <nv_qldpc_decoder_api_python>` API reference.

Usage Example
~~~~~~~~~~~~~

The following example shows how to exercise the decoder using non-trivial pre-generated test data. 
The test data was generated using scripts originating from the GitHub repo for
`BivariateBicycleCodes <https://github.com/sbravyi/BivariateBicycleCodes>`_ [#f1]_; 
it includes parity check matrices (PCMs) and test syndromes to exercise a decoder.

The example demonstrates:

1. **Basic decoder configuration** with OSD post-processing
2. **All BP methods** including Sequential Relay BP
3. **Batched decoding** for improved performance

.. literalinclude:: ../../examples/qec/python/nv-qldpc-decoder.py
    :language: python
    :start-after: [Begin Documentation]

.. rubric:: Footnotes

.. [#f1] [BCGMRY] Sergey Bravyi, Andrew Cross, Jay Gambetta, Dmitri Maslov, Patrick Rall, Theodore Yoder, High-threshold and low-overhead fault-tolerant quantum memory https://arxiv.org/abs/2308.07915

.. _tensor_network_decoder_example:

Exact Maximum Likelihood Decoding with NVIDIA Tensor Network Decoder
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Where belief propagation trades exactness for speed, the tensor network decoder computes the exact maximum-likelihood correction — valuable as an accuracy baseline against which the faster decoders can be measured.

Starting with CUDA-Q QEC v0.4.0, a GPU-accelerated Maximum Likelihood Decoder is included with the
CUDA-Q QEC library. The library follows the CUDA-Q decoder Python interface, namely :class:`cudaq_qec.Decoder`.
At this time, we only support the Python interface for the decoder, which is
available at :ref:`TensorNetworkDecoder <tensor_network_decoder_api_python>`.
As documented in the API sections :ref:`tensor_network_decoder_api_python`, there are many configuration options
that can be passed to the constructor. The decoder requires Python 3.11 or higher.

In the following example, we show how to use the `TensorNetworkDecoder` class from the `cudaq_qec` library to decode a circuit-level noise problem derived from a Stim surface code circuit.

.. literalinclude:: ../../examples/qec/python/tensor_network_decoder.py
    :language: python
    :start-after: [Begin Documentation]

Output:

The decoder returns the probability that the logical observable has flipped for each syndrome. This can be used to assess the performance of the code and the decoder under different error scenarios.

See Also:

- ``cudaq_qec.plugins.decoders.tensor_network_decoder``

.. _deploying-ai-decoders:

Deploying AI Decoders with TensorRT
+++++++++++++++++++++++++++++++++++++++++++++++++

The decoders above are algorithmic. CUDA-Q QEC can also deploy a *learned* decoder — a neural network trained on a specific code and noise model.

Starting with CUDA-Q QEC v0.5.0, a GPU-accelerated TensorRT-based decoder is included with the
CUDA-Q QEC library. The TensorRT decoder (``trt_decoder``) enables users to leverage custom AI
models for quantum error correction, providing a flexible framework for deploying trained models
with optimized inference performance on NVIDIA GPUs.

Unlike traditional algorithmic decoders, neural network decoders can be trained on specific error
models and code structures, potentially achieving superior performance for certain noise regimes.
The TensorRT decoder supports loading models in ONNX format and provides configurable precision
modes (fp16, bf16, int8, fp8, tf32) to balance accuracy and inference speed.

This tutorial demonstrates the complete workflow for training a simple multi-layer perceptron (MLP)
to decode surface code syndromes using PyTorch and Stim, exporting the model to ONNX format, and
deploying it with the TensorRT decoder for accelerated inference.

Overview of the Training-to-Deployment Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow consists of three main stages:

1. **Data Generation**: Use Stim to generate synthetic quantum error correction data by simulating
   surface code circuits with realistic noise models. This produces detector measurements (syndromes)
   and observable flips (logical errors) that serve as training data.

2. **Model Training**: Train a neural network (in this case, an MLP) using PyTorch to learn the
   mapping from syndromes to logical error predictions. The model is trained with standard deep
   learning techniques including dropout regularization, learning rate scheduling, and validation monitoring.

3. **ONNX Export and Deployment**: Export the trained PyTorch model to ONNX format, which can then
   be loaded by the TensorRT decoder for optimized GPU inference in production QEC workflows.

Training a Neural Network Decoder with PyTorch and Stim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows how to generate training data using Stim's built-in surface code
generator, train an MLP decoder with PyTorch, and export the model to ONNX format.
For instructions on installing PyTorch, see :ref:`Installing PyTorch <installing-pytorch>`.

.. literalinclude:: ../../examples/qec/python/train_mlp_decoder.py
   :language: python
   :start-after: [Begin Documentation]

Using the TensorRT Decoder in CUDA-Q QEC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have a trained ONNX model, you can load it with the TensorRT decoder for accelerated
inference. The decoder can be used in both C++ and Python workflows.

**Loading from ONNX (with automatic TensorRT optimization)**:

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec
      import numpy as np

      # Note: The AI decoder doesn't use the parity check matrix.
      # A placeholder matrix is provided here to satisfy the API.
      H = np.array([[1, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)

      # Create TensorRT decoder from ONNX model
      decoder = qec.get_decoder("trt_decoder", H,
                                onnx_load_path="ai_decoder.onnx")

      # Decode a syndrome
      syndrome = np.array([1.0, 0.0, 1.0], dtype=np.float32)
      result = decoder.decode(syndrome)
      print(f"Predicted error: {result}")

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/qec/decoder.h"
      #include "cuda-qx/core/tensor.h"
      #include "cuda-qx/core/heterogeneous_map.h"

      int main() {
          // Note: The AI decoder doesn't use the parity check matrix.
          // A placeholder matrix is provided here to satisfy the API.
          std::vector<std::vector<uint8_t>> H_vec = {
              {1, 0, 0, 1, 0, 1, 1},
              {0, 1, 0, 1, 1, 0, 1},
              {0, 0, 1, 0, 1, 1, 1}
          };
          
          // Convert to tensor
          cudaqx::tensor<uint8_t> H({3, 7});
          for (size_t i = 0; i < 3; ++i) {
              for (size_t j = 0; j < 7; ++j) {
                  H.at({i, j}) = H_vec[i][j];
              }
          }

          // Create decoder parameters
          cudaqx::heterogeneous_map params;
          params.insert("onnx_load_path", "ai_decoder.onnx");
          params.insert("precision", "fp16");

          // Create TensorRT decoder
          auto decoder = cudaq::qec::get_decoder("trt_decoder", H, params);

          // Decode syndrome
          std::vector<cudaq::qec::float_t> syndrome = {1.0, 0.0, 1.0};
          auto result = decoder->decode(syndrome);

          return 0;
      }

**Loading a pre-built TensorRT engine (for fastest initialization)**:

If you've already converted your ONNX model to a TensorRT engine using the provided utility script,
you can load it directly:

.. tab:: Python

   .. code-block:: python

      decoder = qec.get_decoder("trt_decoder", H,
                                engine_load_path="surface_code_decoder.trt")

Converting ONNX Models to TensorRT Engines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production deployments where initialization time is critical, you can pre-build a TensorRT
engine from your ONNX model using the ``trtexec`` command-line tool that comes with TensorRT:

.. code-block:: bash

   # Build with FP16 precision
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --fp16

   # Build with best precision for your GPU
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --best

   # Build with specific input shape (optional, for optimization)
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --fp16 \
           --shapes=detectors:1x24

Pre-built engines offer several advantages:

- **Faster initialization**: Engine loading is significantly faster than ONNX parsing and optimization
- **Reproducible optimization**: The same optimization decisions are made every time
- **Version control**: Engines can be versioned alongside code for reproducible deployments


Dependencies and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TensorRT decoder requires:

- **TensorRT**: Version 10.13.3.9 or higher
- **CUDA**: Version 12.0 or higher for x86 and 13.0 for ARM.
- **GPU**: NVIDIA GPU with compute capability 6.0+ (Pascal architecture or newer)

For training:

- **PyTorch**: Version 2.0+ recommended
- **Stim**: For quantum circuit simulation and data generation

See Also
~~~~~~~~

- :class:`cudaq_qec.Decoder` - Base decoder interface
- `ONNX <https://onnx.ai/>`_ - Open Neural Network Exchange format
- `TensorRT Documentation <https://docs.nvidia.com/deeplearning/tensorrt/>`_ - NVIDIA TensorRT
- `Stim Documentation <https://github.com/quantumlib/Stim>`_ - Fast stabilizer circuit simulator

.. _pymatching_decoder_example:

Matching-Based Decoding with PyMatching
+++++++++++++++++++++++++++++++++++++++

For codes whose errors pair up into a matching graph, a dedicated matching decoder is often the simplest and fastest choice. Starting with CUDA-Q QEC v0.7.0, CUDA-Q QEC bundles a minimum-weight perfect matching (MWPM) decoder built on the
open-source `PyMatching <https://github.com/oscarhiggott/PyMatching>`_ library,
suitable for matchable codes such as the surface code. It is selected by name
through :func:`cudaq_qec.get_decoder` and takes a parity-check matrix whose
columns each have one or two set entries:

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec
      import numpy as np

      H = np.array([[1, 1, 0],
                    [0, 1, 1]], dtype=np.uint8)

      dec = qec.get_decoder("pymatching", H,
                            error_rate_vec=[0.1, 0.1, 0.1],
                            merge_strategy="smallest_weight")
      result = dec.decode(syndrome)

Per-error priors are supplied via ``error_rate_vec`` (values in ``(0, 0.5]``),
and parallel edges are combined according to ``merge_strategy``. See the
:ref:`PyMatching Decoder API <pymatching_decoder_api_python>` for the full list
of options.

.. _chromobius_decoder_example:

Color-Code Decoding with Chromobius
+++++++++++++++++++++++++++++++++++

Matching applies to surface-code-like codes; color codes call for a decoder built around their structure. Starting with CUDA-Q QEC v0.7.0, CUDA-Q QEC bundles a color-code decoder built on the open-source
`Chromobius <https://github.com/quantumlib/chromobius>`_ Möbius decoder. Unlike
the matrix-based decoders, Chromobius is *detector-error-model native*: it is
constructed from Stim detector-error-model (DEM) text rather than a
parity-check matrix, and predicts logical observable flips directly.

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec

      with open("color_code.dem") as f:
          dem_text = f.read()

      dec = qec.get_decoder("chromobius", dem_text)
      corrections = dec.decode(syndrome)  # predicted observable flips

Constructing Chromobius from a parity-check matrix is rejected with an error.
See the :ref:`Chromobius Decoder API <chromobius_decoder_api_python>` for the
available options.

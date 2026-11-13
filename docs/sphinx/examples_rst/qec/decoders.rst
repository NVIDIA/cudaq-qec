Decoders
========

In quantum error correction, decoders are responsible for interpreting measurement outcomes (syndromes) to identify and correct quantum errors. 
We measure a set of stabilizers that give us information about what errors might have happened. The pattern of these measurements is called a syndrome, 
and the decoder's task is to determine what errors most likely caused that syndrome.

The relationship between errors and syndromes is captured mathematically by the parity check matrix. Each row of this matrix represents a 
stabilizer measurement, while each column represents a possible error. When we multiply an error pattern by this matrix, we get the syndrome 
that would result from those errors.

For examples that construct, sample, and decode with detector error models (DEMs),
see :doc:`Working with Detector Error Models <detector_error_models>`.

.. _qldpc_decoder_example:

Getting Started with the NVIDIA QLDPC Decoder
+++++++++++++++++++++++++++++++++++++++++++++

The following sections describe built-in decoders that consume parity check matrices and detector error models. Each is selected by name through :func:`cudaq_qec.get_decoder` and targets a different regime, trading off speed, accuracy, and the class of codes it supports. We begin with the most general.

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
available at :ref:`Tensor Network Decoder <tensor_network_decoder_api_python>`.
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
The TensorRT decoder supports loading models in ONNX format. Its strongly typed network preserves
the data types and quantization encoded in the model; its ``precision`` option controls only the
TensorRT TF32 build policy.

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
inference. The decoder can be used in both C++ and Python workflows. Keep the ``.dem`` file emitted
by the training example beside the model: it fixes the detector and observable dimensions and the
error-mechanism ordering that the model was trained against.

**Loading from ONNX (with automatic TensorRT optimization)**:

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec
      import numpy as np

      with open("surface_code_decoder.dem") as dem_file:
          dem_text = dem_file.read()
      dem = qec.dem_from_stim_text(dem_text)

      # The training model predicts logical observables, so both the engine
      # tensor format and the public decoder result are explicit.
      decoder = qec.get_decoder(
          "trt_decoder",
          dem_text,
          onnx_load_path="surface_code_decoder.onnx",
          engine_output_format="observables",
          output="observables")

      # Decode a syndrome
      syndrome = np.zeros(dem.num_detectors(), dtype=np.float32)
      result = decoder.decode(syndrome)
      print(f"Predicted observable: {result.result}")

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/qec/decoder.h"
      #include "cuda-qx/core/heterogeneous_map.h"

      #include <fstream>
      #include <iterator>
      #include <string>
      #include <utility>
      #include <vector>

      int main() {
          std::ifstream dem_file("surface_code_decoder.dem");
          std::string dem_text((std::istreambuf_iterator<char>(dem_file)),
                               std::istreambuf_iterator<char>());
          auto inputs = cudaq::qec::decoder_init::from_stim_dem(dem_text);
          const auto syndrome_size = inputs.num_detectors();

          cudaqx::heterogeneous_map params;
          params.insert("onnx_load_path", "surface_code_decoder.onnx");
          params.insert("engine_output_format", std::string("observables"));

          // The training model predicts logical observables, so request that
          // result basis explicitly.
          auto decoder = cudaq::qec::get_decoder(
              "trt_decoder", std::move(inputs),
              cudaq::qec::decode_result_type::observables, params);

          std::vector<cudaq::qec::float_t> syndrome(syndrome_size, 0.0);
          auto result = decoder->decode(syndrome);

          return 0;
      }

**Loading a pre-built TensorRT engine (for fastest initialization)**:

If you've already converted your ONNX model to a TensorRT engine using the provided utility script,
you can load it directly:

.. tab:: Python

   .. code-block:: python

      with open("surface_code_decoder.dem") as dem_file:
          dem_text = dem_file.read()

      decoder = qec.get_decoder(
          "trt_decoder",
          dem_text,
          engine_load_path="surface_code_decoder.trt",
          engine_output_format="observables",
          output="observables")

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

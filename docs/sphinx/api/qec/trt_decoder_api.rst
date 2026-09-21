.. class:: trt_decoder

    A GPU-accelerated quantum error correction decoder based on NVIDIA TensorRT.
    This decoder leverages TensorRT's optimized inference engine to perform fast
    neural network-based decoding of quantum error correction syndromes.

    The TRT decoder supports loading pre-trained neural network models in ONNX
    format or directly loading pre-built TensorRT engine files for maximum
    performance. It automatically optimizes the model for the target GPU
    architecture. The network is strongly typed, so TensorRT preserves the
    data types and quantization encoded in the ONNX model.

    Neural network-based decoders can be trained to perform syndrome decoding
    for specific quantum error correction codes and noise models. The TRT decoder
    provides a high-performance inference engine for these models, with automatic
    CUDA graph optimization for reduced latency.

    Requires a CUDA-capable GPU and TensorRT installation. See the `CUDA-Q GPU
    Compatibility List
    <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_
    for a list of valid GPU configurations.

    .. warning::
      **Breaking change:** ``engine_output_format`` is required. It declares
      the semantic layout of the TensorRT model's output and is validated
      against ``H``, ``O``, and any configured global decoder. Existing callers
      must add the value matching the model they load.

    .. note::
      It is required to create decoders with the `get_decoder` API from the CUDA-QX
      extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np
            
            # H defines the error-frame width and must match the model output.
            H = np.array([[1, 0, 0, 1, 0, 1, 1],
                          [0, 1, 0, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)
            
            # Option 1: Load from ONNX model (builds TRT engine)
            trt_dec = qec.get_decoder('trt_decoder', H,
                                      onnx_load_path='model.onnx',
                                      engine_output_format='errors',
                                      engine_save_path='model.engine')
            
            # Option 2: Load pre-built TRT engine (faster startup)
            trt_dec = qec.get_decoder('trt_decoder', H,
                                      engine_load_path='model.engine',
                                      engine_output_format='errors')

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/decoder.h"
            
            std::size_t block_size = 7;
            std::size_t syndrome_size = 3;
            cudaqx::tensor<uint8_t> H;

            // H defines the error-frame width and must match the model output.
            std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1, 
                                          0, 1, 0, 1, 1, 0, 1,
                                          0, 0, 1, 0, 1, 1, 1};
            H.copy(H_vec.data(), {syndrome_size, block_size});
            
            // Option 1: Load from ONNX model (builds TRT engine)
            cudaqx::heterogeneous_map params1;
            params1.insert("onnx_load_path", "model.onnx");
            params1.insert("engine_output_format", std::string("errors"));
            params1.insert("engine_save_path", "model.engine");
            auto trt_dec1 = cudaq::qec::get_decoder("trt_decoder", H, params1);
            
            // Option 2: Load pre-built TRT engine (faster startup)
            cudaqx::heterogeneous_map params2;
            params2.insert("engine_load_path", "model.engine");
            params2.insert("engine_output_format", std::string("errors"));
            auto trt_dec2 = cudaq::qec::get_decoder("trt_decoder", H, params2);
      
    .. note::
      The `"trt_decoder"` implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    .. note::
      The TensorRT engine performs inference, but ``H`` is still model data.
      Its column count defines the error-frame width, validates an
      ``engine_output_format="errors"`` engine, and supplies the detector model
      to an optional global decoder. ``H`` and the optional ``O`` must therefore
      describe the code and index ordering used to train the engine.

    .. note::
      **Batch Processing**: The TRT decoder automatically handles batch size
      optimization. Models trained with batch_size > 1 will receive
      zero-padded inputs when using `decode()` on a single syndrome. When
      using `decode_batch()`, provide syndromes in multiples of the model's
      batch size for optimal performance.

    :param H: Parity check matrix for the code and error-mechanism ordering used
              by the TensorRT model.
    :param params: Heterogeneous map of parameters:

        **Required:**

        - `engine_output_format` (string): Semantic layout of one model output.
          This describes the engine tensor independently of the decoder result
          basis. Accepted values are:

          - ``"errors"``: one error-frame value per column of ``H``.
          - ``"residual_detectors"``: residual detector values consumed by the
            required ``global_decoder``.
          - ``"observables"``: one value per row of ``O``.
          - ``"observables_and_residual_detectors"``: an observable prefix
            followed by residual detectors. A standalone TensorRT decoder
            requires ``global_decoder`` for this format. A task-graph
            ``predecoder`` with ``residual_from: model`` may instead route the
            observable prefix and residual detectors to separate downstream
            graph edges.

        **Required model source (choose one):**

        - `onnx_load_path` (string): Path to ONNX model file. The decoder will
          build a TensorRT engine from this model. Cannot be used together with
          `engine_load_path`.
        - `engine_load_path` (string): Path to pre-built TensorRT engine file.
          Provides faster initialization since the engine is already optimized.
          Cannot be used together with `onnx_load_path`.

        **Optional:**

        - `engine_save_path` (string): Path to save the built TensorRT engine.
          Only applicable when using `onnx_load_path`. Saving the engine allows
          for faster initialization in subsequent runs by using `engine_load_path`.
        - `precision` (string): TensorRT build policy (defaults to ``"best"``).
          ``"best"`` preserves TensorRT's default TF32 policy, ``"tf32"``
          explicitly enables TF32, and ``"noTF32"`` disables it. The legacy
          values ``"fp16"``, ``"bf16"``, ``"int8"``, and ``"fp8"`` remain
          accepted for configuration compatibility, but are ignored because
          the network is strongly typed. To use those types or quantization,
          encode them in the ONNX model before loading it.

        - `memory_workspace` (size_t): Memory workspace size in bytes for TensorRT
          engine building (defaults to 1GB = 1073741824 bytes). Larger workspaces
          may allow TensorRT to explore more optimization strategies.

        - `use_cuda_graph` (bool): Enable CUDA graph optimization for improved
          performance (defaults to True). CUDA graphs capture inference operations
          and replay them with reduced kernel launch overhead, providing ~20%
          speedup. The optimization is applied automatically on the first decode
          call. Automatically disabled for models with dynamic shapes or
          multiple optimization profiles. Set to False to force traditional
          execution path.

        - `batch_size` (automatic): The decoder automatically detects the model's
          batch size from the first input dimension. For models with batch_size > 1,
          the `decode()` method automatically zero-pads single syndromes to fill
          the batch. The `decode_batch()` method requires the number of syndromes
          to be an integral multiple of the model's batch size.

        - `global_decoder` (string): Name of a second-stage "global" decoder to
          chain after the TensorRT model (composite decoding). It is required
          for standalone ``"residual_detectors"`` and
          ``"observables_and_residual_detectors"`` output. The TRT model acts
          as a first-stage predecoder whose residual output is passed to the
          named global decoder (for example ``"pymatching"`` or
          ``"chromobius"``). It may be omitted for output formats that need no
          second stage, or when a task graph explicitly routes model residuals
          to a downstream decoder. See :ref:`the real-time decoding API
          <python_realtime_decoding_api>` for configuring composite decoding
          from YAML. Introduced in 0.7.0.

        - `global_decoder_params` (map): Parameters forwarded to the
          `global_decoder`. The accepted keys follow the parameter schema of the
          named global decoder. This is optional; when omitted, the named
          global decoder is constructed with an empty parameter map.
          Introduced in 0.7.0.

        - `cuda_device_id` (int): Zero-based CUDA device ordinal on which to
          construct the decoder and run every decode. Must be ``>= 0`` and less
          than the number of visible GPUs. When omitted, the decoder is not
          pinned to a specific device and runs on the default device (GPU 0).
          Introduced in 0.7.0.

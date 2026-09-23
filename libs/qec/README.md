# CUDA-Q QEC Library

CUDA-Q QEC is a high-performance quantum error correction library
that leverages NVIDIA GPUs to accelerate classical decoding and
processing of quantum error correction codes. The library provides optimized
implementations of common QEC tasks including syndrome extraction,
decoding, and logical operation tracking.

**Note**: CUDA-Q QEC is currently only supported on Linux operating systems
using `x86_64` processors or `aarch64`/`arm64` processors. CUDA-Q QEC does
not require a GPU to use, but some components are GPU-accelerated.

## Features

- Fast syndrome extraction and processing on GPUs
- Common decoders for surface codes and other topological codes
- Real-time decoding capabilities for quantum feedback
- Integration with CUDA-Q quantum program execution

## Optional Dependencies

Some decoders require additional dependencies to operate. You can install them with

- `pip install cudaq-qec[tensor-network-decoder]` for the Tensor Network Decoder
- `pip install cudaq-qec[trt-decoder]` for the TensorRT Decoder

## Getting Started

For detailed documentation, tutorials, and API reference, visit the
[CUDA-Q QEC Documentation](https://nvidia.github.io/cudaq-qec/components/qec/introduction.html).

## License

Most components of CUDA-Q QEC are open source. The source code is available on
[GitHub][github_link] and licensed under [Apache License
2.0](https://github.com/NVIDIA/cudaq-qec/blob/main/LICENSE).

The `libcudaq-qec-nv-qldpc-decoder.so` library (distributed with CUDA-Q QEC) is
closed source and is subject to the [NVIDIA Software License Agreement][github_qec_license]

[github_link]: https://github.com/NVIDIA/cudaq-qec/tree/main/libs/qec
[github_qec_license]: https://github.com/NVIDIA/cudaq-qec/blob/main/libs/qec/LICENSE

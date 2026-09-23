# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import tensorrt as trt


def build_engine(onnx_file,
                 engine_file,
                 fp16=False,
                 int8=False,
                 workspace_size=1 << 30,
                 max_batch_size=1):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # Parse the ONNX model
    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse the ONNX file.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    # Precision settings
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("Warning: FP16 not supported on this platform.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            print("Warning: INT8 not supported on this platform.")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            # In practice, you'd need a calibrator for INT8
            print(
                "Note: INT8 requires calibration – using dummy mode for example."
            )

    # Build serialized engine
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Example dynamic shape support: use batch dimension
    if input_shape[0] == -1:
        profile.set_shape(input_tensor.name, (1, *input_shape[1:]),
                          (max_batch_size, *input_shape[1:]),
                          (max_batch_size, *input_shape[1:]))
        config.add_optimization_profile(profile)

    print("Building TensorRT engine. This may take a while...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine.")

    # Save engine
    with open(engine_file, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx_file", type=str, help="Path to input ONNX file")
    parser.add_argument("--engine_file",
                        type=str,
                        help="Path to save TensorRT engine")
    parser.add_argument("--fp16",
                        action="store_true",
                        help="Enable FP16 precision")
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision (needs calibration in practice)")
    parser.add_argument("--workspace_size",
                        type=int,
                        default=1 << 30,
                        help="Workspace size in bytes (default: 1GB)")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Max batch size (default: 1)")

    args = parser.parse_args()

    build_engine(onnx_file=args.onnx_file,
                 engine_file=args.engine_file,
                 fp16=args.fp16,
                 int8=args.int8,
                 workspace_size=args.workspace_size,
                 max_batch_size=args.batch_size)

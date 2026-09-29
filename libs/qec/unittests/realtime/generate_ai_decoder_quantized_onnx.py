#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NUM_ELEMENTS = 8


def make_qdq_model(output_path, name, zero_point_tensor):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                               [NUM_ELEMENTS])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                                [NUM_ELEMENTS])
    scale = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "scale")

    quantize = helper.make_node("QuantizeLinear",
                                ["input", "scale", "zero_point"], ["quantized"])
    dequantize = helper.make_node("DequantizeLinear",
                                  ["quantized", "scale", "zero_point"],
                                  ["output"])

    graph = helper.make_graph(
        [quantize, dequantize],
        name,
        [input_info],
        [output_info],
        [scale, zero_point_tensor],
    )
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiny Q/DQ ONNX fixtures for ai_decoder_service.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    make_qdq_model(
        os.path.join(args.out_dir, "qdq_int8.onnx"),
        "qdq_int8",
        numpy_helper.from_array(np.array([0], dtype=np.int8), "zero_point"),
    )
    make_qdq_model(
        os.path.join(args.out_dir, "qdq_fp8.onnx"),
        "qdq_fp8",
        helper.make_tensor("zero_point",
                           TensorProto.FLOAT8E4M3FN, [1],
                           vals=[0]),
    )


if __name__ == "__main__":
    main()

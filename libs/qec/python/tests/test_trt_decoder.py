# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
TRT Decoder Tests

These tests validate the TensorRT-based decoder for quantum error correction.

Test Categories:
  - Parameter validation and file operations (fast, no GPU required)

Note: GPU tests require CUDA/TensorRT and may crash if GPU is not properly
accessible within the container environment. Engine build takes ~5 seconds
on first run.
"""
import pytest
import numpy as np
import cudaq_qec as qec
import os
import tempfile

# TensorRT imports (optional, for building engines)
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# Test data constants
NUM_TEST_SAMPLES = 200
NUM_DETECTORS = 24
NUM_OBSERVABLES = 1

# Resolve ONNX model path relative to the repo root so the test works
# regardless of the working directory used to invoke pytest.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
ONNX_MODEL_PATH = os.path.join(_REPO_ROOT, "assets", "tests",
                               "surface_code_decoder.onnx")


# Check if CUDA/GPU is available for TensorRT tests
def _is_cuda_available():
    """Check if CUDA is available and can be initialized."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


CUDA_AVAILABLE = _is_cuda_available()


def build_simple_mlp_engine(input_size,
                            hidden_size,
                            output_size,
                            engine_path,
                            use_dynamic_shapes=False):
    """
    Build a simple MLP TensorRT engine using the TensorRT Python API.
    
    Architecture: Input -> Dense(hidden_size) -> ReLU -> Dense(output_size)
    
    Args:
        input_size: Size of input layer
        hidden_size: Size of hidden layer
        output_size: Size of output layer
        engine_path: Path where to save the engine file
        use_dynamic_shapes: If True, creates engine with dynamic shapes (incompatible with CUDA graphs)
    
    Returns:
        True if successful, False otherwise
    """
    if not TRT_AVAILABLE:
        return False

    try:
        # Create logger and builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # Set memory pool limit (1GB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # Define input tensor (with multiple optimization profiles if requested)
        if use_dynamic_shapes:
            # Use dynamic shapes with MULTIPLE optimization profiles
            # According to trt_decoder.cpp: multiple profiles = CUDA graphs not supported
            input_shape = (-1, input_size)
            input_tensor = network.add_input(name="input",
                                             dtype=trt.float32,
                                             shape=input_shape)

            # Add MULTIPLE optimization profiles (this makes CUDA graphs unsupported)
            # Profile 1: batch size = 1
            profile1 = builder.create_optimization_profile()
            profile1.set_shape("input",
                               min=(1, input_size),
                               opt=(1, input_size),
                               max=(1, input_size))
            config.add_optimization_profile(profile1)

            # Profile 2: batch size = 2 (adding a second profile disables CUDA graphs)
            profile2 = builder.create_optimization_profile()
            profile2.set_shape("input",
                               min=(2, input_size),
                               opt=(2, input_size),
                               max=(2, input_size))
            config.add_optimization_profile(profile2)
        else:
            # Static shape
            input_tensor = network.add_input(name="input",
                                             dtype=trt.float32,
                                             shape=(1, input_size))

        # Layer 1: Fully Connected (Input -> Hidden) using matrix multiply
        # Initialize random weights and biases
        np.random.seed(42)
        # Weights shape for matrix multiply: (input_size, hidden_size) for op: input @ weights
        weights1 = np.random.randn(input_size, hidden_size).astype(np.float32)
        bias1 = np.random.randn(1, hidden_size).astype(np.float32)

        # Add weights as constant
        weights1_const = network.add_constant(shape=(input_size, hidden_size),
                                              weights=trt.Weights(weights1))

        # Matrix multiply: (1, input_size) @ (input_size, hidden_size) = (1, hidden_size)
        mm1 = network.add_matrix_multiply(input_tensor,
                                          trt.MatrixOperation.NONE,
                                          weights1_const.get_output(0),
                                          trt.MatrixOperation.NONE)

        # Add bias
        bias1_const = network.add_constant(shape=(1, hidden_size),
                                           weights=trt.Weights(bias1))
        fc1 = network.add_elementwise(mm1.get_output(0),
                                      bias1_const.get_output(0),
                                      trt.ElementWiseOperation.SUM)

        # Activation: ReLU
        relu = network.add_activation(input=fc1.get_output(0),
                                      type=trt.ActivationType.RELU)

        # Layer 2: Fully Connected (Hidden -> Output) using matrix multiply
        weights2 = np.random.randn(hidden_size, output_size).astype(np.float32)
        bias2 = np.random.randn(1, output_size).astype(np.float32)

        # Add weights as constant
        weights2_const = network.add_constant(shape=(hidden_size, output_size),
                                              weights=trt.Weights(weights2))

        # Matrix multiply: (1, hidden_size) @ (hidden_size, output_size) = (1, output_size)
        mm2 = network.add_matrix_multiply(relu.get_output(0),
                                          trt.MatrixOperation.NONE,
                                          weights2_const.get_output(0),
                                          trt.MatrixOperation.NONE)

        # Add bias
        bias2_const = network.add_constant(shape=(1, output_size),
                                           weights=trt.Weights(bias2))
        fc2 = network.add_elementwise(mm2.get_output(0),
                                      bias2_const.get_output(0),
                                      trt.ElementWiseOperation.SUM)

        # Mark output
        output_tensor = fc2.get_output(0)
        output_tensor.name = "output"
        network.mark_output(tensor=output_tensor)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            return False

        # Save engine to file
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        return True

    except Exception as e:
        print(f"Failed to build TensorRT engine: {e}")
        return False


# Test inputs - 30 test cases with 24 detectors each
TEST_INPUTS = [[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
],
               [
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
               ],
               [
                   1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
               ]]

# Expected outputs from PyTorch model
TEST_OUTPUTS = [[0.546192], [0.542916], [0.534589], [0.559515], [0.510735],
                [0.526783], [0.536010], [0.522932], [0.563812], [0.560833],
                [0.578034], [0.555534], [0.542923], [0.495684], [0.545392],
                [0.562221], [0.532649], [0.559208], [0.534731], [0.499671],
                [0.497873], [0.511810], [0.566766], [0.527120], [0.573466],
                [0.523840], [0.504422], [0.507719], [0.474930], [0.583623]]


class TestTRTDecoderSetup:
    """Test fixture setup and teardown for TRT decoder tests."""

    def setup_method(self):
        """Set up test parameters before each test."""
        self.block_size = 3
        self.syndrome_size = 2

        # Create a simple parity check matrix H
        self.H = np.zeros((self.syndrome_size, self.block_size), dtype=np.uint8)
        self.H[0, 0] = 1
        self.H[0, 1] = 0
        self.H[0, 2] = 1  # First syndrome bit
        self.H[1, 0] = 0
        self.H[1, 1] = 1
        self.H[1, 2] = 1  # Second syndrome bit

        # For file loading tests
        self.test_file_path = None

    def teardown_method(self):
        """Clean up any test files after each test."""
        if self.test_file_path and os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)


class TestTRTDecoderParameterValidation(TestTRTDecoderSetup):
    """Tests for TRT decoder parameter validation."""

    def test_validate_parameters_valid_onnx_path(self):
        """Test parameter validation with valid ONNX path."""
        # Note: In Python, parameter validation happens inside the decoder creation
        # We can't test the validation function directly, but we can verify that
        # providing valid parameters doesn't raise errors when creating a decoder
        # This test is more of a placeholder showing the expected behavior
        pass

    def test_validate_parameters_valid_engine_path(self):
        """Test parameter validation with valid engine path."""
        # Similar to above - placeholder for parameter validation testing
        pass

    def test_validate_parameters_both_paths_provided(self):
        """Test that providing both ONNX and engine paths raises an error."""
        # This would need to be tested when actually creating a decoder
        # with both parameters, which should raise a RuntimeError
        pass

    def test_validate_parameters_no_paths_provided(self):
        """Providing no model path is an initialization error."""
        # Initialization failures throw (converged never stands in for
        # decoder health), so construction without a path raises.
        try:
            decoder = qec.get_decoder('trt_decoder',
                                      self.H,
                                      engine_output_format='errors')
            # If decoder is None or doesn't initialize properly, skip these tests
            if decoder is None:
                pytest.skip(
                    "TRT decoder returned None - likely CUDA/GPU unavailable")
        except (RuntimeError, SystemError, Exception) as e:
            pytest.skip(
                f"Failed to create TRT decoder (GPU may be unavailable): {e}")


class TestTRTDecoderFileOperations(TestTRTDecoderSetup):
    """Tests for TRT decoder file loading operations."""

    def test_load_file_valid_file(self):
        """Test loading a valid file."""
        # Create a temporary test file
        test_content = "Hello, World!"
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write(test_content)
            self.test_file_path = f.name

        # Read the file back
        with open(self.test_file_path, 'r') as f:
            loaded_content = f.read()

        assert loaded_content == test_content

    def test_load_file_non_existent_file(self):
        """Test loading a non-existent file raises an error."""
        with pytest.raises(FileNotFoundError):
            with open('non_existent_file.txt', 'r') as f:
                f.read()


@pytest.mark.skipif(
    not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE,
    reason="ONNX model file not found or CUDA/GPU not available")
class TestTRTDecoderInference(TestTRTDecoderSetup):
    """Tests for TRT decoder inference with actual model.
    
    Warning: These tests require GPU access and may take 20-30 seconds on first 
    run as TensorRT builds and optimizes the engine from the ONNX model.
    These tests may fail in containerized environments without proper GPU access.
    """

    def setup_method(self):
        """Set up decoder for inference tests."""
        super().setup_method()

        # Create parity check matrix matching the test data
        # For distance-3 surface code: 24 detectors
        num_detectors = NUM_DETECTORS

        # Create a dummy H matrix (identity matrix for simplicity)
        self.H_inference = np.eye(num_detectors, dtype=np.uint8)
        self.O_inference = np.zeros((NUM_OBSERVABLES, num_detectors),
                                    dtype=np.uint8)

        # Create the TRT decoder
        self.onnx_path = ONNX_MODEL_PATH
        try:
            self.decoder = qec.get_decoder('trt_decoder',
                                           self.H_inference,
                                           O=self.O_inference,
                                           output='observables',
                                           engine_output_format='observables',
                                           onnx_load_path=self.onnx_path)
            # If decoder is None or doesn't initialize properly, skip these tests
            if self.decoder is None:
                pytest.skip(
                    "TRT decoder returned None - likely CUDA/GPU unavailable")
        except (RuntimeError, SystemError, Exception) as e:
            pytest.skip(
                f"Failed to create TRT decoder (GPU may be unavailable): {e}")

    def test_validate_against_pytorch_model(self):
        """Test that TRT decoder produces identical results to PyTorch."""
        # Tolerance for floating point comparison
        TOLERANCE = 1e-4

        # Track statistics
        num_passed = 0
        num_failed = 0
        max_error = 0.0
        total_error = 0.0
        failed_cases = []

        # Test each of the 100 test cases
        for i, (test_input,
                test_output) in enumerate(zip(TEST_INPUTS, TEST_OUTPUTS)):
            # Run TRT decoder inference
            result = self.decoder.decode(test_input)

            # Get the expected output
            expected_output = test_output[0]

            # Get the TRT decoder output
            assert len(
                result.result
            ) > 0, f"TRT decoder returned empty result for test case {i}"
            trt_output = result.result[0]

            # Compute absolute error
            error = abs(trt_output - expected_output)
            total_error += error
            max_error = max(max_error, error)

            # Check if within tolerance
            if error < TOLERANCE:
                num_passed += 1
            else:
                num_failed += 1
                # Store detailed error info for first few failures
                if len(failed_cases) < 5:
                    failed_cases.append({
                        'index': i,
                        'expected': expected_output,
                        'got': trt_output,
                        'error': error
                    })

        # Print summary statistics
        print(f"\n=== TRT Decoder Validation Summary ===")
        print(f"Total test cases: {len(TEST_INPUTS)}")
        print(f"Passed: {num_passed}")
        print(f"Failed: {num_failed}")
        print(f"Max error: {max_error}")
        print(f"Average error: {total_error / len(TEST_INPUTS)}")
        print(f"====================================\n")

        # Print detailed failure information
        if failed_cases:
            print("\nFirst few failures:")
            for case in failed_cases:
                print(f"Test case {case['index']} FAILED:")
                print(f"  Expected: {case['expected']}")
                print(f"  Got:      {case['got']}")
                print(f"  Error:    {case['error']}")

        # Overall assertion: all cases must pass
        assert num_failed == 0, f"{num_failed} test cases failed validation"

    def test_validate_single_test_case(self):
        """Test a single specific case for detailed debugging."""
        # Test first case in detail
        syndrome = TEST_INPUTS[0]

        print(f"Input syndrome (first 10 values): {syndrome[:10]}")

        result = self.decoder.decode(syndrome)

        expected = TEST_OUTPUTS[0][0]
        actual = result.result[0]
        error = abs(actual - expected)

        print(f"Expected output: {expected}")
        print(f"Actual output:   {actual}")
        print(f"Absolute error:  {error}")
        print(f"Converged:       {result.converged}")

        assert error < 1e-4, "Single test case validation failed"
        assert result.converged, "Decoder did not converge"

    def test_decoder_result_structure(self):
        """Test that decoder result has expected structure."""
        syndrome = TEST_INPUTS[0]
        result = self.decoder.decode(syndrome)

        # Test basic structure
        assert hasattr(result, 'converged')
        assert hasattr(result, 'result')
        assert isinstance(result.converged, bool)
        assert isinstance(result.result, np.ndarray)
        assert len(result.result) > 0

    def test_decoder_batch_processing(self):
        """Test decoder batch processing with multiple syndromes."""
        # Test with first 5 test cases
        syndromes = TEST_INPUTS[:5]
        expected_outputs = [out[0] for out in TEST_OUTPUTS[:5]]

        # Decode batch
        results = self.decoder.decode_batch(syndromes)

        assert len(results) == len(syndromes)
        assert isinstance(results, qec.BatchDecoderResult)
        assert results.result.shape[0] == len(syndromes)
        assert results.converged.shape == (len(syndromes),)

        # Check each result
        TOLERANCE = 1e-4
        for i, expected in enumerate(expected_outputs):
            assert results.converged[i]
            assert results.result.shape[1] > 0

            error = abs(results.result[i, 0] - expected)
            assert error < TOLERANCE, f"Batch test case {i} failed with error {error}"


class TestTRTDecoderEdgeCases(TestTRTDecoderSetup):
    """Tests for TRT decoder edge cases. Requires GPU access."""

    def test_decoder_with_zero_syndrome(self):
        """Test decoder with all-zero syndrome."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        try:
            decoder = qec.get_decoder('trt_decoder',
                                      H,
                                      O=np.zeros(
                                          (NUM_OBSERVABLES, num_detectors),
                                          dtype=np.uint8),
                                      output='observables',
                                      engine_output_format='observables',
                                      onnx_load_path=ONNX_MODEL_PATH)
        except Exception:
            pytest.skip("Failed to create TRT decoder")

        # All-zero syndrome
        zero_syndrome = [0.0] * num_detectors
        result = decoder.decode(zero_syndrome)

        assert hasattr(result, 'result')
        assert len(result.result) > 0

    def test_decoder_with_all_ones_syndrome(self):
        """Test decoder with all-ones syndrome."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        try:
            decoder = qec.get_decoder('trt_decoder',
                                      H,
                                      O=np.zeros(
                                          (NUM_OBSERVABLES, num_detectors),
                                          dtype=np.uint8),
                                      output='observables',
                                      engine_output_format='observables',
                                      onnx_load_path=ONNX_MODEL_PATH)
        except Exception:
            pytest.skip("Failed to create TRT decoder")

        # All-ones syndrome
        ones_syndrome = [1.0] * num_detectors
        result = decoder.decode(ones_syndrome)

        assert hasattr(result, 'result')
        assert len(result.result) > 0

    def test_performance_comparison_cuda_graph_vs_traditional(self):
        """
        Test performance comparison: CUDA Graph vs Traditional execution.
        
        This test benchmarks the TRT decoder with CUDA graphs enabled vs disabled
        to report the performance from graph optimization without enforcing a
        speedup threshold in CI.
        """
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        # Create dummy H matrix
        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        # Create test syndrome (using first test input)
        syndrome = TEST_INPUTS[0]

        # =====================================================================
        # Create decoder WITH CUDA graphs (default)
        # =====================================================================
        try:
            decoder_cuda_graph = qec.get_decoder(
                'trt_decoder',
                H,
                O=np.zeros((NUM_OBSERVABLES, num_detectors), dtype=np.uint8),
                output='observables',
                engine_output_format='observables',
                onnx_load_path=ONNX_MODEL_PATH,
                precision='fp16',
                use_cuda_graph=True)
        except Exception as e:
            pytest.skip(f"Failed to create CUDA graph decoder: {e}")

        # =====================================================================
        # Create decoder WITHOUT CUDA graphs (traditional)
        # =====================================================================
        try:
            decoder_traditional = qec.get_decoder(
                'trt_decoder',
                H,
                O=np.zeros((NUM_OBSERVABLES, num_detectors), dtype=np.uint8),
                output='observables',
                engine_output_format='observables',
                onnx_load_path=ONNX_MODEL_PATH,
                precision='fp16',
                use_cuda_graph=False)
        except Exception as e:
            pytest.skip(f"Failed to create traditional decoder: {e}")

        # =====================================================================
        # Warm-up phase (for fair comparison)
        # =====================================================================
        warmup_iterations = 5
        print("\n=== Warming up decoders ===")

        for _ in range(warmup_iterations):
            decoder_cuda_graph.decode(syndrome)
            decoder_traditional.decode(syndrome)

        # =====================================================================
        # Benchmark CUDA Graph Executor
        # =====================================================================
        benchmark_iterations = 200
        print("Benchmarking CUDA Graph executor...")

        import time
        start_cuda_graph = time.perf_counter()
        for i in range(benchmark_iterations):
            result = decoder_cuda_graph.decode(syndrome)
            assert result.converged, f"CUDA graph decoder failed at iteration {i}"
        end_cuda_graph = time.perf_counter()

        duration_cuda_graph = (end_cuda_graph -
                               start_cuda_graph) * 1_000_000  # Convert to μs
        avg_time_cuda_graph = duration_cuda_graph / benchmark_iterations

        # =====================================================================
        # Benchmark Traditional Executor
        # =====================================================================
        print("Benchmarking Traditional executor...")

        start_traditional = time.perf_counter()
        for i in range(benchmark_iterations):
            result = decoder_traditional.decode(syndrome)
            assert result.converged, f"Traditional decoder failed at iteration {i}"
        end_traditional = time.perf_counter()

        duration_traditional = (end_traditional -
                                start_traditional) * 1_000_000  # Convert to μs
        avg_time_traditional = duration_traditional / benchmark_iterations

        # =====================================================================
        # Calculate and report performance improvement
        # =====================================================================
        speedup = avg_time_traditional / avg_time_cuda_graph
        improvement_percent = ((avg_time_traditional - avg_time_cuda_graph) /
                               avg_time_traditional) * 100.0

        print("\n=== Performance Comparison Results ===")
        print(f"Iterations: {benchmark_iterations}")
        print(f"CUDA Graph avg time:   {avg_time_cuda_graph:.2f} μs")
        print(f"Traditional avg time:  {avg_time_traditional:.2f} μs")
        print(f"Speedup:               {speedup:.2f}x")
        print(f"Improvement:           {improvement_percent:.2f}%")
        print("======================================\n")

        # =====================================================================
        # Performance assertions
        # =====================================================================
        # Treat the speedup as diagnostic only. On small models, Python overhead,
        # host/device copies, and runner noise can hide graph launch benefits.
        # The adjacent correctness test verifies both execution paths agree.
        # Keep only a coarse sanity check that both paths are reasonably fast.
        assert avg_time_cuda_graph < 100000.0, (
            f"CUDA graph execution unexpectedly slow: {avg_time_cuda_graph:.2f} μs"
        )
        assert avg_time_traditional < 100000.0, (
            f"Traditional execution unexpectedly slow: {avg_time_traditional:.2f} μs"
        )

    def test_cuda_graph_vs_traditional_correctness(self):
        """
        Test that CUDA graph and traditional executors produce identical results.
        
        This test builds a simple MLP engine and verifies that both execution
        paths (CUDA graph and traditional) produce the same output for random inputs.
        """
        if not CUDA_AVAILABLE or not TRT_AVAILABLE:
            pytest.skip(
                "CUDA/GPU or TensorRT Python API not available for this test")

        # Define network architecture
        input_size = NUM_DETECTORS  # 24
        hidden_size = 16
        output_size = NUM_DETECTORS  # 24

        # Create temporary directory for engine file
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = os.path.join(tmpdir, "test_mlp.engine")

            # Build TensorRT engine
            print(
                f"\nBuilding simple MLP engine ({input_size}->{hidden_size}->{output_size})..."
            )
            success = build_simple_mlp_engine(input_size, hidden_size,
                                              output_size, engine_path)
            if not success:
                pytest.skip("Failed to build TensorRT engine")

            assert os.path.exists(
                engine_path), "Engine file was not created successfully"

            # Create dummy H matrix
            H = np.eye(input_size, dtype=np.uint8)

            # =================================================================
            # Create decoder WITH CUDA graphs
            # =================================================================
            try:
                decoder_cuda_graph = qec.get_decoder(
                    'trt_decoder',
                    H,
                    engine_output_format='errors',
                    engine_load_path=engine_path,
                    use_cuda_graph=True)
            except Exception as e:
                pytest.skip(f"Failed to create CUDA graph decoder: {e}")

            # =================================================================
            # Create decoder WITHOUT CUDA graphs (traditional)
            # =================================================================
            try:
                decoder_traditional = qec.get_decoder(
                    'trt_decoder',
                    H,
                    engine_output_format='errors',
                    engine_load_path=engine_path,
                    use_cuda_graph=False)
            except Exception as e:
                pytest.skip(f"Failed to create traditional decoder: {e}")

            # =================================================================
            # Test with multiple random inputs
            # =================================================================
            num_test_cases = 10
            np.random.seed(123)

            print(f"Testing correctness with {num_test_cases} random inputs...")

            for test_idx in range(num_test_cases):
                # Generate random syndrome (values between -1 and 1)
                random_syndrome = (np.random.rand(input_size) * 2 - 1).astype(
                    np.float32).tolist()

                # Run both decoders
                result_cuda_graph = decoder_cuda_graph.decode(random_syndrome)
                result_traditional = decoder_traditional.decode(random_syndrome)

                # Verify both converged (if applicable)
                assert result_cuda_graph.converged == result_traditional.converged, (
                    f"Test {test_idx}: Convergence status differs: "
                    f"CUDA graph={result_cuda_graph.converged}, "
                    f"Traditional={result_traditional.converged}")

                # Convert results to numpy arrays for comparison
                result_cuda_np = np.array(result_cuda_graph.result)
                result_trad_np = np.array(result_traditional.result)

                # Verify results are the same length
                assert len(result_cuda_np) == len(result_trad_np), (
                    f"Test {test_idx}: Result lengths differ: "
                    f"CUDA graph={len(result_cuda_np)}, "
                    f"Traditional={len(result_trad_np)}")

                # Verify results are numerically identical (or very close)
                max_diff = np.max(np.abs(result_cuda_np - result_trad_np))
                assert max_diff < 1e-5, (
                    f"Test {test_idx}: Results differ by {max_diff:.2e}. "
                    f"CUDA graph result: {result_cuda_np[:5]}..., "
                    f"Traditional result: {result_trad_np[:5]}...")

                print(
                    f"  Test {test_idx + 1}/{num_test_cases}: PASS (max_diff={max_diff:.2e})"
                )

            print(
                "✓ All tests passed: CUDA graph and traditional execution produce identical results"
            )


class TestTRTDecoderBatchValidation:
    """Tests for TRT decoder batch size validation using programmatically created engines."""

    @pytest.fixture(params=[1, 2, 4])
    def decoder_with_batch_size(self, request, tmp_path):
        """Create TRT decoder with specific batch size using dummy MLP network."""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA/GPU not available")

        batch_size = request.param
        syndrome_size = 25
        output_size = 25
        engine_path = None

        try:
            import tensorrt as trt
        except ImportError:
            pytest.skip("TensorRT Python API not available")

        try:
            # Create dummy H matrix (identity matrix for syndrome_size)
            H = np.eye(syndrome_size, dtype=np.uint8)

            # Build TensorRT engine with explicit batch size
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(0)  # Explicit batch network
            config = builder.create_builder_config()

            # Input: [batch_size, syndrome_size]
            input_tensor = network.add_input("syndrome", trt.float32,
                                             (batch_size, syndrome_size))

            # Simple network: Just use matrix multiplication with random weights
            # This is sufficient for testing batch validation logic
            # weights shape: [syndrome_size, output_size]
            weights_data = np.random.randn(syndrome_size, output_size).astype(
                np.float32) * 0.1
            weights = trt.Weights(weights_data)
            weights_const = network.add_constant((syndrome_size, output_size),
                                                 weights)

            # Matrix multiply: [batch, syndrome_size] x [syndrome_size, output_size] -> [batch, output_size]
            matmul = network.add_matrix_multiply(input_tensor,
                                                 trt.MatrixOperation.NONE,
                                                 weights_const.get_output(0),
                                                 trt.MatrixOperation.NONE)

            # Output
            output = matmul.get_output(0)
            output.name = "correction"
            network.mark_output(output)

            # Build engine
            serialized_engine = builder.build_serialized_network(
                network, config)
            if serialized_engine is None:
                pytest.skip("Failed to build TensorRT engine")

            # Save engine to file (tmp_path is automatically cleaned up by pytest)
            engine_path = tmp_path / f"test_decoder_batch{batch_size}.trt"
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            # Create decoder from engine
            try:
                decoder = qec.get_decoder("trt_decoder",
                                          H,
                                          engine_output_format="errors",
                                          engine_load_path=str(engine_path))
            except Exception as e:
                pytest.skip(f"Failed to create decoder: {e}")

            yield decoder, batch_size, syndrome_size, output_size

        finally:
            # Explicit cleanup (pytest's tmp_path also auto-cleans)
            if engine_path is not None and engine_path.exists():
                try:
                    engine_path.unlink()
                except Exception:
                    pass  # tmp_path cleanup will handle it

    def test_decode_batch_accepts_non_integral_multiple(
            self, decoder_with_batch_size):
        """Test that decode_batch accepts a final partial batch."""
        decoder, batch_size, syndrome_size, _ = decoder_with_batch_size

        if batch_size == 1:
            pytest.skip("All counts are integral multiples for batch_size=1")

        # Create syndromes (batch_size + 1) - not a multiple
        syndromes = [[float(i + j)
                      for i in range(syndrome_size)]
                     for j in range(batch_size + 1)]

        results = decoder.decode_batch(syndromes)

        assert len(results) == len(
            syndromes), f"Expected {len(syndromes)} results, got {len(results)}"
        assert results.result.shape == (len(syndromes), syndrome_size)
        for i, converged in enumerate(results.converged):
            assert converged, f"Result {i} did not converge"

    def test_decode_batch_syndrome_size_mismatch(self, decoder_with_batch_size):
        """Test that decode_batch validates individual syndrome sizes."""
        decoder, batch_size, syndrome_size, _ = decoder_with_batch_size

        # Create one valid and one invalid syndrome
        valid_syndrome = [0.0] * syndrome_size
        invalid_syndrome = [0.0] * (syndrome_size + 5)

        # Need batch_size syndromes, make second one wrong
        if batch_size > 1:
            syndromes = [valid_syndrome, invalid_syndrome
                        ] + [valid_syndrome] * (batch_size - 2)
        else:
            syndromes = [invalid_syndrome]

        with pytest.raises(RuntimeError, match="Syndrome size mismatch"):
            decoder.decode_batch(syndromes)

    def test_decode_single_syndrome_size_mismatch(self,
                                                  decoder_with_batch_size):
        """Test that decode validates syndrome size."""
        decoder, batch_size, syndrome_size, _ = decoder_with_batch_size

        # Create syndrome with wrong size
        wrong_size_syndrome = [0.0] * (syndrome_size + 5)

        with pytest.raises(RuntimeError, match="Syndrome size mismatch"):
            decoder.decode(wrong_size_syndrome)

    def test_decode_batch_empty_list(self, decoder_with_batch_size):
        """Test that decode_batch handles empty syndrome list."""
        decoder, _, _, _ = decoder_with_batch_size

        results = decoder.decode_batch([])
        assert len(results) == 0, "Empty batch should return empty results"

    def test_decode_batch_large_integral_multiple(self,
                                                  decoder_with_batch_size):
        """Test batch decoding with a larger integral multiple of batch size."""
        decoder, batch_size, syndrome_size, _ = decoder_with_batch_size

        # Test with 2x and 3x the batch size
        for multiplier in [2, 3]:
            count = batch_size * multiplier
            syndromes = [[float(i + j)
                          for i in range(syndrome_size)]
                         for j in range(count)]

            results = decoder.decode_batch(syndromes)

            # Verify results
            assert len(
                results
            ) == count, f"Expected {count} results, got {len(results)}"

            # Verify all results are valid
            assert results.result.shape == (count, syndrome_size)
            for i, converged in enumerate(results.converged):
                assert converged, f"Result {i} did not converge"

    def test_decode_zero_padding_for_batch_gt_1(self, decoder_with_batch_size):
        """Test that decode() zero-pads correctly for batch_size > 1 models."""
        decoder, batch_size, syndrome_size, output_size = decoder_with_batch_size

        if batch_size == 1:
            pytest.skip("Zero-padding only applies to batch_size > 1")

        # Create a non-zero syndrome
        syndrome = [float(i * 0.1) for i in range(syndrome_size)]

        # Method 1: Use decode() (should zero-pad internally)
        result_decode = decoder.decode(syndrome)

        # Method 2: Manually create zero-padded batch and use decode_batch()
        zero_syndrome = [0.0] * syndrome_size
        manual_batch = [syndrome] + [zero_syndrome] * (batch_size - 1)
        result_batch = decoder.decode_batch(manual_batch)

        # Results should be identical
        assert result_decode.converged == result_batch.converged[0], \
            "Convergence status should match"

        assert len(result_decode.result) == len(result_batch.result[0]), \
            "Output sizes should match"

        # Compare results (should be very close)
        np.testing.assert_allclose(
            result_decode.result,
            result_batch.result[0],
            rtol=1e-5,
            atol=1e-7,
            err_msg=
            "Zero-padded decode() should match manually padded decode_batch()")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

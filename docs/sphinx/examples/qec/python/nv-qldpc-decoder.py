# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]

import numpy as np
from scipy.sparse import csr_matrix
import cudaq_qec as qec
import json
import time

# For fetching data
import requests
import bz2
import os

# Note: running this script will automatically download data if necessary.

### Helper functions ###


def parse_csr_mat(j, dims, mat_name):
    """
    Parse a CSR-style matrix from a JSON file using SciPy's sparse matrix utilities.
    """
    assert len(dims) == 2, "dims must be a tuple of two integers"

    # Extract indptr and indices from the JSON.
    indptr = np.array(j[f"{mat_name}_indptr"], dtype=int)
    indices = np.array(j[f"{mat_name}_indices"], dtype=int)

    # Check that the CSR structure is consistent.
    assert len(indptr) == dims[0] + 1, "indptr length must equal dims[0] + 1"
    assert np.all(
        indices < dims[1]), "All column indices must be less than dims[1]"

    # Create a data array of ones.
    data = np.ones(indptr[-1], dtype=np.uint8)

    # Build and return the sparse matrix directly — no dense allocation.
    return csr_matrix((data, indices, indptr), shape=dims, dtype=np.uint8)


def parse_H_csr(j, dims):
    """
    Parse a CSR-style parity check matrix from an input file in JSON format"
    """
    return parse_csr_mat(j, dims, "H")


def parse_obs_csr(j, dims):
    """
    Parse a CSR-style observable matrix from an input file in JSON format"
    """
    return parse_csr_mat(j, dims, "obs_mat").toarray()


### Main decoder loop ###


def run_decoder(filename, num_shots, run_as_batched):
    """
    Load a JSON file and decode "num_shots" syndromes.
    """
    t_load_begin = time.time()
    with open(filename, "r") as f:
        j = json.load(f)

    dims = j["shape"]
    assert len(dims) == 2

    # Read the Parity Check Matrix
    H = parse_H_csr(j, dims)
    syndrome_length, block_length = dims
    t_load_end = time.time()

    print(f"{filename} parsed in {1e3 * (t_load_end-t_load_begin)} ms")

    error_rate_vec = np.array(j["error_rate_vec"])
    assert len(error_rate_vec) == block_length
    obs_mat_dims = j["obs_mat_shape"]
    obs_mat = parse_obs_csr(j, obs_mat_dims)
    assert dims[1] == obs_mat_dims[0]
    file_num_trials = j["num_trials"]
    num_shots = min(num_shots, file_num_trials)
    print(
        f'Your JSON file has {file_num_trials} shots. Running {num_shots} now.')

    # osd_method: 0=Off, 1=OSD-0, 2=Exhaustive, 3=Combination Sweep
    osd_method = 1

    # When osd_method is:
    #  2) there are 2^osd_order additional error mechanisms checked.
    #  3) there are an additional k + osd_order*(osd_order-1)/2 error
    #     mechanisms checked.
    # Ref: https://arxiv.org/pdf/2005.07016
    osd_order = 0

    # Maximum number of BP iterations before attempting OSD (if necessary)
    max_iter = 50

    nv_dec_args = {
        "max_iterations": max_iter,
        "error_rate_vec": error_rate_vec,
        "use_sparsity": True,
        "use_osd": osd_method > 0,
        "osd_order": osd_order,
        "osd_method": osd_method
    }

    if run_as_batched:
        # Perform BP processing for up to 1000 syndromes per batch. If there
        # are more than 1000 syndromes, the decoder will chunk them up and
        # process each batch sequentially under the hood.
        nv_dec_args['bp_batch_size'] = min(1000, num_shots)

    try:
        nv_dec_gpu_and_cpu = qec.get_decoder("nv-qldpc-decoder", H,
                                             **nv_dec_args)
    except Exception as e:
        print(
            'The nv-qldpc-decoder is not available with your current CUDA-Q ' +
            'QEC installation.')
        exit(0)
    decoding_time = 0
    bp_converged_flags = []
    num_logical_errors = 0

    # Batched API
    if run_as_batched:
        syndrome_list = []
        obs_truth_list = []
        for i in range(num_shots):
            syndrome = j["trials"][i]["syndrome_truth"]
            obs_truth = j["trials"][i]["obs_truth"]
            syndrome_list.append(syndrome)
            obs_truth_list.append(obs_truth)
        t0 = time.time()
        results = nv_dec_gpu_and_cpu.decode_batch(syndrome_list)
        t1 = time.time()
        decoding_time += t1 - t0
        for r, obs_truth in zip(results, obs_truth_list):
            bp_converged_flags.append(r.converged)
            dec_result = np.array(r.result, dtype=np.uint8)

            # See if this prediction flipped the observable
            predicted_observable = obs_mat.T @ dec_result % 2
            print(f"predicted_observable: {predicted_observable}")

            # See if the observable was actually flipped according to the truth
            # data
            actual_observable = np.array(obs_truth, dtype=np.uint8)
            print(f"actual_observable:    {actual_observable}")

            if np.sum(predicted_observable != actual_observable) > 0:
                num_logical_errors += 1

    # Non-batched API
    else:
        for i in range(num_shots):
            syndrome = j["trials"][i]["syndrome_truth"]
            obs_truth = j["trials"][i]["obs_truth"]

            t0 = time.time()
            results = nv_dec_gpu_and_cpu.decode(syndrome)
            bp_converged = results.converged
            dec_result = results.result
            t1 = time.time()
            trial_diff = t1 - t0
            decoding_time += trial_diff

            dec_result = np.array(dec_result, dtype=np.uint8)
            bp_converged_flags.append(bp_converged)

            # See if this prediction flipped the observable
            predicted_observable = obs_mat.T @ dec_result % 2
            print(f"predicted_observable: {predicted_observable}")

            # See if the observable was actually flipped according to the truth
            # data
            actual_observable = np.array(obs_truth, dtype=np.uint8)
            print(f"actual_observable:    {actual_observable}")

            if np.sum(predicted_observable != actual_observable) > 0:
                num_logical_errors += 1

    # Count how many shots the decoder failed to correct the errors
    print(f"{num_logical_errors} logical errors in {num_shots} shots")
    print(
        f"Number of shots that converged with BP processing: {np.sum(np.array(bp_converged_flags))}"
    )
    print(
        f"Average decoding time for {num_shots} shots was {1e3 * decoding_time / num_shots} ms per shot"
    )


def demonstrate_bp_methods():
    """
    Demonstrate different BP methods available in nv-qldpc-decoder.
    Shows configurations for: sum-product, min-sum, memory BP, 
    disordered memory BP, and sequential relay BP.
    """
    # Simple 3x7 parity check matrix for demonstration
    H_list = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, 1, 1]]
    H = np.array(H_list, dtype=np.uint8)

    print("=" * 60)
    print("Demonstrating BP Methods in nv-qldpc-decoder")
    print("=" * 60)

    # Method 0: Sum-Product BP (default)
    print("\n1. Sum-Product BP (bp_method=0, default):")
    try:
        decoder_sp = qec.get_decoder("nv-qldpc-decoder",
                                     H,
                                     bp_method=0,
                                     max_iterations=30)
    except Exception as e:
        print(
            'The nv-qldpc-decoder is not available with your current CUDA-Q ' +
            'QEC installation.')
        exit(0)
    print("   Created decoder with sum-product BP")

    # Method 1: Min-Sum BP
    print("\n2. Min-Sum BP (bp_method=1):")
    decoder_ms = qec.get_decoder("nv-qldpc-decoder",
                                 H,
                                 bp_method=1,
                                 max_iterations=30,
                                 scale_factor=1.0)
    print("   Created decoder with min-sum BP")

    # Method 2: Min-Sum with uniform Memory (Mem-BP)
    print("\n3. Mem-BP (bp_method=2, uniform memory strength):")
    decoder_mem = qec.get_decoder("nv-qldpc-decoder",
                                  H,
                                  bp_method=2,
                                  max_iterations=30,
                                  use_sparsity=True,
                                  gamma0=0.5)
    print("   Created decoder with Mem-BP (gamma0=0.5)")

    # Method 3: Min-Sum with Disordered Memory (DMem-BP)
    print("\n4. DMem-BP (bp_method=3, disordered memory strength):")
    # Option A: Using gamma_dist (random gammas in range)
    decoder_dmem = qec.get_decoder("nv-qldpc-decoder",
                                   H,
                                   bp_method=3,
                                   max_iterations=30,
                                   use_sparsity=True,
                                   gamma_dist=[0.1, 0.5],
                                   bp_seed=42)
    print("   Created decoder with DMem-BP (gamma_dist=[0.1, 0.5])")

    # Option B: Using explicit_gammas (specify exact gamma for each variable)
    block_size = H.shape[1]
    explicit_gammas = [[0.1 + 0.05 * i for i in range(block_size)]]
    decoder_dmem_explicit = qec.get_decoder("nv-qldpc-decoder",
                                            H,
                                            bp_method=3,
                                            max_iterations=30,
                                            use_sparsity=True,
                                            explicit_gammas=explicit_gammas)
    print("   Created decoder with DMem-BP (explicit gammas)")

    # Method 4: Sequential Relay BP (composition=1)
    print("\n5. Sequential Relay BP (composition=1):")
    print("   Requires bp_method=3 and srelay_config")

    # Configure relay parameters
    srelay_config = {
        'pre_iter': 5,  # Run 5 iterations with gamma0 before relay legs
        'num_sets': 3,  # Use 3 relay legs
        'stopping_criterion': 'FirstConv'  # Stop after first convergence
    }

    # Option A: Using gamma_dist for relay legs
    decoder_relay = qec.get_decoder("nv-qldpc-decoder",
                                    H,
                                    bp_method=3,
                                    composition=1,
                                    max_iterations=50,
                                    use_sparsity=True,
                                    gamma0=0.3,
                                    gamma_dist=[0.1, 0.5],
                                    srelay_config=srelay_config,
                                    bp_seed=42)
    print("   Created decoder with Relay-BP (gamma_dist, FirstConv stopping)")

    # Option B: Using explicit gammas for each relay leg
    num_relay_legs = 3
    explicit_relay_gammas = [
        [0.1 + 0.02 * i for i in range(block_size)],  # First relay leg
        [0.2 + 0.03 * i for i in range(block_size)],  # Second relay leg
        [0.3 + 0.04 * i for i in range(block_size)]  # Third relay leg
    ]

    srelay_config_all = {
        'pre_iter': 10,
        'num_sets': 3,
        'stopping_criterion': 'All'  # Run all relay legs
    }

    decoder_relay_explicit = qec.get_decoder(
        "nv-qldpc-decoder",
        H,
        bp_method=3,
        composition=1,
        max_iterations=50,
        use_sparsity=True,
        gamma0=0.3,
        explicit_gammas=explicit_relay_gammas,
        srelay_config=srelay_config_all)
    print("   Created decoder with Relay-BP (explicit gammas, All legs)")

    # Option C: NConv stopping criterion
    srelay_config_nconv = {
        'pre_iter': 5,
        'num_sets': 10,
        'stopping_criterion': 'NConv',
        'stop_nconv': 3  # Stop after 3 convergences
    }

    decoder_relay_nconv = qec.get_decoder("nv-qldpc-decoder",
                                          H,
                                          bp_method=3,
                                          composition=1,
                                          max_iterations=50,
                                          use_sparsity=True,
                                          gamma0=0.3,
                                          gamma_dist=[0.1, 0.6],
                                          srelay_config=srelay_config_nconv,
                                          bp_seed=42)
    print("   Created decoder with Relay-BP (NConv stopping after 3)")

    print("\n" + "=" * 60)
    print("All decoder configurations created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Demonstrate different BP methods (introduced in v0.5.0)
    print("\n### PART 1: BP Methods Demonstration ###\n")
    demonstrate_bp_methods()

    # Full decoding with test data
    print("\n\n### PART 2: Full Decoding Example with Test Data ###\n")

    # See other test data options in https://github.com/NVIDIA/cudaqx/releases/tag/0.2.0
    filename = 'osd_1008_8785_0.001.json'
    bz2filename = filename + '.bz2'
    if not os.path.exists(filename):
        url = f"https://github.com/NVIDIA/cudaqx/releases/download/0.2.0/{bz2filename}"

        print(f'Downloading data from {url}')

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error if download fails
        with open(bz2filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f'Decompressing {bz2filename} into {filename}')

        # Decompress the file
        with bz2.BZ2File(bz2filename, "rb") as f_in, open(filename,
                                                          "wb") as f_out:
            f_out.write(f_in.read())

        print(f"Decompressed file saved as {filename}")

    num_shots = 100
    run_as_batched = True
    run_decoder(filename, num_shots, run_as_batched)

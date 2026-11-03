Experiments and Noise Modeling
==============================

The CUDA-Q QEC library lets you run numerical error-correction experiments -- studying how a code and decoder behave under noise. This page introduces several of the most common: modeling noise at the **code-capacity** and **circuit-level**, and running full **memory circuit experiments**.

For a walkthrough of the experiments described below, see the :doc:`Experiments and Noise Modeling </examples_rst/qec/modeling_noise>` example, along with the :doc:`C++ </api/qec/cpp_api>` and :doc:`Python </api/qec/python_api>` API reference.

The sections below follow the :doc:`Conventions <conventions>` for errors, syndromes, and logical observables.

Code-Capacity Noise Modeling
----------------------------

Quantum error correction (QEC) describes a set of tools used to detect and correct errors which occur to qubits on quantum computers.
CUDA-Q QEC centers on two of the most common objects in QEC: stabilizer codes, and decoders.
A stabilizer code is the quantum generalization of linear codes in classical error correction, which use parity checks to detect errors on noisy bits.
In QEC, we'll perform stabilizer measurements on ancilla qubits to check the parity of our data qubits.
These stabilizer measurements are non-destructive, and thus allow us to check the relative parity of qubits without destroying our quantum information.

For example, if we prepare two qubits in the state :math:`|\Psi\rangle = a|00\rangle + b|11\rangle`, we may want to check if a bit-flip error happened.
We can measure the stabilizer `ZZ`, which will return 0 if there are no errors or an even number of errors, but will return 1 if either has flipped.
This is how we can perform parity checks in quantum computing, without performing destructive measurements which collapse our superposition.
How these measurements are physically performed is covered in circuit-level noise modeling below.

We can specify a stabilizer code with either a list of stabilizer operators (like `ZZ` above), or equivalently, a parity check matrix.
We can think of the columns of a parity check matrix as the types of errors that can occur. In this case, each qubit can experience a bit flip `X` or a phase flip `Z` error, so the parity check matrix will have 2N columns where N is the number of data qubits.
Each row represents a stabilizer, or a parity check.
The values are either 0 or 1, where a 1 means that the corresponding column does participate in the parity check, and a 0 means it does not.
Therefore, if a single `X/Z` error happens to a qubit, the supported rows of the parity check matrix will trigger.
This is called the syndrome, a string of 0's and 1's corresponding to which parity checks were violated.
A special class of stabilizer codes are called CSS (Calderbank-Shor-Steane) codes, which means the `X` and `Z` components of their parity check matrix can be separated.

This brings us to decoding. Decoding is the act of solving the problem: given a syndrome, which underlying errors are most likely?
There are many decoding algorithms; the code-capacity example uses a simple single-error look-up table.
This means that the decoder will enumerate for each single error bit string, what the resulting syndromes are.
Then given a syndrome, it will look up the error string and return that as a result.

The last ingredient is a way to generate errors. The code-capacity noise model assumes an independent and identical chance that an `X` or `Z` error happens on each qubit with some probability `p`.

For the runnable code, see :ref:`Code-Capacity Noise Modeling <examples_rst/qec/modeling_noise:Code-Capacity Noise Modeling>`.

Circuit-level Noise Modeling
----------------------------

Circuit-level noise modeling builds upon the code-capacity model above.
In the circuit-level noise modeling experiment, we have many of the same components from the CUDA-Q QEC library: QEC codes, decoders, and noisy data.
The primary difference here, is that we can begin to run CUDA-Q kernels to generate noisy data, rather than just generating a random bit string to represent our errors.

Along with the stabilizers, parity check matrices, and logical observables, the QEC code type also has an encoding map.
This map allows codes to define logical gates in terms of gates on the underlying physical qubits.
These encodings operate on the `qec.patch` type, which represents three registers of physical qubits making up a logical qubit.
A data qubit register, an X-stabilizer ancilla register, and a Z-stabilizer ancilla register.

The most notable encoding stored in the QEC map is the `qec.operation.stabilizer_round`, which encodes a `cudaq.kernel` that stores the gate-level information for performing a stabilizer measurement.
These stabilizer rounds are the gate-level way to encode the parity check matrix of a QEC code into quantum circuits.

Circuit-level noise modeling simulates a quantum memory experiment.
These experiments model how well QEC cycles, or rounds of stabilizer measurements, can protect the information encoded in a logical qubit.
If noise is turned off, then the information is protected indefinitely.
The circuit-level example models depolarization noise after each CX gate and tracks how many logical errors occur.

For the runnable code, see :ref:`Circuit-level Noise Modeling <examples_rst/qec/modeling_noise:Circuit-level Noise Modeling>`.

Memory Circuit Experiments
--------------------------

Memory circuit experiments test a QEC code's ability to preserve quantum information over time by:

1. Preparing an initial logical state
2. Performing multiple rounds of stabilizer measurements
3. Measuring data qubits to verify state preservation
4. Optionally applying noise during the process

A memory circuit experiment measures how well a code and decoder preserve a logical qubit through repeated rounds of noisy stabilizer measurement. After a logical state is prepared, each round extracts a syndrome while noise acts on the qubits; the final data-qubit measurement reconstructs the logical observable so the outcome can be compared against the prepared state. Because the same logical information must survive every round, the experiment exercises the full detection-and-correction loop over time rather than a single shot -- it reveals whether the decoder keeps accumulated errors below the code's threshold, and how the logical error rate grows with the number of rounds and the physical error rate. Running many shots yields the statistics used to estimate a code's logical error rate, and ultimately its threshold.

For the runnable code -- the ``sample_memory_circuit`` function variants, a full experiment, and additional noise models -- see :ref:`Memory Circuit Experiments <examples_rst/qec/modeling_noise:Memory Circuit Experiments>`.

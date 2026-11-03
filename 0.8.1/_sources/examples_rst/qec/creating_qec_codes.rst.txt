Creating New QEC Codes
======================

Below, we demonstrate how to use CUDA-Q QEC to define a new QEC code entirely in Python. This powerful feature allows for rapid prototyping and testing of custom error correction schemes.

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/custom_repetition_code_fine_grain_noise.py
      :language: python
      :start-after: [Begin Documentation]

This example illustrates several key concepts for defining custom codes:

* **Define a Code Class**: A new code is defined by creating a Python class decorated with ``@qec.code(...)``, which registers it with the CUDA-Q QEC runtime. The class must inherit from ``qec.Code``.
* **Implement Required Methods**: The class must implement methods that describe the code's structure, such as ``get_num_data_qubits()`` and ``get_num_ancilla_qubits()``.
* **Define Logical Operations as Kernels**: Quantum operations like state preparation (``prep0``, ``prep1``), logical gates (``x_logical``), and stabilizer measurements (``stabilizer_round``) are implemented as standard CUDA-Q kernels.
* **Map Operations to Kernels**: The ``operation_encodings`` dictionary links abstract QEC operations (e.g., ``qec.operation.prep0``) to the concrete CUDA-Q kernels that implement them.
* **Provide Stabilizers and Observables**: The code's stabilizer generators and logical observables must be defined. This is typically done by creating lists of ``cudaq.SpinOperator`` objects representing the Pauli strings for the stabilizers (e.g., "ZZI") and logical operators (e.g., "ZZZ").
* **Specify Fine-Grained Noise**: This example demonstrates applying noise at a specific point within a kernel. Inside ``stabilizer_round``, ``cudaq.apply_noise`` is called on each data qubit, offering precise control over the noise model, in contrast to applying noise globally to all gates of a certain type.

Once defined, the custom code can be instantiated with ``qec.get_code()`` and used with all standard CUDA-Q QEC tools, including ``qec.dem_from_memory_circuit()`` and ``qec.sample_memory_circuit()``.

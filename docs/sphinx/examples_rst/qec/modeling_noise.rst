Experiments and Noise Modeling
==============================

These examples walk through several of the most common numerical error-correction experiments with the CUDA-Q QEC library: modeling noise at the **code-capacity** and **circuit-level**, and running full **memory circuit experiments**. For the background behind each, see :doc:`Experiments and Noise Modeling </components/qec/numerical_experiments>`.

Code-Capacity Noise Modeling
----------------------------

This example implements a code-capacity noise experiment: random ``X``/``Z`` errors are applied directly to the data qubits and decoded with a single-error look-up table. See :ref:`Code-Capacity Noise Modeling <components/qec/numerical_experiments:Code-Capacity Noise Modeling>` for more details.

CUDA-Q QEC Implementation
+++++++++++++++++++++++++++++
Here's how to use CUDA-Q QEC to perform a code capacity noise model experiment in both Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/code_capacity_noise.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/code_capacity_noise.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --target=stim -lcudaq-qec -lcudaq-qec-decoders code_capacity_noise.cpp -o code_capacity_noise
      ./code_capacity_noise


Code Explanation
++++++++++++++++

1. QEC Code type:
    - CUDA-Q QEC centers around the `qec.code` type, which contains the data relevant for a given code.
    - In particular, this represents a collection of qubits which represent a single logical qubit.
    - Here we get one of the most well known QEC codes, the Steane code, with the `qec.get_code` function.
    - We can get the stabilizers from a code with the `code.get_stabilizers()` function.
    - In this example, we get the parity check matrix of the code. Because the Steane code is a CSS code, we can extract just the `Z` components of the parity check matrix.
    - Here, we see this matrix has 3 rows and 7 columns, which means there are 7 data qubits (7 possible single bit-flip errors) and 3 Z-stabilizers (parity checks). Note that `Z` stabilizers check for `X` type errors.
    - Lastly, we get the logical `Z` observable for the code. This will allow us to see if the `Z` observable of our logical qubit has flipped.

2. Decoder type:
    - A single-error look-up table (LUT) decoder can be acquired with the `qec.get_decoder` call.
    - Passing in the parity check matrix gives the decoder the required information to associated syndromes with underlying error mechanisms.
    - Once the decode has been constructed, the `decoder.decode(syndrome)` member function is called, which returns a predicted error given the syndrome.

3. Noise model:
    - To generate noisy data, we call `qec.generate_random_bit_flips(nBits, p)` which will return an array of bits, where each bit has probability `p` to have been flipped into 1, and a `1-p` chance to have remained 0.
    - Since we are using the `Z` parity check matrix `H_Z`, we want to simulate random `X` errors on our 7 data qubits.

4. Logical Errors:
    - Once we have noisy data, we see what the resulting syndromes are by multiplying our noisy data vector with our parity check matrix (mod 2).
    - From this syndrome, we see what errors the decoder predicts occurred in the data.
    - To classify as a logical error, the decoder does not need to exactly identify what happened to the data, but only whether there was a flip in the logical observable.
    - If the decoder guesses this successfully, we have corrected the quantum error. If not, we have incurred a logical error.

5. Further automation:
    - While this workflow is nice for seeing things step by step, the `qec.sample_code_capacity` API is provided to generate a batch of noisy data and their corresponding syndromes.

Circuit-level Noise Modeling
----------------------------
This example runs a circuit-level memory experiment, generating syndromes by executing the stabilizer-measurement circuits under depolarizing noise. See :ref:`Circuit-level Noise Modeling <components/qec/numerical_experiments:Circuit-level Noise Modeling>` for more details.


CUDA-Q QEC Implementation
+++++++++++++++++++++++++++++
Here's how to use CUDA-Q QEC to perform a circuit-level noise model experiment in both Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/circuit_level_noise.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/circuit_level_noise.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --target=stim -lcudaq-qec -lcudaq-qec-decoders circuit_level_noise.cpp -o circuit_level_noise
      ./circuit_level_noise


Code Explanation
++++++++++++++++

1. QEC Code and Decoder types:
    - As in the code capacity example, our central objects are the `qec.code` and `qec.decoder` types.

2. Clifford simulation backend:
    - As the size of QEC circuits can grow quite large, Clifford simulation is often the best tool for these simulations.
    - `cudaq.set_target("stim")` selects the highly performant Stim simulator as the simulation backend.

3. Noise model:
    - To add noisy gates we use the `cudaq.NoiseModel` type.
    - CUDA-Q supports the generation of arbitrary noise channels. Here we use a `cudaq.Depolarization2` channel to add a depolarization channel.
    - This is added to the `CX` gate by adding it to the `X` gate with 1 control.
    - This noisy gate is added to every qubit via the `noise.add_all_qubit_channel` function.

4. Getting circuit-level noisy data:
    - The `qec.code` is the first input parameter here, as the code's `stabilizer_round` determines the circuits executed.
    - Each memory circuit runs for an input number of `nRounds`, which specifies how many `stabilizer_round` kernels are run.
    - After `nRounds` the data qubits are measured and the run is over.
    - This is performed `nShots` number of times.
    - During a shot, each stabilizer round's syndrome is `xor`'d against the preceding syndrome, so that we can track a sparser flow of data showing which round each parity check was violated.
    - The first round returns the syndrome as is, as there is nothing preceding to `xor` against.

5. Data qubit measurements:
    - The data qubits are only read out after the end of each shot, so there are `nShots` worth of data readouts.
    - The basis of the data qubit measurements depends on the state preparation used.
    - Z-basis readout when preparing the logical `|0>` or logical `|1>` state with the `qec.operation.prep0` or `qec.operation.prep1` kernels.
    - X-basis readout when preparing the logical `|+>` or logical `|->` state with the `qec.operation.prepp` or `qec.operation.prepm` kernels.

6. Logical Errors:
    - From here, the decoding procedure is again similar to the code capacity case, except that we use a Pauli frame to track errors that happen each QEC cycle.
    - The final values of the Pauli frame tell us how our logical state flipped during the experiment, and what needs to be done to correct it.
    - We compare our known initial state (corrected by the Pauli frame), against our measured data qubits to determine if a logical error occurred.


The CUDA-Q QEC library thus provides a platform for numerical QEC experiments. The `qec.code` can be used to analyze a variety of QEC codes (both library or user provided), with a variety of decoders (both library or user provided).
The CUDA-Q QEC library also provides tools to speed up the automation of generating noisy data and syndromes.


Memory Circuit Experiments
--------------------------

The ``sample_memory_circuit`` API runs a memory circuit experiment end to end -- preparing a logical state, running rounds of stabilizer measurement under noise, and measuring the data qubits. See :ref:`Memory Circuit Experiments <components/qec/numerical_experiments:Memory Circuit Experiments>` for more details.

Function Variants
+++++++++++++++++

.. tab:: Python

    .. code-block:: python

        import cudaq
        import cudaq_qec as qec

        # Use the stim backend for performance in QEC settings
        cudaq.set_target("stim")

        # Get a code instance
        code = qec.get_code("steane")

        # Basic memory circuit with |0⟩ state
        syndromes, measurements = qec.sample_memory_circuit(
            code,           # QEC code instance
            numShots=1000,  # Number of circuit executions
            numRounds=1     # Number of stabilizer rounds
        )

        # Memory circuit with custom initial state
        syndromes, measurements = qec.sample_memory_circuit(
            code,                     # QEC code instance
            op=qec.operation.prep1,   # Initial state
            numShots=1000,            # Number of shots
            numRounds=1               # Number of rounds
        )

        # Memory circuit with noise model
        noise = cudaq.NoiseModel()
        # Configure noise
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
        syndromes, measurements = qec.sample_memory_circuit(
            code,             # QEC code instance
            numShots=1000,    # Number of shots
            numRounds=1,      # Number of rounds
            noise=noise       # Noise model
        )

.. tab:: C++

    .. code-block:: cpp

        // Basic memory circuit with |0⟩ state
        auto [syndromes, measurements] = qec::sample_memory_circuit(
            code,       // QEC code instance
            numShots,   // Number of circuit executions
            numRounds   // Number of stabilizer rounds
        );

        // Memory circuit with custom initial state
        auto [syndromes, measurements] = qec::sample_memory_circuit(
            code,               // QEC code instance
            operation::prep1,   // Initial state preparation
            numShots,           // Number of circuit executions
            numRounds           // Number of stabilizer rounds
        );

        // Memory circuit with noise model
        auto noise_model = cudaq::noise_model();
        noise_model.add_channel(...);  // Configure noise
        auto [syndromes, measurements] = qec::sample_memory_circuit(
            code,         // QEC code instance
            numShots,     // Number of circuit executions
            numRounds,    // Number of stabilizer rounds
            noise_model   // Noise model to apply
        );

Return Values
+++++++++++++

The functions return a tuple containing:

1. **Syndrome Measurements** (:code:`tensor<uint8_t>`):

   * Shape: :code:`(num_shots, num_detectors)`
   * Columns follow the layout ``[ B  S  S  …  S  B ]``, where:

     - ``B`` (boundary block) = ``numAncZ = code.get_num_z_stabilizers()`` for Z-basis
       preparations (``prep0``/``prep1``), or ``numAncX = code.get_num_x_stabilizers()``
       for X-basis preparations (``prepp``/``prepm``)
     - ``S`` (inter-round block) = ``numAncZ + numAncX`` detectors per round transition
       (``num_rounds - 1`` blocks total)
     - Total: ``num_detectors = 2*B + (num_rounds - 1)*S``
   * Values are 0 or 1 representing measurement outcomes

2. **Data Measurements** (:code:`tensor<uint8_t>`):

   * Shape: :code:`(num_shots, block_size)`
   * Contains final data qubit measurements
   * Used to verify logical state preservation

Example Usage
+++++++++++++

Example of running a memory experiment:

.. tab:: Python

    .. code-block:: python

        import cudaq
        import cudaq_qec as qec

        # Use the stim backend for performance in QEC settings
        cudaq.set_target("stim")

        # Create code and decoder
        code = qec.get_code('steane')
        decoder = qec.get_decoder('single_error_lut',
                                  code.get_parity())

        # Configure noise
        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)

        # Run memory experiment
        syndromes, measurements = qec.sample_memory_circuit(
            code,
            op=qec.operation.prep0,
            numShots=1000,
            numRounds=10,
            noise=noise
        )

        # Analyze results
        for shot in range(1000):
            # Get syndrome for this shot
            syndrome = syndromes[shot].tolist()

            # Decode syndrome
            result = decoder.decode(syndrome)
            if result.converged:
                # Process correction
                pass

.. tab:: C++

    .. code-block:: cpp

        // Compile and run with:
        // nvq++ --target=stim -lcudaq-qec -lcudaq-qec-decoders example.cpp
        // ./a.out

        #include "cudaq.h"
        #include "cudaq/qec/decoder.h"
        #include "cudaq/qec/experiments.h"
        #include "cudaq/qec/noise_model.h"

        int main(){
          // Create a Steane code instance
          auto code = cudaq::qec::get_code("steane");

          // Configure noise model
          cudaq::noise_model noise;
          noise.add_all_qubit_channel("x", cudaq::depolarization2(0.1),
                              /*num_controls=*/1);

          // Run memory experiment
          auto [syndromes, data] = cudaq::qec::sample_memory_circuit(
              *code,                          // Code instance
              cudaq::qec::operation::prep0,   // Prepare |0⟩ state
              1000,                           // 1000 shots
              1,                              // 1 rounds
              noise                           // Apply noise
          );

          // Analyze results
          auto decoder = cudaq::qec::get_decoder("single_error_lut", code->get_parity());
          for (std::size_t shot = 0; shot < 1000; shot++) {
            // Get syndrome for this shot
            std::vector<cudaq::qec::float_t> syndrome(syndromes.shape()[1]);
            for (std::size_t i = 0; i < syndrome.size(); i++)
              syndrome[i] = syndromes.at({shot, i});

            // Decode syndrome
            auto results = decoder->decode(syndrome);
            // Process correction
            // ...
          }
        }

Additional Noise Models
+++++++++++++++++++++++

.. tab:: Python

  .. code-block:: python

     noise = cudaq.NoiseModel()

     # Add multiple error channels
     noise.add_all_qubit_channel('h', cudaq.BitFlipChannel(0.001))

     # Specify two qubit errors
     noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)

.. tab:: C++

    .. code-block:: cpp

      cudaq::noise_model noise;

      // Add multiple error channels
      noise.add_all_qubit_channel(
          "x", cudaq::bit_flip_channel(/*probability*/ 0.01));

      // Specify two qubit errors
      noise.add_all_qubit_channel(
          "x", cudaq::depolarization2(/*probability*/ 0.01),
          /*numControls*/ 1);

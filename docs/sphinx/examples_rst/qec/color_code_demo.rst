Ising Color-Code Decoder: Accuracy vs. Runtime
==============================================

This example reproduces the model card figure of the
`Ising-Decoder-ColorCode-1-Fast <https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast>`__
neural-network predecoder. It plots the logical error rate (LER) per round
against the decode runtime per round, for two decoders:

- ``chromobius`` — the Chromobius color-code decoder on its own.
- ``composed`` — the Ising predecoder, run on TensorRT (``trt_decoder``),
  followed by Chromobius on the residual syndrome.

The syndromes come from Stim color-code memory experiments (distance
:math:`d`, :math:`d` rounds, Z basis, physical error rate ``--p``) and are
replayed by the playback emulator (``qec.playback``). Each runtime is measured
once per *session*, the path between the emulator and the decoder:

- ``inproc`` — the decoder runs inside the emulator process.
- ``server`` — a ``decoding_server`` over UDP on the local machine.

For each distance, Stim samples ``--shots`` shots (default 100,000) into one
schedule, which every decoder and session replays once. Each run gives both
numbers:

- The runtime per round: the time until the last correction returns, divided
  by the rounds decoded. Up to 4 shots are in flight at once, so this is the
  throughput.
- The LER per round: the fraction of shots whose correction differs from
  Stim's true observable flip, converted to a per-round rate.

Requirements
------------

- A CUDA-Q QEC install with the decoding server, the playback emulator, and
  the TensorRT decoder plugin; an NVIDIA GPU for ``composed``.
- Python packages: ``chromobius``, ``matplotlib``, ``tensorrt``, and the
  Ising-Decoding inference requirements
  (``code/requirements_public_inference.txt`` in its repository).
- The model weights,
  ``ising_decoder_color_code_1_fast_r13_v1.0.400_fp16.safetensors``, downloaded
  from the gated model page above (for example with ``hf download``).

Running
-------

From ``examples/qec/color_code_demo``, with the CUDA-Q QEC Python environment
set up:

.. code-block:: bash

   ./run_color_code_demo.sh --weights <weights>

The script fetches Ising-Decoding into ``./deps``, then runs the demo; ``--help`` lists the options. The demo has three steps, chosen
with ``--steps`` (default: all):

- ``prepare`` — writes each distance's detector error model, exports the
  predecoder to ONNX, and builds its TensorRT engine.
- ``run`` — replays each distance's schedule through every decoder and
  session.
- ``plot`` — writes ``color_code_demo.png``.

All files go to the current directory; results
accumulate in ``results.json``, so steps can be run separately.
In the figure, grey is raw Chromobius and green is the Ising predecoder;
hollow markers have fewer than 25 logical errors behind them.

The example source
------------------

.. literalinclude:: ../../examples/qec/color_code_demo/color_code_demo.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]

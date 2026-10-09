Ising Color-Code Predecoder
==============================================

This example prepares and runs the 
`Ising-Decoder-ColorCode-1-Fast <https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast>`__
neural-network predecoder using the CUDA-Q QEC realtime decoding server.
It plots the logical error rate (LER) per round against the decode 
runtime per round, for two configurations:

- ``chromobius`` — the Chromobius color-code decoder on its own.
- ``trt_decoder`` — the Ising predecoder, run on TensorRT, with Chromobius as
  the global decoder for the residual syndrome.

The syndromes come from Stim color-code memory experiments (distance
:math:`d`, :math:`d` rounds, Z basis, physical error rate ``--p``) and are
replayed by the playback emulator. Each runtime is measured
once per path between the emulator and the decoder (``session``):

- ``inproc`` — the decoder runs inside the emulator process.
- ``server`` — a ``decoding_server`` over UDP on the local machine.

For each distance, Stim samples ``--shots`` shots (default 100,000) into one
schedule, which every decoder and session replays once. Each run gives:

- The runtime per round: the time until the last correction returns, divided
  by the rounds decoded.
- The LER per round: the fraction of shots whose correction differs from
  Stim's true observable flip, converted to a per-round rate.

Requirements
------------

- A CUDA-Q QEC install with the decoding server, the playback emulator, and
  the TensorRT decoder plugin
  (built only if TensorRT 10.x is found; see :doc:`realtime_predecoder_pymatching`).
- Python packages: ``chromobius``, ``matplotlib``, ``tensorrt``, and the
  Ising-Decoding inference requirements
  (``code/requirements_public_inference.txt`` in its repository).
- ``git``, and access to the gated model: accept its terms on the model page
  above and authenticate the ``hf`` CLI (``hf auth login``), as for the
  :doc:`realtime decoding demo <realtime_decoding_demo>`.

Running
-------

From any scratch directory, with the CUDA-Q QEC Python environment set up:

.. code-block:: bash

   python3 /path/to/examples/qec/color_code_demo/color_code_demo.py

The demo fetches pinned versions of Ising-Decoding and the model weights into
``./deps`` (``--weights`` uses a local copy instead); ``--help`` lists the
options. For each distance, the demo writes the detector
error model, exports the predecoder to ONNX, builds its TensorRT engine, and
replays the schedule through every decoder and session. It then writes the
results to ``results.json`` and the figure to ``color_code_demo.png``.

All files go to the current directory.
In the figure, grey is raw Chromobius and green is the Ising predecoder.

The example source
------------------

.. literalinclude:: ../../examples/qec/color_code_demo/color_code_demo.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]

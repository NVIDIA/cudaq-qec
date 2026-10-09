Ising Color-Code Emulator with Decoding Server
=======================================================

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

.. note::

   :doc:`/examples_rst/qec/realtime_decoding_demo` serves the surface-code Ising
   predecoder from ``decoding_server``, with syndromes sourced from a QPU kernel or an
   FPGA. Here, the playback emulator replays the pre-sampled shots of the superdense color-code
   memory circuit to every the decoder on a fixed timing loop using the playback emulator.

Prerequisites
-------------

Hardware
^^^^^^^^

- CUDA-capable GPU (for the TensorRT engine)

Software
^^^^^^^^

- **CUDA Toolkit** 12.6 or later
- **TensorRT** 10.x libraries (``libnvinfer``, ``libnvonnxparser``), also
  needed when CUDA-Q QEC is built: the TensorRT decoder plugin is built only if
  TensorRT is found (see :doc:`realtime_predecoder_pymatching`)
- **CUDA-Q SDK** pre-installed (``cudaq_qec`` imports the ``cudaq`` Python
  package)
- **CUDA-Q Realtime** libraries (``libcudaq-realtime`` and the UDP bridge
  ``libcudaq-realtime-bridge-udp``), used by ``decoding_server``
- **CUDA-Q QEC** installed with ``decoding_server``, the playback emulator, and
  the TensorRT decoder plugin
- **Python packages**: the Ising-Decoding inference requirements
  (``code/requirements_public_inference.txt`` in its repository, including
  ``chromobius``, ``matplotlib``, ``onnx``, ``stim``, and ``torch``), and
  ``huggingface_hub`` for the ``hf`` CLI
- **Model access**: ``git``, plus the gated model's terms accepted on its page
  above and the ``hf`` CLI authenticated (``hf auth login``), as for the
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

Reading the Output
------------------

For each distance, Stim samples ``--shots`` shots (default 100,000) into one
schedule, which every decoder and session replays once. Each run prints and
records:

- The runtime per round: the time until the last correction returns, divided
  by the rounds decoded.
- The LER per round: the fraction of shots whose correction differs from
  Stim's true observable flip, converted to a per-round rate.

In the figure, grey is raw Chromobius and green is the Ising predecoder.

The example source
------------------

.. literalinclude:: ../../examples/qec/color_code_demo/color_code_demo.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]

See Also
--------

* :doc:`Realtime Decoding </components/qec/realtime_decoding>` -- concept and workflow
* :doc:`Getting Started with Realtime Decoding </examples_rst/qec/getting_started_realtime_decoding>`
* :ref:`C++ <cpp_realtime_decoding_api>` and :ref:`Python <python_realtime_decoding_api>` realtime decoding API

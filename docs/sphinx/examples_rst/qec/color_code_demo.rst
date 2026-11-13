Comparing Ising and Chromobius for Color-Code Decoding
======================================================

This experiment compares Chromobius alone with an Ising neural predecoder
followed by Chromobius. It replays the same sampled measurements through both
configurations, measuring logical errors and runtime. Running each configuration
in process and through a local decoding server also shows how the deployment
path affects runtime.

Use this example to evaluate the complete predecoding pipeline and compare it
with a conventional decoder on the same workload. See
:doc:`AI Predecoding </components/qec/ai_predecoding>` for how the two decoding
stages work together. The experiment runs entirely in software with a GPU;
you do not need a QPU or FPGA.

What the Experiment Compares
----------------------------

The script prepares the
`Ising-Decoder-ColorCode-1-Fast <https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast>`__
neural network using code from
`NVIDIA Ising-Decoding <https://github.com/NVIDIA/Ising-Decoding>`__.
It exports the network and its input and output processing to ONNX, then
builds a TensorRT engine. The two decoder configurations are:

- ``chromobius`` — the Chromobius color-code decoder on its own.
- ``trt_decoder`` — the Ising predecoder, run on TensorRT, with Chromobius as
  the global decoder for the residual syndrome. This uses the TensorRT decoder
  plugin, not a separate Ising decoder plugin.

The measurements come from Stim color-code memory experiments (distance
:math:`d`, :math:`d` rounds, Z basis, physical error rate ``--p``) and are
replayed by the :doc:`Playback Emulator </components/qec/playback_emulator>`.
For each distance, both decoders process the same sampled shots through two
deployment paths (``session``):

- ``inproc`` — the decoder runs inside the emulator process.
- ``server`` — a ``decoding_server`` over UDP on the local machine.

The script uses the superdense color-code memory circuit from Ising-Decoding
and allows up to four shots in flight. Measurements are sent as soon as the
schedule's dependencies allow; this is not a simulation of a fixed QPU cycle
time. Both sessions exercise the realtime request interface, even though the
measurements were sampled beforehand.

Prerequisites
-------------

Hardware
^^^^^^^^

- CUDA-capable GPU (for the TensorRT engine)

Software
^^^^^^^^

- **CUDA Toolkit** 12.6 or later
- **TensorRT** 10.x libraries (``libnvinfer``, ``libnvonnxparser``), plus headers
  if you build the TensorRT decoder plugin from source
- **CUDA-Q SDK** pre-installed (``cudaq_qec`` imports the ``cudaq`` Python
  package)
- **CUDA-Q Realtime** libraries (``libcudaq-realtime`` and the UDP bridge
  ``libcudaq-realtime-bridge-udp``), used by ``decoding_server``
- **CUDA-Q QEC** with the Python playback interface (``cudaq_qec.playback``),
  the ``decoding_server`` executable, and the ``chromobius`` and ``trt_decoder``
  plugins available in the same environment
- **Python packages**: the
  `Ising-Decoding inference requirements at the script's pinned revision
  <https://github.com/NVIDIA/Ising-Decoding/blob/33acb152e403bc189f2effdb07f1a87b34c745f1/code/requirements_public_inference.txt>`__
  (including ``chromobius``, ``matplotlib``, ``onnx``, ``stim``, and ``torch``), and
  ``huggingface_hub`` for the ``hf`` CLI
- **Model access**: ``git``, plus the gated model's terms accepted on its page
  above and the ``hf`` CLI authenticated (``hf auth login``), as for the
  :doc:`realtime_decoding_demo`. An existing local weights file can be supplied
  with ``--weights`` instead.

.. note::

   The 0.9.0 pip wheels do not include the Python playback interface.
   The decoding-server image provides the server and TensorRT plugin, but
   does not by itself provide the Python environment needed to run this
   experiment. If building an environment from source, follow
   `Building CUDA-Q QEC from Source
   <https://github.com/NVIDIA/cudaq-qec/blob/main/Building.md>`__ with CUDA-Q
   realtime support and ``CUDAQ_QEC_BUILD_TRT_DECODER=ON``. Both the Python
   process and its local server must be able to load the decoder plugins.

Run the Comparison
------------------

Download :download:`color_code_demo.py
<../../examples/qec/color_code_demo/color_code_demo.py>` or use the copy under
``docs/sphinx/examples/qec/color_code_demo/`` in your CUDA-Q QEC checkout.
With the prerequisites above available, run it from a separate scratch
directory. Start with one distance and a small number of shots to check the
setup:

.. code-block:: bash

   python3 /path/to/color_code_demo.py --distances 5 --shots 100

The script starts and stops the local server itself. If ``decoding_server``
is not on ``PATH``, pass ``--decoding-server /path/to/decoding_server``.
Each distance produces four result lines: one for each decoder and session.
The run fails if any shot does not return a correction successfully; logical
errors are counted as experiment results, not transport failures.

For the full comparison, the defaults are distances 5, 7, 9, 11, and 13,
100,000 shots per distance, and :math:`p = 0.001`:

.. code-block:: bash

   python3 /path/to/color_code_demo.py

The script fetches Ising-Decoding at commit
``33acb152e403bc189f2effdb07f1a87b34c745f1`` and the model weights at revision
``c5775431d0ebb06ffecc3836935708b787db51ee`` into ``./deps``. Use a fresh
dependency directory, or ensure that an existing ``deps/Ising-Decoding``
checkout matches that commit: the script reuses an existing checkout.
``--deps`` changes the dependency directory; ``--weights`` selects an existing
copy of ``ising_decoder_color_code_1_fast_r13_v1.0.400_fp16.safetensors``.
``--help`` lists the remaining options.

For each distance, the script writes the detector error model, exports ONNX,
builds a TensorRT engine, and replays the sampled schedule through all four
configurations. It writes the measurements to ``schedule.txt``, saves results
in ``results.json``, and draws ``color_code_demo.png``. Generated files go to
the current directory; a subsequent run replaces files with the same names.

Reading the Output
------------------

For each distance, Stim samples ``--shots`` shots (default 100,000) into one
schedule, which every decoder and session replays once. Each run prints and
records:

- **Runtime per round**, in microseconds: the elapsed time to the last
  correction, divided by ``shots * d``. This is an amortized runtime, including
  playback scheduling, transport, and decoding, not isolated TensorRT
  inference time or the latency of an individual feedback request.
- **Logical errors**: the number of shots whose correction differs from Stim's
  true observable flip, together with the number of shots sampled.
- **LER per round**: the shot error rate converted using
  :math:`1 - (1 - p_{\mathrm{shot}})^{1/d}`, with an approximate binomial
  uncertainty. For the estimate and uncertainty, the script uses a minimum
  shot error rate of ``0.5 / shots``; the recorded error count is unchanged.

The generated figure uses grey for Chromobius and green for the Ising
predecoder followed by Chromobius. Points with no logical errors are not drawn,
so a short setup check may produce an empty plot. Use larger samples for an
accuracy comparison; 100 shots only checks that the workflow runs.
The figure below
shows example output from an earlier run with 100,000 shots per distance at
:math:`p = 0.001`.

.. image:: ../../../../assets/docs/color_code_demo.png
   :align: center
   :alt: Example logical error rate per round versus runtime per round for Chromobius and the Ising predecoder, for distances 5 to 13, in process and over UDP

Use the figure generated by your run when comparing configurations. Hardware,
software versions, and the selected deployment path affect runtime; the
earlier figure is not a performance guarantee for your environment.

The Example Source
------------------

.. literalinclude:: ../../examples/qec/color_code_demo/color_code_demo.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]

See Also
--------

* :doc:`AI Predecoding </components/qec/ai_predecoding>` -- the two-stage pipeline and other evaluation paths
* :ref:`Color-Code AI Predecoding in the Decoding Server <decoding-server-color-code>` -- deployment with a prepared model and configuration
* :doc:`Playback Emulator </components/qec/playback_emulator>` -- schedules, sessions, and timing records
* :doc:`Realtime Decoding from a Quantum Kernel or FPGA <realtime_decoding_demo>` -- a surface-code example with circuit or FPGA input
* :ref:`C++ <cpp_realtime_decoding_api>` and :ref:`Python <python_realtime_decoding_api>` realtime decoding API

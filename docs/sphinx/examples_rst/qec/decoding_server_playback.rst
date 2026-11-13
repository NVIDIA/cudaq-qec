.. _decoding-server-playback-example:

Realtime Decoding with the Playback Emulator
============================================

The :doc:`Playback Emulator </components/qec/playback_emulator>` sends
measurements and correction requests on a timed schedule, without a QPU or
FPGA. These examples show two ways to use it:

* :ref:`playback-in-process-example` runs a lookup-table decoder in the same
  Python process and checks shots with and without an error.
* :ref:`playback-server-example` sends a synthetic surface-code shot to a
  separate decoding server over UDP, checking configuration and communication.

Neither example measures decoding accuracy over a noise distribution or
establishes a hardware feedback-latency guarantee.

.. _playback-in-process-example:

In-Process Repetition-Code Playback
-----------------------------------

The example below configures a lookup-table decoder for a distance-3
repetition code, then plays back ten fixed shots against it, alternating
between no error and a flipped data qubit. It prints each shot's correction
(``0``, then ``1``) and decoding latency.

Prerequisites
^^^^^^^^^^^^^

Use a realtime-enabled CUDA-Q QEC Python installation. The combined
:ref:`CUDA-Q/QEC Docker container <qec-docker-container>`
includes this interface; the 0.9.0 pip wheels do not.
This example uses the built-in lookup-table decoder and needs no GPU or
running decoding server. Before running it, check that Python playback is
available:

.. code-block:: bash

   python3 -c 'import cudaq_qec as qec; assert hasattr(qec, "playback"), "A realtime-enabled CUDA-Q QEC Python installation is required"'

Run the Example
^^^^^^^^^^^^^^^

Download :download:`playback_emulator.py
<../../examples/qec/python/playback_emulator.py>` into your Python environment,
or use ``docs/sphinx/examples/qec/python/playback_emulator.py`` from a matching
source checkout. Run it from the directory containing the file:

.. code-block:: bash

   python3 playback_emulator.py

The program builds a detector error model and measurement mapping for the
repetition-code memory circuit. It then replays fixed measurements, resetting
the decoder before each shot and sending the final data-qubit readout before
requesting a correction:

.. literalinclude:: ../../examples/qec/python/playback_emulator.py
   :language: python
   :start-after: # [Begin Documentation]
   :end-before: # [End Documentation]

Check the Result
^^^^^^^^^^^^^^^^

Expect ten lines with ``status=0`` and correction bits alternating
``0``, ``1``, repeated five times. These are logical correction bits: the
flipped shots require a correction of ``1``. The reported latency is the
elapsed time from the correction request to its reply; it varies between runs
and does not include the preceding measurement stream.

A message saying the example was skipped is not a successful check; use an
installation that passes the prerequisite command above.

Use the Same Configuration with a Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To play the same schedule against a decoding server instead, save the decoders
with ``decoders.to_yaml_str()`` to ``config.yaml``, start the server, and pass
``udp_endpoints={0: "127.0.0.1:50000"}`` to ``run()`` in place of ``decoders=``:

.. code-block:: bash

   decoding_server --config=config.yaml --transport=udp --port=50000

This alternative requires the matching ``decoding_server`` executable and
access to the saved YAML. The next example supplies the complete server setup
and command-line playback sequence for a surface-code experiment.

.. _playback-server-example:

Playback Against a Decoding Server
----------------------------------

The software playback emulator acts as the client of a standalone decoding
server, replaying measurement and correction requests on a timed schedule.
This exercises the realtime decoding path without running a quantum circuit
or using a QPU or FPGA. You will generate a surface-code memory-experiment
configuration, replay one synthetic all-zero shot, and check the returned
correction. The result verifies configuration and communication; it does not
measure decoding accuracy or establish a hardware feedback-latency guarantee.

Prerequisites
^^^^^^^^^^^^^

Follow :ref:`decoding-server-container-setup` to obtain the prebuilt image,
create ``QEC_RUN_DIR`` and its ``memory-experiment`` subdirectory, and open the
reusable ``qec-decoding`` container. These commands use CPU PyMatching over UDP;
no GPU or host CUDA-Q SDK is required. Start in the container's shared run
directory, ``$QEC_RUN_DIR``.

Generate the Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the container's shared run directory, enter the ``memory-experiment``
subdirectory created on the host. Generate a distance-3, Z-basis experiment
with 26 extraction rounds and PyMatching:

.. code-block:: bash

   cd memory-experiment
   memory-experiment-generator \
     --distance=3 --rounds=26 --basis=Z --error-rate=0.001 \
     --decoder=pymatching --output=memory.yaml

Wait for ``Generated server configuration: ...`` and exit status zero before
using the file. The output directory must exist; an existing file or symlink
is never overwritten. Choose another filename when changing an experiment.

For the supported geometries, decoder profiles, and argument meanings, see
:ref:`decoding-server-memory-configuration`.

Run Playback
^^^^^^^^^^^^

This example uses the software playback emulator as the client, so no QPU is
required. Real QPU and CUDA-Q clients send measurements and correction requests
directly; they do not need a playback schedule. See
:ref:`decoding-server-measurements` for that workflow.
For schedule syntax, CLI and Python entry points, and source and session
configuration, see :doc:`Playback Emulator </components/qec/playback_emulator>`.

In this container terminal, start the CPU PyMatching server using the generated
file. Leave it running in the foreground:

.. code-block:: bash

   decoding_server \
     --config=memory.yaml --transport=udp --port=50000 --timeout=3600

Wait for ``QEC_DECODING_SERVER_READY`` before sending measurements. If port
50000 is unavailable, choose another free port and use it in both commands.

The schedule below is read by the **playback emulator**, not the decoding
server. On the **host**, save it as
``$QEC_RUN_DIR/memory-experiment/memory_schedule.txt``. That directory is
shared with the container.
The schedule resets the decoder, sends 26 eight-bit stabilizer rounds and a
nine-bit final readout, then requests the correction. These synthetic, all-zero
measurements match the distance-3 configuration above. They test the
configuration-to-server handoff, not accuracy or performance:

.. literalinclude:: ../../examples/qec/decoding_server/memory_schedule.txt
   :language: text
   :caption: memory_schedule.txt

Open a second container shell as described in
:ref:`decoding-server-container-shell`. It starts in the shared run directory.
Run:

.. code-block:: bash

   cd memory-experiment
   playback-emulator \
     --config=memory.yaml --schedule=memory_schedule.txt \
     --backend=udp --udp-endpoint=0:127.0.0.1:50000 --tick=1ms \
     --out=memory-result.csv

Check the Result
^^^^^^^^^^^^^^^^

In ``get_corrections 0 return_size=1``, ``0`` is the expected correction bit;
``return_size=1`` requests one bit. Pass ``--config`` to playback too: its
``enqueue_data`` operation uses the YAML's FINAL mapping. Check that the
correction row has ``status=0``, ``read_completed=1`` and
``correction_bits=0``, matching the expected zero correction.
Stop the server with Ctrl+C in its original terminal. The shell remains
available to generate another configuration and start the server with a
different ``--config`` path. Use new
filenames for each experiment's YAML and results.

The YAML, schedule and CSV remain on the host under
``$QEC_RUN_DIR/memory-experiment``. These are functional checks, not an
end-to-end latency or throughput benchmark.

Next Steps
----------

* Run a simulated quantum circuit against the server with
  :doc:`Realtime Decoding from a Quantum Kernel or FPGA <realtime_decoding_demo>`.
* See :doc:`Playback Emulator </components/qec/playback_emulator>` for schedule
  syntax, measurement sources, sessions, run options, and output fields.
* See :doc:`Decoding Server </components/qec/decoding_server>` for deployment,
  configuration, and third-party client integration.

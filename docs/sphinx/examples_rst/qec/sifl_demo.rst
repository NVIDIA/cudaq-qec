Streaming Interleaved Feed-forward Latency (SIFL)
=================================================

This example runs a Streaming Interleaved Feed-forward Latency (SIFL) experiment: 
a surface-code memory experiment streams one stabilizer round every
:math:`T` µs to two decoders that alternate shots, and each shot keeps
streaming rounds until the previous shot's correction lands. A slow decode
therefore makes the next shot longer, and so slower to decode. A decoder must
therefore be able to keep up with the cadence of stabilizer extraction, or else 
the decoder latency will grow without bound.

The syndromes come from Stim and are streamed by the playback emulator. 
``per_round_decoder.cpp`` is a small decoder plugin that hands each shot to 
a sub-decoder built from the full detector error model of a circuit with 
exactly that many rounds.

Running
-------

The demo needs a CUDA-Q QEC install built with CUDA-Q Realtime. It provides the 
playback emulator and the headers the plugin is built against.

From ``examples/qec/sifl_demo``, with that install's Python environment set up:

.. code-block:: bash

   ./run_sifl_demo.sh

The script builds the plugin, runs the demo, and removes the plugin on exit.
``--help`` lists the options. For each cadence it prints the decode time per
round (the total time from dispatching each shot's data readout to its
correction, divided by the total rounds streamed) and the rounds streamed per
shot:

.. code-block:: text

   Decode time per round and rounds streamed per shot (stream cap 50):
     T =    2 us  decode time per round  7.17 us  [10, 50, 50, 50, 50, 50, 50, ...]
     T =   10 us  decode time per round  9.87 us  [10, 16, 13, 12, 18, 12, 16, ...]
     T =   50 us  decode time per round 12.84 us  [10, 3, 2, 2, 2, 1, 2, ...]

Where the transition falls depends on how fast the decoder runs.

The example source
------------------

.. literalinclude:: ../../examples/qec/sifl_demo/sifl_demo.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]

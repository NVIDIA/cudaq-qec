Feedback Latency and Decoder Backlog
====================================

A realtime decoder must return corrections in time for the computation that
needs them. In a feedback-dependent workflow, error-correction rounds may
continue while a later operation waits for a correction. A longer wait then
produces more measurements to decode. This connects decoder response time to
the amount of work the pipeline receives next.

The Streaming Interleaved Feed-forward Latency (SIFL) experiment models this
dependency in software. Two decoder instances alternate shots: while one
finishes decoding, the other receives new measurement rounds. Varying the
interval between rounds lets you study whether the resulting shot lengths
settle down or grow toward the experiment's limit. See
:doc:`Measuring Streaming Interleaved Feed-forward Latency (SIFL)
</examples_rst/qec/sifl_demo>` for the runnable experiment.

How the Feedback Dependency Works
---------------------------------

The first shot has a fixed number of stabilizer rounds. After its final data
readout is sent for decoding, the next shot starts collecting rounds on the
other decoder instance. That shot ends when the previous correction returns,
or when it reaches a configured round cap. Its final readout then triggers
decoding, and the two instances exchange roles again.

If decoding takes longer, the following shot can accumulate more rounds.
Decoding that larger shot can take longer in turn. Increasing the time between
rounds gives the decoder more time to finish before additional measurements
arrive. The experiment exposes this relationship between the input cadence
and the work created during the wait for feedback.

The :doc:`Playback Emulator <playback_emulator>` implements the dependency
with completion signals: a correction request raises a signal, and the next
shot's stream waits for that signal before ending. The SIFL example uses
in-process decoders and Stim-generated measurements, so no decoding server,
QPU, or FPGA is required.

What the Experiment Measures
----------------------------

The example reports the number of stabilizer rounds in each shot and the
time from dispatching each final data readout to receiving its correction,
normalized by the total rounds streamed. Together, these show how feedback
waiting and shot length change as the round cadence changes.

The example's ``per_round_decoder`` plugin buffers a shot's measurements and
decodes them when the final readout arrives. It selects a sub-decoder built
from the detector error model for exactly that number of rounds; PyMatching
is the default. Although measurements arrive through the streaming interface,
this plugin does not perform incremental decoding as each round arrives.

Use the experiment to study this feedback mechanism and compare configurations
under the same conditions. It does not compare corrections against expected
logical values or estimate a logical error rate. For a decoder's accuracy,
use :doc:`Experiments and Noise Modeling <numerical_experiments>` or an example
that checks corrections against known observables.

Interpreting Growth and Limits
------------------------------

Repeatedly reaching the round cap means that the wait for feedback is often
long enough to fill the permitted shot. A stream terminated by the cap stops
because of the experiment's limit rather than the normal completion signal. A finite,
capped run cannot demonstrate unbounded growth, and a short sequence of small
shots cannot establish a long-term latency guarantee.

Consider the shot-length sequence alongside the reported average. Host
scheduling, measurement generation, and decoding all affect the experiment;
the requested round cadence does not guarantee that software dispatch meets
every deadline. For individual dispatch and reply timestamps, see the
:ref:`Playback Emulator output format <playback-output-format>`. For studies of decoder
latency under specified workloads, see :doc:`Performance Studies
</performance/index>`.

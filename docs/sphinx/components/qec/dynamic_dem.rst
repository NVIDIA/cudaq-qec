.. _dynamic_dem:
.. _dynamic_dem_construction:

Dynamic Detector Error Models
=============================

A dynamic detector error model (DEM) describes a quantum error correction
experiment in reusable pieces, one measurement round at a time. Each piece
records the faults that can occur, the detectors and logical observables they
affect, and how it connects to neighboring rounds. CUDA-Q QEC represents this
description as ``dem_chunks`` in YAML, ``DemChunksSpec`` in Python, and
``dem_chunks_spec`` in C++.

This is useful when an experiment repeats the same error-correction cycle
many times. You can describe its first round, repeating rounds, and final
readout once, then reuse that description for different shot lengths or for
windows of a measurement stream. The boundaries remain part of the DEM, so
the decoder can account for faults whose effects cross from one round to the
next.

Why Use a Dynamic DEM?
----------------------

A DEM for a complete circuit fixes the number of rounds in that circuit.
That is convenient for decoding complete shots. For a long or variable-length
experiment, however, most rounds may repeat the same structure while the
beginning and end need different treatment. A final data-qubit readout, for
example, measures different quantities from a round of stabilizer checks.

A dynamic DEM makes that structure explicit. The repeating phase is specified
once, and its connections describe where it can repeat and how the shot ends.
Task graphs can use these pieces to prepare complete-shot decoders or the
smaller detector error models needed by windowed decoders. You can therefore
use the same description when testing a fixed number of rounds offline and
when configuring a pipeline that processes rounds as they arrive.

The description also keeps the fault information together with the
measurement and logical-readout mappings. Those mappings let a task graph
convert raw measurements into detection events and, when a logical output is
configured, return corrected logical measurements. Fault probabilities are
supplied in the phase specifications; the dynamic part is how the phases are
assembled for the shot or decoding window.

Phases and Connections
----------------------

A *phase* describes one kind of round. Its ``num_faults`` and ``error_rates``
identify the possible faults and their probabilities. Its sparse matrices
describe which detectors and logical observables each fault flips. An
occurrence of that phase contributes one DEM *chunk* to the shot.

For a repetition-code memory experiment, three phases are enough:

* ``init`` describes the first round of stabilizer checks.
* ``bulk`` describes a round that can repeat.
* ``final`` describes the data-qubit readout that closes the shot.

``connections`` specifies their order: ``init`` leads to ``bulk``, ``bulk``
can follow itself, and ``bulk`` leads to ``final``. A six-round shot expands
to:

.. code-block:: text

   init -> bulk -> bulk -> bulk -> bulk -> final

The initial and final rounds each occur once; the repeating phase supplies
the remaining rounds. CUDA-Q QEC 0.9.0 supports a linear phase sequence with
at most one repeating phase. Phase connections describe the experiment's
rounds; the separate task graph describes how decoders process their data.

Detectors Across Round Boundaries
---------------------------------

Some faults affect detectors in two adjacent rounds. A measurement error is
a useful example: if a detector compares consecutive check measurements, one
incorrect measurement changes both the current comparison and the next one.
Splitting the experiment into rounds must preserve both effects.

A *seam* records the detector rows at a chunk boundary. In the example's
``seam_specs``, ``prev_round`` lists the current round's detectors and the
faults in this phase that flip them. ``next_round`` lists the next round's
detectors affected by those same faults. The connection
``seam: {from: next_round, to: prev_round}`` joins the matching rows of
adjacent chunks, combining their contributions to each detector.

Use separate ``seam_specs`` when the two boundaries have different fault
effects. If every seam has identical rows, a phase-level ``H_sparse`` is a
shorthand for that shared matrix. The last phase needs no outgoing
``next_round`` rows. Connected seams must have matching widths.

.. _dynamic_dem_measurements:

Measurements and Logical Readout
--------------------------------

A detector error model describes how faults affect detectors and observables.
To connect that description to a stream of raw measurements, a dynamic DEM
can also carry the measurement mappings. These fields serve different roles:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - What It Describes
   * - ``H_sparse``
     - Which faults flip each detector. Each row lists fault indices.
   * - ``O_sparse``
     - Which faults flip each logical observable. Each row lists fault indices.
   * - ``D_sparse``
     - Which measurements combine by XOR to form each detector. Each row lists
       measurement indices, counting the previous round's measurements first
       and then the current round's.
   * - ``m2o_sparse``
     - Which measurements in this phase's own round contribute to each raw
       logical observable. Contributions from all readout rounds combine by XOR.

The sparse row encoding ends each row with ``-1``. For example,
``[0, 4, -1, 1, 5, -1]`` contains two rows. In ``D_sparse``, these rows
compare measurements 0 and 4, then 1 and 5; in ``H_sparse``, the same numbers
would identify faults instead.

Without explicit ``D_sparse`` rows, CUDA-Q QEC uses the memory-experiment
convention: one measurement per detector per round, compared with the
preceding round, or with zero in the first round. Supply ``D_sparse`` for
each phase when the experiment needs a different mapping. Set
``measurements_per_round`` as the default measurement count and override it
with a phase's ``num_measurements`` where necessary. This allows, for example,
four check measurements in each round followed by five data-qubit measurements
in the final round of a distance-5 repetition-code experiment. Each detector's
mapping may refer only to its own round and the immediately preceding round.

``O_sparse`` supplies the observable-flip information used to calculate a
correction. ``m2o_sparse`` supplies the raw readout to which that correction
can be applied. A non-repeating phase can read observables, including during
the circuit; a repeating phase cannot carry ``m2o_sparse``. No later phase
may contain a fault that flips an observable after its last readout.
``observable_names`` can give the observables descriptive names.

To obtain corrected logical measurements, configure the task graph's logical
output as well as the readout mapping. See
:ref:`Logical measurements <task_graph_logical_measurements>` for the output
configuration and when those results become available.

Constructing a Dynamic DEM
--------------------------

For a custom experiment, begin with its fault effects and measurement
relations, then organize them into phases:

1. Identify the distinct rounds: initialization, any repeating cycle, and
   readout or other rounds that occur once.
2. For each phase, specify its faults, probabilities, detector-boundary rows,
   and observable-flip rows.
3. Connect the phases in order and identify the seams to join. Add a
   self-connection to the repeating phase, if there is one.
4. Add the measurement counts and ``D_sparse`` rows when the default
   measurement convention does not apply. Add ``m2o_sparse`` where the
   experiment reads logical observables.
5. Choose the shot length or streaming configuration, then pass the
   specification to a task graph and select the decoder it will use.

You can write these fields in a YAML ``dem_chunks`` block or construct the
equivalent Python or C++ objects. The
:doc:`Decoding with a Dynamic DEM </examples_rst/qec/dyn_dem>` shows all three
representations and decodes a shot with a known injected error.

For surface-code memory experiments, ``memory-experiment-generator`` prepares
the dynamic DEM and a parallel-window task graph together. It derives reusable
phases from bounded representative circuits, so preparing a longer requested
experiment does not require characterizing a circuit of that full length.
See :ref:`Memory Experiment Configuration Generator <memory_experiment_generator>`
for the supported parameters and command. In C++,
:cpp:func:`cudaq::qec::memory_dem_chunks_from_decoder_inputs` also provides
memory-circuit phase extraction; its API describes the input requirements.

Choosing a Decoding Path
------------------------

For complete shots with a repeating phase, set ``num_rounds`` to the total
number of rounds, including initialization and final readout. The
``DecodingTaskGraph.from_spec`` factory then builds a decoder for that fixed
length. If there is no repeating phase, the connections determine the length
of the phase sequence.

For streaming, the dynamic DEM can omit ``num_rounds``. Configure the task
graph's window sizes and ending behavior for the experiment. In particular,
:doc:`Parallel Window </examples_rst/qec/task_graph_parallel_window>`
uses the repeating phase to handle variable-length shots and the ``FINAL``
marker to close each shot. Shot lengths must accommodate the non-repeating
phases and the graph's window requirements. Support for different measurement
counts in different phases also depends on the chosen graph and decoder;
the DEM describes those counts, while the decoding path determines how to
process them.

For a complete circuit that does not need reusable phases or windowed
decoding, the :doc:`detector error model <detector_error_models>` and Stim DEM
interfaces remain useful starting points.

See Also
--------

* :doc:`Decoding with a Dynamic DEM </examples_rst/qec/dyn_dem>`
* :doc:`Decoding task graphs <decoding_task_graph>`
* :ref:`Dynamic DEM Python API <dyn_dem_python_api>` and
  :ref:`C++ API <dyn_dem_cpp_api>`

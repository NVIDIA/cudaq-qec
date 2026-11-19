.. _dyn_dem_example:

Decoding with a Dynamic DEM
----------------------------

.. raw:: html

   <span id="dynamic-dem"></span>

This example constructs a dynamic detector error model (DEM) for a distance-5
repetition-code memory experiment, then uses PyMatching to decode a complete
six-round shot of synthetic measurements containing a known data-qubit error.
It shows the same phase description in YAML, Python,
and C++. For phases, seams, measurement mappings, and when to use them, start
with :doc:`Dynamic Detector Error Models </components/qec/dynamic_dem>`.

Use a CUDA-Q QEC installation with the PyMatching decoder available. The C++
example also requires the ``nvq++`` compiler and CUDA-Q QEC development
libraries; see the :doc:`Installation Guide </quickstart/installation>`.

.. _a-model:

Describe the Repetition-Code Experiment
+++++++++++++++++++++++++++++++++++++++

The experiment has five data qubits and four stabilizer checks. Its DEM
includes data-qubit flips and measurement errors, with a probability of 0.02
for each fault. The final round reads the five data qubits:

.. literalinclude:: ../../examples/qec/yaml/task_graph_parallel_window.yaml
   :language: yaml
   :start-at: dem_chunks:
   :end-before: task_graph:
   :dedent: 4

* ``init`` and ``bulk`` each contain nine faults: five data-qubit flips and
  four check-measurement errors. ``final`` contains five data-qubit flips.
* ``init`` starts the shot, ``bulk`` repeats, and ``final`` ends it. The YAML
  leaves ``num_rounds`` unset for streaming with the
  :doc:`Parallel Window <task_graph_parallel_window>` example.
* The Python and C++ programs below set ``num_rounds = 6``: one ``init``
  round, four ``bulk`` rounds, and one ``final`` round.

.. _detectors-and-seams:

Check the Round Boundaries
++++++++++++++++++++++++++

Faults 5 through 8 are check-measurement errors. They affect detectors in
both their own round and the next, so they appear in both ``prev_round``
and ``next_round`` seam rows. Faults 0 through 4 are data-qubit flips and
appear only in the current round's rows in this DEM. The final phase has no
outgoing ``next_round`` seam.

See :doc:`Dynamic Detector Error Models </components/qec/dynamic_dem>` for
how seams join these effects across round boundaries.

.. _measurements-and-readout:

Follow the Measurements
+++++++++++++++++++++++

The DEM's ``D_sparse`` rows form each detector from the measurements:

* ``init`` compares each check measurement with its expected value of zero.
* ``bulk`` compares each check with the same check in the previous round.
* ``final`` compares each previous check with the parity of its two adjacent
  data-qubit readouts. Its five measurements replace the four check
  measurements of a bulk round.

The final phase's ``m2o_sparse: [0, -1]`` reads data qubit 0 as the raw
logical observable. Its ``O_sparse: [0, -1]`` identifies fault 0 as a flip
of that observable. The matching indices refer to different things:
``m2o_sparse`` indexes measurements, while ``O_sparse`` indexes faults.
See :ref:`Measurements and Logical Readout <dynamic_dem_measurements>` for
the field definitions.

.. _building-a-model-in-code:

Build the DEM and Decode a Shot
+++++++++++++++++++++++++++++++

These programs construct the same detector error model and use PyMatching to
decode a fixed-length shot. The shot contains a flip of data qubit 0 in round
2, with rounds counted from zero:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/dyn_dem.py
      :language: python
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/dyn_dem.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec -lcudaq-qec-decoders dyn_dem.cpp
      ./a.out

.. code-block:: text

   corrections: [1]

The result says that the logical observable needs a flip, as expected from
the injected error. ``run()`` returns the correction; the
:doc:`Parallel Window <task_graph_parallel_window>` example also configures
an output for the corrected logical measurement.

In Python, assign whole lists, such as ``spec.seam_specs = [...]``: appending
to a field changes a copy.

See Also
++++++++

- :doc:`Dynamic Detector Error Models </components/qec/dynamic_dem>`: concepts
  and construction choices.
- :ref:`dyn_dem_python_api` and :ref:`dyn_dem_cpp_api`: the DEM specification types.
- :doc:`task_graph_parallel_window`: decoding streams of any length.

.. _dyn_dem_example:

Dynamic DEM
-----------

A ``dem_chunks`` model describes a detector error model one round at a time.
Each kind of round is a *phase*, with its own faults; connections say which
phase follows which. A phase that follows itself repeats, so the same model
describes shots of any length. Task graphs and streaming decoders read their
models in this form (see :doc:`decoding_task_graph`).

For surface-code memory experiments, ``memory-experiment-generator`` writes
the model for you; see :doc:`/components/qec/realtime_decoding`.

A model
+++++++

This model is a distance-5 repetition-code memory experiment with
measurement errors:

.. literalinclude:: ../../examples/qec/yaml/task_graph_parallel_window.yaml
   :language: yaml
   :start-at: dem_chunks:
   :end-before: task_graph:
   :dedent: 4

* ``phases`` lists each kind of round. ``num_faults`` and ``error_rates``
  give its faults and their probabilities, and ``O_sparse`` lists, for each
  logical observable, the faults that flip it.
* ``connections`` order the phases. ``bulk`` follows itself, so it repeats;
  ``final`` ends the shot.
* ``num_rounds`` sets the shot length for decoders that decode complete
  shots. Leave it out, as here, to stream shots of any length.

``H_sparse`` and ``O_sparse`` hold one row per detector or observable: the
indices of the faults that flip it, ended by ``-1``.

Detectors and seams
+++++++++++++++++++

A fault can flip detectors of its own round and of the next one, as a
measurement error does. ``seam_specs`` say which:

* ``prev_round`` rows are this round's detectors. Each lists the phase's
  faults that flip it.
* ``next_round`` rows are the next round's detectors, listing the faults of
  this phase that flip them too.

``seam: {from: next_round, to: prev_round}`` joins each round's
``next_round`` rows to the next round's ``prev_round`` rows. The last phase
needs no ``next_round`` rows.

When every seam has the same rows, one ``H_sparse`` per phase replaces
``seam_specs``.

Measurements and readout
++++++++++++++++++++++++

By default, each round measures one value per detector, and a detector
compares its measurement with the round before; in the first round, with
zero. These optional fields change that:

* ``D_sparse`` on every phase gives each detector's measurements explicitly,
  from this round and the one before. Set ``measurements_per_round``, and
  ``num_measurements`` on a phase whose round differs, such as a final
  data-qubit readout.
* ``m2o_sparse`` on the last phase lists, for each observable, the
  measurements of that round whose parity is its raw value.
* ``observable_names`` names the observables.

Building a model in code
++++++++++++++++++++++++

The same model, decoding a fixed-length shot:

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

In Python, assign whole lists, such as ``spec.seam_specs = [...]``: appending
to a field changes a copy.

See also
++++++++

- :ref:`dyn_dem_python_api` and :ref:`dyn_dem_cpp_api`: the model types.
- :doc:`task_graph_parallel_window`: decoding streams of any length.

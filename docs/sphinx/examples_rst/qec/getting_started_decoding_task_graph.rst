.. _decoding_task_graph_example:

Getting Started with a Decoding Task Graph
==========================================

This example builds task graphs for a distance-5 repetition-code DEM using
the ``single_error_lut`` decoder. It demonstrates complete-shot and batch
decoding, streaming, and combining two graphs. The input measurements are
synthetic zeros, so the examples demonstrate the interfaces and return zero
logical corrections.

Start with :doc:`Decoding Task Graphs </components/qec/decoding_task_graph>`
for the pipeline concepts. Node options, execution rules, and result fields
are covered by its :ref:`Reference section <task_graph_reference>`.

Use a CUDA-Q QEC installation with the ``single_error_lut`` decoder available.
The C++ example also requires ``nvq++`` and the CUDA-Q QEC headers and
libraries. Download the full :download:`Python program
<../../examples/qec/python/decoding_task_graph.py>` or :download:`C++ program
<../../examples/qec/cpp/decoding_task_graph.cpp>`; both include the
``rep5_spec`` helper used below. See the
:doc:`Installation Guide </quickstart/installation>` for the environment.

Run the Python program with:

.. code-block:: bash

   python3 decoding_task_graph.py

Construct the Graph
-------------------

The simplest graph is one sliding-window node around an inner decoder. Its
detector error model uses ``dem_chunks`` to describe the experiment's rounds;
see :doc:`Dynamic Detector Error Models </components/qec/dynamic_dem>`.

.. code-block:: yaml

   decoders:
     - id: 0
       task_graph:
         nodes:
           - id: sw
             type: sliding_window
             window_size: 3                 # rounds per decode
             step_size: 1                   # rounds to advance per decode
             inner: {type: single_error_lut}
       dem_chunks:
         seam: {from: next_round, to: prev_round}
         connections:
           - {from: init, to: bulk}
           - {from: bulk, to: bulk}   # bulk repeats
           - {from: bulk, to: final}
         phases:
           - name: init
             spec:
               num_faults: 9
               H_sparse: [0,1,5,-1, 1,2,6,-1, 2,3,7,-1, 3,4,8,-1]
               O_sparse: [0,-1]
               error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
           - name: bulk
             spec:
               num_faults: 9
               H_sparse: [0,1,5,-1, 1,2,6,-1, 2,3,7,-1, 3,4,8,-1]
               O_sparse: [0,-1]
               error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
           - name: final
             spec:
               num_faults: 5
               H_sparse: [0,1,-1, 1,2,-1, 2,3,-1, 3,4,-1]
               O_sparse: [0,-1]
               error_rates: [0.02,0.02,0.02,0.02,0.02]

* The graph streams: it first decodes once it holds three rounds, then on
  every round after. ``window_size`` must be at least the fewest rounds the
  phases allow, three here.
* For shots of a fixed length, set ``dem_chunks.num_rounds`` and write
  ``type: single_window``: the graph decodes each complete shot once.
* ``inner``'s other keys configure the inner decoder.

The same graph built in code. ``from_spec`` picks the window itself: the
fewest rounds the phases allow, advancing one round at a time, or the whole
shot when ``num_rounds`` is set:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/decoding_task_graph.py
      :language: python
      :start-after: [Begin from_spec]
      :end-before: [End from_spec]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/decoding_task_graph.cpp
      :language: cpp
      :start-after: [Begin from_spec]
      :end-before: [End from_spec]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec -lcudaq-qec-decoders decoding_task_graph.cpp
      ./a.out

Decode Complete Shots
---------------------

:meth:`~cudaq_qec.DecodingTaskGraph.run` decodes one shot:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/decoding_task_graph.py
      :language: python
      :start-after: [Begin one-shot]
      :end-before: [End one-shot]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/decoding_task_graph.cpp
      :language: cpp
      :start-after: [Begin one-shot]
      :end-before: [End one-shot]

:meth:`~cudaq_qec.DecodingTaskGraph.run_batch` decodes many:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/decoding_task_graph.py
      :language: python
      :start-after: [Begin batch]
      :end-before: [End batch]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/decoding_task_graph.cpp
      :language: cpp
      :start-after: [Begin batch]
      :end-before: [End batch]

A graph with more than one output returns one result per output from
``run_all``, in ``output_names()`` order.

Stream Measurement Rounds
-------------------------

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/decoding_task_graph.py
      :language: python
      :start-after: [Begin streaming]
      :end-before: [End streaming]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/decoding_task_graph.cpp
      :language: cpp
      :start-after: [Begin streaming]
      :end-before: [End streaming]

* Send one round per call: ``get_num_msyn_per_round()`` measurements. Larger
  chunks also work; the graph splits them.
* Tag the shot's last round ``FINAL``, as in
  ``enqueue_syndrome(syndrome, round_type.FINAL)``. It decodes whatever
  windows still hold, so the end of the shot is corrected. Untagged rounds
  never end a shot.
* Corrections accumulate across the shot. After ``FINAL``, the graph drops
  further rounds until ``clear_corrections()``, which consumes the
  corrections and starts the next shot. ``reset_decoder()`` also starts a
  new shot, discarding rounds not yet decoded.

See :ref:`Reading Streaming Results <task_graph_streaming_results>` for
status and coverage, and :ref:`Logical Measurements
<task_graph_logical_measurements>` for graphs with a logical output.

Connect Several Nodes
---------------------

``task_graph.nodes`` lists the nodes and ``task_graph.edges`` wires them. A
node names its kind in ``type``, its inner decoder in ``inner``, and the kind's
parameters inline. The predecoder pipeline shown in
:doc:`Decoding Task Graphs </components/qec/decoding_task_graph>` uses the same
detector error model as the construction example:

.. literalinclude:: ../../examples/qec/yaml/task_graph_predecoder.yaml
   :language: yaml
   :start-at: task_graph:
   :dedent: 4

* ``measurements`` is the incoming shot and ``output`` the graph's result.
* An edge endpoint is a node id, with ``.port`` for a node that has named ports,
  such as ``xor``'s ``a`` and ``b`` and ``predecoder``'s ``pre_L`` (its
  corrections) and ``residual``.
* A node fed by another decodes that node's output, so ``pre`` and
  ``global`` decode the same rounds: their ``window_size`` must match.
* A single node needs no edges: it is wired
  ``measurements → node → output``.
* List nodes in the order data flows. A node without its own ``dem_chunks``
  uses the decoder's.

:meth:`~cudaq_qec.DecodingTaskGraph.combine` joins two single-node graphs
with an ``xor``, for two decoders whose corrections add:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/decoding_task_graph.py
      :language: python
      :start-after: [Begin combine]
      :end-before: [End combine]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/decoding_task_graph.cpp
      :language: cpp
      :start-after: [Begin combine]
      :end-before: [End combine]

For any other topology, use
:meth:`~cudaq_qec.DecodingTaskGraph.from_topology` with a list of
:class:`~cudaq_qec.TaskGraphNodeConfig` nodes and their edges. In code, a
node's ``kind``, ``decoder_type``, ``params`` and ``node_params`` hold what
YAML writes as ``type``, ``inner.type``, ``inner``'s other keys and the
inline parameters.

The ``combine`` snippets use identical decoders to demonstrate XOR
composition: identical corrections cancel. For a pipeline that selects one
of several competing answers, see :doc:`task_graph_ensemble`.

Expected Results
----------------

The complete-shot examples report convergence and a zero correction. The
batch examples process eight shots. The streaming examples send ten rounds,
report a completed shot, and return zero corrections. The combined graph also
returns zero corrections for these inputs.

Next Steps
----------

* :ref:`Reference <task_graph_reference>`: node kinds, guards,
  asynchronous execution, and result semantics.
* :doc:`task_graph_parallel_window`: decode a stream in parallel windows.
* :doc:`task_graph_ensemble`: choose an answer from several member graphs.

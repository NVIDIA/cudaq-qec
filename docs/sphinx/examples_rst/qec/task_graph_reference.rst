.. _task_graph_reference:

Task Graph Reference
--------------------

Details of task-graph execution, for tuning and for writing new node kinds.
Start with :doc:`decoding_task_graph`.

.. _task_graph_async_options:

Asynchronous streaming options
++++++++++++++++++++++++++++++

Set these in the ``task_graph`` block, or on
:class:`~cudaq_qec.TaskGraphOptions` in code. The other options apply only
with ``async_streaming``, and each graph gets threads of its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``async_streaming``
     - ``true``: ``enqueue_syndrome`` queues the round and returns at once,
       and the graph's threads decode it. ``false`` (default): the call
       decodes on the caller's thread and returns when done.
   * - ``async_num_workers``
     - Worker threads that decode. Default: enough for everything that may
       decode at once (independent branches and ``max_in_flight`` copies), up
       to the machine's hardware threads.
   * - ``async_worker_cpus``
     - CPUs to pin the workers to, one worker per CPU (Linux). With fewer
       CPUs than workers, the pool shrinks to match. Default: none.
   * - ``async_coordinator_cpu``
     - CPU to pin the thread that schedules decodes to (Linux). Default
       ``-1``: keep the affinity of the thread that builds the graph. It must
       not be in ``async_worker_cpus``.
   * - ``async_spin_pause_iters``
     - How long an idle coordinator polls before it sleeps. Default ``0``:
       no polling.
   * - ``async_queue_capacity``
     - Records that may be queued or decoding before ``enqueue_syndrome``
       waits. Default ``32`` in code, ``8`` in YAML.

* Each ``enqueue_syndrome`` call is queued as a record; a windowed graph may
  split a long call into several. Records overlap: a later record
  runs on nodes that have finished earlier ones, up to 16 or
  ``async_queue_capacity`` at once, whichever is smaller, but each node still
  sees records in order.
* A windowed graph that falls behind takes the rounds queued for it together,
  to catch up. A ``FINAL`` round returns only after the queue drains, so a
  deeper queue helps a graph that falls behind but lengthens its tail.
* Keep producer threads, workers and the coordinator on separate cores for
  low latency. Configuring decoders warns when two graphs pin the same CPU or
  together need more threads than the machine has.

Fire pools
++++++++++

``max_in_flight`` on a node builds that many copies of it, so several records
can decode on it at once. Results equal one-at-a-time decoding: each record
uses one copy, and corrections fold in record order.

* Only kinds that keep no state between decodes take ``max_in_flight`` above
  one. ``sliding_window``, ``single_window``, ``streaming``, ``predecoder``,
  ``ingest`` and ``logical_readout`` do not.
* While streaming, a fire pool cannot take raw measurements, so it must be
  fed by another node, as ``decode_window`` is fed by ``ingest``.
* Each copy builds its own inner decoder, so the node needs a decoder type,
  not an already-built decoder.
* Copies decode at once only with ``async_streaming``. Each copy beyond the
  first adds a worker to the default pool.

Streaming results in detail
+++++++++++++++++++++++++++

``streaming_result()`` returns the values of every output with their
progress:

* ``status`` and ``closure``, per output and for the graph.
* ``coverage_begin_round`` and ``coverage_end_round`` per output: the rounds
  its latest update committed.
* ``generation``: a counter that advances each time an output publishes. The
  graph's generation advances once every output but the logical one has.
  ``clear_corrections()``
  keeps it, so use ``status``, not ``generation``, to decide whether a value
  can be read.
* ``result_revision``: advances on any change of value or status.
* ``shot_epoch``: advances on ``reset_decoder()``.
* ``error``: why the graph failed, when ``status`` is ``failed``. Reset to
  continue.

With ``async_streaming``, ``streaming_result()`` first waits for every
queued round, then returns a snapshot. Without it, the C++ result is a live
reference owned by the graph; copy it to keep it across later calls. Python
always returns a copy.

Staged outputs
++++++++++++++

Nodes that feed one output normally decode on the same schedule, and the
output's coverage follows that schedule. Window nodes such as
``decode_window`` instead finish arbitrary round intervals in any order. An
output fed only by such nodes is *staged*: the graph holds each result until
every earlier round is done, folds them in round order, and reports coverage
from the frontier at the last ``clear_corrections()`` (round 0 in a new shot)
to the current frontier. An output that mixes both kinds of node cannot
stream; wire them to separate outputs. Nodes that commit the same rounds,
such as two ``parallel_window`` patterns, cannot share an output either;
building the graph refuses them.

Edge values
+++++++++++

Each port declares the type of value it produces or accepts, and building a
graph rejects an edge whose types do not match. The types you wire most are
``measurement_results``, ``detection_events`` and ``logical_outcome``; others
carry values between built-in nodes, such as the corrections one window sends
another. ``registered_edge_value_types()`` (C++) lists an installation's
types.

A node fed by another node decodes that node's output. For example, a
predecoder passes its residual detectors to a global decoder, and an ``xor``
adds both answers:

.. code-block:: yaml

   edges:
     - {from: measurements, to: pre}
     - {from: pre.pre_L, to: combine.a}
     - {from: pre.residual, to: global}
     - {from: global, to: combine.b}
     - {from: combine, to: output}

.. _task_graph_logical_output:

Logical outputs
+++++++++++++++

A ``logical_readout`` node reads the raw logical value through the model's
``m2o_sparse``, and an ``xor`` applies the corrections to it. The output that
the node reaches, whatever its name, is the graph's logical output. On a
model whose rounds all have the same width:

.. code-block:: yaml

   nodes:
     - {id: sw, type: sliding_window, inner: {type: pymatching}}
     - {id: R, type: logical_readout}
     - {id: L, type: xor}
   edges:
     - {from: measurements, to: sw}
     - {from: measurements, to: R}
     - {from: sw, to: output.corrections}
     - {from: sw, to: L.a}
     - {from: R, to: L.b}
     - {from: L, to: output.logical}

* A graph has at most one logical output, and needs a corrections output
  beside it.
* One ``logical_readout`` node reaches the logical output, and it reaches no
  other output.
* The node joins any stream, whatever the other nodes' schedules. Here the
  corrections publish on the window's schedule, so the logical measurements
  are released at ``FINAL``; see :ref:`task_graph_logical_measurements`.

Native streaming nodes
++++++++++++++++++++++

A ``streaming`` node passes each ``enqueue_syndrome`` call to its inner
decoder's own ``enqueue_syndrome``, for decoders that process rounds
incrementally. Its complete-shot path decodes the whole shot at once.

Rounds may differ in width, such as a narrower readout round, if the inner
decoder implements ``streaming_correction_provider`` and so ends its own
shots. The node then reports the rounds each update covers, its output is
staged, and a shot may end at any ``FINAL`` round. Two such nodes cannot
share an output.

.. code-block:: yaml

   - id: native
     type: streaming
     num_syndromes_per_round: 24
     inner: {type: <streaming-decoder>}

Arbitrary-length sliding windows
++++++++++++++++++++++++++++++++

``arbitrary_length: true`` on a ``sliding_window`` node lets a shot end at any
round, not only on a window boundary. ``FINAL`` decodes the actual last
rounds, without padding.

* The model is ``dem_chunks`` with one measurement count for every round, a
  prefix, one repeating phase and one or more final phases. The prefix and
  final phases together must be fewer than ``window_size`` rounds, and
  ``window_size`` at least 3.
* The node needs a trailing guard: ``step_size`` equal to
  ``window_size - start_guard - end_guard``, with ``end_guard`` of at least
  one. Set no guards and ``end_guard`` defaults to
  ``window_size - step_size``. With ``t > 1`` final phases, ``start_guard``
  must be at least ``t``, and results lag ``t`` rounds behind.
* The ``FINAL`` round must be a complete round, and a shot must have at least
  the model's minimum number of rounds.
* The inner decoder is named by type, so the node can build one per window
  shape; all shapes are built when the graph is.
* Nodes combined into one output must all set it.

``run()`` still decodes a fixed-length shot. For streams of any length whose
windows decode at the same time, see :doc:`task_graph_parallel_window`.

Ensembles
+++++++++

An ensemble runs several member graphs on the same input and keeps one
answer:

.. code-block:: yaml

   nodes:
     - id: ens
       type: ensemble
       join_strategy: deadline
       join_params: {deadline_us: 200, soft_stop_before_us: 50}
       members:
         - nodes: [{id: a, type: single_window, inner: {type: pymatching}}]
           edges: [{from: input, to: a}, {from: a, to: output}]
         - nodes:
             - {id: b, type: single_window, inner: {type: nv-qldpc-decoder}}
           edges:
             - {from: input, to: b}
             - {from: b, to: output}
   edges:
     - {from: measurements, to: ens}
     - {from: ens, to: output}

* Members name the ensemble's input ``input`` and their answer ``output``.
* ``join_strategy`` picks the answer: ``first`` keeps the first answer;
  ``deadline`` keeps the cheapest answer before ``deadline_us``, or the first
  if answers have no cost. ``soft_stop_before_us`` asks members to stop early.
* Once an answer is chosen, the other members are asked to stop. A member that
  does not check its cancellation token delays the result until it finishes.
* Fed measurements directly, as here, an ensemble decodes complete shots
  only (``run``, ``run_batch``). It streams when an upstream node delivers
  windows to it; it then runs once per window and adds each winner to the
  corrections.

``decoder_param_schema("selection_strategy.<name>")`` describes a strategy's
parameters, and ``register_selection_strategy()`` (C++) adds a strategy.

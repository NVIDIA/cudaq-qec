.. _task_graph_node_kinds:

Node Kinds
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Kind
     - What it does
   * - ``sliding_window``
     - Decodes ``window_size`` rounds and advances ``step_size`` rounds. The
       default ``kind`` in code; YAML nodes always name a ``type``.
   * - ``single_window``
     - Decodes a complete fixed-length shot.
   * - ``streaming``
     - Passes each round to an inner decoder that streams natively.
   * - ``parallel_window``
     - Decodes streams of any length in windows that decode at the same time.
       See :doc:`/examples_rst/qec/task_graph_parallel_window`.
   * - ``ingest``, ``decode_window``
     - The nodes ``parallel_window`` builds: ``ingest`` cuts rounds into
       windows and ``decode_window`` decodes them.
   * - ``xor``
     - XORs its inputs: ports ``a`` and ``b``, or ``in_0`` to ``in_{N-1}``
       with ``num_inputs: N``.
   * - ``logical_readout``
     - Reads the raw logical value through the DEM's ``m2o_sparse``. See
       :ref:`task_graph_logical_output`.
   * - ``predecoder``
     - Runs a predecoder and passes its residual detectors on through its
       ``residual`` port.
   * - ``ensemble``
     - Runs several member graphs and keeps one answer. See
       :doc:`/examples_rst/qec/task_graph_ensemble`.

Kinds are a registry, so plugins can add more.
``registered_task_graph_node_kinds()`` lists an installation's kinds,
``decoder_param_schema("task_graph_node.<kind>")`` describes a kind's
parameters, and ``decoder_config_json_schema()`` returns a JSON Schema for
checking configurations with your own tools.

.. _task_graph_guards:

Guarded Sliding Windows
^^^^^^^^^^^^^^^^^^^^^^^

Overlapping windows see each round more than once. Guard rounds are context
only: the window decodes with them but does not correct them, because a
neighboring window does.

.. code-block:: yaml

   - id: sw
     type: sliding_window
     window_size: 5
     step_size: 4
     end_guard: 1            # the last round is context only
     inner: {type: single_error_lut}

* ``start_guard`` and ``end_guard`` (or ``guard_rounds`` for both) set the
  guards.
* With a guard, ``step_size`` must equal
  ``window_size - start_guard - end_guard``, so each round is corrected
  exactly once.
* The ``FINAL`` round corrects the last window's ``end_guard``.

.. _task_graph_async_options:

Asynchronous Streaming Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``enqueue_syndrome`` returns after the decodes it starts. With
``async_streaming``, it copies the round and returns, and the graph decodes on
its own worker threads; it waits only when the graph's queue is full. Reads
(``get_obs_corrections``, ``streaming_result``) wait for rounds already sent,
and a ``FINAL`` round returns once the whole shot is decoded.

.. code-block:: yaml

   task_graph:
     async_streaming: true
     async_num_workers: 4          # worker threads; default: sized to the graph
     async_worker_cpus: [4, 5, 6, 7]
     nodes: [...]


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

Fire Pools
^^^^^^^^^^

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

.. _task_graph_streaming_results:

Reading Streaming Results
^^^^^^^^^^^^^^^^^^^^^^^^^

``get_obs_corrections()`` returns the corrections, one byte per observable.
``streaming_result()`` also says whether they are ready and which rounds they
cover. For each graph output it reports:

* ``status``: ``PENDING`` before the output has published or after
  ``clear_corrections()``, ``AVAILABLE`` mid-shot, ``COMPLETE`` after
  ``FINAL``, or ``FAILED``. (In C++, lowercase.)
* ``coverage_begin_round`` and ``coverage_end_round``: the rounds the latest
  update committed, as a half-open range. For a parallel window, this is every
  round committed since the last ``clear_corrections()``.
* ``closure``: whether decoded windows were checked against their detectors
  (see :doc:`/examples_rst/qec/task_graph_parallel_window`).

The result also reports aggregate ``status`` and ``closure`` for the graph's
correction outputs; a logical output does not contribute to these aggregates.

A zero correction with status ``AVAILABLE`` is a real result. Synchronously,
``enqueue_syndrome`` also returns whether any node decoded; with
``async_streaming``, an untagged call returns ``False``, so read ``status``
instead.

Additional result fields describe changes to the output and the shot:

* ``generation``: a counter that advances each time an output publishes,
  once every node that feeds it has updated. Fed only by native streaming
  nodes, it advances whenever coverage grows. The graph's generation advances
  once every output but the logical one has.
  ``clear_corrections()`` keeps it, so use ``status``, not ``generation``, to
  decide whether a value can be read.
* ``result_revision``: advances on any change of value or status.
* ``shot_epoch``: advances on ``reset_decoder()``.
* ``error``: why the graph failed, when ``status`` is ``failed``. Reset to
  continue.

With ``async_streaming``, ``streaming_result()`` first waits for every
queued round, then returns a snapshot. Without it, the C++ result is a live
reference owned by the graph; copy it to keep it across later calls. Python
always returns a copy.

Staged Outputs
^^^^^^^^^^^^^^

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

Edge Values
^^^^^^^^^^^

Each port declares the type of value it produces or accepts, and building a
graph rejects an edge whose types do not match. The types you wire most are
``measurement_results``, ``detection_events`` and ``logical_outcome``; others
carry values between built-in nodes, such as the corrections one window sends
another. ``registered_edge_value_types()`` (C++) lists an installation's
types.

A node fed by another node decodes that node's output, as the global decoder
in :doc:`/components/qec/decoding_task_graph`'s first figure decodes the predecoder's
``residual``.

.. _task_graph_logical_measurements:

Logical Measurements
^^^^^^^^^^^^^^^^^^^^

A graph can return two results:

* **Corrections**, the flips of the logical observables:
  ``get_obs_corrections()``, ``run()`` and the decoding server's
  ``get_corrections``.
* **Logical measurements**, the raw logical readout with the corrections
  applied: ``get_logical_measurements()`` and the server's
  ``get_logical_measurements``.

Logical measurements need a detector error model that maps measurements to
the raw logical observables through ``m2o_sparse`` (see
:ref:`dynamic_dem_measurements`), and a graph with a logical
output. ``parallel_window`` wires one with
``output: corrections_and_logical`` (see :doc:`/examples_rst/qec/task_graph_parallel_window`);
other graphs wire a ``logical_readout`` node (see
:ref:`task_graph_logical_output`).

* A logical measurement is *released* once its readout has arrived and every
  correction that can flip it is committed. A final readout is released at
  ``FINAL``. A mid-circuit readout is released mid-shot when the corrections
  are staged, as ``parallel_window``'s are; otherwise at ``FINAL``.
* ``get_logical_measurements(first, count)`` returns ``None`` (C++:
  ``nullptr``) until every measurement in the range is released, and again
  once the shot is consumed.
* The logical output is the last of ``streaming_result().outputs``, with a
  ``released`` flag per measurement. It is not in ``output_names()``, and
  ``get_num_logical_measurements()`` gives its width.
* The logical measurements apply every correction of the shot, so the shot
  can be consumed only after ``FINAL``: ``clear_corrections()`` raises before
  then. ``reset_decoder()`` still abandons a shot.
* ``run`` and ``run_batch`` return corrections only.

.. _task_graph_logical_output:

Configuring Logical Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``logical_readout`` node uses a detector error model (DEM) with an
``m2o_sparse`` mapping to read the raw logical value. An ``xor`` applies the
corrections to it. See :ref:`dynamic_dem_measurements` for the mapping.
The output that the node reaches, whatever its name, is the graph's logical
output. For a DEM whose rounds all have the same measurement count:

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

Native Streaming Nodes
^^^^^^^^^^^^^^^^^^^^^^

A ``streaming`` node passes each ``enqueue_syndrome`` call to its inner
decoder's own ``enqueue_syndrome``, for decoders that process rounds
incrementally. Its complete-shot path decodes the whole shot at once.

Rounds may differ in width, such as a narrower readout round, if the inner
decoder implements ``streaming_correction_provider`` and so ends its own
shots; a shot may then end at any ``FINAL`` round. Coverage is the rounds
the decoder has committed, not the node's schedule:

* Mid-shot, the node publishes whenever ``committed_round_count()`` grows.
  Coverage starts where ``clear_corrections()`` last consumed the
  corrections. A decoder without the count publishes nothing before
  ``FINAL``.
* ``FINAL`` covers the rounds ``finalized_round_count()`` reports, or else
  the committed ones.
* A shot the decoder ends without ``FINAL`` covers the rounds
  ``finalized_round_count()`` reports, and the stream continues.
* Several such nodes may feed one output, which covers the rounds all of them
  have committed. If one reports no count, or they disagree on where the shot
  begins, the output follows the nodes' schedule.

.. code-block:: yaml

   - id: native
     type: streaming
     num_syndromes_per_round: 24
     inner: {type: <streaming-decoder>}

Arbitrary-Length Sliding Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``arbitrary_length: true`` on a ``sliding_window`` node lets a shot end at any
round, not only on a window boundary. ``FINAL`` decodes the actual last
rounds, without padding.

* The detector error model uses ``dem_chunks`` with one measurement count for
  every round, a prefix, one repeating phase and one or more final phases.
  The prefix and final phases together must be fewer than ``window_size`` rounds, and
  ``window_size`` at least 3.
* The node needs a trailing guard: ``step_size`` equal to
  ``window_size - start_guard - end_guard``, with ``end_guard`` of at least
  one. Set no guards and ``end_guard`` defaults to
  ``window_size - step_size``. With ``t > 1`` final phases, ``start_guard``
  must be at least ``t``, and results lag ``t`` rounds behind.
* The ``FINAL`` round must be a complete round, and a shot must have at least
  the DEM's minimum number of rounds.
* The inner decoder is named by type, so the node can build one per window
  shape; all shapes are built when the graph is.
* Nodes combined into one output must all set it.

``run()`` still decodes a fixed-length shot. For streams of any length whose
windows decode at the same time, see :doc:`/examples_rst/qec/task_graph_parallel_window`.

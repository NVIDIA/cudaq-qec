.. _decoding_task_graph_example:

Decoding Task Graphs
--------------------

A :class:`~cudaq_qec.DecodingTaskGraph` is a composable decode pipeline that
owns its decoders and D/O matrices.  Internally it is a directed acyclic graph
(DAG) of nodes connected by typed edges, but from the outside it exposes two
interfaces:

* **One-shot** — :meth:`~cudaq_qec.DecodingTaskGraph.run` processes one
  complete shot.
* **Batch** — :meth:`~cudaq_qec.DecodingTaskGraph.run_batch` processes a
  collection of shots.
* **Streaming** — :meth:`~cudaq_qec.DecodingTaskGraph.enqueue_syndrome` /
  :meth:`~cudaq_qec.DecodingTaskGraph.get_obs_corrections` accumulate
  per-round syndrome bytes and fire when a window is full.  This is the path
  used during real-time quantum error correction.

Task graphs are constructed either **programmatically** (via factory methods)
or **declaratively** (via the ``task_graph:`` block in a decoder YAML config).

.. note::

   The streaming interface is the primary use case for real-time QEC. One-shot
   execution is also useful for offline analysis and unit testing.

Arbitrary-length fixed-width memory streams
+++++++++++++++++++++++++++++++++++++++++++

For a DEM-backed ``sliding_window`` node, set ``arbitrary_length: true`` in
its node parameters to allow FINAL between regular firings. Middle deliveries
can use a fixed predefined chunk size; the total round count need not be a
multiple of that size or the window stride. FINAL decodes the actual physical
tail and publishes its exact committed coverage, without zero-padding.

This mode requires equal measurement counts per round, an initial prefix,
one repeating phase, one or more terminal phases, and a named fault-output
decoder that can be rebuilt from the DEM recipe. Explicit per-phase D maps or
the existing memory-experiment D convention are supported. A positive trailing guard and
``step_size = window_size - start_guard - end_guard`` preserve fault ownership.
Shots must contain the recipe's minimum number of physical rounds. FINAL
carries the shot's final measurements: a partial or empty FINAL round is an
error. Byte-fragmented ordinary input is supported. Detection-event edges,
flat matrices and prebuilt decoder instances do not supply the required recipe.
The finite-shot ``run()`` still uses the configured closed W-round model.
Enable this mode consistently for branches combined into one streaming output;
their firing/commit schedules and exact committed coverage must agree.
A native ``streaming`` node and an ``arbitrary_length`` guarded-window node
cannot share a streaming output: their schedules differ, so streaming is
rejected before input is delivered. They can run in one graph through separate
named outputs. At FINAL, native coverage describes the whole shot, after any
untagged shots before it, while window coverage describes its last committed
interval; both end at the physical round count since the stream began. Joining
those publication policies is not supported. Failures mark the result failed
and require reset before resuming; values in that state are diagnostic, not a
completed shot.

Models are prepared eagerly: ``floor(p / s) + p + 4 + W + lookahead - m`` plus
the base decoder, where p is the prefix length, s the stride, W the window and
m the minimum shot length. Lookahead is zero for one terminal phase and t for
t > 1 terminal phases. Short-shot models can occupy quadratic storage in W.
Nonterminal models store H by fault column and a detector-to-round lookup,
prepared at construction. Carry updates scan the committed fault range and
visit only the detector edges of active faults; guard ownership and decoder
thresholding are unchanged. The lookup adds one index per window detector,
and the column layout replaces, rather than duplicates, the node's row layout.
The raw ring has W+1 rounds for a single terminal phase. With t > 1 terminal
phases, regular inference waits for t further records so it never treats an
unidentified terminal phase as bulk. The ring
then has W+t+1 rounds and the leading guard must be at least t. Initial FINAL
models cover up to W+t rounds if FINAL preempts the first regular fire. The
reported delivery schedule includes this lookahead; regular inference still
uses W-round models.
This delays regular publications by t input rounds, not work proportional to
shot length. FINAL is processed without waiting for additional input records.
The lookahead prevents unresolved terminal records from being decoded as bulk;
it must not be removed just to reduce reported latency.
Model and buffer storage do not grow with shot duration. Inner-decoder
allocations are backend-dependent. Legacy mode is
unchanged. Production LER, tail latency and hardware qualification must be
measured for the deployed decoder and window configuration.

Single-node graphs (sliding window)
++++++++++++++++++++++++++++++++++++

The most common topology is a single sliding-window node wrapping an inner
decoder.  The ``task_graph:`` YAML block expresses this explicitly.

**YAML configuration**

.. code-block:: yaml

   decoders:
     - id: 0
       task_graph:
         type: sliding_window
         inner_type: pymatching      # inner decoder registry name
       dem_chunks:
         seam: {from: next_round, to: prev_round}
         connections:
           - {from: init, to: bulk}
           - {from: bulk, to: bulk}  # self-loop = repeating round
           - {from: bulk, to: final}
         # num_rounds absent → streaming: window sized to minimum valid rounds
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

Key points:

* ``type: sliding_window`` names the node kind; ``inner_type`` names the
  inner decoder plugin.  ``decoder_custom_args`` apply to the inner decoder.
* When ``num_rounds`` is absent and a self-loop connection exists, the graph
  is built in **streaming mode**: the window fires once per round (stride = 1).
* When ``num_rounds`` is present, the graph uses a **fixed-T window** that
  fires once per full shot.  Write ``type: single_window`` to say so
  explicitly; that spelling then requires the fixed round count, so a spec
  that would stream is rejected rather than quietly given a sliding window.
* Error rates are derived automatically from the phase ``error_rates`` fields.
* This two-key form covers one node.  Write ``nodes:`` instead to size the
  window yourself, add guard rounds, or build a multi-node topology; see
  `Declarative topologies`_ below.

**Programmatic construction**

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

One-shot graph execution
++++++++++++++++++++++++

:meth:`~cudaq_qec.DecodingTaskGraph.run` processes one complete shot without
touching the streaming accumulation buffer.

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

Batch execution
+++++++++++++++

:meth:`~cudaq_qec.DecodingTaskGraph.run_batch` processes a collection of
complete shots.

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

Streaming interface
+++++++++++++++++++

In real-time QEC, syndrome measurements arrive one round at a time.
:meth:`~cudaq_qec.DecodingTaskGraph.enqueue_syndrome` accumulates bytes into
the window buffer and returns ``True`` / ``true`` when a decode fires.

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

Streaming contract:

* Deliver ``get_num_msyn_per_round()`` bytes per call.  A window that slides
  has only that much room once it is full, so a larger chunk no longer fits.
* The window fires when it first holds ``get_num_msyn_per_decode()`` bytes,
  then every ``stride`` rounds after that.  A fixed-T window empties on each
  fire, so it also accepts a full window per call.
* Task graphs meter arbitrary chunks to each node's available room; input is
  not dropped when one call spans multiple window fires.
* Corrections XOR-accumulate across windows within a shot.
  ``clear_corrections()`` consumes the published accumulator while preserving
  sliding-window overlap and publication generations. It marks statuses
  ``pending`` and zeros values, so a preserved nonzero generation is only a
  watermark and does not mean that a result remains readable.
* ``reset_decoder()`` starts a clean experiment epoch and resets inner decoder
  state.

For a graph with independently progressing outputs, the enqueue return value
only means that some node fired.  Use ``streaming_result()`` for result
readiness.  It returns the named output values together with each output's
status, generation, value range, and half-open round coverage.  This makes an
available all-zero correction distinguishable from an output that has not
fired.  The graph-level generation advances only after every named output
passes the corresponding publication barrier.  ``result_revision`` advances
on any output or graph-status change, including progress before that aggregate
barrier.  A ``FINAL`` round marks published outputs complete; outputs that
never published and the aggregate graph remain pending.  Reset starts a new
shot epoch and restarts its revision at zero.  C++ returns a coherent live
graph-owned reference, so copy it before later graph mutations when historical
state is needed.  Python returns a detached result snapshot.  In either API,
readability is determined by ``available`` or ``complete`` status, never by
generation alone.

Opt-in asynchronous streaming
++++++++++++++++++++++++++++++

Callers select asynchronous streaming when constructing a graph with
``task_graph_options::async_streaming`` in C++ or
``TaskGraphOptions.async_streaming`` in Python. The existing streaming calls
remain the interface: ``enqueue_syndrome`` copies BULK input into a bounded
queue, ``streaming_result`` and ``get_obs_corrections`` wait for records
accepted before their call, and both return coherent snapshots without closing
the session. A FINAL enqueue waits for all accepted input and the final flush.
``clear_corrections`` drains accepted input without flushing guarded tails
before consuming the result. Each public enqueue remains one queue record, so
inner streaming decoders observe the same call boundaries as synchronous
graphs.

Decoder-server YAML exposes the same policy as ``task_graph.async_streaming``,
with optional ``async_queue_capacity``, ``async_coordinator_cpu``, and
``async_spin_pause_iters`` fields. A nonnegative coordinator CPU requests
Linux thread affinity and fails graph construction if it cannot be honored.
It pins the coordinator thread only; decoder-pool workers keep their own
configured affinity.
The spin count keeps an active coordinator polling briefly before it parks;
zero, the default, disables spinning. Reserve that CPU from producer and
decoder-worker affinity masks when using this low-latency policy.

Multi-node graphs (combine)
++++++++++++++++++++++++++++

:meth:`~cudaq_qec.DecodingTaskGraph.combine` runs two independent decoders and
XORs their logical outcomes.  Each branch applies its own D matrix.  In
streaming mode, ``enqueue_syndrome`` routes the syndrome to both branches
simultaneously.

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

.. _Declarative topologies:

Declarative topologies
++++++++++++++++++++++

``task_graph.nodes`` describes the graph node by node, and ``task_graph.edges``
wires them.  A node names its kind in ``type``, its inner decoder in ``inner:``,
and the kind's own parameters inline:

.. code-block:: yaml

   decoders:
     - id: 0
       task_graph:
         nodes:
           - id: sw
             type: sliding_window
             window_size: 3               # rounds per decode
             step_size: 1                 # advance one round per decode
             num_syndromes_per_round: 4    # checked against the DEM
             inner:
               type: pymatching           # inner decoder args go here too
         # edges omitted: a single node is wired measurements → sw → output
       dem_chunks: {...}

An edge endpoint is a node id, optionally with a port: ``{from: pre.residual,
to: global}``. ``measurements`` is the incoming shot and ``output`` is the
graph's result; the legacy source spelling ``dets`` is also accepted. An
omitted port means ``output`` on the source and ``input`` on the destination,
which is what a single-port node declares, so most edges name only the nodes; a
node with named ports, such as an ``xor`` node's ``a`` and ``b``, has to be
named explicitly. This is the X/Z split that
:meth:`~cudaq_qec.DecodingTaskGraph.combine` builds in C++:

.. code-block:: yaml

   task_graph:
     nodes:
       - id: x
         type: sliding_window
         window_size: 3
         step_size: 1
         inner: {type: pymatching}
       - id: z
         type: sliding_window
         window_size: 3
         step_size: 1
         inner: {type: pymatching}
         dem_chunks: {...}          # optional: this node's own DEM
       - id: combine
         type: xor
     edges:
       - {from: measurements, to: x}
       - {from: measurements, to: z}
       - {from: x, to: combine.a}
       - {from: z, to: combine.b}
       - {from: combine, to: output}

Nodes are listed in topological order, and a node without its own
``dem_chunks`` uses the entry-level one. Every node fed from ``measurements``
receives the same bytes. Nodes contributing to one output must use the same
streaming schedule; nodes feeding separate outputs may progress independently.
A node fed by an edge consumes detection events instead.

Available node kinds are a registry, not a fixed list.  Query an installation
with ``registered_task_graph_node_kinds()`` and inspect one kind's
parameters with ``decoder_param_schema("task_graph_node.sliding_window")``.
``decoder_config_json_schema()`` includes the whole ``task_graph`` block, so
configs can be validated offline with standard JSON Schema tooling.

The ``streaming`` kind transparently forwards raw chunks to its inner decoder's
native ``enqueue_syndrome()`` implementation. It is useful for decoders that
incrementally process each round themselves, instead of asking a
``single_window`` node to buffer the complete shot and call ``decode()`` at the
end. Its one-shot graph path still applies D, calls ``decode()``, and projects
through O normally:

When the inner decoder declares native DEM-chunk support and produces
observables, graph preparation keeps that recipe instead of closing the
configured full-shot H and error-rate model. D and the graph's observable
projection remain available, including through ``d_matrix()`` and
``o_matrix()``. These are configured-template metadata, not constant-size
objects: D has one row per configured detector and O enumerates the configured
fault columns that flip each observable. Their storage can grow with configured
round count even though the native decoder does not close a full-shot H or
error-rate model. Bounded decoder-window storage does not imply that every
graph allocation is independent of the configured template length.
Matrix-backed and guarded-window kinds still prepare their
finite window models. A bounded-DEM deployment must use a compatible native
streaming recipe or a fixed-size window, rather than a full-shot matrix model.

.. code-block:: yaml

   - id: native
     type: streaming
     num_syndromes_per_round: 24
     inner: {type: <streaming-decoder>}   # implements enqueue_syndrome()
     dem_chunks: {...}                    # or inherit the entry-level DEM

Ensembles
+++++++++

An ensemble races several member subgraphs and keeps one answer. It is written
as a pattern in place of a node, with the members as node lists of their own:

.. code-block:: yaml

   nodes:
     - id: ens
       ensemble:
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

Members name the ensemble's input edge as
``input`` and the answer as ``output``; every member lists its edges.
Every member node that accepts a ``candidate_sink`` as an input is wired to the
run's shared candidate sink and every one that accepts a ``cancellation_token`` to
the run's shared cancellation token. Cancellation is cooperative: a member that does
not poll its token delays the result until it finishes. Members can push candidate solutions either by
accepting the ``candidate_sink`` as an input, or by pushing a ``logical_outcome`` to
``output``.

The decision to commit to a specific logical outcome is made by the run's ``selection_strategy``.
Every candidate push is shown, on the pushing member's thread, to the run's ``selection_strategy``:

- ``first`` commits to the first candidate
- ``deadline`` optionally requests a soft stop ``soft_stop_before_us`` before ``deadline_us``
  and a hard stop at the deadline, then picks the cheapest costed candidate seen
  before the deadline, falling back to the first one if candidates do not have costs available.

A strategy's parameters
are described by ``decoder_param_schema("selection_strategy.<name>")``; a module adds one with
``register_selection_strategy()`` and holds the returned registration while its code is loaded. A commit hard-stops
the other members through their token. If a strategy fails to commit to solution, the run throws.
When streaming, the fan-out runs one ensemble per window its producer delivers and the join XORs
each window's winner into the corrections, sized from the first member answer
or member node ``output`` of known width.

Guarded windows
+++++++++++++++

Overlapping windows see each round more than once.  ``start_guard`` and
``end_guard`` (or ``guard_rounds`` for both) declare how many rounds at each
edge of the window are **context only**: the decode still uses them, but they
contribute nothing to that window's corrections, because a neighboring window
commits them.  Without guards, every round of every window commits and an
overlapped round is counted twice.

.. code-block:: yaml

   - id: sw
     type: sliding_window
     window_size: 5
     step_size: 4
     end_guard: 1            # the last round is context for this decode only
     inner: {type: pymatching}

Guards must leave at least one round to commit and tile the stream exactly
once: ``step_size`` must equal ``window_size - start_guard - end_guard``.
The first decode of a shot commits its leading guard, since no earlier window
covered those rounds; ``reset_decoder()`` starts a new shot and restores that
behavior.

.. note::

   Mark the shot's last round ``FINAL`` to flush and commit the final window's
   ``end_guard``. Without that marker, resetting the graph discards the
   withheld tail.

Streaming edge routing
++++++++++++++++++++++

The graph's streaming path supports typed edges between streaming nodes.  When
node A fires, its outputs are routed to downstream nodes according to the graph
edges, and those nodes accumulate the routed values rather than raw syndrome
bytes.

Each node declares the path-independent type it produces on each output slot
(``output_type``) and which types each input slot accepts
(``accepts_input_type``). ``compile_wires`` validates every one-shot and
streaming edge and rejects mismatches before the graph goes live. Nodes
separately declare whether an input supports edge-fed streaming.

Edge-value types are a registry, like node kinds. The built-in types are
``measurement_results``, ``detection_events`` and ``logical_outcome``, plus
those built by plugins and lowering operations.
``registered_edge_value_types()`` lists what an installation provides.

A pre-decoder + global decoder topology illustrates this: the ``predecoder``
kind fires once per configured window, emits its pre-logical outcome on one
output port and a residual ``detection_events`` vector on another. The residual
flows into a downstream ``window_node`` (e.g. PyMatching) via a streaming edge:

.. code-block:: yaml

   edges:
     - {from: measurements, to: pre}
     - {from: pre.pre_L, to: combine.a}
     - {from: pre.residual, to: global}
     - {from: global, to: combine.b}
     - {from: combine, to: output}

.. note::

   ``predecoder`` is a registered node kind. It produces both a
   ``logical_outcome`` and a ``detection_events`` residual on each fire through
   its ``pre_L`` and ``residual`` ports.

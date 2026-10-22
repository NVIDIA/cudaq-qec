Playback Emulator
=================

The playback emulator replays a scripted sequence of decoder calls (a
*schedule*) against one or more realtime decoders, on a precise timing loop
that does not need a quantum processor (QPU). Each line of the schedule says
*when* to call a decoder and *what* to send it. The emulator records when each
call was made, when its reply arrived, and what came back, so it can be used
to measure decoder latency or check decoder output.

.. note::
   The playback emulator is only built when CUDA-Q QEC is built against a 
   CUDA-Q install with ``cudaq-realtime`` enabled.

A schedule is a text file, where each line is a single event representing a 
decoder call, or sequence of decoder calls. For example, the following schedule 
resets decoder 0, enqueues three rounds of syndromes, and asks for a
one-bit correction:

.. code-block:: text

   # <trigger> <op> [key=value...]
   0 reset
   1 stream source=0 rounds=3 every=2
   - get_corrections return_size=1

Save it to a file (e.g. ``schedule.txt``) and run it from the command line or
from Python:

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec

      decoders = qec.multi_decoder_config.from_yaml_str(open("config.yaml").read())
      result = qec.playback.run(open("schedule.txt").read(),
                                tick_ns=1000,
                                sources={0: {"type": "static",
                                             "rounds": [[0, 1]] * 3}},
                                decoders=decoders)
      print(result.write_csv())

.. tab:: Command Line

   .. code-block:: bash

      # syndromes.txt holds one 0/1 bit string per round, one round per line.
      playback-emulator --schedule=schedule.txt --config=config.yaml \
        --source=0:syndromes.txt --tick=1us --out=result.csv

``config.yaml`` is a decoder configuration, as described in
:doc:`Getting Started with Realtime Decoding </examples_rst/qec/getting_started_realtime_decoding>`.
Both entry points produce the same result: one record per schedule line,
written as CSV by ``write_csv()`` (Python) or ``--out`` (command line); see
`Output Format`_. Run ``playback-emulator --help`` for
all command-line options.

Schedule Format
^^^^^^^^^^^^^^^

Each line is ``<trigger> <op> [key=value...]``. Blank lines are ignored, and
``#`` starts a comment.

Timing is tracked in ticks, where one tick is defined by the ``tick_ns`` run option (default 1 µs). The **trigger** says when the line runs:

* ``N`` (an integer): at tick ``N`` from the start of the run. One tick
  defaults to 1 µs (see `Run Options`_).
* ``+N``: ``N`` ticks after the previous line finished sending. 
* ``-``: same as ``+0``, i.e. execute immediately.

Lines run in file order. Plain tick numbers must not decrease, and once a line
uses ``+N`` or ``-``, every later line must too.

The **operations** are listed below. Arguments in ``[brackets]`` are optional.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Operation
     - Description
   * - ``reset``
     - Resets the decoder.
   * - ``stream source=<source_id> [rounds=N] [every=N]``

       ``stream source=<source_id> until=NAME [min_rounds=N] [max_rounds=N]
       [every=N]``
     - Sends syndromes from ``source=``, one call per round. ``rounds=N``
       sends exactly ``N`` rounds (default 1). ``every=N`` waits ``N`` ticks
       between rounds (default 1; ``0`` sends as fast as possible). With
       ``until=NAME``, the stream instead runs until signal ``NAME`` is raised,
       sending at least ``min_rounds`` (default 1) and at most ``max_rounds``
       (default 1000) rounds.
   * - ``enqueue source=<source_id>``
     - Shorthand for a one-round ``stream``.
   * - ``enqueue_data source=<source_id>``
     - Like ``enqueue``, but sends the source's final data-qubit readout,
       which ends the shot.
   * - ``get_corrections [<bits>] [return_size=N]``
     - Requests corrections. ``<bits>`` (e.g. ``get_corrections 01``) is the
       expected correction; a different reply sets ``correction_mismatch``.
       ``return_size=N`` sets the number of correction bits (default: the
       length of ``<bits>``).

``source=`` sets the syndrome source by its integer ID (see below), or gives one
round of bits inline, e.g. ``enqueue source=0b0110``.

Every operation also takes these optional keys:

* ``[session=N]``: the decoder ID to call (default ``0``).
* ``[signal=NAME]``: raise signal ``NAME`` once all of this line's replies
  have arrived.
* ``[after=NAME]``: wait for signal ``NAME`` before running this line.

A signal must be raised by an earlier line than any line that waits on it.
Signals let one decoder's schedule react to another's, e.g. stream syndromes
to decoder 0 until decoder 1 has replied. 

If a decoder reply reports an error, the run stops. Lines that never ran have
``status`` ``-1``, and the reason is in ``result.warnings``.

Output Format
^^^^^^^^^^^^^

The output has one CSV row per schedule line, in schedule order. Times are in
nanoseconds from the start of the run. Columns that list one value per request
(one per round for ``stream``) are space-separated.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - ``event_index``, ``decoder_id``, ``op``
     - The schedule line (0-based), the decoder it called, and its operation.
   * - ``deadline_ns``, ``call_ns``, ``return_ns``
     - When the line was due, when it started sending, and when its last
       reply arrived.
   * - ``status``
     - ``0`` is OK. Other values below ``100`` are decoder errors
       (``1`` invalid decoder, ``2`` bad request, ``3`` internal error,
       ``4`` not ready, ``5`` busy, ``6`` syndromes dropped). A ``stream``
       instead reports ``100`` (OK), ``101`` (source ran out), ``102``
       (reached ``max_rounds`` before ``until=``), or ``103`` (error).
       ``-1`` means the line never ran.
   * - ``rounds_streamed``
     - Rounds actually sent (``stream`` only).
   * - ``read_completed``
     - ``1`` if a correction was received.
   * - ``syndrome_bits``, ``correction_bits``
     - Bits sent and bits received, as ``0``/``1`` strings.
   * - ``correction_mismatch``
     - ``1`` if the correction differs from the expected bits.
   * - ``request_ids``
     - Each request's ID, for matching against decoder server logs.
   * - ``dispatched``
     - ``1`` if the line ran.
   * - ``request_dispatch_ns``, ``request_return_ns``
     - When each request was sent, and when its reply arrived (``0`` if
       never).
   * - ``request_status``
     - Each request's status: the decoder status codes above (``0`` to
       ``6``), or ``-1`` if no reply was recorded. A request with no reply
       before its timeout is ``3``.

In Python, ``result.records`` holds one dict per CSV row, keyed by the column
names above.

Syndrome Sources
^^^^^^^^^^^^^^^^

A syndrome source produces one round of syndrome bits at a time. Each source
has an integer ID, which ``source=`` refers to. Any source can be used with
``until=``; a stream whose source runs out first stops early with status
``101``.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Type
     - Description
     - How to create
   * - ``static``
     - Replays fixed rounds in order. Best for exactly reproducible tests.
     - Python: ``{"type": "static", "rounds": [[0, 1], ...]}``

       CLI: ``--source=ID:FILE``, one round per line.
   * - ``stim_memory``
     - Generates rounds on demand from a Stim memory circuit (``code`` is
       ``surface_code``, ``repetition_code``, or ``color_code``). Every key
       except ``type`` and ``seed`` is passed unchanged to Stim's circuit
       generator (as in ``stim.Circuit.generated``, with ``code`` and
       ``task`` forming ``code_task``). Its ``rounds`` must be at least 3. 
       Never runs out of rounds.
     - Python: ``{"type": "stim_memory", "seed": 1, "code": ..., "task": ...,
       "distance": ..., "rounds": 3, <noise>}``

       CLI: ``--stim-source=ID:code=...,task=...,distance=...``
   * - ``cudaq_memory``
     - Raw measurements from a CUDA-Q QEC ``memory_circuit``, up to
       ``max_rounds`` rounds per shot.
     - Python only: ``{"type": "cudaq_memory", "code": qec.get_code(...),
       "state_prep": "prep0", "max_rounds": 3, "seed": 1, "noise": ...}``

Sessions
^^^^^^^^

A session connects the emulator to one decoder, chosen by ``session=N`` in
the schedule. 

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Session
     - Description
     - How to select
   * - ``inproc``
     - Runs the decoders from a decoder configuration in the same process.
     - Python: ``decoders=<multi_decoder_config>``

       CLI: ``--backend=inproc --config=FILE`` (default)
   * - ``udp``
     - Sends calls over UDP to a running decoding server.
     - Python: ``udp_endpoints={ID: "host:port"}``

       CLI: ``--udp-endpoint=ID:HOST:PORT``
   * - ``cpu_roce``
     - Sends calls over RDMA (CPU RoCE) to a decoding server started with
       ``--transport=cpu_roce``.
     - Python: ``cpu_roce_endpoints={ID: "host:port"}``, plus
       ``cpu_roce_options={"device": ..., "local_ip": ...}``

       CLI: ``--cpu-roce-endpoint=ID:HOST:PORT``, plus ``--cpu-roce-device``
       and ``--cpu-roce-local-ip``
   * - ``null``
     - Discards every call. Measures the emulator's own overhead.
     - Python: ``null_decoder_ids=[ID, ...]``

       CLI: ``--backend=null --config=FILE``

Run Options
^^^^^^^^^^^

These options behave the same in Python and on the command line. Python takes
whole nanoseconds (``_ns``) or milliseconds (``_ms``). The command line takes a
duration with a unit (``ns``, ``us``, ``ms``, or ``s``); timeouts must be a
whole number of milliseconds. The tick and every timeout must be positive.

.. list-table::
   :header-rows: 1
   :widths: 30 32 10 28

   * - Python
     - Command line
     - Default
     - Description
   * - ``tick_ns``
     - ``--tick``
     - 1 µs
     - Length of one schedule tick.
   * - ``lead_in_ns``
     - ``--lead-in``
     - 20 ms
     - Setup time between starting the run and tick 0.
   * - ``udp_timeout_ms``
     - ``--udp-timeout``
     - 200 ms
     - How long a UDP request waits for its reply.
   * - ``cpu_roce_timeout_ms``
     - ``--cpu-roce-timeout``
     - 200 ms
     - How long a CPU RoCE request waits for its reply.
   * - ``cpu_roce_options["connect_timeout_ms"]``
     - ``--cpu-roce-connect-timeout``
     - 5 s
     - How long to wait for the connection to a CPU RoCE server.
   * - ``cpu_roce_options["slots"]``, ``["slot_size"]``
     - ``--cpu-roce-slots``, ``--cpu-roce-slot-size``
     - 8, 256
     - CPU RoCE ring size, in slots and bytes per slot. Must match the
       server's ``--num-slots`` and ``--slot-size``.

A request with no reply within its timeout is an error, which stops the run.

Example
^^^^^^^

The example below configures a lookup-table decoder for a distance-3
repetition code, then plays back ten fixed shots against it, alternating
between no error and a flipped data qubit. It prints each shot's correction
(``0``, then ``1``) and decoding latency.

.. literalinclude:: ../../examples/qec/python/playback_emulator.py
   :language: python
   :start-after: # [Begin Documentation]
   :end-before: # [End Documentation]

To play the same schedule against a decoding server instead, save the decoders
with ``decoders.to_yaml_str()`` to ``config.yaml``, start the server, and pass
``udp_endpoints={0: "127.0.0.1:50000"}`` to ``run()`` in place of ``decoders=``:

.. code-block:: bash

   decoding_server --config=config.yaml --transport=udp --port=50000

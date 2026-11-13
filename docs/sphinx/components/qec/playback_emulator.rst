Playback Emulator
=================

Overview
--------

The playback emulator is a software client for testing realtime decoding
without a quantum processor (QPU). It sends measurements and correction
requests according to a text *schedule*, which specifies when each call should
run and what to send. You can replay recorded measurements or generate them
from a simulated memory experiment, check the returned corrections, and
inspect the time taken to receive each reply.

Decoders can run in the same process as the emulator or in a separate
:doc:`Decoding Server <decoding_server>`. Use the in-process mode to test a
decoder configuration locally; use a server session to exercise communication
as well as decoding. Both modes exercise the realtime request interface, even
when the measurements were produced offline.

A run connects three pieces:

* A **schedule** orders resets, measurement rounds, final readouts, and
  correction requests, with timing and optional dependencies between events.
* **Syndrome sources** supply the measurements. A static source replays fixed
  rounds; memory-circuit sources generate measurements for the experiment.
* **Sessions** connect each decoder ID in the schedule to an in-process
  decoder, a server endpoint, or a null backend for measuring emulator overhead.

The emulator records the scheduled deadline, actual dispatch and reply times,
status, and correction bits. Compare these to distinguish late dispatch from
time spent waiting for a decoder reply. Timing depends on the host and selected
transport; software playback alone does not establish a hardware feedback-latency
guarantee. This tool is separate from the FPGA playback described in
:doc:`Realtime Decoding from a Quantum Kernel or FPGA
</examples_rst/qec/realtime_decoding_demo>`.

.. note::
   The playback emulator is only built when CUDA-Q QEC is built against a
   CUDA-Q install with ``cudaq-realtime`` enabled. The prebuilt decoding-server
   image includes the command-line tool. The Python interface requires a
   realtime-enabled CUDA-Q QEC Python installation, such as the combined
   CUDA-Q/QEC image; it is not included in the 0.9.0 pip wheels.

Start with :ref:`In-Process Repetition-Code Playback <playback-in-process-example>`
to check ten shots with alternating corrections, or
:ref:`Playback Against a Decoding Server <playback-server-example>` to generate
a surface-code configuration and replay a shot over UDP.

For an accuracy and runtime comparison, see
:doc:`Comparing Ising and Chromobius for Color-Code Decoding
</examples_rst/qec/color_code_demo>`. It replays the same sampled measurements
through Chromobius alone and an Ising predecoder followed by Chromobius, using
both in-process and server sessions.

To make the next shot's length depend on the previous correction's arrival,
see :doc:`Measuring Streaming Interleaved Feed-forward Latency (SIFL)
</examples_rst/qec/sifl_demo>`. It uses completion signals and alternating
in-process decoders to study :doc:`Feedback Latency and Decoder Backlog
<feedback_latency>` as the measurement cadence changes.

Reference
---------

Python and Command-Line Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A schedule is a text file, where each line is a single event representing a 
decoder call, or sequence of decoder calls. For example, the following schedule 
resets decoder 0, enqueues three rounds of syndromes, and asks for a
one-bit correction. This illustrates the syntax; the required rounds and any
final data-qubit readout depend on the decoder configuration:

.. code-block:: text

   # <trigger> <op> [key=value...]
   0 reset session=0
   1 stream session=0 source=0 rounds=3 every=2
   - get_corrections session=0 return_size=1

The interfaces below assume that you have prepared a matching ``config.yaml``,
``schedule.txt``, and syndrome source. For complete configurations and inputs,
use the examples linked above.

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
   * - ``stream source=<source_id> [rounds=N | until=NAME] [every=N]``
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

If a decoder reply reports an error other than ``4`` (not ready), the run
stops. Lines that never ran have ``status`` ``-1``, and the reason is in
``result.warnings``.

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
     - Replays fixed rounds in order.
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

.. _playback-output-format:

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
       ``4`` not ready, ``5`` busy, ``6`` syndromes dropped).
       ``stream`` and ``enqueue_data`` instead report ``100`` (OK), ``101``
       (source ran out), ``102`` (reached ``max_rounds`` before ``until=``), or
       ``103`` (error).
       ``-1`` means the line never ran.
   * - ``rounds_streamed``
     - Rounds actually sent (``stream`` and ``enqueue_data`` only).
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

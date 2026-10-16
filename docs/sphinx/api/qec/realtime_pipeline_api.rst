.. _realtime_pipeline_api:

Realtime Pipeline API
=====================

The realtime pipeline API provides the reusable host-side runtime for
low-latency QEC pipelines that combine GPU inference with optional CPU
post-processing. The published reference is generated from
``cudaq/qec/realtime/pipeline.h``.

.. note::

   This API is experimental and subject to change.


Configuration
-------------

.. doxygenstruct:: cudaq::qec::realtime::experimental::core_pinning
   :members:

.. doxygenstruct:: cudaq::qec::realtime::experimental::pipeline_stage_config
   :members:


GPU Stage
---------

.. doxygenstruct:: cudaq::qec::realtime::experimental::gpu_worker_resources
   :members:

.. doxygentypedef:: cudaq::qec::realtime::experimental::gpu_stage_factory


CPU Stage
---------

.. doxygenstruct:: cudaq::qec::realtime::experimental::cpu_stage_context
   :members:

.. doxygentypedef:: cudaq::qec::realtime::experimental::cpu_stage_callback

.. doxygenvariable:: cudaq::qec::realtime::experimental::DEFERRED_COMPLETION


Completion
----------

.. doxygenstruct:: cudaq::qec::realtime::experimental::completion
   :members:

.. doxygentypedef:: cudaq::qec::realtime::experimental::completion_callback


Ring Buffer Injector
--------------------

.. doxygenclass:: cudaq::qec::realtime::experimental::ring_buffer_injector
   :members:


Pipeline
--------

.. doxygenclass:: cudaq::qec::realtime::experimental::realtime_pipeline
   :members:

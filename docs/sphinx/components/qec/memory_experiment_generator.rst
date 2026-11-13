.. _memory_experiment_generator:

Memory Experiment Configuration Generator
=========================================

``memory-experiment-generator`` prepares a
:doc:`dynamic detector error model <dynamic_dem>` and writes one YAML file
for a Surface Code memory experiment with a requested finite horizon using the
public ``parallel_window`` configuration. It supports X/Z bases and independent
logical patches. The decoding-server Docker image includes the generator;
there is no need to clone or build the library to generate a configuration.
Inside the image, run:

.. code-block:: bash

   memory-experiment-generator \
     --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
     --output=memory.yaml

To keep the YAML on the host, mount an existing output directory when starting
the container. For example, with ``DECODING_IMAGE`` set to the decoding-server
image tag for your CUDA version:

.. code-block:: bash

   docker run --rm --volume "$PWD:/configs" \
     --entrypoint memory-experiment-generator "$DECODING_IMAGE" \
     --distance=3 --rounds=26 --basis=Z --decoder=pymatching \
     --output=/configs/memory.yaml

``--rounds=T`` records ``dem_chunks.num_rounds=T+1``: T extraction cycles plus
terminal data-qubit readout. At least four cycles and an odd distance of at least
three are required. Characterization uses bounded circuits independent of T.
The parallel-window ingest contract determines actual shot length at FINAL;
the repeating DEM accepts valid lengths at or above its minimum rather than
enforcing the recorded horizon. The generated YAML
embeds a shared dynamic DEM, one graph per independent patch and BULK/FINAL
syndrome mappings. The DEM embeds the readout map, and the generator writes
``output: corrections``, so each graph returns its corrections through
``get_corrections``. ``--pattern-arg=output=corrections_and_logical`` also
wires the graph's ``logical`` output, whose corrected logical measurements
``get_logical_measurements`` returns.
``--help`` lists geometry, decoder, pattern and per-graph worker options.
See :doc:`/examples_rst/qec/task_graph_parallel_window` for how the
``parallel_window`` graph decodes and its options.

``--workers-per-graph=N`` sets the graph's worker-pool size and the pattern node's
``max_in_flight`` field to N, giving each A/B layer the same capacity. The layers
share the graph's N workers.

The generator is a standalone tool and requires ``--output=PATH``. It has no
default filename or implicit output location. Generation never overwrites an
existing file or symlink. Wait for successful completion,
then pass the file explicitly to ``decoding_server --config=PATH``. The server
executes the ``parallel_window`` graph and closes each shot through the emitted
``final`` mapping. Exit status is zero for successful generation or help
and one for an error. A forced termination may leave a
partial output file, which must be removed before retrying.

.. important::

   Official release archives omit both the standalone server and generator;
   use the decoding-server image for this workflow.

See Also
--------

* :ref:`decoding-server-memory-configuration` — supported profiles and input mappings
* :ref:`Playback Against a Decoding Server <playback-server-example>` — generate a configuration and replay a shot against the server

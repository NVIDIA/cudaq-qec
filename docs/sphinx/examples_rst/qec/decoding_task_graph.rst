Decoding Task Graph
===================

These examples show how to build and run decoding task graphs, then use
parallel windows and ensembles for more specialized pipelines. For the
concepts and configuration reference, see
:doc:`Decoding Task Graphs </components/qec/decoding_task_graph>`.

* :doc:`getting_started_decoding_task_graph`: construct a graph, decode
  complete shots, stream measurement rounds, and combine corrections.
* :doc:`Parallel Window <task_graph_parallel_window>`: distribute
  overlapping decoding windows across workers.
* :doc:`Decoder Ensembles <task_graph_ensemble>`: run several member graphs
  and select a result.

.. toctree::
   :hidden:
   :maxdepth: 1

   Getting Started with a Decoding Task Graph <getting_started_decoding_task_graph>
   Parallel Window <task_graph_parallel_window>
   Decoder Ensembles <task_graph_ensemble>

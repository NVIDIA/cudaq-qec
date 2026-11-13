Working with Detector Error Models
==================================

These examples show how to construct a detector error model (DEM), sample from
it, and use it for decoding. For the underlying concepts, see
:doc:`Detector Error Models </components/qec/detector_error_models>`.

* :doc:`stim_dem`: construct a decoder from Stim DEM text and predict
  observable flips.
* :doc:`multi_round_pcm`: generate a parity check matrix for multiple
  measurement rounds.
* :doc:`DEM Sampling <dem_sampling>`: sample synthetic errors and syndromes
  on the CPU or GPU.
* :doc:`dyn_dem`: construct a DEM from reusable phases and decode a
  measurement record with a task graph.

.. toctree::
   :hidden:
   :maxdepth: 1

   Decoding From Stim DEM Text <stim_dem>
   Generating a Multi-Round Parity Check Matrix <multi_round_pcm>
   DEM Sampling <dem_sampling>
   Decoding with a Dynamic DEM <dyn_dem>

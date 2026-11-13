.. _detector_error_model:

Detector Error Models
=====================

A detector error model (DEM) describes how faults in a quantum circuit flip
detectors and logical observables. For a complete circuit, CUDA-Q QEC provides
:class:`~cudaq_qec.DetectorErrorModel` in Python and
:cpp:class:`cudaq::qec::detector_error_model` in C++. Functions such as
``dem_from_memory_circuit()`` construct a DEM from a QEC circuit and a noise
model. For circuit-level noise, the DEM can be put into a canonical form
organized by measurement rounds, making it suitable for multi-round decoding.

The parity check matrix a decoder consumes is derived from the DEM: each row
is a detector and each column a possible error mechanism. For a runnable
example that generates a DEM from a surface code and decodes with it, see
:doc:`Experiments and Noise Modeling </examples_rst/qec/modeling_noise>`.

.. toctree::
   :hidden:
   :maxdepth: 2

   Decoding from Stim DEM Text <stim_dem>
   DEM Sampling <dem_sampling>
   Dynamic DEM <dynamic_dem>

The following guides cover different ways to use detector error models:

* .. _decoding-from-stim-dem-text:

  :doc:`Decoding from Stim DEM Text <stim_dem>` explains how to construct a
  decoder from an existing Stim DEM and how its fault information is parsed.

* .. _dem-sampling:

  :doc:`DEM Sampling <dem_sampling>` explains how to generate synthetic errors
  and syndromes from a DEM to test a decoder, using CPU or GPU sampling.

* .. _dynamic-dem:
  .. _dynamic-dem-construction:

  :doc:`Dynamic DEM <dynamic_dem>` explains how reusable phases describe
  repeating error-correction cycles, preserve effects across round boundaries,
  and support fixed-length shots or streaming decoding.

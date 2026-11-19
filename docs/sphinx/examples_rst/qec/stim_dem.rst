.. _stim_dem_text_example:

Decoding From Stim DEM Text
===========================

This example constructs a decoder from raw Stim ``.dem`` text and uses the matching parsed matrix for observable predictions. For what a detector error model is and how the text is parsed, see :ref:`Decoding from Stim DEM text <decoding_from_stim_dem_text>`.

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/stim_dem_decoder.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/stim_dem_decoder.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec -lcudaq-qec-decoders stim_dem_decoder.cpp -o stim_dem_decoder
      ./stim_dem_decoder

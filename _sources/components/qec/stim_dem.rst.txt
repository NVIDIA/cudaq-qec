.. _decoding_from_stim_dem_text:

Decoding from Stim DEM Text
============================

A detector error model (DEM) does not have to be produced inside CUDA-Q.
Decoders can be constructed from either a parity-check matrix or raw Stim
DEM text, which is useful when the DEM is already available in Stim's
``.dem`` format — from a saved file, a Stim workflow, or CUDA-Q DEM generation.

For native decoders, raw DEM text is authoritative:
:cpp:class:`cudaq::qec::decoder_init` retains the text and exposes its parsed
``H``, ``O``, and error rates, and custom parameters cannot override those
model inputs. Python-registered decoders cannot consume ``decoder_init``
directly, so the binding instead passes the parsed ``H`` and supplies
DEM-derived ``O`` and ``error_rate_vec`` keyword defaults when the Python
caller did not provide them.

By default, ``get_decoder(..., dem_text)`` and
``dem_from_stim_text(dem_text)`` parse with ``use_decomp_suggestions=False`` —
Stim ``^`` decomposition hints are ignored and each ``error(...)`` instruction
becomes one matrix column; passing ``use_decomp_suggestions=True`` splits
``^``-separated components into separate columns.

For a runnable example, see
:ref:`Decoding from Stim DEM Text <stim_dem_text_example>`.

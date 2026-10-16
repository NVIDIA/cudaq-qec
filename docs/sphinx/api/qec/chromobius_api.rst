.. class:: chromobius

    A decoder for color codes built on the open-source
    `Chromobius <https://github.com/quantumlib/chromobius>`_ Möbius decoder.
    Unlike the matrix-based decoders, Chromobius is *detector-error-model native*:
    it is constructed directly from Stim detector-error-model (DEM) text and
    predicts logical observable flips directly.

    .. note::
      To use the decoder, use the `get_decoder` API with the Stim DEM text as
      the decoder input:

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec

            with open("color_code.dem") as f:
                dem_text = f.read()

            dec = qec.get_decoder("chromobius", dem_text)
            corrections = dec.decode(syndrome)  # predicted observable flips

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/decoder.h"

            std::string dem_text = /* Stim detector error model text */;
            cudaqx::heterogeneous_map params;
            auto dec = cudaq::qec::get_decoder("chromobius", dem_text, params);

    .. note::
      Chromobius is DEM-native: constructing it from a parity-check matrix is
      rejected with an error. Use ``get_decoder("chromobius", dem_text, params)``.
      The DEM must describe a color code whose errors carry the color/basis
      annotations that Chromobius decomposes.

    .. note::
      The `"chromobius"` decoder implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface for
      C++. The wrapper currently returns observable flips as a 64-bit mask, so at
      most 64 logical observables are supported; ``decode()`` returns one bit per
      observable.

    :param dem_text: Stim detector error model, as text. Detectors become the
                     decoder syndrome bits; observables become the returned
                     ``block_size`` correction bits.
    :param params: Heterogeneous map of parameters (all optional and boolean),
                   forwarded to the Chromobius decoder configuration:

        - `drop_mobius_errors_involving_remnant_errors` (bool)
        - `ignore_decomposition_failures` (bool)
        - `include_coords_in_mobius_dem` (bool)
        - `return_weight` (bool): When true, the match weight is returned in the
          decode result's ``opt_results`` under the key ``"weight"``.
        - `write_mobius_match_to_stderr` (bool)

.. class:: pymatching

    A minimum-weight perfect matching (MWPM) decoder for matchable quantum error
    correction codes (such as the surface code), built on the open-source
    `PyMatching <https://github.com/oscarhiggott/PyMatching>`_ library. It is a
    CPU decoder: each syndrome bit becomes a detector node, and each error (column
    of the parity-check matrix) with one or two set entries becomes a (boundary)
    edge whose weight is derived from the error prior.

    .. warning::
      **Breaking change:** Supplying an observable matrix ``O`` no longer
      selects observable output. ``O`` is model data and may be used even when
      the requested result is an error frame. PyMatching returns errors when
      ``output`` is omitted. Pass ``output="observables"`` in Python or
      :cpp:enumerator:`cudaq::qec::decode_result_type::observables` in C++ to
      request observable flips explicitly.

    .. note::
      To use the decoder, use the `get_decoder` API with a parity-check matrix
      as the decoder input:

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np

            # Parity check matrix. Each column (error mechanism) must have one
            # or two set entries so the graph is matchable.
            H = np.array([[1, 1, 0],
                          [0, 1, 1]], dtype=np.uint8)
            O = np.array([[1, 0, 1]], dtype=np.uint8)

            # O does not change the default: this decoder returns an error
            # frame with one entry per column of H.
            error_dec = qec.get_decoder(
                "pymatching", H, O=O,
                error_rate_vec=[0.1, 0.1, 0.1],
                merge_strategy="smallest_weight")

            # Request observable flips explicitly.
            observable_dec = qec.get_decoder(
                "pymatching", H, O=O, output="observables",
                error_rate_vec=[0.1, 0.1, 0.1],
                merge_strategy="smallest_weight")

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/decoder.h"

            cudaqx::tensor<uint8_t> H;
            const std::vector<uint8_t> h_data{1, 1, 0, 0, 1, 1};
            H.copy(h_data.data(), {2, 3});
            cudaqx::tensor<uint8_t> O;
            const std::vector<uint8_t> o_data{1, 0, 1};
            O.copy(o_data.data(), {1, 3});

            cudaqx::heterogeneous_map params;
            params.insert("merge_strategy", std::string("smallest_weight"));
            std::vector<double> error_rates{0.1, 0.1, 0.1};
            cudaq::qec::decoder_init inputs(
                cudaq::qec::sparse_binary_matrix(H),
                cudaq::qec::sparse_binary_matrix(O),
                error_rates);

            // O is present, but omitting the result type still requests the
            // PyMatching default: an error frame.
            auto error_dec = cudaq::qec::get_decoder(
                "pymatching", inputs, params);

            // Observable output is explicit and fixed at construction.
            auto observable_dec = cudaq::qec::get_decoder(
                "pymatching", inputs,
                cudaq::qec::decode_result_type::observables, params);

    .. note::
      The `"pymatching"` decoder implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface for
      C++, so it supports all the methods in those respective classes.

    :param H: Parity check matrix. Each column must have one or two set entries
              (matchable graph). In Python, a ``scipy.sparse`` matrix or a dense
              NumPy ``uint8`` array may be passed.
    :param params: Decoder model inputs and heterogeneous parameters:

        - `error_rate_vec` (vector<double>): Per-error prior probabilities, one
          per column of ``H`` (length ``block_size``). Python accepts this as a
          keyword. C++ supplies it through :cpp:class:`cudaq::qec::decoder_init`,
          not the heterogeneous parameter map. Each value must lie in
          ``(0, 0.5]`` and sets the matching edge weight ``-log(p / (1 - p))``.
          When omitted, all edge weights default to ``1.0``.
        - `merge_strategy` (string): How to combine parallel edges that map to
          the same pair of detectors. One of ``"disallow"`` (default for the
          ``H``-only path), ``"independent"``, ``"smallest_weight"``,
          ``"keep_original"``, or ``"replace"``.
        - `O` (tensor, optional): A ``num_observables x block_size`` binary
          matrix used to project error frames into observable flips. Its
          presence does not select the result basis. Request observable output
          explicitly with ``output="observables"`` in Python or
          :cpp:enumerator:`cudaq::qec::decode_result_type::observables` in C++.
          When observable output is requested, ``merge_strategy`` defaults to
          ``"independent"`` to match PyMatching's detector-error-model
          construction. Supplying ``O`` while requesting error output leaves
          the default as ``"disallow"``.

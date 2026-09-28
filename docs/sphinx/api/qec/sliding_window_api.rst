.. class:: sliding_window

    The Sliding Window Decoder is a wrapper around a standard decoder that
    introduces two key differences:
    
    1. **Sliding Window Decoding**: The decoding process is performed
    incrementally, one window at a time.  The window size is specified by the
    user. This allows decoding to begin before all syndromes have been received,
    potentially reducing overall latency in multi-round QEC codes.
    
    2. **Partial Syndrome Support**: Unlike standard decoders, the
    :code:`decode` function (and its variants like :code:`decode_batch`) can
    accept partial syndromes. If partial syndromes are provided, the return
    vector will be empty, the decoder will not complete the processing and
    remain in an intermediate state, awaiting future syndromes. The return
    vector is only non-empty once enough data has been provided to match the
    original syndrome size (calculated from the Parity Check Matrix).

    The cancellable :code:`decode` and :code:`decode_batch` overloads forward
    the token to the inner decoders. They return the same empty result while a
    block is incomplete, and :code:`std::nullopt` if an inner decoder honors a
    stop; the partial block is dropped and the next round starts a new one.
    :code:`reset_decoder()` also drops a partial block.
    
    Sliding window decoders are advantageous in QEC codes subject to
    circuit-level noise across multiple syndrome extraction rounds. These
    decoders permit syndrome processing to begin before the complete syndrome
    measurement sequence is obtained, potentially reducing the overall decoding
    latency. However, this approach introduces a trade-off: the reduction in
    latency typically comes at the cost of increased logical error rates.
    Therefore, the viability of sliding window decoding depends critically on
    the specific code parameters, noise model, and latency requirements of the
    system under consideration.
    
    Sliding window decoding imposes only a single structural constraint on the
    parity check matrices: each syndrome extraction round must produce a
    constant number of syndrome measurements. Notably, the decoder makes no
    assumptions about temporal correlations or periodicity in the underlying
    noise process.

    **Streaming Syndrome Interface**
    
    For real-time applications, the decoder provides an ``enqueue_syndrome()``
    method that accepts syndrome data one round at a time. This allows the host
    to feed syndrome measurements as they arrive without waiting for all rounds
    to complete. The decoder automatically manages internal buffering and triggers
    window decodes at appropriate boundaries.
    
    References:
    `Toward Low-latency Iterative Decoding of QLDPC Codes Under Circuit-Level Noise <https://arxiv.org/abs/2403.18901>`_

    .. note::
      It is required to create decoders with the `get_decoder` API from the CUDA-QX
      extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq
            import cudaq_qec as qec
            import numpy as np

            cudaq.set_target('stim')
            num_rounds = 5
            code = qec.get_code('surface_code', distance=num_rounds)
            noise = cudaq.NoiseModel()
            noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.001), 1)
            statePrep = qec.operation.prep0
            dem = qec.dem_from_memory_circuit(code, statePrep, num_rounds, noise)
            inner_decoder_params = {'use_osd': True, 'max_iterations': 50}
            opts = {
                'error_rate_vec': np.array(dem.error_rates),
                'window_size': 1,
                'num_syndromes_per_round': code.get_num_z_stabilizers() + code.get_num_x_stabilizers(),
                'num_boundary_syndromes': code.get_num_z_stabilizers(),
                'inner_decoder_name': 'single_error_lut',
                'inner_decoder_params': inner_decoder_params,
            }
            swdec = qec.get_decoder('sliding_window', dem.detector_error_matrix, **opts)

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/code.h"
            #include "cudaq/qec/decoder.h"
            #include "cudaq/qec/experiments.h"
            #include "common/NoiseModel.h"

            int main() {
                // Generate a Detector Error Model.
                int num_rounds = 5;
                auto code = cudaq::qec::get_code(
                    "surface_code", cudaqx::heterogeneous_map{{"distance", num_rounds}});
                cudaq::noise_model noise;
                noise.add_all_qubit_channel("x", cudaq::depolarization2(0.001), 1);
                auto statePrep = cudaq::qec::operation::prep0;
                auto dem = cudaq::qec::dem_from_memory_circuit(*code, statePrep, num_rounds,
                                                                noise);
                // Use the DEM to create a sliding window decoder.
                auto inner_decoder_params =
                    cudaqx::heterogeneous_map{{"use_osd", true}, {"max_iterations", 50}};
                auto opts = cudaqx::heterogeneous_map{
                    {"window_size", 1},
                    {"num_syndromes_per_round", code->get_num_z_stabilizers() + code->get_num_x_stabilizers()},
                    {"num_boundary_syndromes", code->get_num_z_stabilizers()},
                    {"inner_decoder_name", "single_error_lut"},
                    {"inner_decoder_params", inner_decoder_params}};
                auto swdec = cudaq::qec::get_decoder("sliding_window",
                                                    cudaq::qec::decoder_init{dem}, opts);

                return 0;
            }

    .. note::
      The `"sliding_window"` decoder implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    :param H: Parity check matrix (tensor format)
    :param params: Heterogeneous map of parameters:

        - `error_rate_vec` (vector<double>): Vector of length ``block_size``
          containing the probability of each error. Python accepts this as a
          keyword. C++ supplies it through :cpp:class:`cudaq::qec::decoder_init`
          (for example, by constructing it from the DEM as above), not the
          heterogeneous parameter map. It is sliced to each window before the
          inner decoder is constructed.
        - `window_size` (int): The number of rounds of syndrome data in each window. (Defaults to 1.)
        - `step_size` (int): The number of rounds to advance the window by each time. (Defaults to 1.)
        - `num_syndromes_per_round` (int): The number of syndromes per round. (Must be provided.)
        - `num_boundary_syndromes` (int): The number of detectors in the first and last
          (boundary) round of the memory experiment. (Defaults to 0, meaning all layers
          have `num_syndromes_per_round` detectors.) For a single-basis DEM from
          :func:`~cudaq_qec.z_dem_from_memory_circuit` (respectively
          :func:`~cudaq_qec.x_dem_from_memory_circuit`), every layer has
          ``code.get_num_z_stabilizers()`` (respectively
          ``code.get_num_x_stabilizers()``) detectors, so this may be left at its
          default. For a full DEM from :func:`~cudaq_qec.dem_from_memory_circuit`,
          ``num_syndromes_per_round`` is ``code.get_num_z_stabilizers() +
          code.get_num_x_stabilizers()`` while the boundary rounds only carry the
          stabilizer type fixed by the state prep, so set this to
          ``code.get_num_z_stabilizers()`` for Z-basis preps (``prep0``/``prep1``) or
          ``code.get_num_x_stabilizers()`` for X-basis preps (``prepp``/``prepm``).
        - `straddle_start_round` (bool): When forming a window, should error
          mechanisms that span the start round and any preceding rounds be included? (Defaults to False.)
        - `straddle_end_round` (bool): When forming a window, should error
          mechanisms that span the end round and any subsequent rounds be included? (Defaults to True.)
        - `inner_decoder_name` (string): The name of the inner decoder to use.
        - `inner_decoder_params` (Python dict or C++ `heterogeneous_map`): A
          dictionary of parameters to pass to the inner decoder.

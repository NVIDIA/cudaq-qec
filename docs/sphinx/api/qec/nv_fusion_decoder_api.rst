.. class:: nv_fusion_decoder

    A multi-threaded minimum-weight perfect matching (MWPM) decoder based on
    the NV Fusion Brickwall algorithm.  It is in essence a combination of
    `fusion blossom <https://arxiv.org/abs/2305.08307>`_ with
    `sparse blossom <https://arxiv.org/abs/2303.15933>`_: the detector matching
    graph is partitioned into temporal blocks that are solved independently and
    then fused across their boundaries (fusion blossom), while each individual
    block is solved by PyMatching's sparse blossom implementation.  Blocks
    and the fuses between them form a dependency DAG that is dispatched to a
    worker pool as syndrome data arrives, so the decoder is designed to
    maximize parallel processing utilization and reach minimal latency in a
    streaming, realtime decoding environment.  Offline batch decoding is
    supported through the same scaffold.

    Thread count is set by ``num_threads``, which defaults to ``1``; pass
    ``0`` to size the pool from the hardware.  That pool parallelizes the
    blocks and fuses *within* one shot; it does not make a decoder callable
    from several threads at once.

    .. important::
      **One decoder decodes one shot at a time.**  Both ``decode()`` and the
      realtime enqueue path keep per-shot state on the decoder -- syndrome
      routing, block and fuse matching state, and the herald flags -- so two
      threads sharing a decoder corrupt each other.  Use one decoder per
      thread.

      This applies to ``decode_async()``, which runs ``decode()`` on a new
      thread: keeping two futures outstanding on the *same* decoder is not
      supported.  ``decode_batch()`` is sequential and therefore safe.

      Concurrent ``decode()`` calls on one nv-fusion decoder raise
      ``std::logic_error`` rather than interleaving, so the misuse surfaces as
      an exception instead of a wrong answer.

    The decoder is **graphlike**: every error mechanism must touch exactly
    one (boundary edge) or two (bulk edge) detectors.  Constructing from
    ``H``, that is a constraint on its columns; constructing from a DEM,
    Stim's ``^`` decomposition suggestions are applied first, so a hyperedge
    is accepted only when the DEM suggests a graphlike decomposition for it.
    Block size is set by ``block_leaf_size``, blocks overlap so
    that fusing adjacent pairs propagates information across their shared
    boundary, and the observable corrections that come out are
    XOR-accumulated into a running Pauli frame.

    .. important::
      A **single-leaf** schedule is one block with no fuses, and is exact
      monolithic MWPM.  Any **fused** schedule is a windowed approximation:
      the brickwall has two fuse layers, so each block's corrections are
      decided within a window of at most four leaves (two for the first and
      last blocks) rather than against the whole shot.  An error chain longer
      than that window can be matched differently than monolithic MWPM would
      match it, and the result is not always flagged -- the heralded-failure
      check fires on unresolved virtual-boundary matches, which an undersized
      window does not reliably produce.

      Sizing the leaf from the code distance is what keeps the window large
      enough for this to be safe; see ``block_leaf_size`` below.

    References:

    - `Sparse Blossom: correcting a million errors per core second with minimum-weight matching <https://arxiv.org/abs/2303.15933>`_
    - `Fusion Blossom: Fast MWPM Decoders for QEC <https://arxiv.org/abs/2305.08307>`_
    - `Quantum error correction below the surface code threshold <https://arxiv.org/abs/2408.13687>`_

    .. note::
      It is required to create decoders with the ``get_decoder`` API from the
      CUDA-Q QEC extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np

            # Two boundary edges carrying one observable each, plus a timelike
            # edge between the detectors carrying none.
            H = np.array([[1, 0, 1],              # rows = detectors
                          [0, 1, 1]], dtype=np.uint8)
            O = np.array([[1, 0, 0],              # rows = observables
                          [0, 1, 0]], dtype=np.uint8)
            # int32, one round index per detector
            opts = {"detector_round": np.array([0, 1], dtype=np.int32)}
            decoder = qec.get_decoder('nv-fusion-decoder', H, O=O,
                                      output='observables', **opts)
            decoder.decode([1.0, 0.0])            # -> [1.0, 0.0]

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/decoder.h"

            // Two boundary edges carrying one observable each, plus a
            // timelike edge between the detectors carrying none.
            cudaqx::tensor<uint8_t> H, O;
            std::vector<uint8_t> H_vec = {1, 0, 1,   // rows = detectors
                                          0, 1, 1};
            std::vector<uint8_t> O_vec = {1, 0, 0,   // rows = observables
                                          0, 1, 0};
            H.copy(H_vec.data(), {2, 3});
            O.copy(O_vec.data(), {2, 3});

            cudaqx::heterogeneous_map opts;
            opts.insert("detector_round", std::vector<int32_t>{0, 1});
            cudaq::qec::decoder_init inputs(
                cudaq::qec::sparse_binary_matrix(H),
                cudaq::qec::sparse_binary_matrix(O));
            auto decoder = cudaq::qec::get_decoder(
                "nv-fusion-decoder", std::move(inputs),
                cudaq::qec::decode_result_type::observables, opts);
            decoder->decode({1.0, 0.0});          // -> {1.0, 0.0}

    .. note::
      The ``"nv-fusion-decoder"`` implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    :param H: Parity-check matrix (sparse binary matrix or dense
      ``tensor<uint8_t>``), shape ``(num_detectors, num_error_mechanisms)``.
      Each column must have exactly one or two non-zero rows (graphlike error
      mechanisms). Pass Stim DEM text in this position instead to construct
      from an authoritative DEM, which carries ``H``, ``O``, and error rates.
    :param O: Optional observable matrix with shape
      ``(num_observables, num_error_mechanisms)``. It is model data and does
      not select the result basis. Python accepts it as a keyword adapter; C++
      supplies it through :cpp:class:`cudaq::qec::decoder_init`.
    :param output: Result basis, ``"errors"`` or ``"observables"`` in Python
      and :cpp:enum:`cudaq::qec::decode_result_type` in C++. Supplying ``O`` or
      DEM text does not select observable output.
    :param error_rate_vec: Optional physical error probability per error
      mechanism, with length ``H.num_cols()`` and values in ``(0, 0.5]``.
      Python accepts it as a keyword adapter; C++ supplies it through
      :cpp:class:`cudaq::qec::decoder_init`. The H construction path uses it
      to compute log-likelihood-ratio edge weights; otherwise unit weights are
      used. A DEM already carries its own probabilities.
    :param params: Heterogeneous map of parameters:

        .. note::
          These are the C++ and Python parameters.  The realtime YAML surface
          is narrower: ``decoder_custom_args`` accepts only ``num_threads``,
          ``block_leaf_size``, ``fusion_strategy`` and ``error_rate_vec``, and
          rejects any other key.  A YAML configuration supplies the matrices
          and the temporal layout through the top-level ``H_sparse``,
          ``O_sparse`` and ``D_sparse`` fields instead.

        **Temporal layout (one of the following is required, unless a DEM
        supplies detector coordinates):**

        - ``detector_round`` (``vector<int32_t>``): Per-detector temporal
          round index, length ``H.num_rows()``.  Maps each parent detector to
          its round in 0-based integer coordinates.  Takes highest priority
          over all automatic derivation paths.

        - ``D`` in :cpp:class:`cudaq::qec::decoder_init`: Sparse
          measurement-to-detector map with shape ``(num_detectors,
          num_measurements)``. The decoder infers the per-detector round from
          the column stride of the two-entry (timelike) rows. Rows with more
          than two entries are treated as terminal boundary detectors placed
          in the last round. The realtime configuration's top-level
          ``D_sparse`` field is normalized into this model input.

        The three sources are consulted in that order: an explicit
        ``detector_round`` first, then DEM coordinates, then ``D``. A DEM
        therefore takes precedence over ``D``, and one missing a detector
        coordinate fails rather than falling back to it. Without a DEM and
        with neither input given, construction is rejected.

        **Blocking and threading:**

        - ``block_leaf_size`` (uint64): Number of temporal rounds per leaf
          block.  Smaller values reduce per-block latency but increase fuse
          overhead; larger values amortize fuse cost at the expense of
          latency.  An explicit value is always honored as given; if it is
          below the measured code distance and the schedule fuses more than
          one block, construction logs a warning, because such a leaf cannot
          contain a logical error chain and inflates the logical error rate
          without raising a heralded failure.

          When omitted, the schedule is selected automatically:

          - **192 rounds or fewer** -- a single leaf spanning the shot, which
            is exact monolithic MWPM.
          - **More than 192 rounds** -- a fused schedule with a leaf of
            ``2 * d``, where ``d`` is the code distance measured from the
            matching graph as the shortest undetectable logical error.  This
            bounds tail latency, which under a single leaf grows linearly with
            the number of rounds.
          - **Distance not measurable** (a model with no observables, or none
            admitting an undetectable logical error) -- a single leaf at any
            shot length, since fusing on an unverified leaf height inflates the
            logical error rate silently.

          .. warning::
            The second case trades exactness for latency, and it is the
            default.  A shot longer than 192 rounds is decoded by the windowed
            approximation described above, not by monolithic MWPM.  A leaf of
            ``2 * d`` puts four times the margin over the smallest leaf
            measured to be safe (``0.5 * d``), and over d=5..21 on rotated
            surface and repetition codes it reproduced single-leaf corrections
            shot for shot -- but that is a measured result on those codes and
            noise models, not a guarantee for every code.

            To keep exact monolithic MWPM at any shot length, set
            ``block_leaf_size`` to the shot's full round count explicitly.

          This differs from earlier releases, where omitting
          ``block_leaf_size`` always produced a single leaf.

        - ``num_threads`` (uint64): Number of CPU threads to use during
          scaffold construction and parallel fuse operations.  Defaults to
          ``1``.

        **Fusion schedule:**

        - ``fusion_strategy`` (str): Fusion schedule to use.  Currently
          only ``"brickwall"`` is supported.

    **Decode result format:**

    ``decode()`` returns a :class:`~cudaq_qec.DecoderResult` where:

    - ``result`` — the explicitly selected result basis. Observable output has
      length ``num_observables`` and each entry indicates a predicted logical
      flip. Error output has length ``block_size`` and each entry indicates
      whether the corresponding H column was selected as a matching edge.

    - ``converged`` — ``True`` if the fuse pass raised no herald flag.  A
      ``False`` value indicates a suspect or ambiguous match; the correction
      is still populated and may be used.

    - ``opt_results`` — heterogeneous map with key:

      - ``heralded`` (bool): Raw herald flag from the brickwall fuse pass,
        corresponding to ``converged = not heralded``.

    **Realtime streaming:**

    The decoder supports the base-class realtime API
    (``enqueue_syndrome``, ``get_obs_corrections``, ``reset_decoder``,
    ``clear_corrections``).  The realtime path accepts raw per-round
    *measurement* bits rather than pre-differenced detector bits; the
    decoder accumulates measurements internally and computes detector events
    via ``D_sparse`` as each detector round's measurements become complete.
    Observable corrections are XOR-accumulated into a running Pauli frame
    and exposed via ``get_obs_corrections()``.

    .. note::
      ``D`` must be supplied in :cpp:class:`cudaq::qec::decoder_init` before
      the first call to ``enqueue_syndrome()``. The realtime layer normalizes
      its top-level ``D_sparse`` configuration into that model input.

    .. note::
      Observable corrections accumulate across shots.  Call
      ``clear_corrections()`` or ``reset_decoder()`` between shots to
      reset the Pauli frame.  ``reset_decoder()`` also rewinds the
      streaming session; ``clear_corrections()`` does not.

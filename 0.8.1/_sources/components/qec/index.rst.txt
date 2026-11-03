CUDA-Q QEC - Quantum Error Correction Library
=============================================

The ``cudaq-qec`` library provides a comprehensive framework for quantum
error correction research and development. It leverages GPU acceleration
for efficient syndrome decoding and error correction simulations (coming soon).

The library supports both offline analysis and realtime error correction on quantum hardware,
enabling low-latency decoding for practical quantum computing applications.

``cudaq-qec`` is composed of three main interfaces:

1. **QEC Codes** (:code:`cudaq::qec::code`) - Define quantum error correcting codes with logical operations
2. **Decoders** (:code:`cudaq::qec::decoder`) - Implement syndrome decoding algorithms
3. **Realtime Decoding** (:code:`cudaq::qec::decoding`) - Enable online error correction on quantum hardware

These types are meant to be extended by developers to provide new error correcting codes and decoding strategies.

The pages below document each of these interfaces in depth — the abstraction, how to extend it, and what ships built in. For runnable, copy-pasteable programs, see the :doc:`CUDA-Q QEC examples </examples_rst/qec/examples>`.

.. toctree::
   :maxdepth: 2

   QEC Codes <codes>
   QEC Decoders <decoders>
   Realtime Decoding <realtime_decoding>
   Experiments and Noise Modeling <numerical_experiments>
   Conventions <conventions>

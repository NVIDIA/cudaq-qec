.. _edem_composition_example:

EDEM Composition and Sampling
-----------------------------

An extended detector error model (EDEM) describes a gadget's fault responses
and its input and output code boundaries. The ``cudaq-edem`` library composes
these models by connecting compatible boundaries. A complete experiment with
no exposed code boundaries can be projected to a detector error model (DEM)
and converted to CUDA-Q QEC's :cpp:struct:`cudaq::qec::detector_error_model`.

This C++ example composes preparation, three syndrome-extraction cycles (SEC),
and a final Z measurement:

.. code-block:: text

   prep -> SEC -> SEC -> SEC -> mz

The supplied gadgets are hand-authored linear models. Each occurrence has
one independent fault, with probability 0.01 for preparation, 0.02 for each
SEC, and 0.03 for readout. Reusing the SEC definition creates three distinct
fault occurrences.

Build and run
+++++++++++++

With CUDA-Q QEC installed, download
:download:`edem_monolithic.cpp <../../examples/qec/cpp/edem_monolithic.cpp>`,
:download:`edem_memory_gadgets.h <../../examples/qec/cpp/edem_memory_gadgets.h>`,
and :download:`edem_example_utils.h <../../examples/qec/cpp/edem_example_utils.h>`
into the same directory. When using the ``cudaq-edem`` wheel, add its native
headers and libraries to the compiler paths:

.. code-block:: bash

   EDEM_ROOT=$(python3 -c 'from pathlib import Path; import cudaq_edem; print(Path(cudaq_edem.__file__).resolve().parent)')
   nvq++ -std=c++20 -I"${EDEM_ROOT}/include" -L"${EDEM_ROOT}/lib" \
     -Wl,-rpath,"${EDEM_ROOT}/lib" \
     -lcudaq-qec -lcudaq-qec-decoders -lcudaq-edem -lcudaq-gf2 \
     edem_monolithic.cpp -o edem_monolithic
   ./edem_monolithic

The example prints ``is_DEM: true``, an H matrix with four detector rows and
five fault columns, and an O matrix with one observable row and five fault
columns. It generates 1,024 shots with seed 13 by default.

Compose, project, and sample
++++++++++++++++++++++++++++

The fixture's ``memory_experiment`` helper creates the occurrence list. The
library's ``compose_chain`` connects the gadgets, and ``cudaq_edem_is_dem``
checks that the resulting model has no exposed input or output code boundary.
Strict ``project_DEM`` requires this closure and preserves the original fault
columns and their exact laws.

:cpp:func:`cudaq::qec::dem_from_edem` copies the projected detector and
observable responses into QEC's H and O matrices and extracts the fault rates.
The CPU sampler returns detector and fault samples; multiplying the fault
samples by ``O.transpose()`` gives observable flips over GF(2).

.. literalinclude:: ../../examples/qec/cpp/edem_monolithic.cpp
   :language: cpp
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]
   :dedent: 4

``edem_example::check`` translates a failed native status into an exception.
``PackedEDEM`` owns the composed and projected buffers. Input views borrow
the gadget definitions during composition; the resulting owners and the
converted QEC model have independent lifetimes.

Using supplied gadget EDEMs
+++++++++++++++++++++++++++

To use packed EDEM files produced by ``cudaq-edem``, pass the preparation,
SEC, and readout paths, followed optionally by shot count and seed:

.. code-block:: bash

   ./edem_monolithic prep.edem sec.edem mz.edem 10000 42

Preparation must have no input code block and one output; SEC must have one
of each; readout must have one input and no output. Adjacent code
presentations must match. The example repeats the supplied SEC three times.

Fault-law conversion
++++++++++++++++++++

QEC conversion preserves every original fault column, including columns
with no detector response, and assigns its marginal probability to an
independent Bernoulli mechanism. This is exact for the independent faults
used here. Exclusive fault alternatives use an **independent marginal
approximation**: alternatives can co-occur after conversion, changing both
detector and observable distributions. The native EDEM and its projection
retain their exact laws.

Models with nonempty success outputs are rejected by QEC conversion because
it has no success or postselection policy. The generated observable values
are flips relative to the ideal readout.

For the conversion interface, see :ref:`edem_conversion_cpp_api`. For more
sampling examples, see :ref:`dem_sampling_example`.

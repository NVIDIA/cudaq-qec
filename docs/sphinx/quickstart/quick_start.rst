Quick Start
===========

Check your installation by running a small offline decoding experiment. You
will construct a decoder, calculate syndromes for known bit flips, and verify
that applying its corrections recovers the original bits.

Before You Start
----------------

Complete the :doc:`Installation Guide <installation>`, then use the same Python
environment or container shell for the commands below. This example uses the
built-in CPU look-up table decoder and NumPy. It needs no GPU, QPU, or optional
decoder packages.

Run the Experiment
------------------

A three-bit repetition code represents a bit as ``000`` or ``111`` and can
correct one bit flip. Its two parity checks compare neighboring bits: a check
returns ``1`` when those bits disagree. The pair of check results is the
*syndrome* supplied to the decoder.

Start from ``000`` and try four cases: no error and a single bit flip at each
position. The decoder receives only the syndrome; it does not receive the
injected error. After decoding, the script applies the suggested correction
and checks that the result is ``000``.

Save the following as ``quick_start.py``, or
:download:`download the script <../examples/qec/python/quick_start.py>`:

.. literalinclude:: ../examples/qec/python/quick_start.py
   :language: python
   :start-after: [Begin Documentation]

Run it from the directory where you saved the file:

.. code-block:: bash

   python3 quick_start.py

Expected Result
---------------

The script prints:

.. code-block:: text

   error=[0, 0, 0]  syndrome=[0, 0]  corrected=[0, 0, 0]
   error=[1, 0, 0]  syndrome=[1, 0]  corrected=[0, 0, 0]
   error=[0, 1, 0]  syndrome=[1, 1]  corrected=[0, 0, 0]
   error=[0, 0, 1]  syndrome=[0, 1]  corrected=[0, 0, 0]
   Installation check passed: all 4 cases recovered.

Import-time notices from CUDA-Q may appear before these lines.

The final line confirms that Python loaded CUDA-Q QEC, constructed the decoder,
and recovered all four cases. This checks the basic CPU decoding path; GPU
decoders and realtime integration have their own requirements.

Next Steps
----------

See :doc:`QEC Decoders </components/qec/decoders>` to choose a decoder for your
workload. For experiments with sampled noise, continue with
:doc:`Experiments and Noise Modeling </examples_rst/qec/modeling_noise>`.

:tocdepth: 1

.. _qldpc_noise_learning:

Noise Learning
==============

A detector error model (DEM) describes which detectors and logical observables
each error mechanism can flip. Decoding also needs the probability of each
mechanism. On a device, these probabilities may be unknown or change over time;
the rates assumed in a simulation may no longer describe the measured data.

Noise learning estimates these probabilities from recorded detector samples.
It needs the DEM's mechanism structure, but no logical labels or known device
error rates. The result is one learned probability per mechanism, ready to use
as a decoder prior. It estimates rates for existing mechanisms; it does not
discover mechanisms missing from the DEM.

CUDA-Q QEC provides GPU noise learning through
``cudaq_qec.plugins.nv_qldpc``. It uses a Trajectory-Calibrated BP (TCBP) model
to interpret belief-propagation information. Learning does not require Jittered
OSD Beam Search (JOBS): JOBS is one choice for the subsequent decoding step.
See :ref:`qldpc_tcbp_jobs` for the relationship between these components.

Noise learning is a CUDA-Q QEC feature. The
:ref:`standalone decoder <nv_qldpc_standalone>` includes TCBP model fitting
and decoding, but does not include ``learn_noise`` or the DEM utilities.

For the complete Python workflow, see :doc:`Learning Noise and Decoding with
Learned Rates </examples_rst/qec/noise_learning>`. For measured learning time,
rate accuracy, and decoding performance, see :ref:`noise_learning_user_guide`.

When to Use Noise Learning
--------------------------

Use noise learning when you have a detector error model and detector samples
from the corresponding experiment, but need to estimate the model's mechanism
probabilities. This separates constructing the possible fault mechanisms from
measuring how often they occur. The learned rates can then replace assumed
rates when comparing decoder performance on a separate set of shots.

The method can use recorded device data or samples from a simulator. The
example uses simulated samples so it can compare learned rates with the known
rates that generated the data. Those known rates are held aside for evaluation
and are not supplied to the learning step.

Inputs and Learning Workflow
----------------------------

Use a CUDA-Q QEC installation with the ``nv-qldpc-decoder`` plugin and a
compatible NVIDIA GPU. The Python learning functions need NumPy; the ``tying``
and ``p25`` helpers and command-line tools also need Stim.

The detector error matrix ``H`` has shape ``[m detectors, n mechanisms]``.
Each column records which detectors a mechanism flips. Detector samples have
shape ``[N shots, m detectors]``, with their columns in the same order as the
rows of ``H``. Samples use ``uint8`` values.

The workflow has two stages:

.. list-table::
   :header-rows: 1
   :widths: 22 38 20 20

   * - Function
     - Purpose
     - Input
     - Output
   * - ``init_tcbp_model``
     - Fit a TCBP model for a code family. Run once and reuse.
     - ``H``
     - TCBP model bytes; with ``return_report=True``, also a report
   * - ``learn_noise``
     - Estimate the probability of every mechanism in ``H``.
     - ``H``, detector samples, and the TCBP model
     - Rates ``[n]`` and a report

``init_tcbp_model`` draws its own training rates by default, so fitting the
TCBP model requires no information about the device's rates. Although it
accepts ``error_rate_vec``, do not use true device rates or rates learned from
the device's data when preparing a model for noise learning: the resulting
estimate would no longer be independent of those rates. Fit the model once
per code family and reuse it. See the Trajectory-Calibrated BP section of the
:ref:`API reference <nv_qldpc_decoder_api_python>` for fitting options and
recommendations.

``learn_noise`` takes no error rates. It updates its estimates over successive
passes through the detector data. ``passes`` limits the number of passes
(default 4); ``tol`` controls early stopping as the estimated rates stop
changing (default 0.1). Set ``tol=0`` to run exactly the requested number of
passes. More shots and passes increase the work; the
:ref:`performance study <noise_learning_user_guide>` measures how they affect
accuracy and runtime.

Choosing a Learning Mode
------------------------

The ``mode`` argument controls which rates are estimated independently. Sharing
information between related mechanisms reduces the amount of data needed, but
adds assumptions about the noise.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Mode
     - Use When
   * - ``"per_mechanism"`` (default)
     - You trust the mechanisms of ``H`` but know nothing about their rates.
       Learn one rate per column; this mode needs the most shots.
   * - ``"tied"``
     - Noise is unchanged from round to round. Repeated mechanisms share a
       rate, lowering the shot-noise floor. Supply ``class_ids``, for example
       from ``tying.round_classes(circuit).class_ids``.
   * - ``"parametric"``
     - The circuit follows the supported 25-parameter circuit-level noise
       model. Supply a ``design`` from ``p25.design_map``. Estimating these
       shared parameters can improve accuracy per shot when the assumption
       is appropriate.

Using and Checking the Learned Rates
------------------------------------

Pass the learned rates as ``error_rate_vec`` to an NVIDIA QLDPC decoder built
from the same detector error matrix. The array order matters: each rate must
still describe the corresponding column of ``H``. Evaluate decoding on
held-out shots so the measurements test how well the learned prior generalizes.

The example compares uniform, learned, and known rates using the same decoder
configuration and the same held-out shots. It uses TCBP with JOBS for decoding;
that choice is separate from learning the rates. If you reuse the TCBP model
for decoding, supply it as a writable one-dimensional ``uint8`` array or a
file path, and match its ``bp_method``, ``max_iterations``, ``clip_value``, and
``proc_float`` settings. See :ref:`qldpc_tcbp_jobs` for output modes and JOBS
requirements, and the :ref:`API reference <nv_qldpc_decoder_api_python>` for
decoder options.

The learning report includes per-pass changes, held-out detector statistics,
the starting rate, and wall time. A small change between passes indicates
that the estimates have stabilized; it does not by itself establish that the
detector error model describes the device correctly. Check the report and
evaluate decoding separately.

The same learning functions are available through command-line tools in
``cudaq_qec.plugins.nv_qldpc.cli``; each tool's ``--help`` lists its options.

Requirements and Limits
-----------------------

* ``H`` must be a two-dimensional ``uint8`` array or a ``scipy.sparse`` matrix.
  A Stim ``DetectorErrorModel`` or an ``(H, rates)`` tuple is rejected.
* A fitted TCBP model is required. Detector samples must match the ordering
  and number of detectors in ``H``.
* Only rates of mechanisms already in ``H`` are learned. Missing mechanisms
  and incorrect structural assumptions need separate investigation.
* Rates outside the model's training range are reported as
  ``rates_outside_training_range``. Set ``min_p`` and ``max_p`` when fitting
  the TCBP model if the expected rates lie outside the default range.
* Tied and parametric modes rely on their stated noise assumptions. Use the
  per-mechanism mode when those additional assumptions are not justified.

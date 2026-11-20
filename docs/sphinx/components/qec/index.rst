:tocdepth: 1

CUDA-Q QEC - Quantum Error Correction Library
=============================================

CUDA-Q QEC is a Python and C++ library for quantum error correction. It provides
tools to build and simulate quantum codes, model noise, and decode measurements
using CPU or GPU decoders. These tools support both offline experiments and
realtime decoding during quantum circuit execution.

.. _qec_v091_highlights:

Highlights in 0.9.1
-------------------

* :ref:`Relay BP <qldpc_decoder>` now processes groups of shots together on
  the GPU when using ``decode_batch()`` for offline decoding.
* :ref:`Jittered OSD Beam Search <qldpc_tcbp_jobs>` now supports models with
  more than 8192 detectors. The decoder selects the kernels automatically;
  the maximum detector count depends on the GPU's shared-memory capacity.

.. _qec_v090_highlights:

Highlights in 0.9.0
-------------------

* :ref:`Trajectory-Calibrated BP and Jittered OSD Beam Search
  <qldpc_tcbp_jobs>` expand the NVIDIA QLDPC decoder's GPU decoding pipeline.
  TCBP turns BP trajectories into calibrated error probabilities; JOBS uses
  parallel searches over alternative corrections and can report a per-shot
  confidence measure for post-selection.
* :ref:`Noise learning <qldpc_noise_learning>` estimates a detector error
  model's mechanism probabilities from recorded detector samples, without
  logical labels or known device error rates. Use the learned probabilities
  as decoder priors when the model's mechanisms are known but their rates
  need to be measured.
* :doc:`Decoding task graphs <decoding_task_graph>` let you connect
  decoders and combine their results in a single pipeline. Build pipelines with
  parallel windows or decoder ensembles to decode complete shots or
  process measurement rounds as they arrive.
* :doc:`Dynamic detector error models (DEMs) <dynamic_dem>` now describe how
  measurements form detectors and how logical observables are read out,
  alongside the existing fault information. Task graphs use these mappings to
  return corrected logical measurements as well as correction bits.
* The :doc:`playback emulator </components/qec/playback_emulator>` lets you test
  realtime decoders without a live QPU. Replay recorded or simulated measurements
  on a timed schedule, check the returned corrections, and measure latency,
  either within one process or through a :ref:`decoding server <decoding-server>`.

C++ decoder plugin APIs changed in 0.9.0; see the
:ref:`migration guide <qec_decoder_plugin_migration>` when updating a plugin.

What the Library Provides
-------------------------

* **Codes and models:** Define :doc:`QEC codes <codes>`, their logical
  operations, and stabilizer-measurement circuits. Use
  :ref:`detector error models <detector_error_model>` to describe how faults
  affect detectors and logical observables.
* **CPU and GPU decoders:** Choose from built-in :doc:`decoders <decoders>` for
  different codes and workloads, or integrate your own through the decoder
  plugin interface.
* **Decoding task graphs:** Connect decoders into a
  :doc:`task graph <decoding_task_graph>`, distribute work across
  :doc:`parallel windows </examples_rst/qec/task_graph_parallel_window>`, or run
  :doc:`decoder ensembles </examples_rst/qec/task_graph_ensemble>`.
* **Realtime integration:** Connect your quantum-control application to
  :doc:`realtime decoding <realtime_decoding>` so it can send measurements and
  retrieve corrections during circuit execution.
* **Decoding server and playback:** Deploy a configured decoding pipeline in a
  standalone :ref:`server <decoding-server>`. Use the
  :doc:`playback emulator </components/qec/playback_emulator>` to test decoder
  behavior and timing, either locally or through the server.
* **Experiments and noise modeling:** Run :doc:`experiments
  <numerical_experiments>` to simulate code-capacity or circuit-level noise,
  sample detector error models, and estimate logical error rates.
* **Noise learning:** Estimate a detector error model's mechanism
  probabilities from recorded detector samples with :doc:`noise learning
  <noise_learning>`, then use those estimates as decoder priors.

Explore the :doc:`examples </examples_rst/qec/examples>` for working programs
and the :doc:`performance studies </performance/index>` for benchmark results.
The :doc:`conventions guide <conventions>` explains syndrome and correction
formats.

.. rubric:: Library Guides

* :doc:`QEC Decoders <decoders>`
* :doc:`Detector Error Models <detector_error_models>`
* :doc:`Decoding Task Graphs <decoding_task_graph>`
* :doc:`Realtime Decoding <realtime_decoding>`
* :doc:`Experiments and Noise Modeling <numerical_experiments>`
* :doc:`Noise Learning <noise_learning>`
* :doc:`QEC Codes <codes>`
* :doc:`Conventions <conventions>`

Performance Studies
===================

In-depth performance studies of CUDA-Q QEC decoders on NVIDIA GPUs -- measuring
decode latency, logical error rate, and the trade-offs behind decoder tuning knobs.

* :ref:`Gamma ensembles <ensemble_gamma_user_guide>` -- how ensembling Relay BP gamma
  trajectories narrows the decode-latency tail, improving the logical error rate under
  hard decode deadlines by up to **~89x** on bivariate-bicycle codes (measured on a
  single GB200 with CUDA-Q QEC 0.7.0).
* :ref:`Relay solution recording <relay_solutions_user_guide>` -- how recording every
  Relay BP convergence replaces a per-``stop_nconv`` sweep of full decode runs with one
  recording run plus offline post-processing, reproducing every RelayBP-N result exactly.
* :ref:`Min-LLR OSD initialization <minllr_osd_user_guide>` -- how
  ``osd_init_method="min_llr"`` lets a 10-iteration BP+OSD decoder beat a 60-iteration
  one by up to **~100x** in logical error rate on joint-XYZ circuit-level DEMs of
  bivariate-bicycle codes, and makes correlated (joint XYZ) BP+OSD decoding more accurate
  than uncorrelated (split X/Z) decoding.
* :ref:`Trajectory-Calibrated BP and JOBS <tcbp_jobs_user_guide>` -- how TCBP
  turns BP trajectories into calibrated soft error probabilities, how JOBS
  uses those scores in a GPU-parallel search over jittered OSD orderings, and
  how the search exposes complementary logical-class information.
* :ref:`NV-Fusion latency <nv_fusion_latency_user_guide>` -- how overlapping decoding
  with syndrome acquisition bounds the final-round response latency of long-running
  surface-code experiments.
* :ref:`Noise learning <noise_learning_user_guide>` -- how the rates of all error mechanisms
  of a circuit-level noise model are learned from detector data alone, and how the logical
  error rate of TCBP-JOBS decoding with learned rates compares with uniform and oracle rates:
  on the ``[[144,12,12]]`` bivariate-bicycle code with ``si1000`` noise the observed logical error
  rate with learned rates is **1.00x to 1.05x** that of the oracle and up to **~30x** lower
  than with a uniform prior (measured on a single GB200).

.. toctree::
   :maxdepth: 1

   Improving Relay BP Decoding With Gamma Ensembles <nv_qldpc_gamma_ensemble_user_guide>
   Sweeping Relay BP Stopping Criteria From a Single Run <nv_qldpc_relay_solutions_user_guide>
   Improving BP+OSD Decoding With Min-LLR OSD Initialization <nv_qldpc_minllr_osd_user_guide>
   Trajectory-Calibrated BP and Jittered OSD Beam Search <nv_qldpc_tcbp_jobs_user_guide>
   Bounding Real-Time Decode Latency With NV-Fusion <nv_fusion_latency_user_guide>
   Learning Noise From Detector Data and Decoding With the Learned Rates <nv_qldpc_noise_learning_user_guide>

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
* :ref:`NV-Fusion latency <nv_fusion_latency_user_guide>` -- how overlapping decoding
  with syndrome acquisition bounds the final-round response latency of long-running
  surface-code experiments.

.. toctree::
   :maxdepth: 1

   Improving Relay BP Decoding With Gamma Ensembles <nv_qldpc_gamma_ensemble_user_guide>
   Sweeping Relay BP Stopping Criteria From a Single Run <nv_qldpc_relay_solutions_user_guide>
   Improving BP+OSD Decoding With Min-LLR OSD Initialization <nv_qldpc_minllr_osd_user_guide>
   Bounding Real-Time Decode Latency With NV-Fusion <nv_fusion_latency_user_guide>

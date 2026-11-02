# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line front ends of the in-plugin noise learning (python -m cudaq_qec.plugins.nv_qldpc.cli.<command>):

    init_tcbp_model         circuit.stim --out surface.tcbp          # init_tcbp_model() -> TCBP model (once per code family)
    learn_dem_rates         circuit.stim --tcbp-model surface.tcbp [--dets samples_Z.dets --metadata metadata_Z.json]
    learn_tied_rates        circuit.stim --tcbp-model surface.tcbp [...] [--boundary-layers 2] [--round-sizes 72 144 ...]
    learn_physical_params   circuit.stim --tcbp-model surface.tcbp --tag-circuit tag.stim [--yaml noise_model.yaml] | --print-tags

Without ``--dets`` the device is simulated from the circuit's own DEM (stim), and the circuit's rates are then the truth
the reported error is measured against. A port of the reference package's CLI over the plugin (numpy + stim only).
"""

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Command-line front ends of the in-plugin noise learning (python -m cudaq_qec.plugins.nv_qldpc.cli.<command>):

    calibrate               circuit.stim --out surface.cal           # calibrate() -> calibration data (once per code family)
    learn_dem_rates         circuit.stim --calibration surface.cal [--dets samples_Z.dets --metadata metadata_Z.json]
    learn_tied_rates        circuit.stim --calibration surface.cal [...] [--boundary-layers 2] [--round-sizes 72 144 ...]
    learn_physical_params   circuit.stim --calibration surface.cal --tag-circuit tag.stim [--yaml noise_model.yaml] | --print-tags

Without ``--dets`` the device is simulated from the circuit's own DEM (stim), and the circuit's rates are then the truth
the reported error is measured against. A port of the reference package's CLI over the plugin (numpy + stim only).
"""

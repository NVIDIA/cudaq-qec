# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import argparse

# Use argument parser to get the filename
parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    help="The filename of the QIR output from the surface code circuit")
parser.add_argument("--distance",
                    type=int,
                    help="The distance of the surface code")
args = parser.parse_args()

# Read the file, looking for one "OUTPUT INT ### i64" per shot, like this:
# OUTPUT	INT	26469089	i64
shots_i64 = []
with open(args.filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("OUTPUT	INT	"):
            shots_i64.append(int(line.split()[2]))

observables = []
num_corrections_x = []
num_corrections_z = []
for s in shots_i64:
    # Logical observable.
    bits_to_xor = [args.distance * i for i in range(args.distance)]
    observable = 0
    for b in bits_to_xor:
        observable ^= ((s >> b) & 1)
    observables.append(observable)
    # Number of corrections applied during the shot (should be max 1)
    num_corrections_x.append((s >> (args.distance * args.distance)) & 65535)
    num_corrections_z.append(s >> (args.distance * args.distance + 16))

print(f'shots_i64:         {shots_i64}')
print(f'observables:       {observables}')
print(f'num_corrections_x: {num_corrections_x}')
print(f'num_corrections_z: {num_corrections_z}')

# Assert that at least 90% of the observables are 0.
assert sum(observables) <= len(observables) * 0.1

# Assert that a correction was applied at most once per shot
for c in num_corrections_x:
    assert c <= 1
for c in num_corrections_z:
    assert c <= 1
# For distance 3, we expect at least 90% of the shots to have a correction applied.
if args.distance == 3:
    assert sum(num_corrections_x) >= len(shots_i64) * 0.9 or \
           sum(num_corrections_z) >= len(shots_i64) * 0.9

# If we made it this far, we passed.
print("Passed all assertions.")

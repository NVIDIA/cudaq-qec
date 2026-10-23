/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Link-and-load canary. main does not name the CUDA-Q stubs. The link fails
// when the Quantinuum library needs a symbol the stubs do not export. Running
// it loads that library, and startup calls the stubs the library actually
// uses.

int main() { return 0; }

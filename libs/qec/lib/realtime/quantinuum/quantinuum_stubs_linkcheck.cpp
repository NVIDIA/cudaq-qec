/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Link-and-load canary. main does not name the CUDA-Q stubs. The link fails
// when the Quantinuum library needs a symbol the stubs do not export. Running
// it loads that library, and startup calls the stubs the library actually
// uses.

int main() { return 0; }

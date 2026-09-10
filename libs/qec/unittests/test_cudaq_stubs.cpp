/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

extern "C" {
void __quantum__qis__x__ctl();
void __quantum__qis__y__ctl();
void __quantum__qis__z__ctl();
}

TEST(CudaqStubs, CallsTheThreeExportedQisCtlSymbols) {
  __quantum__qis__x__ctl();
  __quantum__qis__y__ctl();
  __quantum__qis__z__ctl();
}

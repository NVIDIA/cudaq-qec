/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../lib/realtime/quantinuum/quantinuum_decoding.h"

#include <gtest/gtest.h>

TEST(QuantinuumPrivate, PublicUi64CallsReachTheTestPrivateDso) {
  enqueue_syndromes_ui64(0, 1, 1, 0);
  EXPECT_EQ(get_corrections_ui64(0, 3, 1), 0x5u);
  reset_decoder_ui64(0);
}

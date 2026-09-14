/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExtraPayloadProvider.h"
#include "cudaq/qec/decoder_config_payload.h"

#include <gtest/gtest.h>
#include <string>

// The stub ExtraPayloadProvider captures the registered unique_ptr so the
// public publisher can be exercised without a CUDA-Q job.
TEST(DecoderConfigPayload, PublishRegistersProviderAndYaml) {
  cudaq::qec::publish_decoder_config_payload("known-yaml");
  ASSERT_EQ(cudaq::g_providers.size(), 1u);
  auto *provider = cudaq::g_providers.front().get();
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider->name(), "decoder");
  EXPECT_EQ(provider->getPayloadType(), "gpu_decoder_config");
  cudaq::RuntimeTarget target;
  EXPECT_EQ(provider->getExtraPayload(target), "known-yaml");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

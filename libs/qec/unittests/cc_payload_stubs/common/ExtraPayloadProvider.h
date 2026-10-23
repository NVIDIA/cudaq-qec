/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace cudaq {

struct RuntimeTarget {};

class ExtraPayloadProvider {
public:
  ExtraPayloadProvider() = default;
  virtual ~ExtraPayloadProvider() = default;
  virtual std::string name() const = 0;
  virtual std::string getPayloadType() const = 0;
  virtual std::string getExtraPayload(const RuntimeTarget &target) = 0;
};

inline std::vector<std::unique_ptr<ExtraPayloadProvider>> g_providers;

inline void
registerExtraPayloadProvider(std::unique_ptr<ExtraPayloadProvider> provider) {
  g_providers.push_back(std::move(provider));
}

inline const std::vector<std::unique_ptr<ExtraPayloadProvider>> &
getExtraPayloadProviders() {
  return g_providers;
}

} // namespace cudaq

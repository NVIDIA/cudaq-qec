/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file selection_strategy.h
/// @brief How an ensemble picks its answer, as a pluggable policy.

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/cancellation.h"
#include "cudaq/qec/candidate_sink.h"
#include "cudaq/qec/decoder_config_schema.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// @brief One instance serves one fan-out. `begin()` is called once per
/// run when it is armed, and takes the cancellation source and candidate sink
/// by reference, so that the strategy can request stops and poll candidates. On
/// the pushing member's thread and possibly concurrently, `offer()` is called
/// for every push, where a value commits the run to it and hard-stops the
/// other members; a strategy commits at most once per run and ignores every
/// candidate after that. `finish()` once every member has finished and nothing
/// committed, where a value commits and `std::nullopt` is a miss.
class selection_strategy {
public:
  virtual ~selection_strategy() = default;
  virtual void begin(cancellation_source /*cancel*/, candidate_sink /*sink*/) {}
  virtual std::optional<logical_outcome> offer(const candidate &c) = 0;
  virtual std::optional<logical_outcome> finish() { return std::nullopt; }
};

/// The decoder-schema registry name a strategy's parameter schema is
/// registered under, namespaced like `task_graph_node_schema_name()` so a
/// strategy name is never taken for a decoder.
inline std::string selection_strategy_schema_name(const std::string &name) {
  return "selection_strategy." + name;
}

using selection_strategy_factory =
    std::function<std::unique_ptr<selection_strategy>(
        const cudaqx::heterogeneous_map &)>;

/// @brief One registration, released when destroyed: the factory is removed
/// if it is still the current one under its name, so a module that holds this
/// at namespace scope unregisters its strategy before its code is unloaded.
/// The parameter schema stays, as every registered schema does.
class __attribute__((visibility("default"))) selection_strategy_registration {
public:
  selection_strategy_registration(std::string name, std::uint64_t id)
      : name_(std::move(name)), id_(id) {}
  selection_strategy_registration(selection_strategy_registration &&o) noexcept
      : name_(std::exchange(o.name_, {})), id_(o.id_) {}
  ~selection_strategy_registration();

private:
  std::string name_;
  std::uint64_t id_;
};

/// @brief Register (or replace) the strategy `name`: its parameter schema,
/// registered under `selection_strategy_schema_name(name)`, and the factory
/// that builds it from parameters validated against that schema.
[[nodiscard]]
__attribute__((visibility("default"))) selection_strategy_registration
register_selection_strategy(std::string name,
                            decoding::config::decoder_schema params,
                            selection_strategy_factory factory);

/// @brief Build the strategy registered under `name` from its parameters,
/// which have been validated against the strategy's schema.
/// @throws std::invalid_argument for an unknown name.
__attribute__((visibility("default"))) std::unique_ptr<selection_strategy>
make_selection_strategy(const std::string &name,
                        const cudaqx::heterogeneous_map &params);

/// @brief Names of the registered strategies, sorted.
__attribute__((visibility("default"))) std::vector<std::string>
registered_selection_strategies();

} // namespace cudaq::qec

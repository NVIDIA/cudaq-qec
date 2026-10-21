/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "decoding_task_edge_value.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cudaq::qec {

namespace {
/// Assign into a payload that already holds the registered type, so an edge
/// value can be refilled in place instead of reallocated.
struct edge_value_ops {
  void (*copy_assign)(std::any &, const std::any &);
  void (*move_assign)(std::any &, std::any &);
};
} // namespace

/// Assignment hooks for one registered C++ type; names live in the alias map.
struct edge_value_descriptor {
  edge_value_type type;
  edge_value_ops ops;
};

namespace {

/// A registered name.  Counts its registrations so modules that share a
/// payload type can unregister independently, and keeps its own ops so the
/// descriptor never points into a module that has unloaded.
struct alias {
  edge_value_descriptor *desc;
  edge_value_ops ops;
  int owners = 1;
};

// Node-based so the descriptor addresses edge_value caches stay valid across
// later registrations, and intentionally leaked for the same reason the
// node-kind registry is: a type may be registered by a plugin that gets
// dlclose'd at exit, and no destructor of ours may run afterwards.
std::unordered_map<edge_value_type, edge_value_descriptor> &types() {
  static auto *r =
      new std::unordered_map<edge_value_type, edge_value_descriptor>();
  return *r;
}

std::map<std::string, alias, std::less<>> &aliases() {
  static auto *r = new std::map<std::string, alias, std::less<>>();
  return *r;
}

std::mutex &registry_mutex() {
  static auto *m = new std::mutex();
  return *m;
}

void add_registration(std::string name, edge_value_type type,
                      edge_value_ops ops) {
  if (name.empty())
    throw std::invalid_argument(
        "edge_value_registration: type name must not be empty");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto iter = aliases().find(name);
  if (iter != aliases().end()) {
    if (iter->second.desc->type != type)
      throw std::invalid_argument("edge_value_registration: '" + name +
                                  "' is registered for another type");
    ++iter->second.owners;
    return;
  }
  auto &desc =
      types().try_emplace(type, edge_value_descriptor{type, ops}).first->second;
  aliases().emplace(std::move(name), alias{&desc, ops});
}

void remove_registration(std::string_view name) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto iter = aliases().find(name);
  if (iter == aliases().end())
    return;
  if (--iter->second.owners > 0)
    return;
  auto *desc = iter->second.desc;
  aliases().erase(iter);
  auto survivor =
      std::find_if(aliases().begin(), aliases().end(), [&](const auto &entry) {
        return entry.second.desc == desc;
      });
  if (survivor == aliases().end())
    types().erase(desc->type);
  else
    desc->ops = survivor->second.ops;
}

} // namespace

/// The built-in payloads (decoding_task_types.h).
CUDAQ_REGISTER_EDGE_VALUE_TYPE(measurement_results, "measurement_results")
CUDAQ_REGISTER_EDGE_VALUE_TYPE(detection_events, "detection_events")
CUDAQ_REGISTER_EDGE_VALUE_TYPE(logical_outcome, "logical_outcome")
CUDAQ_REGISTER_EDGE_VALUE_TYPE(completion, "completion")

edge_value_type::edge_value_type(std::string_view name) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto iter = aliases().find(name);
  if (iter == aliases().end())
    throw std::invalid_argument("edge_value_type: '" + std::string(name) +
                                "' is not a registered edge-value type name");
  info_ = iter->second.desc->type.info_;
}

detail::edge_value_registration::edge_value_registration(
    std::string name, edge_value_type type,
    void (*copy_assign)(std::any &, const std::any &),
    void (*move_assign)(std::any &, std::any &))
    : name_(name) {
  add_registration(std::move(name), type, {copy_assign, move_assign});
}

detail::edge_value_registration::~edge_value_registration() {
  remove_registration(name_);
}

std::vector<edge_value_type> registered_edge_value_types() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  std::vector<edge_value_type> ids;
  ids.reserve(types().size());
  for (const auto &[type, desc] : types())
    ids.push_back(type);
  return ids;
}

const edge_value_descriptor *edge_value::descriptor_for(edge_value_type type) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto iter = types().find(type);
  if (iter == types().end())
    throw std::invalid_argument("edge_value: payload type was never passed to "
                                "register_edge_value_type");
  return &iter->second;
}

edge_value &edge_value::operator=(const edge_value &other) {
  if (this == &other)
    return *this;
  if (desc_ && desc_ == other.desc_) {
    desc_->ops.copy_assign(value_, other.value_);
    return *this;
  }
  desc_ = other.desc_;
  value_ = other.value_;
  return *this;
}

edge_value &edge_value::operator=(edge_value &&other) {
  if (this == &other)
    return *this;
  if (desc_ && desc_ == other.desc_) {
    desc_->ops.move_assign(value_, other.value_);
    return *this;
  }
  desc_ = std::exchange(other.desc_, nullptr);
  value_ = std::move(other.value_);
  other.value_.reset();
  return *this;
}

std::optional<edge_value_type> edge_value::type() const {
  return desc_ ? std::optional<edge_value_type>(desc_->type) : std::nullopt;
}

void edge_value::throw_type_mismatch() const {
  throw std::invalid_argument(
      desc_ ? "edge_value: does not hold the requested type"
            : "edge_value: is empty");
}

} // namespace cudaq::qec

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decoding_task_node_kind_registry.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

namespace cudaq::qec {

namespace decoding::config {
void unregister_task_graph_node_schema(const std::string &name);
}

namespace {

struct plugin_registration_state {
  bool accepting = false;
  bool rejected = false;
  bool committed = false;
  std::vector<std::pair<std::string, uint64_t>> registrations;
};

thread_local plugin_registration_state *active_plugin_registration = nullptr;

// Node-based maps so pointers handed out by find_task_graph_node_kind() stay
// valid across later registrations, and intentionally leaked for the same
// reason the decoder-schema registry is: a kind may be registered by a plugin
// that gets dlclose'd at exit, and no destructor of ours may run afterwards.
std::map<std::string, task_graph_node_kind_spec> &kind_registry() {
  static auto *registry =
      new std::map<std::string, task_graph_node_kind_spec>();
  return *registry;
}

/// alias -> canonical name
std::map<std::string, std::string> &alias_registry() {
  static auto *registry = new std::map<std::string, std::string>();
  return *registry;
}

std::map<std::string, node_factory> &factory_registry() {
  static auto *registry = new std::map<std::string, node_factory>();
  return *registry;
}

std::map<std::string, uint64_t> &registration_registry() {
  static auto *registry = new std::map<std::string, uint64_t>();
  return *registry;
}

std::recursive_mutex &registry_mutex() {
  static auto *m = new std::recursive_mutex();
  return *m;
}

uint64_t next_registration_id_locked() {
  // Zero denotes unconditional removal rather than a scoped registration.
  static uint64_t next_registration_id = 1;
  return next_registration_id++;
}

/// Canonical name for a name or alias; empty when nothing is registered.
/// Caller holds registry_mutex().
std::string canonical_locked(const std::string &name) {
  if (kind_registry().count(name))
    return name;
  auto iter = alias_registry().find(name);
  return iter == alias_registry().end() ? std::string{} : iter->second;
}

/// @brief Register node metadata and its schema while holding registry_mutex().
/// @param spec Public metadata describing the node kind.
/// @param params Parameter schema registered for the node kind.
/// @return Identity used to prevent an old plugin from removing a replacement.
uint64_t
register_task_graph_node_kind_locked(task_graph_node_kind_spec spec,
                                     decoding::config::decoder_schema params) {
  std::string name;
  uint64_t registration_id = 0;
  if (active_plugin_registration) {
    auto &state = *active_plugin_registration;
    if (!state.accepting) {
      // Do not throw through a DSO static initializer. The loader observes the
      // refusal and closes the module without publishing any registration.
      state.rejected = true;
      return 0;
    }
    if (!canonical_locked(spec.name).empty())
      throw std::invalid_argument(
          "plugin cannot replace an existing node kind");
    std::vector<std::string> names{spec.name};
    for (const auto &alias : spec.aliases) {
      if (alias.empty() || !canonical_locked(alias).empty() ||
          std::find(names.begin(), names.end(), alias) != names.end())
        throw std::invalid_argument(
            "plugin cannot replace or repeat a node alias");
      names.push_back(alias);
    }
  }
  if (spec.name.empty())
    throw std::invalid_argument(
        "register_task_graph_node_kind: kind name must not be empty");
  if (spec.native_dem_chunks && (!spec.requires_dem || spec.requires_decoder))
    throw std::invalid_argument(
        "register_task_graph_node_kind: native chunks require a DEM and "
        "factory-owned decoder state");
  // The schema name is derived, not supplied, so a kind cannot register its
  // parameters under a name the config layer would not look them up by.
  params.name = task_graph_node_schema_name(spec.name);
  decoding::config::register_decoder_schema(std::move(params));

  for (const auto &alias : spec.aliases)
    alias_registry().insert_or_assign(alias, spec.name);
  name = spec.name;
  registration_id = next_registration_id_locked();
  kind_registry().insert_or_assign(name, std::move(spec));
  registration_registry().insert_or_assign(name, registration_id);
  if (active_plugin_registration)
    active_plugin_registration->registrations.emplace_back(name,
                                                           registration_id);
  return registration_id;
} // end - register_task_graph_node_kind_locked()

/// @brief Remove a current node-kind registration while its image is mapped.
/// @param name Canonical kind name or alias to unregister.
/// @param registration_id Expected owner, or zero for unconditional removal.
void unregister_task_graph_node_kind_locked(const std::string &name,
                                            uint64_t registration_id) {
  const std::string canonical = canonical_locked(name);
  const auto owner = registration_registry().find(canonical);
  if (canonical.empty() ||
      (registration_id != 0 && (owner == registration_registry().end() ||
                                owner->second != registration_id)))
    return;

  auto alias = alias_registry().begin();
  while (alias != alias_registry().end())
    if (alias->second == canonical)
      alias = alias_registry().erase(alias);
    else
      ++alias;

  // Destroy the factory first because std::function dispatches its destructor
  // through plugin code that becomes invalid as soon as dlclose returns.
  factory_registry().erase(canonical);
  kind_registry().erase(canonical);
  registration_registry().erase(canonical);
  decoding::config::unregister_task_graph_node_schema(
      task_graph_node_schema_name(canonical));
} // end - unregister_task_graph_node_kind_locked()

} // namespace

struct node_kind_plugin_registration::impl {
  node_kind_registry_lock lock{registry_mutex()};
  plugin_registration_state state;

  impl() {
    if (active_plugin_registration)
      throw std::logic_error("nested task-graph plugin loading is unsupported");
    active_plugin_registration = &state;
  }

  ~impl() {
    if (!state.committed)
      for (auto item = state.registrations.rbegin();
           item != state.registrations.rend(); ++item)
        unregister_task_graph_node_kind_locked(item->first, item->second);
    active_plugin_registration = nullptr;
  }
};

node_kind_plugin_registration::node_kind_plugin_registration()
    : pimpl_(std::make_unique<impl>()) {}
node_kind_plugin_registration::~node_kind_plugin_registration() = default;

bool node_kind_plugin_registration::initialization_rejected() const noexcept {
  return pimpl_->state.rejected;
}

void node_kind_plugin_registration::begin_registration() {
  if (pimpl_->state.rejected)
    throw std::logic_error("plugin registered from a static initializer");
  pimpl_->state.accepting = true;
}

bool node_kind_plugin_registration::ready_to_commit() const noexcept {
  return pimpl_->state.accepting && !pimpl_->state.rejected &&
         !pimpl_->state.registrations.empty();
}

void node_kind_plugin_registration::commit() noexcept {
  pimpl_->state.committed = true;
}

namespace {
bool may_unregister_locked(const std::string &name) {
  if (!active_plugin_registration)
    return true;
  const auto &entries = active_plugin_registration->registrations;
  const auto canonical = canonical_locked(name);
  if (active_plugin_registration->accepting &&
      std::any_of(entries.begin(), entries.end(), [&](const auto &entry) {
        return entry.first == canonical &&
               registration_registry().at(canonical) == entry.second;
      }))
    return true;
  active_plugin_registration->rejected = true;
  return false;
}
} // namespace

/// @brief Lock the node-kind registry across a multi-step pointer operation.
/// @return Recursive registry lock owned by the caller.
node_kind_registry_lock lock_node_kind_registry() {
  return node_kind_registry_lock(registry_mutex());
}

void register_task_graph_node_kind_v2(task_graph_node_kind_spec spec,
                                      decoding::config::decoder_schema params) {
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  register_task_graph_node_kind_locked(std::move(spec), std::move(params));
}

const task_graph_node_kind_spec *
find_task_graph_node_kind(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  const auto canonical = canonical_locked(name);
  const auto kind = kind_registry().find(canonical);
  if (canonical.empty())
    return nullptr;
  return kind == kind_registry().end() ? nullptr : &kind->second;
}

std::vector<std::string> registered_task_graph_node_kinds() {
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  std::vector<std::string> names;
  names.reserve(kind_registry().size());
  for (const auto &[name, spec] : kind_registry())
    names.push_back(name);
  return names;
}

void register_node_kind_v2(task_graph_node_kind_spec spec,
                           decoding::config::decoder_schema params,
                           node_factory factory) {
  const std::string name = spec.name;
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  if (register_task_graph_node_kind_locked(std::move(spec), std::move(params)))
    factory_registry().insert_or_assign(name, std::move(factory));
}

/// @brief Remove a node kind while plugin-owned callbacks remain callable.
/// @param name Canonical kind name or alias to unregister.
void unregister_task_graph_node_kind(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  if (may_unregister_locked(name))
    unregister_task_graph_node_kind_locked(name, 0);
} // end - unregister_task_graph_node_kind()

/// @brief Remove a node kind only while a scoped token still owns it.
/// @param name Canonical kind name or alias to unregister.
/// @param registration_id Identity assigned to the scoped registration.
void unregister_task_graph_node_kind_if_current(const std::string &name,
                                                uint64_t registration_id) {
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  if (may_unregister_locked(name))
    unregister_task_graph_node_kind_locked(name, registration_id);
}

/// @brief Register a plugin-owned node kind with an unload cleanup token.
/// @param spec Public metadata describing the node kind.
/// @param params Parameter schema registered for the node kind.
/// @param factory Factory that constructs nodes of this kind.
/// @return Move-only token that unregisters the kind when destroyed.
node_kind_registration
register_scoped_node_kind_v2(task_graph_node_kind_spec spec,
                             decoding::config::decoder_schema params,
                             node_factory factory) {
  std::string name = spec.name;
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  const uint64_t registration_id =
      register_task_graph_node_kind_locked(std::move(spec), std::move(params));
  if (!registration_id)
    return {};
  factory_registry().insert_or_assign(name, std::move(factory));
  return node_kind_registration(std::move(name), registration_id);
}

const node_factory *find_node_factory(const std::string &kind) {
  std::lock_guard<std::recursive_mutex> lock(registry_mutex());
  const auto canonical = canonical_locked(kind);
  if (canonical.empty())
    return nullptr;
  auto iter = factory_registry().find(canonical);
  return iter == factory_registry().end() ? nullptr : &iter->second;
}

} // namespace cudaq::qec

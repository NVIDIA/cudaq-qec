/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <any>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "cuda-qx/core/tuple_utils.h"
#include "cuda-qx/core/type_traits.h"

namespace cudaqx {

/// @brief A class that implements a heterogeneous map allowing string keys to
/// be mapped to any value type
class heterogeneous_map {
private:
  std::unordered_map<std::string, std::any> items;

  /// @brief Check if an std::any object can be cast to a specific type
  /// @tparam T The type to cast to
  /// @param t The std::any object to check
  /// @return true if castable, false otherwise
  template <typename T>
  bool isCastable(const std::any &t) const {
    try {
      std::any_cast<T>(t);
    } catch (...) {
      return false;
    }
    return true;
  }

public:
  // Type aliases for iterators
  using iterator = std::unordered_map<std::string, std::any>::iterator;
  using const_iterator =
      std::unordered_map<std::string, std::any>::const_iterator;

  // Iterator methods
  iterator begin() { return items.begin(); }
  iterator end() { return items.end(); }
  const_iterator begin() const { return items.begin(); }
  const_iterator end() const { return items.end(); }
  const_iterator cbegin() const { return items.cbegin(); }
  const_iterator cend() const { return items.cend(); }

  /// @brief Default constructor
  heterogeneous_map() = default;

  /// @brief Constructor from initializer list
  /// @param list The initializer list of key-value pairs
  heterogeneous_map(
      const std::initializer_list<std::pair<std::string, std::any>> &list) {
    for (auto &l : list)
      insert(l.first, l.second);
  }

  /// @brief Clear the map
  void clear() { items.clear(); }

  /// @brief Insert a key-value pair into the map
  /// @tparam T The type of the value
  /// @param key The key
  /// @param value The value
  template <typename T>
  void insert(const std::string &key, const T &value) {

    if constexpr (is_bounded_char_array<T>{}) {
      // Never insert a raw char array or char ptr,
      // auto conver to a string
      items.insert_or_assign(key, std::string(value));
    } else {
      items.insert_or_assign(key, value);
    }
  }

  /// @brief Insert a key-value pair into the map, moving the value in
  /// @tparam T The type of the value
  /// @param key The key
  /// @param value The value, moved into the map
  /// @note Only participates in overload resolution for rvalues; lvalues go to
  /// the copying overload above. Useful for large payloads where the extra
  /// deep copy is expensive.
  template <typename T, typename = std::enable_if_t<
                            !std::is_lvalue_reference_v<T> &&
                            !is_bounded_char_array<std::remove_cv_t<T>>::value>>
  void insert(const std::string &key, T &&value) {
    items.insert_or_assign(key, std::move(value));
  }

  /// @brief Get a value from the map
  /// @tparam T The type of the value to retrieve
  /// @param key The key of the value to retrieve
  /// @return The value associated with the key
  /// @throw std::runtime_error if the key is invalid or the type doesn't match
  template <typename T, typename KeyT,
            std::enable_if_t<std::is_convertible_v<KeyT, std::string>, int> = 0>
  const T get(const KeyT &key) const {
    const std::string keyStr{key}; // build once to avoid repeated conversions
    auto iter = items.find(keyStr);
    if (iter == items.end())
      throw std::runtime_error("Invalid key.");

    if (isCastable<T>(iter->second))
      return std::any_cast<T>(iter->second);

    // It may be that user has requested a value of
    // a type that is "related" to its actual type, e.g.
    // we have a value of type int, but request here is std::size_t.
    // Handle that case, by getting T's map of related types, and checking
    // if any of them are valid.
    using RelatedTypes = typename RelatedTypesMap<
        std::remove_cv_t<std::remove_reference_t<T>>>::types;
    std::optional<T> opt;
    cudaqx::tuple_for_each(RelatedTypes(), [&](auto &&el) {
      if (!opt.has_value() &&
          isCastable<std::remove_cv_t<std::remove_reference_t<decltype(el)>>>(
              iter->second))
        opt = std::any_cast<
            std::remove_cv_t<std::remove_reference_t<decltype(el)>>>(
            iter->second);
    });

    if (opt.has_value())
      return opt.value();

    // Can't find it, throw an exception
    throw std::runtime_error(
        "heterogeneous_map::get() error - Invalid type or key (" + keyStr +
        ").");

    return T();
  }

  /// @brief Get a value from the map, search for the value
  /// from any of the provided string keys
  /// @tparam T The type of the value to retrieve
  /// @param keys The keys to search for the desired value.
  /// @return The value associated with the key
  /// @throw std::runtime_error if the key is invalid or the type doesn't match
  template <typename T>
  const T get(const std::vector<std::string> &keys) const {
    for (auto &key : keys) {
      try {
        return get<T>(key);
      } catch (...) {
        // do nothing
      }
    }
    // Can't find it, throw an exception
    auto keyStr = std::accumulate(keys.begin(), keys.end(), std::string(),
                                  [](std::string ss, std::string s) {
                                    return ss.empty() ? s : ss + "," + s;
                                  });

    throw std::runtime_error(
        "heterogeneous_map::get(keys) error - Invalid keys (" + keyStr + ").");

    return T();
  }

  template <typename T>
  const T get(const std::vector<std::string> &keys,
              const T &defaultValue) const {
    for (auto &key : keys) {
      try {
        return get<T>(key);
      } catch (...) {
        // do nothing
      }
    }
    return defaultValue;
  }

  /// @brief Get a value from the map with a default value
  /// @tparam T The type of the value to retrieve
  /// @param key The key of the value to retrieve
  /// @param defaultValue The default value to return if the key is not found
  /// @return The value associated with the key or the default value
  template <typename T>
  const T get(const std::string key, const T &defaultValue) const {
    try {
      return get<T>(key);
    } catch (...) {
    }
    return defaultValue;
  }

  /// @brief Get the size of the map
  /// @return The number of key-value pairs in the map
  std::size_t size() const { return items.size(); }

  /// @brief Check if the map is empty
  /// @return true if the map is empty, false otherwise
  bool empty() const { return items.empty(); }

  /// @brief Check if the map contains a key
  /// @param key The key to check
  /// @return true if the key exists, false otherwise
  bool contains(const std::string &key) const {
    return items.find(key) != items.end();
  }
  bool contains(const std::vector<std::string> &keys) const {
    for (auto &key : keys)
      if (items.find(key) != items.end())
        return true;

    return false;
  }
};

} // namespace cudaqx

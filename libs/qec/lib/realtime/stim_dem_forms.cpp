/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// How the host reads a decoder's or task-graph node's Stim DEM before it is
// built: from a file, from a data_artifacts entry named in place of model
// text, and from a pinned entry's file, read once and checked against its
// sha256. configure_decoders(), the decoding server and the data artifact
// store all read models through here, so a configuration built in code is
// read as a parsed one is.

#include "stim_dem_forms.h"
#include "realtime_decoding.h"
#include "../decoding_task/decoding_task_data_artifacts.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::qec::decoding::host {

// Opens the file in binary, so a pinned sha256 hashes the file's own bytes.
std::string read_stim_dem_file(const std::filesystem::path &path,
                               std::string_view where) {
  // A directory opens as a stream and fails only when read, past the check
  // below.
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec))
    throw std::runtime_error(
        fmt::format("{} is not a readable file: {}", where, path.string()));
  // Binary, so the bytes read are the file's: a pinned sha256 hashes them.
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error(
        fmt::format("{} could not be opened: {}", where, path.string()));
  try {
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  } catch (const std::exception &error) {
    throw std::runtime_error(fmt::format("{} could not be read: {} ({})", where,
                                         path.string(), error.what()));
  }
}

// A decoder's own stim_dem_path, named so in errors.
std::string read_stim_dem(const std::filesystem::path &dem_path) {
  return read_stim_dem_file(dem_path, "stim_dem_path");
}

// A pin as sha256_hex() writes a hash: lowercase.
static std::string lowercase_pin(std::string pin) {
  std::transform(pin.begin(), pin.end(), pin.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return pin;
}

// A pin needs a file, as parsing requires, and is checked here again since a
// configuration built in code skips the parser.
stim_dem_read read_stim_dem_entry(const config::data_artifact &entry,
                                  const std::filesystem::path &base_dir) {
  const auto where = "data_artifacts." + entry.name;
  const auto pinned = lowercase_pin(entry.sha256);
  std::filesystem::path path(entry.stim_dem_path);
  stim_dem_read read;
  if (entry.stim_dem_path.empty() && !entry.sha256.empty())
    throw std::runtime_error(where + ": sha256 pins a file, so it needs a "
                                     "file-backed entry such as stim_dem_path");
  if (entry.stim_dem_path.empty()) {
    read.text = entry.stim_dem;
    if (!read.text.empty())
      read.sha256 = cudaq::qec::sha256_hex(read.text);
    return read;
  }
  if (path.is_relative())
    path = std::filesystem::absolute(base_dir / path).lexically_normal();
  read.text = read_stim_dem_file(path, where);
  read.sha256 = cudaq::qec::sha256_hex(read.text);
  if (!pinned.empty() && pinned != read.sha256)
    throw std::runtime_error(where + ": " + entry.stim_dem_path +
                             " has sha256 " + read.sha256 +
                             ", but the entry pins " + entry.sha256);
  return read;
} // end - read_stim_dem_entry()

// Hashes compare in lowercase, as lowercase_pin() writes a pin.
const stim_dem_read &pinned_read(const pinned_stim_dems &pinned,
                                 const config::data_artifact &entry) {
  const auto where = "data_artifacts." + entry.name;
  const auto it = pinned.find(entry.name);
  if (it == pinned.end())
    throw std::logic_error(where + " is pinned, but its text was not read "
                                   "with read_pinned_stim_dems()");
  if (it->second.sha256 != lowercase_pin(entry.sha256))
    throw std::logic_error(where + " pins sha256 " + entry.sha256 +
                           ", but its text was read against " +
                           it->second.sha256 +
                           "; read the pins of this configuration");
  return it->second;
}

pinned_stim_dems
read_pinned_stim_dems(const config::multi_decoder_config &config,
                      const std::filesystem::path &base_dir) {
  pinned_stim_dems pinned;
  for (const auto &entry : config.data_artifacts)
    if (!entry.sha256.empty())
      pinned[entry.name] = read_stim_dem_entry(entry, base_dir);
  return pinned;
}

config::multi_decoder_config
with_pinned_stim_dems(const config::multi_decoder_config &config,
                      const std::filesystem::path &base_dir) {
  return with_pinned_stim_dems(config, read_pinned_stim_dems(config, base_dir));
}

namespace {

// Resolves each decoder's and node's stim_dem that names a data_artifacts
// entry, as parsing does, so a configuration built in code names entries as
// a parsed one does. A parsed copy holds the entry's text or path, never a
// name, so it is left as it is. The entries are checked as parsing checks
// them, in name order, which code need not keep.
void resolve_stim_dem_names(config::multi_decoder_config &config) {
  auto artifacts = config.data_artifacts;
  std::sort(artifacts.begin(), artifacts.end(),
            [](const config::data_artifact &a, const config::data_artifact &b) {
              return a.name < b.name;
            });
  config::validate_data_artifacts(artifacts);
  for (auto &decoder : config.decoders) {
    const auto where = "decoder " + std::to_string(decoder.id);
    config::resolve_stim_dem(artifacts, decoder.stim_dem, decoder.stim_dem_path,
                             decoder.stim_dem_artifact, where);
    if (decoder.task_graph)
      for_each_node(decoder.task_graph->nodes, [&](auto &node) {
        config::resolve_stim_dem(artifacts, node.stim_dem, node.stim_dem_path,
                                 node.stim_dem_artifact,
                                 where + ", task_graph node '" + node.id + "'");
      });
  }
} // end - resolve_stim_dem_names()

} // namespace

// Names are resolved and pinned entries looked up in a copy, so a failure
// leaves the caller's configuration untouched. A copy names an entry only
// while its path is still the entry's: one edited after parsing is read as
// it stands.
config::multi_decoder_config
with_pinned_stim_dems(const config::multi_decoder_config &config,
                      const pinned_stim_dems &pinned) {
  auto out = config;
  resolve_stim_dem_names(out);
  // Each pinned entry's path and checked text, by name.
  std::map<std::string, std::pair<std::string, std::string>> by_name;
  const auto use_text = [&](std::string &text, std::string &path,
                            const std::string &artifact) {
    const auto it = by_name.find(artifact);
    if (it == by_name.end() || !text.empty() || path != it->second.first)
      return;
    text = it->second.second;
    path.clear();
  };
  for (const auto &entry : config.data_artifacts)
    if (!entry.sha256.empty())
      by_name[entry.name] = {entry.stim_dem_path,
                             pinned_read(pinned, entry).text};
  for (auto &decoder : out.decoders) {
    use_text(decoder.stim_dem, decoder.stim_dem_path,
             decoder.stim_dem_artifact);
    if (decoder.task_graph)
      for_each_node(decoder.task_graph->nodes, [&](auto &node) {
        use_text(node.stim_dem, node.stim_dem_path, node.stim_dem_artifact);
      });
  }
  return out;
} // end - with_pinned_stim_dems()

} // namespace cudaq::qec::decoding::host

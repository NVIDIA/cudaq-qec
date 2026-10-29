/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the rules a decoder's or task-graph node's Stim DEM keys
// share, so the parser and the host builder, which checks a configuration
// built in code again, name them alike; and how the host reads those models,
// defined in stim_dem_forms.cpp. Never included from a public header.

#pragma once

#include "realtime_decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::qec::decoding::config {

/// The whitespace that tells a data_artifacts name, which holds none, from
/// Stim model text, which always holds some: Stim separates targets with
/// spaces or tabs and instructions with line breaks. Other whitespace, such
/// as Unicode's, may appear in a name.
inline constexpr const char *name_breaks = " \t\r\n";

/// Checks data_artifacts entries each on its own and against the others, as
/// parsing does. @p list is sorted by name.
/// @throws std::runtime_error naming the first bad entry.
void validate_data_artifacts(const std::vector<data_artifact> &list);

/// Resolves a stim_dem that names an entry of @p artifacts as parsing does: a
/// value with none of name_breaks, which Stim model text always holds, moves
/// to @p artifact, and @p text or @p path becomes a copy of the entry's text
/// or path. @p where names the decoder or node in errors.
/// @throws std::runtime_error if no Stim DEM entry has that name, or if the
/// entry's file meets a @p path of the decoder's or node's own.
void resolve_stim_dem(const std::vector<data_artifact> &artifacts,
                      std::string &text, std::string &path,
                      std::string &artifact, const std::string &where);

/// The key a Stim DEM is given under: stim_dem_path when @p stim_dem_path
/// names a file, else stim_dem.
inline const char *stim_dem_key(const std::string &stim_dem_path) {
  return stim_dem_path.empty() ? "stim_dem" : "stim_dem_path";
}

/// The error for a decoder or node, named by @p where, that sets both
/// stim_dem and stim_dem_path.
inline std::runtime_error both_stim_forms(const std::string &where) {
  return std::runtime_error(where +
                            " sets both stim_dem and stim_dem_path; the model "
                            "comes in exactly one form, inline or by file");
}

} // namespace cudaq::qec::decoding::config

namespace cudaq::qec::decoding::host {

/// Read a Stim DEM file into the text a decoder is built from. @p where names
/// the key being resolved, since both the entry-level `stim_dem_path` and a
/// node's come through here and a bare path is not enough to place the error.
/// @throws std::runtime_error naming @p where if the file cannot be read.
std::string read_stim_dem_file(const std::filesystem::path &path,
                               std::string_view where);

/// Read the DEM a decoder's own `stim_dem_path` names.
std::string read_stim_dem(const std::filesystem::path &dem_path);

/// Read a data_artifacts entry's Stim DEM, hashing it once: its inline text,
/// or its file, resolved against @p base_dir as a decoder's is and checked
/// against the entry's pin. Other kinds of entry give no text and no hash.
/// @throws std::runtime_error naming the entry if its file cannot be read,
/// does not match its pin, or an inline entry is pinned.
stim_dem_read read_stim_dem_entry(const config::data_artifact &entry,
                                  const std::filesystem::path &base_dir);

/// What read_pinned_stim_dems() read for pinned entry @p entry. The read is
/// the entry's only while its hash is the entry's current pin: by name alone,
/// a map read for another configuration would pass bytes this pin does not
/// name, and a missing read would leave the file unchecked.
/// @throws std::logic_error naming the entry if @p pinned lacks it or read it
/// against another pin.
const stim_dem_read &pinned_read(const pinned_stim_dems &pinned,
                                 const config::data_artifact &entry);

/// Resolves each decoder's and node's stim_dem in @p config that names a
/// data_artifacts entry, as parsing does, so a configuration built in code
/// names entries as a parsed one does.
/// @throws std::runtime_error as parsing does for a bad entry or a name no
/// Stim DEM entry holds.
void resolve_stim_dem_names(config::multi_decoder_config &config);

/// Calls @p visit on every node of a node list, const or not, including those
/// inside ensemble members.
template <typename Nodes, typename Visit>
void for_each_node(Nodes &nodes, Visit &&visit) {
  for (auto &node : nodes) {
    visit(node);
    if (node.ensemble)
      for (auto &member : node.ensemble->members)
        for_each_node(member.nodes, visit);
  }
}

} // namespace cudaq::qec::decoding::host

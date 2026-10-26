/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the data artifact store, which holds the templates, Stim
// DEMs and window variants a configuration names, once each. Identity is
// content: each entry is keyed by the sha256 of its canonical bytes, so equal
// content from different decoders or nodes is one entry and one handle. The
// store is filled at configuration time and never evicted; lookups return
// handles and never build. Never included from a public header.

#pragma once

#include "decoding_task_decoding_window.h"
#include "decoding_task_experiment_template.h"
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::qec {

/// SHA-256 of @p bytes, as 64 lowercase hexadecimal digits.
std::string sha256_hex(std::string_view bytes);

/// The canonical bytes of a template: every field of @p spec, in declaration
/// order, with doubles by their bit patterns. Equal specs give equal bytes.
std::string canonical_bytes(const dem_chunks_spec &spec);

/// The canonical bytes of a window request: every field, as for a template.
std::string canonical_bytes(const window_request &request);

/// One store entry's provenance: its kind (dem_chunks, stim_dem or window),
/// and the size and sha256 of its canonical bytes. A window's canonical bytes
/// are its template's sha256, its request's and its variant key's.
struct data_artifact_record {
  std::string kind;
  std::size_t bytes = 0;
  std::string sha256;
  bool operator==(const data_artifact_record &) const = default;
};

/// Templates, Stim DEMs and window variants, each held once and shared.
class data_artifact_store {
public:
  /// The template with @p spec's content, built and recorded if new.
  /// @throws std::invalid_argument as experiment_template does.
  std::shared_ptr<const experiment_template>
  add_template(const dem_chunks_spec &spec);

  /// Records a Stim DEM's text, if new.
  void add_stim_dem(std::string_view text);

  /// As above, with @p sha256 the sha256_hex() of @p text, which a caller
  /// that has already hashed the text passes rather than hash it again.
  void add_stim_dem(std::string_view text, std::string sha256);

  /// Builds every variant of @p request on the store's template with
  /// @p context's content not held yet: one window per variant key, over
  /// the shot lengths that hold every key. A window holding the readout
  /// round carries m2o when the template has a readout map.
  /// @throws std::invalid_argument as window_from_dem_chunks() does.
  /// @throws std::logic_error if no template with @p context's content was
  /// added.
  void add_windows(const std::shared_ptr<const experiment_template> &context,
                   const window_request &request);

  /// The template with @p spec's content.
  /// @throws std::logic_error if it was not added.
  std::shared_ptr<const experiment_template>
  find_template(const dem_chunks_spec &spec) const;

  /// The window variant @p key of @p request on the template with
  /// @p context's content, without building it.
  /// @throws std::logic_error if it was not built at configuration time.
  std::shared_ptr<const decoding_window>
  find_window(const experiment_template &context, const window_request &request,
              const variant_key &key) const;

  /// Every entry, in the order it was added.
  const std::vector<data_artifact_record> &records() const { return records_; }

private:
  /// A template with its spec, kept to confirm a hash hit is the content.
  struct template_entry {
    dem_chunks_spec spec;
    std::shared_ptr<const experiment_template> handle;
  };
  /// A window variant with what it was built from, kept as for templates.
  struct window_entry {
    window_request request;
    variant_key key;
    std::shared_ptr<const decoding_window> handle;
  };
  /// The sha256 of the template this store holds with @p spec's content.
  /// @throws std::logic_error naming @p what if it holds no such template.
  std::string template_hash(const dem_chunks_spec &spec,
                            const std::string &what) const;

  std::map<std::string, template_entry> templates_;
  /// The sha256 of each Stim DEM held. A Stim DEM's bytes are its text, so a
  /// hash hit is the content and the text need not be kept.
  std::set<std::string> stim_dems_;
  std::map<std::string, window_entry> windows_;
  std::vector<data_artifact_record> records_;
};

} // namespace cudaq::qec

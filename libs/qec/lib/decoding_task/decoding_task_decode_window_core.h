/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the decode_window node's core, apart from the executor.
// It holds one model and one decoder for each variant of its layer, taken
// from the data artifact store, and fires any instance of a variant from
// ingest's view of its rows and the sends its receives name: the canonical
// workload, into a slot sized beforehand for every variant. Sends carry
// their rows' place in the shot, so a receive needs neither instance's
// model. Never included from a public header.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "decoding_task_data_artifacts.h"
#include "decoding_task_decoding_window.h"
#include "decoding_task_ingest_core.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cudaq::qec {

/// A send as a fire publishes it: its values, its definition in the sending
/// window's variant, whose rows count from that window's first row, and
/// where that row lies in the shot. A send the variant lacks has neither
/// values nor a definition.
struct window_send_value {
  std::span<const std::uint8_t> values;
  const window_send *send = nullptr;
  std::size_t first_row = 0;
};

/// One fire's results, held in its slot until the slot fires again.
struct window_fire {
  std::int64_t k = 0;
  bool fired = false; ///< False until the slot first fires
  round_range commit; ///< The commit zone's rounds: the frame's interval
  /// d after receiving: the window's own rows, then its imported ones.
  std::span<const std::uint8_t> detectors;
  /// The estimate, convergence, whether H e = d, and the outputs.
  const window_solution *solution = nullptr;
  std::vector<window_send_value> sends; ///< One per send of the request
};

/// Fires one layer's windows. Not thread-safe: the slots share one decoder
/// per variant.
class decode_window_core {
public:
  /// @param store The store holding every variant of @p request on
  /// @p context.
  /// @param context The experiment template.
  /// @param request The layer's request.
  /// @param decoder_name The inner decoder, one built per variant.
  /// @param params The inner decoder's parameters.
  /// @param slots The fires whose results may be held at once.
  /// @throws std::invalid_argument if @p slots is 0, or as
  /// make_window_decoder() does; std::logic_error if the store lacks a
  /// variant.
  decode_window_core(const data_artifact_store &store,
                     std::shared_ptr<const experiment_template> context,
                     window_request request, const std::string &decoder_name,
                     const cudaqx::heterogeneous_map &params,
                     std::size_t slots);

  /// Fire window view.k into @p slot: copy the view's rows from its ring,
  /// apply each receive, a delta to the window's own rows and a residual to
  /// its imported ones, then decode and commit. The view and every received
  /// row are checked first, so a rejected fire leaves the slot holding what
  /// it held; a decoder that fails leaves it empty. Allocates nothing
  /// beyond what the decoder's decode() does.
  /// @param receives One per receive port: the producer instance's send, or
  /// null when that instance does not exist.
  /// @throws std::invalid_argument if the slot does not exist, the view is
  /// of no variant of this layer or holds other rows than its variant, or a
  /// received send reaches a row the window does not take in its mode;
  /// std::out_of_range if the view's rounds are not in its ring.
  const window_fire &fire(std::size_t slot, const window_view &view,
                          std::span<const window_send_value *const> receives);

  /// The fire of window k still held in a slot, or null.
  const window_fire *find(std::int64_t k) const;
  /// The index of send @p id among the request's sends.
  /// @throws std::invalid_argument if the request has no such send.
  std::size_t send_index(const std::string &id) const;
  /// The layer's variants.
  std::size_t num_variants() const { return variants_.size(); }
  /// The slots.
  std::size_t num_slots() const { return fires_.size(); }

private:
  /// One variant's model, its decoder, and where each of the request's
  /// sends is among the model's, or none.
  struct variant {
    variant_key key;
    std::shared_ptr<const decoding_window> window;
    std::unique_ptr<decoder> dec;
    std::vector<std::size_t> send_of;
  };

  /// Throws std::invalid_argument for @p what.
  [[noreturn]] void fail(const std::string &what) const;

  std::shared_ptr<const experiment_template> context_;
  window_request request_;
  std::vector<variant> variants_;
  // Per slot, a workspace and a detector buffer for every variant.
  std::vector<std::vector<window_workspace>> workspaces_;
  std::vector<std::vector<std::vector<std::uint8_t>>> detectors_;
  std::vector<window_fire> fires_;
};

} // namespace cudaq::qec

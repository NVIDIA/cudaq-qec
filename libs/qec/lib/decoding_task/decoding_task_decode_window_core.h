/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the decode_window node's core, apart from the executor.
// It holds one model for each variant of its layer, taken from the data
// artifact store, with its decoders, and fires any instance of a variant
// from its rows, as ingest hands them over, and the sends its receives
// name: the canonical workload, into a slot sized beforehand for every
// variant. Fires into different slots may run at once. Sends carry their
// rows' place in the shot, so a receive needs neither instance's model.
// Never included from a public header.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "decoding_task_data_artifacts.h"
#include "decoding_task_decoding_window.h"
#include "decoding_task_window_kind_params.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cudaq::qec {

/// A send as a fire publishes it: its values, its definition in the sending
/// window's variant, whose rows count from that window's first row, and
/// where that row lies in the shot. A send whose rounds the window clips
/// away has no rows, and so no values.
struct window_send_value {
  std::span<const std::uint8_t> values;
  const window_send *send = nullptr;
  std::size_t first_row = 0;
};

/// A send as an edge carries it: a window_send_value whose values it owns,
/// so an executor may copy it into a join and keep it after the sending
/// slot fires again. Its definition lives in the sending window, which
/// the producing decode_window core holds for its life. It is the
/// edge-value type "window_send".
struct window_send_payload {
  std::vector<std::uint8_t> values;
  const window_send *send = nullptr;
  std::size_t first_row = 0;
};

/// Copy a fire's send @p from into @p to. Allocates nothing once @p to has
/// held as many values.
inline void copy_send(const window_send_value &from, window_send_payload &to) {
  to.values.assign(from.values.begin(), from.values.end());
  to.send = from.send;
  to.first_row = from.first_row;
}

/// @p payload as a receive takes it; valid while @p payload is.
inline window_send_value receive_of(const window_send_payload &payload) {
  return {payload.values, payload.send, payload.first_row};
}

/// One fire's results, held in its slot until the slot fires again.
struct window_fire {
  std::int64_t k = 0;
  bool fired = false;   ///< False until the slot first fires
  bool checked = false; ///< Whether verify_residual checked the solution
  round_range commit;   ///< The commit zone's rounds: the frame's interval
  /// d after receiving: the window's own rows, then its imported ones.
  std::span<const std::uint8_t> detectors;
  /// The estimate, convergence, whether H e = d, and the outputs.
  const window_solution *solution = nullptr;
  std::vector<window_send_value> sends; ///< One per send of the request
};

/// Fires one layer's windows. Fires into different slots may run at once,
/// each on its own thread: each slot has its own buffers and bulk decoder,
/// and a prefix or terminal variant, which only one window of a shot uses,
/// has a single decoder, so a window fires into one slot at a time: a fire
/// of a prefix or terminal variant while another decodes is rejected. While
/// a slot fires, held() may read other slots, but nothing may read or fire
/// that slot, and reset() may not run. A fire's sends are leased to the
/// fires that receive them: its slot must not fire again until they have
/// finished. The caller chooses the slots, so it knows which holds which
/// window, and orders all of this. decode_window_node uses one slot; the
/// executor runs copies of the node for concurrency.
class decode_window_core {
public:
  /// @param store The store holding every variant of @p request on
  /// @p context.
  /// @param context The experiment template.
  /// @param request The layer's request.
  /// @param decoder_name The inner decoder, one built per slot for the
  /// bulk and one for each other variant.
  /// @param params The inner decoder's parameters.
  /// @param slots The fires whose results may be held at once, each with
  /// its own bulk decoder, so decoder memory, a GPU decoder's device memory
  /// included, grows with it.
  /// @param verify What a fire does with a solution that does not satisfy
  /// its detectors.
  /// @throws std::invalid_argument if @p slots is 0, or as
  /// make_window_decoder() does; std::logic_error if the store lacks a
  /// variant, or a variant's sends are not the request's.
  decode_window_core(const data_artifact_store &store,
                     std::shared_ptr<const experiment_template> context,
                     window_request request, const std::string &decoder_name,
                     const cudaqx::heterogeneous_map &params, std::size_t slots,
                     verify_residual_mode verify = verify_residual_mode::none);

  /// Fire @p window into @p slot: copy its rows, apply each receive, a
  /// delta to the window's own rows and a residual to its imported ones,
  /// then decode and commit. The window and every received row are checked
  /// first, so a rejected fire leaves the slot holding what it held; a decoder
  /// that fails, or a solution that verify_residual fail rejects, leaves it
  /// empty. Allocates nothing beyond what the decoder's decode() does, except
  /// for the message of a solution it rejects.
  /// @param receives One per receive port: the producer instance's send, or
  /// null when that instance does not exist.
  /// @throws std::invalid_argument if the slot does not exist, the window is
  /// not this layer's window k, of one of its variants, holding that
  /// variant's rows, a received send reaches a row the window does not take
  /// in its mode, or an imported row receives no residual;
  /// std::logic_error if the window's variant is a prefix or terminal one
  /// that another slot is decoding; std::runtime_error if verify_residual
  /// is fail and the solution does not satisfy the window's detectors.
  const window_fire &fire(std::size_t slot, const window_rows &window,
                          std::span<const window_send_value *const> receives);

  /// Start a new shot: no slot holds a fire, so held() shows none of an
  /// earlier shot's windows.
  void reset();
  /// The fire @p slot holds, unfired when it holds none. Reads only that
  /// slot, so it may run while other slots fire.
  /// @throws std::invalid_argument if the slot does not exist.
  const window_fire &held(std::size_t slot) const;
  /// The index of send @p id among the request's sends.
  /// @throws std::invalid_argument if the request has no such send.
  std::size_t send_index(const std::string &id) const;
  /// The layer's variants.
  std::size_t num_variants() const { return variants_.size(); }
  /// The slots.
  std::size_t num_slots() const { return fires_.size(); }
  /// The most rows send @p s has in any variant.
  std::size_t max_send_rows(std::size_t s) const;

private:
  /// One variant's model and its decoders: one per slot for the bulk, else
  /// one.
  struct variant {
    variant_key key;
    std::shared_ptr<const decoding_window> window;
    std::vector<std::unique_ptr<decoder>> decoders;
  };

  /// Throws std::invalid_argument for @p what.
  [[noreturn]] void fail(const std::string &what) const;
  /// Throws std::invalid_argument unless @p slot exists.
  void check_slot(std::size_t slot) const;
  /// The shot row where @p window's first round starts.
  std::size_t first_row(const window_rows &window) const;

  std::shared_ptr<const experiment_template> context_;
  window_request request_;
  verify_residual_mode verify_ = verify_residual_mode::none;
  round_range span_; ///< The rounds a window spans, from its anchor
  std::vector<variant> variants_;
  // Per slot, a workspace and a detector buffer for every variant.
  std::vector<std::vector<window_workspace>> workspaces_;
  std::vector<std::vector<std::vector<std::uint8_t>>> detectors_;
  std::vector<window_fire> fires_;
  /// Per slot, which imported rows a fire's residuals reach, checked before
  /// it writes.
  std::vector<std::vector<std::uint8_t>> covered_;
  /// Per variant, whether a fire holds its one decoder; set only for prefix
  /// and terminal variants.
  std::unique_ptr<std::atomic<bool>[]> decoding_;
};

} // namespace cudaq::qec

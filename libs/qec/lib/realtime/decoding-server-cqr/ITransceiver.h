/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace cudaq::qec::decoding_server {

/// Peer identity — the address to which the server sends a response.
struct PeerId {
  std::array<uint8_t, 16> addr; ///< GID / IPv6 (16 bytes)
  uint16_t port;

  bool operator==(const PeerId &) const = default;
};

/// Move-only wrapper that invokes the wrapped callback exactly once, when the
/// holder is destroyed.  Frames are dropped on many paths (dispatcher
/// validation failures, session queue full, handler exceptions); tying the
/// release to the frame's lifetime guarantees the ring slot is returned on
/// every one of them instead of relying on each path to call it by hand.
class ReleaseFn {
public:
  ReleaseFn() = default;
  explicit ReleaseFn(std::function<void()> fn) : fn_(std::move(fn)) {}
  ReleaseFn(const ReleaseFn &) = delete;
  ReleaseFn &operator=(const ReleaseFn &) = delete;
  ReleaseFn(ReleaseFn &&other) noexcept
      : fn_(std::exchange(other.fn_, nullptr)) {}
  ReleaseFn &operator=(ReleaseFn &&other) noexcept {
    if (this != &other) {
      if (fn_)
        fn_();
      fn_ = std::exchange(other.fn_, nullptr);
    }
    return *this;
  }
  ~ReleaseFn() {
    if (fn_)
      fn_();
  }
  explicit operator bool() const noexcept { return static_cast<bool>(fn_); }

private:
  std::function<void()> fn_;
};

/// A received frame: owns the wire bytes (RPCHeader + payload) plus transport
/// metadata needed for syndrome scatter and response routing.
///
/// release_fn: when non-null, returns the transport ring slot; it fires
/// automatically when the frame is destroyed.  For host-copy transports it is
/// always null — the copy itself constitutes "release."  For GPU ring buffer
/// transports (full RelayBP path), the frame must be kept alive until GPU
/// decode completes so the slot is not returned to GpuRoceTransceiver early.
struct RxFrame {
  std::vector<uint8_t> buf; ///< RPCHeader + payload (owned copy)
  uint32_t vp_id = 0;
  PeerId peer{};
  ReleaseFn release_fn; ///< null except on GPU ring-buffer path
};

/// Transport abstraction used by DecodingServer (the device_graph path).
struct ITransceiver {
  /// Block until a frame is available and return it (buf is owned by caller).
  /// After shutdown() this may return a frame with an EMPTY buf -- the
  /// sentinel that unblocks a waiting caller so it can observe the shutdown
  /// flag and exit.
  virtual RxFrame recv() = 0;

  /// Unblock any thread waiting in recv() (which then returns an empty
  /// sentinel frame). Called by DecodingServer::stop().
  virtual void shutdown() {}

  /// Send a response to \p peer.  Thread-safe: called from session worker
  /// threads, which may be concurrent.
  virtual void send(const PeerId &peer, const uint8_t *data, size_t len) = 0;

  /// Optional hook for GPU ring-buffer transports: wire \p graph_resources
  /// (the opaque pointer from decoder::capture_decode_graph()) into an
  /// on-device scheduler and launch it.  Returns false when the transport has
  /// no device scheduler (the default); callers that require one treat that
  /// as an error.  Keeps DecodingServer free of concrete-transceiver types
  /// (and their link-time dependencies).
  virtual bool launch_device_scheduler(void * /*graph_resources*/) {
    return false;
  }

  virtual ~ITransceiver() = default;
};

} // namespace cudaq::qec::decoding_server

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Env-gated latency probes for the decoding-server HOST_CALL path.
///
/// A served RPC runs start to finish on the CUDAQ dispatcher thread that
/// delivered it (DecodingSession::handle_*), so there are no thread-hop
/// probes — only single-thread stage durations:
///
///   total          handler entry -> exit (the HOST_CALL as the CUDAQ
///                  dispatcher sees it; same definition as before the
///                  worker-thread retirement, so reports stay comparable)
///   stage_parse    handler entry -> request parsed + validated
///   stage_decode   parse done -> decoder work done (pin, capture, unpack,
///                  enqueue/decode or corrections pack)
///   stage_respond  decoder work done -> response committed + handler exit
///
/// The volume-completing enqueue runs its decode inline, so decode time
/// shows up in that enqueue's stage_decode (previously it hid between the
/// enqueue ACK and the get_corrections wait).
///
/// Off by default; when disabled every probe is a single predicted branch.
/// Gates (each read once, on first use):
///   QEC_DECODING_SERVER_HOP_STATS      unset/"" = off; "total" = only the
///                                      entry/exit stamps (probe-distortion
///                                      A/B); any other value = full stages.
///   QEC_DECODING_SERVER_HOP_STATS_CSV  per-sample CSV dump path, written at
///                                      report time.
///   QEC_PIN_DISPATCHER                 pin the dispatcher thread(s) to a
///                                      cpu id; intended for single-decoder
///                                      diagnosis runs.
///
/// Samples are stack-resident during the request (StageScope) and appended
/// to a fixed lock-free array at handler exit — no correlation slots, no
/// cross-thread stamps. Diagnostic-quality instrumentation: report/CSV
/// formatting happens only at shutdown, never on the hot path.

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <vector>

namespace cudaq::qec::decoding_server::hopstats {

// ---------------------------------------------------------------------------
// Gates and small utilities
// ---------------------------------------------------------------------------

enum class Mode : int { off = 0, full = 1, total_only = 2 };

inline Mode mode() {
  static const Mode m = [] {
    const char *e = std::getenv("QEC_DECODING_SERVER_HOP_STATS");
    if (!e || !e[0])
      return Mode::off;
    if (std::strcmp(e, "total") == 0)
      return Mode::total_only;
    return Mode::full;
  }();
  return m;
}

inline bool enabled() { return mode() != Mode::off; }
inline bool full() { return mode() == Mode::full; }

inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// ---------------------------------------------------------------------------
// Completed samples (one POD per request, appended lock-free)
// ---------------------------------------------------------------------------

constexpr int32_t kMissing = INT32_MIN; // stamp absent (mode / early return)

struct Sample {
  uint8_t kind; // 0 enqueue, 1 get_corrections, 2 reset, 3 other
  uint32_t rid;
  int32_t total;
  int32_t stage_parse, stage_decode, stage_respond;
};

constexpr std::size_t kMaxSamples = 262144; // ~5 MB; recording stops when full
inline Sample g_samples[kMaxSamples];
inline std::atomic<uint64_t> g_sample_count{0};

inline uint8_t kind_of(uint32_t function_id) {
  using namespace cudaq::qec::decoding::rpc;
  if (function_id == kEnqueueSyndromesFunctionId)
    return 0;
  if (function_id == kGetCorrectionsFunctionId)
    return 1;
  if (function_id == kResetDecoderFunctionId)
    return 2;
  return 3;
}

// b - a in ns, kMissing when either stamp is absent, clamped to int32.
inline int32_t delta_ns(uint64_t a, uint64_t b) {
  if (!a || !b)
    return kMissing;
  const int64_t d = static_cast<int64_t>(b) - static_cast<int64_t>(a);
  if (d > INT32_MAX)
    return INT32_MAX;
  if (d < INT32_MIN + 1)
    return INT32_MIN + 1;
  return static_cast<int32_t>(d);
}

inline void append_sample(const Sample &smp) {
  const uint64_t idx = g_sample_count.fetch_add(1, std::memory_order_relaxed);
  if (idx < kMaxSamples)
    g_samples[idx] = smp;
}

// ---------------------------------------------------------------------------
// StageScope — stack-resident probe for one inline request
// ---------------------------------------------------------------------------

/// Constructed at handle_* entry; parsed()/decoded() mark stage boundaries
/// (full mode only — total mode stamps just entry/exit to bound probe
/// distortion); the destructor closes the sample at handler exit, after the
/// response write.
class StageScope {
public:
  explicit StageScope(uint32_t function_id) noexcept {
    if (!enabled())
      return;
    active_ = true;
    kind_ = kind_of(function_id);
    t_entry_ = now_ns();
  }
  StageScope(const StageScope &) = delete;
  StageScope &operator=(const StageScope &) = delete;

  /// Request parsed + validated; \p rid is the client's request id.
  void parsed(uint32_t rid) noexcept {
    if (!active_)
      return;
    rid_ = rid;
    if (full())
      t_parsed_ = now_ns();
  }

  /// Decoder work done (response not yet written).
  void decoded() noexcept {
    if (active_ && full())
      t_decoded_ = now_ns();
  }

  ~StageScope() {
    if (!active_)
      return;
    const uint64_t t_exit = now_ns();
    Sample smp{};
    smp.kind = kind_;
    smp.rid = rid_;
    smp.total = delta_ns(t_entry_, t_exit);
    smp.stage_parse = delta_ns(t_entry_, t_parsed_);
    smp.stage_decode = delta_ns(t_parsed_, t_decoded_);
    smp.stage_respond = delta_ns(t_decoded_, t_exit);
    append_sample(smp);
  }

private:
  bool active_ = false;
  uint8_t kind_ = 3;
  uint32_t rid_ = 0;
  uint64_t t_entry_ = 0, t_parsed_ = 0, t_decoded_ = 0;
};

// ---------------------------------------------------------------------------
// Thread naming + optional pinning (named threads make ftrace readable)
// ---------------------------------------------------------------------------

inline void name_and_pin(const char *name, const char *pin_env) {
  pthread_setname_np(pthread_self(), name);
  const char *e = std::getenv(pin_env);
  if (!e || !e[0])
    return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(std::atoi(e), &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    std::fprintf(stderr, "hopstats: %s=%s pin failed for %s\n", pin_env, e,
                 name);
}

inline void on_dispatcher_thread() {
  thread_local bool done = false;
  if (!done) {
    done = true;
    name_and_pin("cqr-dispatch", "QEC_PIN_DISPATCHER");
  }
}

// ---------------------------------------------------------------------------
// Shutdown report (formatting cost paid only here)
// ---------------------------------------------------------------------------

inline double percentile_sorted(const std::vector<int32_t> &sorted, double p) {
  if (sorted.empty())
    return 0.0;
  const double pos = p * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return static_cast<double>(sorted[lo]) +
         (static_cast<double>(sorted[hi]) - static_cast<double>(sorted[lo])) *
             frac;
}

inline void report() {
  if (!enabled())
    return;
  static std::atomic<bool> reported{false};
  if (reported.exchange(true))
    return;

  const uint64_t total = g_sample_count.load(std::memory_order_relaxed);
  const std::size_t n =
      static_cast<std::size_t>(std::min<uint64_t>(total, kMaxSamples));
  std::printf("QEC_HOP_STATS mode=%s recorded=%zu dropped=%llu\n",
              full() ? "full" : "total", n,
              static_cast<unsigned long long>(total - n));

  struct MetricDesc {
    const char *name;
    int32_t Sample::*field;
  };
  const MetricDesc metrics[] = {
      {"total", &Sample::total},
      {"stage_parse", &Sample::stage_parse},
      {"stage_decode", &Sample::stage_decode},
      {"stage_respond", &Sample::stage_respond},
  };
  const char *kind_names[] = {"enqueue_syndromes", "get_corrections",
                              "reset_decoder", "other"};

  std::vector<int32_t> vals;
  for (int kind = 0; kind < 4; ++kind) {
    for (const auto &m : metrics) {
      vals.clear();
      for (std::size_t i = 0; i < n; ++i) {
        const Sample &smp = g_samples[i];
        if (smp.kind != kind)
          continue;
        const int32_t v = smp.*(m.field);
        if (v == kMissing)
          continue;
        vals.push_back(v);
      }
      if (vals.empty())
        continue;
      std::sort(vals.begin(), vals.end());
      int64_t sum = 0;
      for (int32_t v : vals)
        sum += v;
      const double avg =
          static_cast<double>(sum) / static_cast<double>(vals.size()) / 1000.0;
      std::printf("QEC_HOP_STATS kind=%s metric=%s subset=all count=%zu "
                  "min_us=%.2f avg_us=%.2f p50_us=%.2f p90_us=%.2f p99_us=%.2f "
                  "max_us=%.2f\n",
                  kind_names[kind], m.name, vals.size(), vals.front() / 1000.0,
                  avg, percentile_sorted(vals, 0.50) / 1000.0,
                  percentile_sorted(vals, 0.90) / 1000.0,
                  percentile_sorted(vals, 0.99) / 1000.0, vals.back() / 1000.0);
    }
  }

  if (const char *path = std::getenv("QEC_DECODING_SERVER_HOP_STATS_CSV");
      path && path[0]) {
    if (FILE *f = std::fopen(path, "w")) {
      std::fprintf(f, "kind,rid,total_ns,stage_parse_ns,stage_decode_ns,"
                      "stage_respond_ns\n");
      for (std::size_t i = 0; i < n; ++i) {
        const Sample &s = g_samples[i];
        std::fprintf(f, "%s,%u,%d,%d,%d,%d\n", kind_names[s.kind], s.rid,
                     s.total, s.stage_parse, s.stage_decode, s.stage_respond);
      }
      std::fclose(f);
      std::printf("QEC_HOP_STATS csv=%s rows=%zu\n", path, n);
    } else {
      std::fprintf(stderr, "hopstats: cannot open csv path %s\n", path);
    }
  }
  std::fflush(stdout);
}

} // namespace cudaq::qec::decoding_server::hopstats

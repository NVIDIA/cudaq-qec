/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

/// @file
/// @brief Shared machinery for the realtime `[DecoderStats]` log stream.
/// @details
/// This is not a public header: it sits beside its implementation in
/// libs/qec/lib rather than in the installed include tree, so it adds nothing
/// to the decoder API. Nothing in decoder.h pulls it in and the base decoder
/// does not own an instance. A decoder that wants latency instrumentation
/// includes it, holds a decoder_stats member and reports through it, which
/// keeps the cost off every decoder that does not. A target that consumes this
/// must add libs/qec/lib to its private include directories.
///
/// It stays within C++17 so that a decoder compiled as CUDA C++17 can consume
/// it: vector references rather than std::span, and the
/// __builtin_FILE/__builtin_LINE default arguments that logger.h already uses
/// in place of std::source_location.
///
/// Every decoder that reports through this header emits the same line format,
/// which is the format `libs/qec/utils/replay_decoder_logs.py` consumes. The
/// parts that are identical for every decoder live here -- level selection,
/// output routing, the line prefix and counter, and the sparse/dense
/// conversions the lines need -- so a decoder only has to decide which fields
/// to report.
///
/// A decoder that reports its submits and decode completions also gets a
/// `stats_summary` line when it is destroyed, aggregating two latencies over
/// the decoder's lifetime: the tail latency from a shot's last measurements
/// reaching enqueue_syndrome() to the end of its decode, and the full-shot
/// latency from that shot's first measurements to the same point. Both include
/// whatever delay the caller adds between the two, since that is what a
/// realtime application experiences, and both are reported as a count, mean,
/// min, max and percentiles -- see latency_series for how the percentiles are
/// estimated. By default the aggregate spans the decoder's whole lifetime. A
/// decoder with an explicit warmup phase may call emit_summary() afterward; a
/// successful emission starts a fresh aggregate for later decodes.
///
/// Two levels are recognized. `info` reports that one summary and nothing else,
/// so an instrumented realtime run costs a clock read per submit and per decode
/// and emits a single line. `debug` adds a line per instrumented call, counts
/// and sparse arrays together, which is the volume a replay capture needs and
/// the volume a streaming path cannot afford. `CUDAQ_QEC_DEBUG_DECODER=1`
/// selects the per-call form through a raw printf, bypassing the logger (and
/// therefore a log forwarder, which truncates long messages and so breaks
/// replay).

namespace cudaq::qec {

/// @brief How much detail an instrumented call should emit.
enum class stats_detail {
  off,     ///< Nothing is logged; skip all measurement and formatting.
  summary, ///< Only the lifetime summary line; each call stays silent but still
           ///< records the timestamps that summary aggregates.
  arrays   ///< A line per call, with the counts, durations and sparse arrays.
};

/// @brief Append the positions of the set bytes in `[bits, bits + count)`.
/// @details Callers own `out` so several arrays can be alive in one line; it is
/// cleared first, and keeping it as a member avoids reallocating per call.
void collect_set_bits(const uint8_t *bits, std::size_t count,
                      std::vector<uint32_t> &out);

/// @brief Reusable storage for the sparse arrays a decode line reports.
/// @details A decoder holds one of these so an enabled log reuses its
/// allocations instead of converting into fresh vectors per decode. Fill the
/// members with collect_set_bits() or directly, then hand the whole struct to
/// decoder_stats::append_replay_fields(). A decoder with no error-space result
/// simply leaves `errors` empty.
struct sparse_scratch {
  /// @brief Sparse indices of the raw measurement bits consumed.
  std::vector<uint32_t> msyn;
  /// @brief Sparse indices of the detectors that fired.
  std::vector<uint32_t> detectors;
  /// @brief Sparse indices of the predicted error mechanisms.
  std::vector<uint32_t> errors;

  /// @brief Drop the previous line's contents, keeping the capacity.
  void clear() {
    msyn.clear();
    detectors.clear();
    errors.clear();
  }
};

/// @brief The array fields `replay_decoder_logs.py` reads from a decode line.
/// @details Every decoder fills the same set so the replay tool needs no
/// per-decoder knowledge. An empty vector renders as an empty value, which the
/// tool reads as "none".
struct replay_fields {
  /// @brief Sparse indices of the raw measurement bits consumed.
  const std::vector<uint32_t> &msyn;
  /// @brief Sparse indices of the detectors that fired.
  const std::vector<uint32_t> &detectors;
  /// @brief Sparse indices of the predicted error mechanisms, if any.
  const std::vector<uint32_t> &errors;
  /// @brief Sparse indices of the observables flipped by this decode.
  const std::vector<uint32_t> &observables;
  /// @brief Dense per-observable flips contributed by this decode.
  const std::vector<uint8_t> &corrections_this_call;
  /// @brief Dense per-observable accumulated Pauli frame.
  const std::vector<uint8_t> &corrections_total;
};

/// @brief Running count, average, min, max and percentiles over a latency
/// series, recorded without allocating or retaining the samples.
///
/// @details A realtime decoder cannot know how many shots it will see, so there
/// is no sample buffer to size up front, and growing one mid-run would stall
/// the path being measured. Instead each sample increments one counter in a
/// log-spaced histogram sized at compile time: `sub_buckets_per_octave`
/// subdivisions of every power of two from `first_octave_ns` up, which is what
/// lets one table cover the sub-microsecond decodes and the millisecond
/// outliers at the same relative resolution. Count, mean, min and max stay
/// exact; the percentiles are estimates whose error is bounded by the bucket
/// width, near 1.6% of the value at the default resolution.
struct latency_series {
  std::size_t count = 0;
  double sum_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;

  void add(double us) {
    if (count == 0 || us < min_us)
      min_us = us;
    if (count == 0 || us > max_us)
      max_us = us;
    sum_us += us;
    ++count;
    ++buckets_[bucket_of(us)];
  }

  double average_us() const { return count ? sum_us / count : 0.0; }

  /// @brief Estimate the latency at `fraction` of the distribution, where 0.5
  /// is the median.
  /// @details Nearest-rank: the smallest sample at or above the requested rank,
  /// reported as the midpoint of its bucket and clamped to the exact min and
  /// max, so the result never falls outside the range actually seen. Returns 0
  /// for an empty series.
  double percentile_us(double fraction) const;

private:
  /// Powers of two are subdivided this finely, which fixes the resolution: a
  /// bucket spans 1/32 of its value, so a midpoint estimate is within 1/64.
  static constexpr int sub_bucket_bits = 5;
  static constexpr std::size_t sub_buckets_per_octave = 1u << sub_bucket_bits;
  /// Nothing below 64ns is resolved, which two clock reads already exceed, and
  /// nothing above 2^27ns (~134ms); both ends clamp into the end buckets while
  /// min_us and max_us keep the exact value.
  static constexpr int first_octave_ns = 6;
  static constexpr int octaves = 21;
  static constexpr std::size_t bucket_count = octaves * sub_buckets_per_octave;

  /// @brief Which counter a sample belongs to: its octave picks the range, the
  /// next `sub_bucket_bits` mantissa bits pick the subdivision within it.
  static std::size_t bucket_of(double us) {
    const double ns = us * 1000.0;
    // Also catches NaN and a clock that ran backwards; min_us keeps the value.
    if (!(ns >= 1.0))
      return 0;
    const auto value = static_cast<uint64_t>(ns);
    const int octave = 63 - __builtin_clzll(value);
    if (octave < first_octave_ns)
      return 0;
    const auto index = static_cast<std::size_t>(octave - first_octave_ns);
    if (index >= octaves)
      return bucket_count - 1;
    const auto sub = static_cast<std::size_t>(
        (value >> (octave - sub_bucket_bits)) & (sub_buckets_per_octave - 1));
    return (index << sub_bucket_bits) | sub;
  }

  static double bucket_midpoint_us(std::size_t bucket);

  /// 21 octaves x 32 sub-buckets of uint64: ~5KB held with the object, so
  /// recording a sample never touches the allocator.
  std::array<uint64_t, bucket_count> buckets_ = {};
};

/// @brief Per-decoder instrumentation state for the `[DecoderStats]` stream.
///
/// One instance lives in the decoder that opts in, so that decoder and its own
/// helpers share one counter and one resolved output mode. Hold it as a member
/// rather than constructing one per call: it carries the latency histograms the
/// summary aggregates, and constructing one reads the environment.
class decoder_stats {
public:
  /// @brief Construct for `owner`, whose address identifies the decoder in the
  /// log. Reads `CUDAQ_QEC_DEBUG_DECODER` once.
  explicit decoder_stats(const void *owner = nullptr);

  /// @brief Emits the latency summary for the decoder being destroyed.
  ~decoder_stats();

  /// @brief Point the log lines at their decoder. Needed when the owner cannot
  /// be known at construction.
  void set_owner(const void *owner) { owner_ = owner; }

  /// @brief Record the decoder id reported by every line.
  void set_decoder_id(uint32_t decoder_id) { decoder_id_ = decoder_id; }

  /// @brief Resolve how much to log right now.
  /// @details Probes `info` first, so a disabled call costs one relaxed atomic
  /// load: a threshold low enough to admit `debug` admits `info` as well.
  stats_detail detail() const;

  /// @brief Start a line: "[DecoderStats][owner] Counter:N DecoderId:M".
  /// @details Advances the counter, so call it once per emitted line.
  std::string prefix();

  /// @brief Emit a finished line at the level `detail` implies.
  /// @details The defaulted file and line make the log report the decoder that
  /// emitted the line rather than this file.
  void emit(stats_detail detail, const std::string &message,
            const char *file_name = __builtin_FILE(),
            int line_no = __builtin_LINE()) const;

  /// @brief Emit a line for a realtime call that only reads or clears the
  /// correction frame, such as `get_obs_corrections` or `reset_decoder`.
  /// @details A per-call line, so it appears at `arrays` and nowhere else; the
  /// level check is done here because these call sites have nothing else to
  /// gate.
  void emit_frame_call(const char *call,
                       const std::vector<uint8_t> &corrections,
                       const char *file_name = __builtin_FILE(),
                       int line_no = __builtin_LINE());

  /// @brief Append the replay field set to `line`.
  /// @details Only meaningful at `stats_detail::arrays`; the field names are
  /// fixed because the replay tool matches on them.
  void append_replay_fields(std::string &line,
                            const replay_fields &fields) const;

  /// @brief Append the replay field set from a decoder's `scratch`.
  /// @details Fills the observable fields from the last diff_frame(), so call
  /// that first. `corrections_total` is the accumulated Pauli frame.
  void
  append_replay_fields(std::string &line, const sparse_scratch &scratch,
                       const std::vector<uint8_t> &corrections_total) const;

  /// @brief Record that measurements were accepted at this instant.
  /// @param first_of_shot Whether these are the shot's first measurements,
  /// which is what starts the full-shot clock. Passing it explicitly means a
  /// level raised part-way through a shot reports no shot latency for that shot
  /// instead of a short one measured from the middle.
  /// @details Only the enabled path should call this: it reads the clock.
  void note_submit(bool first_of_shot);

  /// @brief Microseconds since the last note_submit(), or 0 if there was none.
  double since_last_submit_us() const;

  /// @brief Record that a decode finished, closing this shot's latencies.
  /// @details Accumulates the interval since the last submit and, when the
  /// shot's first submit was seen, the interval since that first submit.
  void note_decode_complete();

  /// @brief Abandon an unfinished shot, as a reset discards its measurements.
  void note_reset();

  /// @brief Emit the aggregate latency line, if any decode was measured.
  /// @details Called on destruction; a decoder may also call it earlier, for
  /// instance to separate a warmup phase from a measured one. A successful
  /// emission clears the aggregate so the next line covers only later decodes;
  /// when logging is disabled, the samples remain available for a later call.
  void emit_summary(const char *file_name = __builtin_FILE(),
                    int line_no = __builtin_LINE());

  /// @brief Diff two Pauli frames into reusable scratch.
  /// @details A size mismatch yields no flips rather than a diff against a
  /// stale frame, which happens when O is replaced mid-stream.
  void diff_frame(const std::vector<uint8_t> &before,
                  const std::vector<uint8_t> &now);

  /// @brief Dense flips from the last diff_frame().
  const std::vector<uint8_t> &frame_flips() const { return frame_flips_; }

  /// @brief Sparse flipped indices from the last diff_frame().
  const std::vector<uint32_t> &frame_flip_ids() const {
    return frame_flip_ids_;
  }

private:
  using clock = std::chrono::steady_clock;

  const void *owner_ = nullptr;
  uint32_t decoder_id_ = 0;
  uint32_t counter_ = 0;
  /// Whether CUDAQ_QEC_DEBUG_DECODER selected the forwarder-proof printf path.
  bool printf_mode_ = false;
  std::vector<uint8_t> frame_flips_;
  std::vector<uint32_t> frame_flip_ids_;
  /// Latency bookkeeping: when the current shot's first and latest
  /// measurements arrived, and the two series closed out by each decode.
  clock::time_point shot_start_;
  clock::time_point last_submit_;
  bool shot_open_ = false;
  bool have_submit_ = false;
  latency_series tail_latency_;
  latency_series shot_latency_;
};

/// @brief Elapsed-time stopwatch for the stage durations a per-call line
/// reports, which measures nothing at the levels that emit no such line.
///
/// Reading a clock costs more than the branch that skips it, and an
/// instrumented path usually wants several stage timings, so this keeps the
/// `detail == arrays ? now() : time_point{}` dance out of decoder code. The
/// lifetime summary's latencies do not come from here; they come from
/// decoder_stats::note_submit() and note_decode_complete().
class stats_timer {
public:
  explicit stats_timer(stats_detail detail)
      : enabled_(detail == stats_detail::arrays) {
    if (enabled_)
      last_ = clock::now();
  }

  /// @brief Microseconds since construction or the previous lap.
  double lap_us() {
    if (!enabled_)
      return 0.0;
    const auto now = clock::now();
    const std::chrono::duration<double, std::micro> elapsed = now - last_;
    last_ = now;
    return elapsed.count();
  }

private:
  using clock = std::chrono::steady_clock;
  bool enabled_;
  clock::time_point last_;
};

} // namespace cudaq::qec

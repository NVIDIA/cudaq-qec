/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Out-of-line half of decoder_stats.h: the shared machinery behind the
// realtime `[DecoderStats]` log stream. Everything a reporting decoder does
// per call -- resolving the level, stamping the line prefix, routing the line
// to the logger or to a raw printf, recording a latency sample and emitting
// the lifetime summary -- is defined here rather than in the header, so a
// decoder that opts in pays for a declaration and a call rather than for
// fmt and chrono instantiations in every translation unit that sees it.
//
// This lives in the same library as decoder.cpp so that a decoder plugin,
// which already links that library to reach decoder itself, resolves these
// symbols without linking anything further.

#include "decoder_stats.h"
#include "cudaq/qec/logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fmt/ranges.h>

namespace cudaq::qec {

void collect_set_bits(const uint8_t *bits, std::size_t count,
                      std::vector<uint32_t> &out) {
  out.clear();
  for (std::size_t i = 0; i < count; ++i)
    if (bits[i])
      out.push_back(static_cast<uint32_t>(i));
}

// The inverse of latency_series::bucket_of(): rebuild the value range a bucket
// covers and return its midpoint, which is the best estimate available for a
// sample whose exact value was not retained.
double latency_series::bucket_midpoint_us(std::size_t bucket) {
  const int octave =
      static_cast<int>(bucket >> sub_bucket_bits) + first_octave_ns;
  const auto sub = static_cast<uint64_t>(bucket & (sub_buckets_per_octave - 1));
  const uint64_t width = uint64_t{1} << (octave - sub_bucket_bits);
  const uint64_t lower = (uint64_t{1} << octave) + sub * width;
  return static_cast<double>(lower + width / 2) / 1000.0;
}

// Nearest-rank over the histogram: walk buckets in increasing value order
// until the cumulative count reaches the requested rank.
double latency_series::percentile_us(double fraction) const {
  if (count == 0)
    return 0.0;
  if (fraction <= 0.0)
    return min_us;
  if (fraction >= 1.0)
    return max_us;
  const auto rank =
      static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(count)));
  uint64_t cumulative = 0;
  for (std::size_t bucket = 0; bucket < buckets_.size(); bucket++) {
    cumulative += buckets_[bucket];
    if (cumulative < rank)
      continue;
    // The bucket is coarser than the extremes we tracked exactly, so a
    // single-sample or narrow series still reports a value it actually saw.
    return std::clamp(bucket_midpoint_us(bucket), min_us, max_us);
  }
  return max_us;
} // end - latency_series::percentile_us()

decoder_stats::decoder_stats(const void *owner) : owner_(owner) {
  if (const char *env = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
    printf_mode_ = env[0] == '1' || env[0] == 'y' || env[0] == 'Y';
}

// Destructors are implicitly noexcept, and emit_summary() formats and logs, so
// an allocation failure there would terminate instead of losing a stats line.
decoder_stats::~decoder_stats() {
  try {
    emit_summary();
  } catch (...) {
  }
} // end - decoder_stats::~decoder_stats

void decoder_stats::note_submit(bool first_of_shot) {
  const auto now = clock::now();
  if (first_of_shot) {
    shot_start_ = now;
    shot_open_ = true;
  }
  last_submit_ = now;
  have_submit_ = true;
}

double decoder_stats::since_last_submit_us() const {
  if (!have_submit_)
    return 0.0;
  return std::chrono::duration<double, std::micro>(clock::now() - last_submit_)
      .count();
}

void decoder_stats::note_decode_complete() {
  if (!have_submit_)
    return;
  const auto now = clock::now();
  tail_latency_.add(
      std::chrono::duration<double, std::micro>(now - last_submit_).count());
  if (shot_open_)
    shot_latency_.add(
        std::chrono::duration<double, std::micro>(now - shot_start_).count());
  // The next submit belongs to the next shot, whose first_of_shot reopens the
  // full-shot clock.
  shot_open_ = false;
  have_submit_ = false;
}

void decoder_stats::note_reset() {
  shot_open_ = false;
  have_submit_ = false;
}

void decoder_stats::emit_summary(const char *file_name, int line_no) {
  if (tail_latency_.count == 0)
    return;
  const auto detail_level = detail();
  if (detail_level == stats_detail::off)
    return;
  // One line for the decoder's whole lifetime, so unlike the per-call lines it
  // reports at info.
  emit(stats_detail::summary,
       // Two decimals because these are microseconds and a realtime decode is
       // often a fraction of one, where a single decimal would quantize away
       // the difference between two decoders.
       fmt::format(
           "{} stats_summary Decodes:{} TailAvg:{:.2f}us "
           "TailP50:{:.2f}us TailP90:{:.2f}us TailP99:{:.2f}us "
           "TailMin:{:.2f}us TailMax:{:.2f}us Shots:{} "
           "ShotAvg:{:.2f}us ShotP50:{:.2f}us ShotP90:{:.2f}us "
           "ShotP99:{:.2f}us ShotMin:{:.2f}us ShotMax:{:.2f}us",
           prefix(), tail_latency_.count, tail_latency_.average_us(),
           tail_latency_.percentile_us(0.50), tail_latency_.percentile_us(0.90),
           tail_latency_.percentile_us(0.99), tail_latency_.min_us,
           tail_latency_.max_us, shot_latency_.count,
           shot_latency_.average_us(), shot_latency_.percentile_us(0.50),
           shot_latency_.percentile_us(0.90), shot_latency_.percentile_us(0.99),
           shot_latency_.min_us, shot_latency_.max_us),
       file_name, line_no);
  // A caller may emit between warmup and measurement. Start a fresh aggregate
  // after every successful emission, and leave disabled summaries intact so
  // they can still be reported if the level is enabled before destruction.
  tail_latency_ = {};
  shot_latency_ = {};
} // end - decoder_stats::emit_summary()

stats_detail decoder_stats::detail() const {
  if (printf_mode_)
    return stats_detail::arrays;
  if (!detail::should_log(detail::log_level::info))
    return stats_detail::off;
  return detail::should_log(detail::log_level::debug) ? stats_detail::arrays
                                                      : stats_detail::summary;
}

std::string decoder_stats::prefix() {
  return fmt::format("[DecoderStats][{}] Counter:{} DecoderId:{}", owner_,
                     ++counter_, decoder_id_);
}

void decoder_stats::emit(stats_detail detail_level, const std::string &message,
                         const char *file_name, int line_no) const {
  if (printf_mode_) {
    std::printf("%s\n", message.c_str());
    return;
  }
  // The level macros hard-code their own level and call site, and CUDA_QEC_DBG
  // compiles away unless CUDAQ_DEBUG is defined, so go to log_message directly;
  // detail() has already checked the level.
  detail::log_message(detail_level == stats_detail::arrays
                          ? detail::log_level::debug
                          : detail::log_level::info,
                      "{}", file_name, line_no, message);
}

void decoder_stats::emit_frame_call(const char *call,
                                    const std::vector<uint8_t> &corrections,
                                    const char *file_name, int line_no) {
  const auto detail_level = detail();
  if (detail_level != stats_detail::arrays)
    return;
  emit(detail_level,
       fmt::format("{} {} called ObservableCorrectionsTotal:{}", prefix(), call,
                   fmt::join(corrections, ",")),
       file_name, line_no);
}

void decoder_stats::append_replay_fields(std::string &line,
                                         const replay_fields &fields) const {
  line += fmt::format(
      " InputMsyn:{} InputDetectors:{} Errors:{} "
      "Observables:{} ObservableCorrectionsThisCall:{} "
      "ObservableCorrectionsTotal:{}",
      fmt::join(fields.msyn, ","), fmt::join(fields.detectors, ","),
      fmt::join(fields.errors, ","), fmt::join(fields.observables, ","),
      fmt::join(fields.corrections_this_call, ","),
      fmt::join(fields.corrections_total, ","));
}

void decoder_stats::append_replay_fields(
    std::string &line, const sparse_scratch &scratch,
    const std::vector<uint8_t> &corrections_total) const {
  append_replay_fields(line,
                       replay_fields{.msyn = scratch.msyn,
                                     .detectors = scratch.detectors,
                                     .errors = scratch.errors,
                                     .observables = frame_flip_ids(),
                                     .corrections_this_call = frame_flips(),
                                     .corrections_total = corrections_total});
}

void decoder_stats::diff_frame(const std::vector<uint8_t> &before,
                               const std::vector<uint8_t> &now) {
  frame_flips_.clear();
  frame_flip_ids_.clear();
  if (before.size() != now.size())
    return;
  frame_flips_.resize(now.size());
  for (std::size_t i = 0; i < now.size(); ++i) {
    frame_flips_[i] = now[i] ^ before[i];
    if (frame_flips_[i])
      frame_flip_ids_.push_back(static_cast<uint32_t>(i));
  }
}

} // namespace cudaq::qec

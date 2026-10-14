/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "HopStats.h"
#include "qec_cc_test_helpers.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
int __real_pthread_setaffinity_np(pthread_t thread, size_t cpusetsize,
                                  const cpu_set_t *cpuset);
int __wrap_pthread_setaffinity_np(pthread_t thread, size_t cpusetsize,
                                  const cpu_set_t *cpuset) {
  if (std::getenv("QEC_TEST_AFFINITY_FAIL"))
    return EINVAL;
  return __real_pthread_setaffinity_np(thread, cpusetsize, cpuset);
}
}

namespace {

using namespace cudaq::qec::decoding_server::hopstats;

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

// report() and the pin helper write with printf/fprintf, so capture the fds.
struct report_text {
  std::string out;
  std::string err;
};

report_text capture_stdio(const std::function<void()> &fn) {
  const auto out_path = qec_cc::write_temp("", ".out");
  const auto err_path = qec_cc::write_temp("", ".err");
  std::fflush(stdout);
  std::fflush(stderr);
  const int old_out = ::dup(STDOUT_FILENO);
  const int old_err = ::dup(STDERR_FILENO);
  const int fd_out = ::open(out_path.c_str(), O_WRONLY | O_TRUNC);
  const int fd_err = ::open(err_path.c_str(), O_WRONLY | O_TRUNC);
  ::dup2(fd_out, STDOUT_FILENO);
  ::dup2(fd_err, STDERR_FILENO);
  fn();
  std::fflush(stdout);
  std::fflush(stderr);
  ::dup2(old_out, STDOUT_FILENO);
  ::dup2(old_err, STDERR_FILENO);
  ::close(fd_out);
  ::close(fd_err);
  ::close(old_out);
  ::close(old_err);
  return {slurp(out_path), slurp(err_path)};
}
using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;

int run_helper(const char *name) {
  if (std::strcmp(name, "off") == 0) {
    {
      StageScope scope(kEnqueueSyndromesFunctionId);
      scope.parsed(1);
      scope.decoded();
    }
    report();
    report();
    return enabled() ? 2 : 0;
  }
  if (std::strcmp(name, "total") == 0) {
    {
      StageScope scope(kEnqueueSyndromesFunctionId);
      scope.parsed(7);
      scope.decoded();
    }
    report();
    return full() ? 3 : 0;
  }
  if (std::strcmp(name, "full_csv") == 0) {
    {
      StageScope a(kEnqueueSyndromesFunctionId);
      a.parsed(1);
      a.decoded();
    }
    {
      StageScope b(kGetCorrectionsFunctionId);
      b.parsed(2);
      b.decoded();
    }
    {
      StageScope c(kResetDecoderFunctionId);
      c.parsed(3);
      c.decoded();
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
          ::setenv("QEC_PIN_DISPATCHER", std::to_string(cpu).c_str(), 1);
          break;
        }
      }
    }
    on_dispatcher_thread();
    on_dispatcher_thread();
    const auto text = capture_stdio([] { report(); });
    const char *csv_path = std::getenv("QEC_DECODING_SERVER_HOP_STATS_CSV");
    const auto csv = csv_path ? slurp(csv_path) : std::string{};
    // One row per request kind, plus the full-mode summary line.
    if (csv.find("kind,rid,total_ns,stage_parse_ns,stage_decode_ns,"
                 "stage_respond_ns\n") == std::string::npos ||
        csv.find("enqueue_syndromes,1,") == std::string::npos ||
        csv.find("get_corrections,2,") == std::string::npos ||
        csv.find("reset_decoder,3,") == std::string::npos ||
        text.out.find("mode=full") == std::string::npos)
      return 2;
    return 0;
  }
  if (std::strcmp(name, "full_csv_fail") == 0) {
    {
      StageScope scope(kEnqueueSyndromesFunctionId);
      scope.parsed(1);
      scope.decoded();
    }
    const auto text = capture_stdio([] { report(); });
    return text.err.find("cannot open csv path") == std::string::npos ? 2 : 0;
  }
  if (std::strcmp(name, "affinity_fail") == 0) {
    const auto text = capture_stdio([] { on_dispatcher_thread(); });
    return text.err.find("pin failed") == std::string::npos ? 2 : 0;
  }
  if (std::strcmp(name, "buffer_full") == 0) {
    g_sample_count.store(kMaxSamples);
    Sample extra{};
    extra.kind = 0;
    extra.total = 1;
    append_sample(extra);
    const auto text = capture_stdio([] { report(); });
    // The extra sample is past the cap, so the report must say it was dropped.
    return text.out.find("dropped=1") == std::string::npos ? 2 : 0;
  }
  if (std::strcmp(name, "missing_and_empty") == 0) {
    Sample missing{};
    missing.kind = 0;
    missing.total = kMissing;
    missing.stage_parse = kMissing;
    missing.stage_decode = kMissing;
    missing.stage_respond = kMissing;
    append_sample(missing);
    Sample enqueue{};
    enqueue.kind = 0;
    enqueue.total = 10;
    append_sample(enqueue);
    const auto text = capture_stdio([] { report(); });
    // kMissing is omitted, so the remaining 10 ns sample is the whole metric.
    return text.out.find("kind=enqueue_syndromes metric=total subset=all "
                         "count=1 min_us=0.01") == std::string::npos
               ? 2
               : 0;
  }
  return 1;
}

TEST(HopStatsMath, KindOfDeltaNsAndPercentile) {
  EXPECT_EQ(kind_of(kEnqueueSyndromesFunctionId), 0);
  EXPECT_EQ(kind_of(kGetCorrectionsFunctionId), 1);
  EXPECT_EQ(kind_of(kResetDecoderFunctionId), 2);
  EXPECT_EQ(kind_of(99), 3);

  EXPECT_EQ(delta_ns(0, 1), kMissing);
  EXPECT_EQ(delta_ns(1, 0), kMissing);
  EXPECT_EQ(delta_ns(1, 5), 4);
  EXPECT_EQ(delta_ns(0, UINT64_MAX), kMissing);
  EXPECT_EQ(delta_ns(1, 1ull + static_cast<uint64_t>(INT32_MAX) + 10),
            INT32_MAX);
  EXPECT_EQ(delta_ns(static_cast<uint64_t>(INT32_MAX) + 20, 1), INT32_MIN + 1);

  EXPECT_EQ(percentile_sorted({}, 0.5), 0.0);
  EXPECT_EQ(percentile_sorted({4}, 0.5), 4.0);
  EXPECT_EQ(percentile_sorted({2, 8}, 0.5), 5.0);
}

TEST(HopStatsHelpers, ScenarioProcesses) {
  EXPECT_EQ(qec_cc::exec_self("off"), 0);
  EXPECT_EQ(
      qec_cc::exec_self("total", {{"QEC_DECODING_SERVER_HOP_STATS", "total"}}),
      0);
  const auto csv = qec_cc::write_temp("", ".csv");
  EXPECT_EQ(qec_cc::exec_self("full_csv",
                              {{"QEC_DECODING_SERVER_HOP_STATS", "full"},
                               {"QEC_DECODING_SERVER_HOP_STATS_CSV", csv}}),
            0);
  EXPECT_EQ(
      qec_cc::exec_self("full_csv_fail",
                        {{"QEC_DECODING_SERVER_HOP_STATS", "full"},
                         {"QEC_DECODING_SERVER_HOP_STATS_CSV",
                          std::filesystem::temp_directory_path().string()}}),
      0);
  EXPECT_EQ(
      qec_cc::exec_self("affinity_fail", {{"QEC_PIN_DISPATCHER", "0"},
                                          {"QEC_TEST_AFFINITY_FAIL", "1"}}),
      0);
  EXPECT_EQ(qec_cc::exec_self("buffer_full",
                              {{"QEC_DECODING_SERVER_HOP_STATS", "full"}}),
            0);
  EXPECT_EQ(qec_cc::exec_self("missing_and_empty",
                              {{"QEC_DECODING_SERVER_HOP_STATS", "full"}}),
            0);
}

} // namespace

int main(int argc, char **argv) {
  if (const char *helper = qec_cc::helper_name(argc, argv))
    return run_helper(helper);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

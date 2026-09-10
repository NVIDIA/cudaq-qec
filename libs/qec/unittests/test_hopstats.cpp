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
#include <cstring>
#include <filesystem>
#include <pthread.h>
#include <sched.h>
#include <string>
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
    report();
    return 0;
  }
  if (std::strcmp(name, "full_csv_fail") == 0) {
    {
      StageScope scope(kEnqueueSyndromesFunctionId);
      scope.parsed(1);
      scope.decoded();
    }
    report();
    return 0;
  }
  if (std::strcmp(name, "affinity_fail") == 0) {
    on_dispatcher_thread();
    return 0;
  }
  if (std::strcmp(name, "buffer_full") == 0) {
    g_sample_count.store(kMaxSamples);
    Sample extra{};
    extra.kind = 0;
    extra.total = 1;
    append_sample(extra);
    report();
    return 0;
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
    report();
    return 0;
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

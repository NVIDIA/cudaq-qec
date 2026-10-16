/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Implementation of the benchmark instance harness: thread fan-out, CPU
// pinning, the setup/run start gate, and the pooling helpers that fold
// per-instance samples into machine-wide numbers.  See
// benchmark_instance_pool.h for the contract.

#include "benchmark_instance_pool.h"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace cudaq::qec::benchmark {

namespace {

using clock_type = std::chrono::steady_clock;

double elapsed_us(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

// "CPU 5" or "CPUs 4-7", for messages a user has to act on.
std::string describe_block(int first, std::size_t count) {
  if (count == 1)
    return "CPU " + std::to_string(first);
  return "CPUs " + std::to_string(first) + "-" +
         std::to_string(first + static_cast<int>(count) - 1);
}

} // namespace

// Pinning runs on the instance's own thread rather than on the spawning thread
// so that everything the instance touches afterwards -- decoder construction
// included -- allocates with the final CPU (and therefore the final NUMA node)
// already in effect.  It also means threads the decoder spawns inherit this
// mask, which is why a block can be wider than a single CPU.
std::string pin_current_thread(const cpu_block &block) {
  if (block.first < 0 || block.count == 0)
    return {};

  const std::string where = describe_block(block.first, block.count);

#if defined(__linux__)
  cpu_set_t set;
  int status = 0;

  CPU_ZERO(&set);
  for (std::size_t i = 0; i < block.count; ++i)
    CPU_SET(block.first + static_cast<int>(i), &set);
  status = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (status != 0)
    return "failed to pin decoder instance thread to " + where + ": " +
           std::strerror(status);
  return {};
#else
  return "instance CPU pinning is only implemented on Linux (requested " +
         where + ")";
#endif
} // end - pin_current_thread()

std::size_t effective_stride(const core_layout &layout) {
  // Stride 0 means "pack instances back to back", i.e. advance by one whole
  // block. A zero width still advances by 1 so no two instances share a
  // first CPU.
  if (layout.stride > 0)
    return layout.stride;
  return layout.width > 0 ? layout.width : 1;
}

std::vector<int> allowed_cpus() {
  std::vector<int> cpus;

#if defined(__linux__)
  cpu_set_t set;

  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0)
    return cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &set))
      cpus.push_back(cpu);
#endif
  return cpus;
} // end - allowed_cpus()

core_layout divide_cpus(std::size_t num_instances,
                        const std::vector<int> &cpus) {
  core_layout layout;

  // Nothing to divide, or too few CPUs to give each instance its own: leave
  // pinning off rather than hand out overlapping blocks, which would be worse
  // than letting the scheduler place the threads.
  if (num_instances == 0 || cpus.size() < num_instances)
    return layout;

  // base + i * stride can only name a contiguous run, so a fragmented cpuset
  // (0-3,8-11 under a cgroup, say) has to be refused: tiling from the bottom
  // would put an instance on a CPU this process may not be allowed to use.
  if (static_cast<std::size_t>(cpus.back() - cpus.front()) + 1 != cpus.size())
    throw std::runtime_error(
        "cannot place decoder instances automatically: the " +
        std::to_string(cpus.size()) + " available CPUs (" +
        std::to_string(cpus.front()) + "-" + std::to_string(cpus.back()) +
        ") are not contiguous; place them by hand with instance_core_base, "
        "instance_core_width and instance_core_stride");

  layout.base = cpus.front();
  // Floor: a remainder stays idle rather than making one instance wider or
  // running the last block past the end of the range.
  layout.width = cpus.size() / num_instances;
  return layout;
} // end - divide_cpus()

std::size_t cpus_per_instance(std::size_t num_instances,
                              const core_layout &layout, unsigned num_cpus) {
  // Pinned: the block is all this instance may touch, whatever the machine
  // looks like, so it is the whole answer.
  if (layout.base >= 0)
    return layout.width > 0 ? layout.width : 1;
  // Unknown CPU count (hardware_concurrency() may return 0): stay at 1 rather
  // than invent a share.
  if (num_cpus == 0)
    return 1;
  if (num_instances == 0)
    num_instances = 1;
  // Floor, not round: dividing 24 CPUs among 5 instances has to give 4, since
  // 5 would put 25 threads on the machine.
  return std::max<std::size_t>(1, num_cpus / num_instances);
} // end - cpus_per_instance()

std::size_t worker_threads_per_instance(std::size_t num_instances,
                                        const core_layout &layout,
                                        unsigned num_cpus) {
  const std::size_t cpus = cpus_per_instance(num_instances, layout, num_cpus);

  // Ceiling on the reserve, matching the four hardware threads a decoder holds
  // back when it sizes a pool from the whole machine: past a few CPUs of slack
  // the barrier has enough room to absorb a preemption, and holding back more
  // only idles the block.
  constexpr std::size_t kMaxReservedCpus = 4;
  // CPUs per reserved CPU, so the reserve scales with the block instead of
  // eating it: a 6-CPU block gives up 1, and 24 CPUs reach the ceiling.
  constexpr std::size_t kCpusPerReservedCpu = 6;
  const std::size_t reserved = std::max<std::size_t>(
      1, std::min(kMaxReservedCpus, cpus / kCpusPerReservedCpu));

  // A block with nothing to spare still has to run: 0 would be read as the
  // machine-sizing sentinel this substitution exists to replace.
  return cpus > reserved ? cpus - reserved : 1;
} // end - worker_threads_per_instance()

cpu_block resolve_instance_cpus(std::size_t instance, const core_layout &layout,
                                unsigned num_cpus) {
  cpu_block block;
  std::size_t first = 0;
  std::size_t last = 0;

  // Negative disables pinning, the same convention core_pinning uses in
  // realtime/pipeline.h.
  if (layout.base < 0)
    return block;

  block.count = layout.width > 0 ? layout.width : 1;
  first = static_cast<std::size_t>(layout.base) +
          instance * effective_stride(layout);
  // The whole block has to fit, not just the CPU it starts on.
  last = first + block.count - 1;

  const auto rejection = [&](const std::string &limit) {
    return "instance_core_base " + std::to_string(layout.base) + " (width " +
           std::to_string(block.count) + ", stride " +
           std::to_string(effective_stride(layout)) +
           ") puts decoder instance " + std::to_string(instance) + " on " +
           describe_block(static_cast<int>(first), block.count) + ", " + limit;
  };

#if defined(__linux__)
  // A cpu_set_t cannot name a CPU at or beyond CPU_SETSIZE, so reject it here
  // rather than let pthread_setaffinity_np fail with a bare EINVAL.
  if (last >= static_cast<std::size_t>(CPU_SETSIZE))
    throw std::runtime_error(rejection("which exceeds CPU_SETSIZE (" +
                                       std::to_string(CPU_SETSIZE) + ")"));
#endif

  // 0 means the CPU count is unknown (hardware_concurrency() is allowed to
  // return 0); skip the check rather than reject every placement.
  if (num_cpus > 0 && last >= static_cast<std::size_t>(num_cpus))
    throw std::runtime_error(rejection("but only " + std::to_string(num_cpus) +
                                       " CPU(s) are visible"));

  block.first = static_cast<int>(first);
  return block;
} // end - resolve_instance_cpus()

instance_pool_result run_instances(std::size_t num_instances,
                                   const core_layout &layout,
                                   const instance_work &work) {
  instance_pool_result result;
  std::vector<cpu_block> cpus;
  std::vector<std::thread> instance_threads;

  if (num_instances == 0)
    return result;

  // Resolve placement up front, on the caller's thread: an impossible core
  // list must abort before any instance spins up, not surface as a per-instance
  // error after the run.
  cpus.reserve(num_instances);
  for (std::size_t instance = 0; instance < num_instances; ++instance)
    cpus.push_back(resolve_instance_cpus(instance, layout,
                                         std::thread::hardware_concurrency()));

  // Sized before any instance starts, so an instance can write its own slot
  // without synchronization and without reallocating mid-run.
  result.instances.resize(num_instances);

  std::barrier<> gate(static_cast<std::ptrdiff_t>(num_instances));

  auto instance_body = [&](std::size_t instance) {
    instance_result &out = result.instances[instance];

    out.error = pin_current_thread(cpus[instance]);
    if (out.error.empty() && work.setup) {
      try {
        work.setup(instance);
      } catch (const std::exception &e) {
        out.error = std::string("setup failed: ") + e.what();
      } catch (...) {
        out.error = "setup failed with a non-standard exception";
      }
    }

    // Reached unconditionally, including on failure: an instance that bails
    // out before arriving would leave every healthy one blocked here forever.
    gate.arrive_and_wait();

    if (!out.error.empty() || !work.run)
      return;
    // Timed from inside the instance, so the window covers exactly the shots
    // this instance took and nothing else: not its thread spawn or teardown,
    // and not the wait a slower sibling would otherwise add to the tail of a
    // shared span. An aggregate rate is then the sum of the per-instance
    // rates rather than the total work over one window.
    const auto entered = clock_type::now();
    try {
      work.run(instance, out);
    } catch (const std::exception &e) {
      out.error = std::string("run failed: ") + e.what();
    } catch (...) {
      out.error = "run failed with a non-standard exception";
    }
    // A failed instance's samples are incomplete, so leave its window at 0
    // and let it drop out of any aggregate instead of weighting it.
    if (out.error.empty())
      out.wall_us = elapsed_us(entered, clock_type::now());
  }; // end - instance_body

  instance_threads.reserve(num_instances);
  for (std::size_t instance = 0; instance < num_instances; ++instance)
    instance_threads.emplace_back(instance_body, instance);
  for (auto &instance_thread : instance_threads)
    instance_thread.join();

  return result;
} // end - run_instances()

std::vector<double>
concat_instance_samples(const std::vector<instance_result> &instances,
                        instance_sample_kind kind) {
  std::vector<double> pooled;
  std::size_t total = 0;

  for (const auto &instance : instances)
    total += (kind == instance_sample_kind::shot ? instance.shot_us
                                                 : instance.tail_us)
                 .size();
  pooled.reserve(total);
  for (const auto &instance : instances) {
    const auto &samples = kind == instance_sample_kind::shot ? instance.shot_us
                                                             : instance.tail_us;
    pooled.insert(pooled.end(), samples.begin(), samples.end());
  }
  return pooled;
} // end - concat_instance_samples()

instance_totals
sum_instance_results(const std::vector<instance_result> &instances) {
  instance_totals totals;

  for (const auto &instance : instances) {
    totals.shots += instance.shots;
    totals.logical_errors += instance.logical_errors;
  }
  return totals;
}

std::string
first_instance_error(const std::vector<instance_result> &instances) {
  for (std::size_t i = 0; i < instances.size(); ++i)
    if (!instances[i].error.empty())
      return "decoder instance " + std::to_string(i) + ": " +
             instances[i].error;
  return {};
}

std::size_t instance_shot_offset(std::size_t instance,
                                 std::size_t num_instances, std::size_t shots) {
  if (num_instances == 0 || shots == 0)
    return 0;
  return (instance % num_instances) * shots / num_instances;
}

} // namespace cudaq::qec::benchmark

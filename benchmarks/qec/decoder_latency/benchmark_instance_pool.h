/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Instance harness shared by the decoder benchmarks: runs N independent
// decoder instances concurrently, one thread each, and collects the
// per-instance latency samples the caller pools into a single report.
//
// An "instance" is one whole decoder, the unit a deployment scales when it
// decodes several logical qubits at once. It is deliberately not called a
// "lane", which commonly describes parallelism within a single decoder.
//
// An instance owns its decoder end to end.  The decoder base class pins the
// constructing thread persistently to the decoder's CUDA device (see
// decoder::get_cuda_device_id()) and the library calls that rule
// "one-thread-owns-one-decoder", so the thread that will drive a decoder is
// also the thread that has to build it.  run_instances() therefore splits an
// instance's work into a setup phase (construction plus warmup, untimed) and a
// run phase (timed), with a start gate in between so that every instance
// measures itself while all the others are loaded.
//
// Everything here is deliberately free of decoder types: the benchmark supplies
// the per-instance work as callables, which keeps the threading, pinning, and
// pooling logic unit-testable on its own.

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace cudaq::qec::benchmark {

/// @brief Timed samples and outcome counters produced by a single instance.
struct instance_result {
  /// @brief Per-shot latency in microseconds, one entry per timed shot.
  std::vector<double> shot_us;
  /// @brief Per-shot tail latency in microseconds (streaming mode: final round
  /// plus corrections readout). Batch mode leaves this empty.
  std::vector<double> tail_us;
  /// @brief Microseconds this instance spent in its own timed region, entering
  /// to leaving its run phase. Construction, warmup, thread spawn and teardown
  /// all fall outside it, as does the wait every other instance's stragglers
  /// impose -- an aggregate rate is therefore the sum of the per-instance
  /// rates, not the total work over one shared span. 0 when the instance never
  /// ran or failed, so a rate taken against it has to guard the division.
  double wall_us = 0.0;
  /// @brief Timed shots this instance completed.
  std::size_t shots = 0;
  /// @brief Shots whose predicted observable mask missed the sampled mask.
  std::size_t logical_errors = 0;
  /// @brief Empty on success, otherwise why this instance produced no samples.
  std::string error;
};

/// @brief The work one instance performs, split so construction and warmup
/// stay outside the timed region.
struct instance_work {
  /// @brief Called on the instance's own thread before the start gate. Owns
  /// decoder construction and warmup. May be empty.
  std::function<void(std::size_t instance)> setup;
  /// @brief Called on the same thread once every instance has finished setup.
  /// Fills the instance's samples and counters.
  std::function<void(std::size_t instance, instance_result &result)> run;
};

/// @brief Results of one run of the pool, one entry per instance.
struct instance_pool_result {
  /// @brief One entry per instance, in instance order.
  std::vector<instance_result> instances;
};

/// @brief How decoder instances are laid out across CPUs.
///
/// Instance i is given @p width consecutive CPUs starting at
/// @p base + i * stride.  Width exists because a decoder's internal threads
/// inherit their creator's affinity mask, so a width of 1 confines all of them
/// to a single CPU; a decoder run with num_threads=8 needs a width to match.
/// Stride exists to control the gap between instances -- most usefully to skip
/// SMT siblings, since consecutive CPU *ids* are not necessarily distinct
/// physical cores.
struct core_layout {
  /// @brief First CPU of instance 0. Negative disables pinning entirely,
  /// matching core_pinning in realtime/pipeline.h.
  int base = -1;
  /// @brief Consecutive CPUs each instance may run on. 0 is treated as 1.
  std::size_t width = 1;
  /// @brief Distance between the first CPUs of consecutive instances. 0 means
  /// "same as width", which packs instances back to back without overlap.
  std::size_t stride = 0;
};

/// @brief Resolve core_layout::stride, applying the "0 means width" default.
std::size_t effective_stride(const core_layout &layout);

/// @brief CPUs one instance effectively owns.
///
/// This is the share a decoder should size itself against when it was asked to
/// size itself from the machine (the num_threads=0 sentinel), since each
/// instance resolving that sentinel on its own would claim the whole machine
/// and N instances would then ask for N machines' worth of threads.
///
/// @param num_instances Concurrent instances sharing the machine; 0 counts as
/// 1.
/// @param layout CPU placement. When pinning is on the answer is exactly the
/// block width, which is all the instance is allowed to use.
/// @param num_cpus Visible CPU count; 0 (unknown) yields 1 rather than a
/// guess.
std::size_t cpus_per_instance(std::size_t num_instances,
                              const core_layout &layout, unsigned num_cpus);

/// @brief Worker threads to substitute for one instance's num_threads=0.
///
/// The instance's CPU share less a reserve, because the instance thread is not
/// a bystander: it runs a slice of every fork-join phase, spins on the join
/// barrier afterwards, and busy-waits between paced rounds, so a pool as wide
/// as the share leaves it competing with its own workers and makes every
/// barrier pay for the preemption. The reserve is at least one CPU and grows
/// to the four hardware threads a decoder holds back from a whole machine,
/// which keeps a narrow block from giving up most of itself.
///
/// Never 0, since that is the sentinel meaning "size yourself from the
/// machine" -- the request being substituted in the first place.
///
/// @param num_instances Concurrent instances sharing the machine; 0 counts as
/// 1.
/// @param layout CPU placement, as for cpus_per_instance().
/// @param num_cpus Visible CPU count; 0 (unknown) yields 1 rather than a
/// guess.
std::size_t worker_threads_per_instance(std::size_t num_instances,
                                        const core_layout &layout,
                                        unsigned num_cpus);

/// @brief CPUs this process is allowed to run on, ascending.
///
/// Automatic placement has to choose CPUs, so it asks the OS which ones it may
/// actually use rather than assuming the machine's whole range: a cgroup
/// cpuset rejects a bind outside itself, and a taskset mask would be silently
/// widened instead. Empty when the query fails or is unsupported.
std::vector<int> allowed_cpus();

/// @brief Divide @p cpus evenly among @p num_instances, a block each.
///
/// Instances tile the range from its first CPU upwards, so the layout depends
/// on the instance count and has to be recomputed for each row of a sweep.
///
/// @param num_instances Instances to place; 0 disables pinning.
/// @param cpus Assignable CPUs, as returned by allowed_cpus(). Fewer CPUs than
/// instances disables pinning, since blocks would have to overlap.
/// @returns A layout with base at the first CPU and width
/// cpus.size() / num_instances, or pinning disabled when there is nothing to
/// divide.
/// @throws std::runtime_error when @p cpus is not a contiguous run, which a
/// base-plus-stride layout cannot describe.
core_layout divide_cpus(std::size_t num_instances,
                        const std::vector<int> &cpus);

/// @brief A contiguous run of CPUs one instance may run on.
struct cpu_block {
  /// @brief First CPU in the block, or -1 when pinning is disabled.
  int first = -1;
  /// @brief How many consecutive CPUs the block covers, 0 when disabled.
  std::size_t count = 0;
};

/// @brief Run @p num_instances copies of @p work concurrently, one thread each.
///
/// Each instance pins itself (when @p layout enables pinning), runs
/// @p work.setup, waits at the start gate until every instance has done the
/// same, then runs @p work.run and records how long that took in its own
/// instance_result::wall_us. An instance that fails to pin or throws during
/// setup records the reason in its instance_result::error and skips the run
/// phase without stalling the others.
///
/// @param num_instances Number of concurrent decoder instances. 0 yields an
/// empty result.
/// @param layout CPU placement for the instances.
/// @param work Per-instance setup and timed run callables.
/// @throws std::runtime_error when the requested cores cannot be honored.
instance_pool_result run_instances(std::size_t num_instances,
                                   const core_layout &layout,
                                   const instance_work &work);

/// @brief Resolve the CPUs an instance pins to under @p layout.
/// @param instance Zero-based instance index.
/// @param layout CPU placement, whose base may be negative to disable pinning.
/// @param num_cpus Visible CPU count used for range checking; 0 skips the
/// check (std::thread::hardware_concurrency() is allowed to return 0).
/// @returns The block to bind to, or a disabled block when pinning is off.
/// @throws std::runtime_error when the block runs past the visible CPUs or
/// cannot be represented in a cpu_set_t.
cpu_block resolve_instance_cpus(std::size_t instance, const core_layout &layout,
                                unsigned num_cpus);

/// @brief Bind the calling thread to @p block.
/// @param block CPUs to allow, or a disabled block for no-op.
/// @returns An empty string on success, otherwise a description of the
/// failure. Reported rather than thrown because this runs on an instance
/// thread.
std::string pin_current_thread(const cpu_block &block);

/// @brief Which sample vector of an instance_result to read.
enum class instance_sample_kind {
  shot, ///< instance_result::shot_us
  tail, ///< instance_result::tail_us
};

/// @brief Concatenate one sample vector from every instance, in order.
///
/// Pooling the samples is what makes the reported percentiles describe the
/// whole machine under load: a p99 taken from one instance says nothing about
/// the other instances contending with it.
std::vector<double>
concat_instance_samples(const std::vector<instance_result> &instances,
                        instance_sample_kind kind);

/// @brief Shot and logical-error counts summed over all instances.
struct instance_totals {
  std::size_t shots = 0;
  std::size_t logical_errors = 0;
};

/// @brief Sum the per-instance shot and logical-error counters.
instance_totals
sum_instance_results(const std::vector<instance_result> &instances);

/// @brief First instance failure, prefixed with the instance index.
/// @returns An empty string when every instance succeeded.
std::string first_instance_error(const std::vector<instance_result> &instances);

/// @brief First timed shot an instance starts on.
///
/// Instance i starts i * shots / num_instances shots into the pool and wraps,
/// so concurrent instances do not walk the shared syndrome pool in lockstep
/// while still decoding the same multiset of shots (which keeps the logical
/// error rate comparable across instance counts).
std::size_t instance_shot_offset(std::size_t instance,
                                 std::size_t num_instances, std::size_t shots);

} // namespace cudaq::qec::benchmark

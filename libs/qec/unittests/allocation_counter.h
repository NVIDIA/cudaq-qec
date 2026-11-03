/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Counting replacements for the global allocation functions, plain and
// aligned, so a test can check that a call allocates nothing. Include this
// from exactly one source file of a test binary: it defines operator new and
// delete, which a second file would define again. Under a sanitizer, whose
// allocator replaces operator new itself, it replaces nothing and
// QEC_TEST_SANITIZED is 1, so a test can skip.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

// Whether a sanitizer runs this build. Its allocator replaces operator new,
// aborts on a refused allocation instead of returning null, and reserves
// address space of its own.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define QEC_TEST_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define QEC_TEST_SANITIZED 1
#endif
#endif
#ifndef QEC_TEST_SANITIZED
#define QEC_TEST_SANITIZED 0
#endif

namespace allocation_counter {

/// Whether this thread's allocations are being counted.
inline thread_local bool counting = false;
/// This thread's allocations counted so far.
inline thread_local std::size_t counted = 0;

/// Whether every thread's allocations are being counted, and how many.
inline std::atomic<bool> counting_all = false;
inline std::atomic<std::size_t> counted_all = 0;

/// The allocations made on any thread while @p run runs, as a graph's pool
/// threads make them. Does not nest. Always 0 under a sanitizer.
template <typename Run>
std::size_t on_every_thread_during(const Run &run) {
  struct restore {
    ~restore() { counting_all = false; }
  } guard;
  counted_all = 0;
  counting_all = true;
  run();
  counting_all = false;
  return counted_all;
}

/// The allocations @p run makes on this thread. Calls nest: an inner count
/// is included in the outer one. Always 0 under a sanitizer.
template <typename Run>
std::size_t during(const Run &run) {
  // Restores the flag however run() returns, an exception included.
  struct restore {
    bool was_counting;
    ~restore() { counting = was_counting; }
  } guard{counting};
  const auto before = counted;
  counting = true;
  run();
  return counted - before;
}

} // namespace allocation_counter

#if !QEC_TEST_SANITIZED
// The global allocation functions, counting; operator new[] and the nothrow
// forms call these.
void *operator new(std::size_t size) {
  void *p = std::malloc(size == 0 ? 1 : size);
  if (allocation_counter::counting)
    ++allocation_counter::counted;
  if (allocation_counter::counting_all.load(std::memory_order_relaxed))
    allocation_counter::counted_all.fetch_add(1, std::memory_order_relaxed);
  if (!p)
    throw std::bad_alloc();
  return p;
}

// aligned_alloc takes a size that is a whole number of alignments.
void *operator new(std::size_t size, std::align_val_t align) {
  const auto alignment = static_cast<std::size_t>(align);
  const auto rounded =
      (std::max(size, std::size_t{1}) + alignment - 1) / alignment * alignment;
  void *p = std::aligned_alloc(alignment, rounded);
  if (allocation_counter::counting)
    ++allocation_counter::counted;
  if (allocation_counter::counting_all.load(std::memory_order_relaxed))
    allocation_counter::counted_all.fetch_add(1, std::memory_order_relaxed);
  if (!p)
    throw std::bad_alloc();
  return p;
}

void operator delete(void *p) noexcept { std::free(p); }

void operator delete(void *p, std::size_t) noexcept { std::free(p); }

void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }

void operator delete(void *p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
#endif

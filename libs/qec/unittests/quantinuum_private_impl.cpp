/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdint>

extern "C" {

__attribute__((visibility("default"))) void
enqueue_syndromes_ui64_private(std::uint64_t, std::uint64_t, std::uint64_t,
                               std::uint64_t) {}

__attribute__((visibility("default"))) std::uint64_t
get_corrections_ui64_private(std::uint64_t, std::uint64_t, std::uint64_t) {
  return 0x5;
}

__attribute__((visibility("default"))) void
reset_decoder_ui64_private(std::uint64_t) {}

} // extern "C"

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FIXME - this is a hack to allow us to dlopen the
// libcudaq-qec-realtime-decoding-quantinuum.so without having to link in the
// whole CUDA-Q runtime when deploying a decoding server to a QPU provider.
// It is expected that one would preload this .so file before loading the
// libcudaq-qec-realtime-decoding-quantinuum.so in that environment.
extern "C" {
__attribute__((visibility("default"))) void __quantum__qis__x__ctl() {}
__attribute__((visibility("default"))) void __quantum__qis__y__ctl() {}
__attribute__((visibility("default"))) void __quantum__qis__z__ctl() {}
}

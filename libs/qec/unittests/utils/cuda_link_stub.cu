/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Stub file: gives CMake a .cu source so the CUDA device-link step
// produces a real link command for targets that only have .cpp sources
// but link against static libraries containing device code.

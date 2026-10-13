/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

// Symbols that libcudaq-qec-realtime-decoding-quantinuum.so leaves undefined.
// The GPU Server preloads the stubs library instead of the CUDA-Q runtime.
extern "C" {
void __cudaq_deviceCodeHolderAdd();
void __cudaq_registerLinkableKernel();
void __cudaq_registerRunnableKernel();
void cudaqRegisterArgsCreator();
void cudaqRegisterKernelName();
}

namespace cudaq::detail {
class YamlTargetConfigDisabler {
public:
  YamlTargetConfigDisabler();
};
} // namespace cudaq::detail

// A return from each call means that symbol is exported by the stubs library.
TEST(CudaqStubs, CallsTheExportedCudaqSymbols) {
  __cudaq_deviceCodeHolderAdd();
  __cudaq_registerLinkableKernel();
  __cudaq_registerRunnableKernel();
  cudaqRegisterArgsCreator();
  cudaqRegisterKernelName();
  cudaq::detail::YamlTargetConfigDisabler disabler;
  (void)disabler;
}

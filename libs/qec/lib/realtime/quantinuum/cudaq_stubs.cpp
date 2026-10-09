/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// FIXME - this is a hack to allow us to dlopen the
// libcudaq-qec-realtime-decoding-quantinuum.so without having to link in the
// whole CUDA-Q runtime when deploying a decoding server to a QPU provider.
// It is expected that one would preload this .so file before loading the
// libcudaq-qec-realtime-decoding-quantinuum.so in that environment.
extern "C" {
__attribute__((visibility("default"))) void __cudaq_deviceCodeHolderAdd() {}
__attribute__((visibility("default"))) void __cudaq_registerLinkableKernel() {}
__attribute__((visibility("default"))) void __cudaq_registerRunnableKernel() {}
__attribute__((visibility("default"))) void cudaqRegisterArgsCreator() {}
__attribute__((visibility("default"))) void cudaqRegisterKernelName() {}
}

// nvq++ always defines CUDAQ_DISABLE_YAML_TARGET_CONFIG, so every kernel
// translation unit gets a static initializer for the inline global in
// cudaq/host_config.h whose constructor lives in libcudaq.so. It only turns
// off YAML target config parsing, which the GPU Server never does.
namespace cudaq::detail {
class YamlTargetConfigDisabler {
public:
  YamlTargetConfigDisabler();
};

__attribute__((visibility("default")))
YamlTargetConfigDisabler::YamlTargetConfigDisabler() {}
} // namespace cudaq::detail

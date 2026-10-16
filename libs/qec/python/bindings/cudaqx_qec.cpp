/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "py_code.h"
#include "py_decoder.h"
#include "py_decoding.h"
#include "py_decoding_config.h"
#include "py_dem_construction.h"
#include "py_dem_sampling.h"
#include "py_extended_dem.h"
#include "py_surface_code.h"

// Only built when cudaq-realtime is available; see qec/python/CMakeLists.txt.
#ifdef CUDAQX_QEC_HAS_CUDAQ_REALTIME
#include "py_playback_emulator.h"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

NB_MODULE(_pycudaqx_qec_the_suffix_matters_cudaq_qec, mod) {
  mod.doc() = "Python bindings for the CUDA-Q QEC Libraries.";
  // Ensure cudaq is loaded so its nanobind-registered types are available
  // before any QEC binding tries to return or consume them.
  nanobind::module_::import_("cudaq");
  cudaq::qec::bindCode(mod);
  cudaq::qec::bindDecoder(mod);
  cudaq::qec::decoding::config::bindDecodingConfig(mod);
  cudaq::qec::decoding::bindDecoding(mod);
  cudaq::qec::dem_sampler::bindDemSampling(mod);
  cudaq::qec::surface_code::bindSurfaceCode(mod);
  cudaq::qec::bindDemConstruction(mod);
  cudaq::qec::bindExtendedDem(mod);
#ifdef CUDAQX_QEC_HAS_CUDAQ_REALTIME
  cudaq::qec::playback::bindPlaybackEmulator(mod);
#endif
  // Suppress nanobind's reference-leak warnings.
  //
  // Background: nanobind runs leak detection when its internal state capsule is
  // destroyed during Python interpreter shutdown. At that point it reports any
  // nanobind-managed objects (instances, types, functions) whose reference
  // counts are still non-zero. This module imports cudaq, so cudaq_runtime
  // types (SpinOperatorTerm, CompiledModule, NoiseModel, …) are visible to
  // nanobind's checker here. Those types are kept alive by objects in the
  // calling script's global scope, which Python only clears *after* extension
  // modules are torn down — hence the ordering mismatch.
  //
  // Reproduction: run
  //   docs/sphinx/examples/qec/python/custom_repetition_code_fine_grain_noise.py
  // without this call and nanobind will print "leaked N instances / types /
  // functions" walls at exit. The warnings disappear if the script explicitly
  // releases its module-level QEC objects before the interpreter shuts down:
  //
  //   import gc
  //   del my_repetition_code, noise_model, dem_rep, syndromes, data
  //   gc.collect()
  //
  // That proves the objects are properly owned and would be freed naturally
  // moments later — this is not a genuine reference-counting bug. Suppressing
  // the warning here is the correct fix for the cross-module cleanup ordering
  // false positive.
  nanobind::set_leak_warnings(false);
}

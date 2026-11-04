/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CUDA-Q-free python module: the decoder API, decoding configuration, DEM
// construction from matrices, extended DEMs and the decoding task graph. It
// links only libcudaq-qec-decoders and libcudaq-qec-realtime-decoding, so
// `import cudaq_qec` works without a CUDA-Q install. The CUDA-Q-dependent
// bindings (codes, experiments, device calls, ...) live in
// _pycudaqx_qec_the_suffix_matters_cudaq_qec, which binds into this module's
// `qecrt` submodule.

#include "py_decoder.h"
#include "py_decoding_config.h"
#include "py_decoding_task_graph.h"
#include "py_dem_construction.h"
#include "py_extended_dem.h"

#include "cudaq/qec/version.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sstream>

NB_MODULE(_qec_core, mod) {
  mod.doc() = "Python bindings for the CUDA-Q QEC Libraries (CUDA-Q-free "
              "part).";
  cudaq::qec::bindDecoder(mod);
  cudaq::qec::decoding::config::bindDecodingConfig(mod);
  cudaq::qec::bindDemConstruction(mod);
  cudaq::qec::bindExtendedDem(mod);
  cudaq::qec::bindDecodingTaskGraph(mod);

  std::stringstream ss;
  ss << "CUDA-Q QEC " << cudaq::qec::getVersion() << " ("
     << cudaq::qec::getFullRepositoryVersion() << ")";
  mod.attr("qecrt").attr("__version__") = ss.str();
  nanobind::set_leak_warnings(false);
}

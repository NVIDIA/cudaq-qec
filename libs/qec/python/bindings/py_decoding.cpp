/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "py_decoding.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cudaq/python/PythonCppInteropDecls.h"

#include "cudaq/qis/qubit_qis.h"

namespace nb = nanobind;

namespace cudaq::qec::decoding {

void bindDecoding(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  // We split the bindings of device kernels into 2 phases:
  // 1. Register the device kernels as valid Python functions: this is done once
  // per module load
  // 2. Register the device kernel Quake code: this is done via
  // `load_device_kernels` function call. As the quake code is populated into
  // the quake registry after an appropriate library is loaded, we need to
  // ensure that the quake code is registered after the library is loaded.
  nb::module_ qecSubMod = [&]() {
    const char *subModName = "qec";
    if (nb::hasattr(mod, subModName))
      return nb::cast<nb::module_>(mod.attr(subModName));
    else
      return mod.def_submodule(subModName);
  }();
  const std::string qecModName =
      nb::cast<std::string>(qecSubMod.attr("__name__"));
  // Define these kernels as part of the qec submodule
  qecSubMod.def(
      "reset_decoder", [](std::uint64_t) {},
      R"pbdoc(Reset the decoder with the given ID.)pbdoc");
  qecSubMod.def(
      "enqueue_syndromes",
      [](std::uint64_t, std::vector<bool>, std::uint64_t) {},
      R"pbdoc(Reset the decoder with the given ID.)pbdoc"
      R"pbdoc(Enqueue a vector of syndrome bit for realtime decoding.
                Parameters
                - decoder_id: The ID of the decoder.
                - syndromes: A vector of syndrome bits (0 or 1).
                - tag: An optional tag for the enqueue operation.
                        )pbdoc");
  // Mirrors the C++ `enqueue_syndromes_test(uint64_t, vector<bool>, uint64_t)`
  // entry point. The Python device kernel must pre-discriminate any
  // `list[measure_handle]` via `cudaq.to_bools(...)` before the call. Used by
  // app examples that already own bool-typed syndrome bits (file replay or
  // explicit kernel-side discrimination) where the production
  // `enqueue_syndromes(vector<measure_result>)` shape is the wrong fit.
  qecSubMod.def(
      "enqueue_syndromes_test",
      [](std::uint64_t, std::vector<bool>, std::uint64_t) {},
      R"pbdoc(Test-only entry point for enqueueing pre-discriminated syndrome
              bits. Production code should use `enqueue_syndromes` instead.
                Parameters
                - decoder_id: The ID of the decoder.
                - syndromes: A vector of syndrome bits (0 or 1).
                - tag: An optional tag for the enqueue operation.
                        )pbdoc");
  qecSubMod.def(
      "get_corrections", [](std::uint64_t, std::uint64_t, bool) {},
      R"pbdoc(Get the corrections from the decoder.
                Parameters
                - decoder_id: The ID of the decoder.
                - return_size: The number of corrections to return.
                - reset: Whether to reset the decoder after getting corrections.
                        )pbdoc");

  qecmod.def("load_device_kernels", [qecModName]() {
    cudaq::python::registerDeviceKernel(
        qecModName, "reset_decoder",
        cudaq::python::getMangledArgsString<std::uint64_t>());
    cudaq::python::registerDeviceKernel(
        qecModName, "enqueue_syndromes",
        cudaq::python::getMangledArgsString<std::uint64_t, std::vector<bool>,
                                            std::uint64_t>());
    cudaq::python::registerDeviceKernel(
        qecModName, "enqueue_syndromes_test",
        cudaq::python::getMangledArgsString<std::uint64_t, std::vector<bool>,
                                            std::uint64_t>());
    cudaq::python::registerDeviceKernel(
        qecModName, "get_corrections",
        cudaq::python::getMangledArgsString<std::uint64_t, std::uint64_t,
                                            bool>());
  });

  qecmod.def("__repr__", [qecmod]() {
    return "<qecrt.decoding (realtime decoding API bindings)>";
  });
}
} // namespace cudaq::qec::decoding

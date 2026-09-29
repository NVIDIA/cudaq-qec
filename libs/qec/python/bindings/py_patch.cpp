/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "py_patch.h"

#include "cudaq/qec/patch.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cudaq::qec {
void bindPatch(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  mod.doc() = "Python bindings for CUDA-Q QEC patch";

  nb::class_<cudaq::qec::patch>(
      qecmod, "patch",
      "Represents a logical qubit patch for quantum error correction.\n"
      "Fields are cudaq.qview objects:\n"
      " - data : data-qubit view\n"
      " - ancx : X-stabilizer ancilla view\n"
      " - ancz : Z-stabilizer ancilla view")
      .def_rw("data", &cudaq::qec::patch::data,
              "View of the data qubits in the patch")
      .def_rw("ancx", &cudaq::qec::patch::data,
              "View of the ancilla qubits for X stabilizers")
      .def_rw("ancz", &cudaq::qec::patch::data,
              "View of the ancilla qubits for Z stabilizers")
      .def("__repr__", [](const cudaq::qec::patch &p) {
        std::string s = "<cudaq.qec.patch";
        try {
          s += " data=" + std::to_string(p.data.size());
          s += " ancx=" + std::to_string(p.ancx.size());
          s += " ancz=" + std::to_string(p.ancz.size());
        } catch (...) {
        }
        s += ">";
        return s;
      });
}
} // namespace cudaq::qec

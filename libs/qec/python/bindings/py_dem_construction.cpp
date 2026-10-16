/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Python bindings for CSS code matrix types and DEM construction functions.
// css_code_matrices and css_noise_params are exposed as simple structs whose
// sparse_binary_matrix fields can be set from numpy arrays or scipy sparse
// matrices. dem_from_css_matrices accepts either a css_code_matrices or a
// code object directly.

#include "py_dem_construction.h"
#include "sparse_matrix_casters.h"
#include "type_casters.h"
#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_construction.h"
#include "cudaq/qec/dem_construction_code.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec {

void bindDemConstruction(nb::module_ &mod) {

  // -------------------------------------------------------------------------
  // css_code_matrices
  // -------------------------------------------------------------------------
  // sparse_binary_matrix is not a Python type, so each matrix field is a
  // property that reads back as a dense NumPy array and accepts either a
  // dense array or any scipy sparse format on assignment.
#define CSS_MATRIX_PROPERTY(name, doc)                                         \
  def_prop_rw(                                                                 \
      #name,                                                                   \
      [](const css_code_matrices &self) {                                      \
        return sparse_binary_matrix_to_numpy(self.name);                       \
      },                                                                       \
      [](css_code_matrices &self, nb::object value) {                          \
        self.name = sparse_binary_matrix_from_python(std::move(value));        \
      },                                                                       \
      nb::rv_policy::move, doc)

  nb::class_<css_code_matrices>(
      mod, "CssCodes",
      "CSS code generator matrices: parity-check and logical operators.\n\n"
      "Each matrix column corresponds to one data qubit. Assign each matrix\n"
      "from a dense NumPy array or a scipy sparse matrix, or build the whole\n"
      "set with css_matrices_from_code(). Reading a matrix back returns a\n"
      "dense uint8 NumPy array.")
      .def(nb::init<>())
      .CSS_MATRIX_PROPERTY(hz, "Z stabilizers [n_z_checks x n_qubits]")
      .CSS_MATRIX_PROPERTY(hx, "X stabilizers [n_x_checks x n_qubits]")
      .CSS_MATRIX_PROPERTY(lz, "Z logical operators [k x n_qubits]")
      .CSS_MATRIX_PROPERTY(lx, "X logical operators [k x n_qubits]");

#undef CSS_MATRIX_PROPERTY

  // -------------------------------------------------------------------------
  // css_noise_params
  // -------------------------------------------------------------------------
  nb::class_<css_noise_params>(
      mod, "CssNoise",
      "Phenomenological noise parameters for CSS code-capacity DEM\n"
      "construction. Scalar rates apply uniformly; per-element vectors\n"
      "override when non-empty (length must equal n_qubits or n_checks).")
      .def(nb::init<>())
      .def_rw("px", &css_noise_params::px, "Uniform X data-qubit error rate")
      .def_rw("py", &css_noise_params::py, "Uniform Y data-qubit error rate")
      .def_rw("pz", &css_noise_params::pz, "Uniform Z data-qubit error rate")
      .def_rw("pm", &css_noise_params::pm,
              "Uniform syndrome measurement error rate per check per round")
      .def_rw("px_per_qubit", &css_noise_params::px_per_qubit,
              "Per-qubit X rates (overrides px when non-empty)")
      .def_rw("py_per_qubit", &css_noise_params::py_per_qubit,
              "Per-qubit Y rates")
      .def_rw("pz_per_qubit", &css_noise_params::pz_per_qubit,
              "Per-qubit Z rates")
      .def_rw("pm_per_check", &css_noise_params::pm_per_check,
              "Per-check measurement error rates (Z-checks first)");

  // -------------------------------------------------------------------------
  // dem_from_css_matrices — matrix-based overload
  // -------------------------------------------------------------------------
  mod.def(
      "dem_from_css_matrices",
      [](const css_code_matrices &code, const css_noise_params &noise,
         std::size_t num_rounds) {
        return dem_from_css_matrices(code, noise, num_rounds);
      },
      nb::arg("code"), nb::arg("noise"), nb::arg("num_rounds") = 1,
      "Build a T-round code-capacity DEM from CSS generator matrices.\n\n"
      "Args:\n"
      "    code:       CssCodes instance with hz/hx/lz/lx matrices.\n"
      "    noise:      CssNoise instance with error rates.\n"
      "    num_rounds: Number of syndrome measurement rounds (default 1).\n"
      "Returns:\n"
      "    detector_error_model ready for any decoder.");

  // -------------------------------------------------------------------------
  // css_matrices_from_code — bridge from code objects
  // -------------------------------------------------------------------------
  mod.def("css_matrices_from_code", &css_matrices_from_code, nb::arg("code"),
          "Extract CSS generator matrices from a QEC code object.\n\n"
          "Calls code.get_parity_z/x() and code.get_observables_z/x() and\n"
          "wraps the results as CssCodes.\n\n"
          "Args:\n"
          "    code: A cudaq_qec.Code instance (repetition, surface, etc.).\n"
          "Returns:\n"
          "    CssCodes instance.");

  // -------------------------------------------------------------------------
  // dem_from_css_matrices — code-object overload
  // -------------------------------------------------------------------------
  mod.def(
      "dem_from_css_matrices",
      [](const code &qec_code, const css_noise_params &noise,
         std::size_t num_rounds) {
        return dem_from_css_matrices(qec_code, noise, num_rounds);
      },
      nb::arg("code"), nb::arg("noise"), nb::arg("num_rounds") = 1,
      "Build a T-round code-capacity DEM directly from a code object.\n\n"
      "Equivalent to dem_from_css_matrices(css_matrices_from_code(code),\n"
      "noise, num_rounds).\n\n"
      "Args:\n"
      "    code:       A cudaq_qec.Code instance.\n"
      "    noise:      CssNoise instance.\n"
      "    num_rounds: Number of syndrome rounds (default 1).\n"
      "Returns:\n"
      "    detector_error_model.");
}

} // namespace cudaq::qec

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_surface_code.h"
#include "cudaq_type_casters.h"
#include "type_casters.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// Disable this warning: "annotate" attribute directive ignored [-Wattributes]
// These warnings occur because GCC does not understand the __qpu__ attribute.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#include "cudaq/qec/codes/surface_code.h"
#pragma GCC diagnostic pop

namespace nb = nanobind;

namespace cudaq::qec::surface_code {
static nb::dict
map_vec2d_to_py_tuple_dict(const std::map<vec2d, std::size_t> &m) {
  nb::dict d;
  for (const auto &kv : m) {
    const auto &k = kv.first;
    d[nb::make_tuple(k.row, k.col)] = kv.second;
  }
  return d;
}

template <typename Fn>
std::string capture_print(Fn &&fn) {
  std::ostringstream oss;
  auto *old_buf = std::cout.rdbuf(oss.rdbuf());
  fn();
  std::cout.rdbuf(old_buf);
  return oss.str();
}

void bindSurfaceCode(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  nb::enum_<surface_role>(qecmod, "surface_role")
      .value("amx", surface_role::amx)
      .value("amz", surface_role::amz)
      .value("empty", surface_role::empty)
      .export_values();

  nb::enum_<sc_orientation>(qecmod, "sc_orientation")
      .value("XV", sc_orientation::XV)
      .value("XH", sc_orientation::XH)
      .value("ZV", sc_orientation::ZV)
      .value("ZH", sc_orientation::ZH)
      .export_values();

  nb::class_<vec2d>(qecmod, "vec2d")
      .def(nb::init<int, int>(), nb::arg("row"), nb::arg("col"))
      .def_rw("row", &vec2d::row)
      .def_rw("col", &vec2d::col)
      .def("__repr__",
           [](const vec2d &v) {
             return "Vec2D(row=" + std::to_string(v.row) +
                    ", col=" + std::to_string(v.col) + ")";
           })
      .def("__add__", [](const vec2d &a, const vec2d &b) { return a + b; })
      .def("__sub__", [](const vec2d &a, const vec2d &b) { return a - b; })
      .def("__eq__", [](const vec2d &a, const vec2d &b) { return a == b; });

  nb::class_<stabilizer_grid>(qecmod, "stabilizer_grid")
      .def(nb::init<>())
      .def(nb::init<std::uint32_t, sc_orientation>(), nb::arg("distance"),
           nb::arg("orientation") = sc_orientation::ZH)
      .def(
          "__init__",
          [](stabilizer_grid &self, std::uint32_t distance,
             const std::string &orientation) {
            new (&self)
                stabilizer_grid(distance, sc_orientation_from_str(orientation));
          },
          nb::arg("distance"), nb::arg("orientation"))
      .def_ro("distance", &stabilizer_grid::distance)
      .def_prop_ro("orientation", &stabilizer_grid::get_orientation)
      .def_ro("grid_length", &stabilizer_grid::grid_length)
      .def_ro("roles", &stabilizer_grid::roles)
      .def_ro("x_stab_coords", &stabilizer_grid::x_stab_coords)
      .def_ro("z_stab_coords", &stabilizer_grid::z_stab_coords)
      .def_ro("data_coords", &stabilizer_grid::data_coords)
      .def_ro("x_stabilizers", &stabilizer_grid::x_stabilizers)
      .def_ro("z_stabilizers", &stabilizer_grid::z_stabilizers)
      .def_prop_ro("x_stab_indices",
                   [](const stabilizer_grid &g) {
                     return map_vec2d_to_py_tuple_dict(g.x_stab_indices);
                   })
      .def_prop_ro("z_stab_indices",
                   [](const stabilizer_grid &g) {
                     return map_vec2d_to_py_tuple_dict(g.z_stab_indices);
                   })
      .def_prop_ro("data_indices",
                   [](const stabilizer_grid &g) {
                     return map_vec2d_to_py_tuple_dict(g.data_indices);
                   })
      .def("format_stabilizer_grid",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizer_grid(); });
           })
      .def("format_stabilizer_coords",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizer_coords(); });
           })
      .def("format_stabilizer_indices",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizer_indices(); });
           })
      .def("format_data_grid",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_data_grid(); });
           })
      .def("format_stabilizers",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizers(); });
           })
      .def("__repr__",
           [](const stabilizer_grid &g) {
             return "<StabilizerGrid distance=" + std::to_string(g.distance) +
                    " grid_length=" + std::to_string(g.grid_length) + ">";
           })
      .def("get_spin_op_stabilizers", &stabilizer_grid::get_spin_op_stabilizers,
           "Return the stabilizers as a list of cudaq::spin_op_term")
      .def("get_spin_op_observables", &stabilizer_grid::get_spin_op_observables,
           "Return the logical observables as [X, Z] cudaq::spin_op_term "
           "entries")
      .def(
          "get_cnot_schedule_x",
          [](const stabilizer_grid &g) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                g.get_cnot_schedule_x());
          },
          "Return the X-stabilizer CNOT schedule matrix as a numpy array "
          "whose rows match the sorted parity-matrix rows. Entry 0 = no "
          "support, k >= 1 = CNOT timestep, ordered so that ancilla (hook) "
          "errors land perpendicular to the logical operators.")
      .def(
          "get_cnot_schedule_z",
          [](const stabilizer_grid &g) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                g.get_cnot_schedule_z());
          },
          "Return the Z-stabilizer CNOT schedule matrix as a numpy array "
          "whose rows match the sorted parity-matrix rows. Entry 0 = no "
          "support, k >= 1 = CNOT timestep, ordered so that ancilla (hook) "
          "errors land perpendicular to the logical operators.")
      .def("get_cnot_schedule_pairs_x",
           &stabilizer_grid::get_cnot_schedule_pairs_x,
           "Return the X-stabilizer CNOT schedule as a flat list of "
           "(stabilizer index, data index) pairs ordered by timestep within "
           "each stabilizer.")
      .def("get_cnot_schedule_pairs_z",
           &stabilizer_grid::get_cnot_schedule_pairs_z,
           "Return the Z-stabilizer CNOT schedule as a flat list of "
           "(stabilizer index, data index) pairs ordered by timestep within "
           "each stabilizer.");

  qecmod.def("role_to_str", [](surface_role r) {
    switch (r) {
    case surface_role::amx:
      return "X";
    case surface_role::amz:
      return "Z";
    default:
      return "e";
    }
  });
}
} // namespace cudaq::qec::surface_code

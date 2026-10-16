/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

// Casters for CUDA-Q runtime result types (spin_op, sample_result,
// observe_result). These need CUDA-Q headers and the cudaq python module at
// runtime, so they live apart from type_casters.h: translation units that
// bind only the decoder API (and are built without a CUDA-Q install) include
// type_casters.h alone.

#include <cstdint>
#include <string>
#include <vector>

#include "common/ObserveResult.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nanobind {
namespace detail {

template <>
struct type_caster<cudaq::spin_op> {
  NB_TYPE_CASTER(cudaq::spin_op, const_name("SpinOperator"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto data = nanobind::cast<std::vector<double>>(src.attr("serialize")());
      value = cudaq::spin_op(data);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::spin_op v, rv_policy, cleanup_list *) noexcept {
    try {
      nanobind::object tv_py = nanobind::module_::import_("cudaq").attr(
          "SpinOperator")(v.get_data_representation());
      return tv_py.release();
    } catch (...) {
      return handle();
    }
  }
};

template <>
struct type_caster<cudaq::sample_result> {
  NB_TYPE_CASTER(cudaq::sample_result, const_name("SampleResult"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto data =
          nanobind::cast<std::vector<std::size_t>>(src.attr("serialize")());
      value = cudaq::sample_result();
      value.deserialize(data);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::sample_result v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nanobind::object tv_py =
          nanobind::module_::import_("cudaq").attr("SampleResult")();
      tv_py.attr("deserialize")(v.serialize());
      return tv_py.release();
    } catch (...) {
      return handle();
    }
  }
};

template <>
struct type_caster<cudaq::observe_result> {
  NB_TYPE_CASTER(cudaq::observe_result, const_name("ObserveResult"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto e = nanobind::cast<double>(src.attr("expectation")());
      value = cudaq::observe_result(e, cudaq::spin_op());
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::observe_result v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nanobind::object tv_py = nanobind::module_::import_("cudaq").attr(
          "ObserveResult")(v.expectation(), v.get_spin(), v.raw_data());
      return tv_py.release();
    } catch (...) {
      return handle();
    }
  }
};

} // namespace detail
} // namespace nanobind

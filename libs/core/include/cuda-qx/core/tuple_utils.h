/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <tuple>

namespace cudaqx {

template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<std::size_t,
                           std::tuple_size<typename std::remove_reference<
                               TupleType>::type>::value>) {}
// Utility function for looping over tuple elements
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
// Utility function for looping over tuple elements
void tuple_for_each(TupleType &&t, FunctionType f,
                    std::integral_constant<std::size_t, I>) {
  f(std::get<I>(t));
  tuple_for_each(std::forward<TupleType>(t), f,
                 std::integral_constant<size_t, I + 1>());
}
// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
  tuple_for_each(std::forward<TupleType>(t), f,
                 std::integral_constant<size_t, 0>());
}

} // namespace cudaqx

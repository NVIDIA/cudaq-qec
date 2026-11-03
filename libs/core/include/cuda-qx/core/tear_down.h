/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace cudaqx {

/// @brief A base class for tear down services to be run at application
/// shutdown.
///
/// This class is designed to be subclassed with concrete tear down routines.
/// Instances of subclasses should be submitted to a global registry which will
/// be run at application shutdown.
class tear_down {
public:
  /// @brief Pure virtual function to be implemented by derived classes.
  ///
  /// This function should contain the specific tear down logic for each
  /// service.
  virtual void runTearDown() const = 0;

  virtual ~tear_down() = default;
};

/// @brief Schedule a new tear down routine
void scheduleTearDown(std::unique_ptr<tear_down>);

} // namespace cudaqx

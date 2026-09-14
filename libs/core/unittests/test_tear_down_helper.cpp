/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cuda-qx/core/tear_down.h"

#include <fstream>
#include <memory>
#include <string>

namespace {

class marker_tear_down : public cudaqx::tear_down {
public:
  explicit marker_tear_down(std::string path) : path_(std::move(path)) {}

  void runTearDown() const override {
    std::ofstream out(path_);
    out << "torn-down\n";
  }

private:
  std::string path_;
};

} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return 2;
  cudaqx::scheduleTearDown(std::make_unique<marker_tear_down>(argv[1]));
  return 0;
}

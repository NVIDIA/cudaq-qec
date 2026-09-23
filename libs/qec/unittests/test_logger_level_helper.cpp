/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/logger.h"

#include <cstring>
#include <iostream>
#include <string>

// Fresh-process helper: the first should_log() parses CUDAQ_LOG_LEVEL.
int main(int argc, char **argv) {
  if (argc < 2)
    return 2;
  using cudaq::qec::detail::log_level;
  using cudaq::qec::detail::should_log;
  const std::string want = argv[1];
  const bool t = should_log(log_level::trace);
  const bool d = should_log(log_level::debug);
  const bool i = should_log(log_level::info);
  const bool w = should_log(log_level::warn);
  const bool e = should_log(log_level::error);
  auto expect = [&](bool et, bool ed, bool ei, bool ew, bool ee) {
    return t == et && d == ed && i == ei && w == ew && e == ee;
  };
  bool ok = false;
  if (want == "trace")
    ok = expect(true, true, true, true, true);
  else if (want == "debug")
    ok = expect(false, true, true, true, true);
  else if (want == "info")
    ok = expect(false, false, true, true, true);
  else if (want == "warning")
    ok = expect(false, false, false, true, true);
  else if (want == "error")
    ok = expect(false, false, false, false, true);
  else if (want == "invalid")
    ok = expect(false, false, false, true, true);
  std::cout << want << " t=" << t << " d=" << d << " i=" << i << " w=" << w
            << " e=" << e << "\n";
  return ok ? 0 : 1;
}

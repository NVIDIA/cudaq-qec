/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace qec_cc {

inline std::string lut_yaml(int id = 0, const char *dispatch = "host") {
  return std::string("decoders:\n  - id: ") + std::to_string(id) +
         "\n    type: single_error_lut\n    dispatch: " + dispatch +
         "\n    block_size: 1\n    syndrome_size: 1\n"
         "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, "
         "-1]\n";
}

inline std::string mixed_lut_yaml() {
  return lut_yaml(0, "host") +
         "  - id: 1\n    type: single_error_lut\n    dispatch: device_graph\n"
         "    block_size: 1\n    syndrome_size: 1\n"
         "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, "
         "-1]\n";
}

inline std::string write_temp(const std::string &contents,
                              const char *suffix = ".yaml") {
  static std::atomic<int> n{0};
  auto path = std::filesystem::temp_directory_path() /
              ("qec-cc-" + std::to_string(::getpid()) + "-" +
               std::to_string(n++) + suffix);
  std::ofstream out(path);
  out << contents;
  return path.string();
}

// Returns the child's exit status, or a negative value on wait/fork failure.
inline int
exec_self(const std::string &helper,
          const std::vector<std::pair<std::string, std::string>> &env = {}) {
  const pid_t pid = ::fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    for (const auto &kv : env)
      ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
    char exe[4096];
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n < 0)
      _exit(127);
    exe[n] = '\0';
    std::string arg = "--qec-helper=" + helper;
    char *argv[] = {exe, arg.data(), nullptr};
    ::execv(exe, argv);
    _exit(127);
  }
  int st = 0;
  if (::waitpid(pid, &st, 0) != pid)
    return -1;
  if (WIFEXITED(st))
    return WEXITSTATUS(st);
  return 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
}

inline const char *helper_name(int argc, char **argv) {
  constexpr const char kPrefix[] = "--qec-helper=";
  const std::size_t n = sizeof(kPrefix) - 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], kPrefix, n) == 0)
      return argv[i] + n;
  }
  return nullptr;
}

} // namespace qec_cc

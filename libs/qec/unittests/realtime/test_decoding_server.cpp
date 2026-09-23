/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// Two-process realtime decoding-server test: this process runs only the
/// caller (simulated QPU) side; the decoder lives in a separate
/// decoding_server process reached through a CUDA-Q device_call
/// channel.  Both the decoder and the transport are configuration:
///   - the decoder comes from the YAML config file handed to the server
///     (swapping decoders is a config-file change, not a code change);
///   - the transport defaults to `udp` (loopback; runs anywhere) and can be
///     switched to the CPU RoCE RDMA wire with
///       QEC_DECODING_SERVER_TRANSPORT=cpu_roce
///     plus the RDMA topology env vars shared with CUDA-Q's
///     CpuRoceChannelTester:
///       CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE / CUDAQ_CPU_ROCE_TEST_CHANNEL_IP
///       CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE  / CUDAQ_CPU_ROCE_TEST_DAEMON_IP
///
/// The server is spawned as a subprocess, its ephemeral port read from the
/// QEC_DECODING_SERVER_READY stdout line, and its dispatch count (printed at
/// shutdown) is the proof the device_calls crossed the process boundary --
/// there is no decoder configured in this process at all.
///
/// The kernel's block/syndrome size and expected correction must stay
/// consistent with the H/O/D matrices in the config file (3-bit identity:
/// syndrome bit 1 set -> correction bit 1 set for any sane decoder).

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/realtime.h"
#include <fstream>
#include <gtest/gtest.h>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kRunShots = 1;
constexpr std::int64_t kExpectedCorrection = 1;

__qpu__ std::int64_t decoding_server_kernel() {
  constexpr std::uint64_t kKernelDecoderId = 0;
  constexpr std::uint64_t kKernelBlockSize = 3;
  constexpr std::uint64_t kKernelSyndromeSize = 3;
  constexpr std::uint64_t kKernelSyndromeTag = 1;
  constexpr std::size_t kKernelActiveSyndromeIndex = 1;

  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/kKernelDecoderId);

  std::vector<bool> syndrome(kKernelSyndromeSize);
  for (std::size_t i = 0; i < kKernelSyndromeSize; ++i)
    syndrome[i] = false;
  syndrome[kKernelActiveSyndromeIndex] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(
      /*decoder_id=*/kKernelDecoderId, syndrome, /*tag=*/kKernelSyndromeTag);

  auto corrections = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/kKernelDecoderId, /*return_size=*/kKernelBlockSize,
      /*reset=*/true);
  return corrections[kKernelActiveSyndromeIndex] ? std::int64_t{1}
                                                 : std::int64_t{0};
}

std::string env_or(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return value && *value ? std::string(value) : fallback;
}

class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    if (const char *old_value = std::getenv(name))
      old_value_ = old_value;
    ::setenv(name, value, 1);
  }

  ~ScopedEnv() {
    if (old_value_)
      ::setenv(name_.c_str(), old_value_->c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> old_value_;
};

// The server binary path is baked in at configure time (the server is built
// from libs/qec/tools/decoding-server); QEC_DECODING_SERVER overrides it. The
// example decoder configs are placed in the same directory as the server.
std::string server_path() {
  return env_or("QEC_DECODING_SERVER", QEC_DECODING_SERVER_PATH);
}

std::string server_dir() {
  const std::string path = server_path();
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

// Spawns decoding_server with the given decoder config file, hands back
// its READY port (udp: the UDP data port; cpu_roce: the TCP rendezvous port),
// and collects its stdout (for the shutdown dispatch-count line).
class ServerProcess {
public:
  // `transport_cli` = false launches the server WITHOUT --transport or its
  // default provider args, exercising configs whose wire is named and
  // configured by the YAML transport section instead of the command line.
  // `capture_stderr` folds the server's stderr into `captured` (used by the
  // conflict-rejection test to see the startup error).
  bool start(const std::string &config_file, std::string &error,
             int ready_timeout_ms = 15000, bool transport_cli = true,
             bool capture_stderr = false,
             const std::vector<std::string> &extra_provider_args = {}) {
    int out_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0) {
      error = "pipe() failed";
      return false;
    }
    pid = ::fork();
    if (pid < 0) {
      error = "fork() failed";
      return false;
    }
    if (pid == 0) {
      ::dup2(out_pipe[1], STDOUT_FILENO);
      if (capture_stderr)
        ::dup2(out_pipe[1], STDERR_FILENO);
      ::close(out_pipe[0]);
      ::close(out_pipe[1]);
      const std::string server = server_path();
      std::vector<std::string> args = {
          server, "--config=" + (!config_file.empty() && config_file[0] == '/'
                                     ? config_file
                                     : server_dir() + "/" + config_file)};
      if (transport_cli) {
        args.push_back("--transport=" +
                       env_or("QEC_DECODING_SERVER_TRANSPORT", "udp"));
        args.push_back("--device=" +
                       env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE", "mlx5_0"));
        args.push_back("--local-ip=" +
                       env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_IP", "10.0.0.2"));
        args.push_back("--port=0");
      }
      args.insert(args.end(), extra_provider_args.begin(),
                  extra_provider_args.end());
      args.push_back("--timeout=60");
      std::vector<char *> argv;
      for (auto &a : args)
        argv.push_back(a.data());
      argv.push_back(nullptr);
      ::execv(server.c_str(), argv.data());
      std::perror("execv decoding_server");
      _exit(127);
    }
    ::close(out_pipe[1]);
    outFd = out_pipe[0];

    // Read stdout until the READY line (or the deadline; decoders that
    // build TensorRT engines at startup need more than the default 15 s).
    std::string ready_line;
    if (!readLineWithPrefix("QEC_DECODING_SERVER_READY", ready_timeout_ms,
                            ready_line)) {
      error = "server did not print QEC_DECODING_SERVER_READY; output so "
              "far: " +
              captured;
      return false;
    }
    if (std::sscanf(ready_line.c_str(), "QEC_DECODING_SERVER_READY port=%hu",
                    &port) != 1) {
      error = "could not parse server port from: " + ready_line;
      return false;
    }
    // One-ring-per-decoder servers additionally publish ring<id>=<port>
    // tokens (one per decoder).
    {
      std::istringstream tokens(ready_line);
      std::string token;
      while (tokens >> token) {
        long long id = -1;
        unsigned ring_port = 0;
        if (std::sscanf(token.c_str(), "ring%lld=%u", &id, &ring_port) == 2)
          ring_ports[id] = static_cast<std::uint16_t>(ring_port);
      }
    }
    return true;
  }

  // Terminate the server and return its dispatched-request count (-1 if the
  // shutdown line never appeared). Also captures the per-decoder-worker
  // concurrency high-water mark into max_concurrent_decoders.
  std::int64_t stopAndGetDispatchCount(int num_rings = 0) {
    if (pid <= 0)
      return -1;
    ::kill(pid, SIGTERM);
    std::string line;
    std::int64_t count = -1;
    // The per-ring QEC_DECODING_SERVER_RING lines precede the DISPATCHED
    // line; collect them when the caller expects a multi-ring server.
    for (int i = 0; i < num_rings; ++i) {
      if (!readLineWithPrefix("QEC_DECODING_SERVER_RING", 10000, line))
        break;
      long long id = -1, dispatched = -1;
      if (std::sscanf(line.c_str(),
                      "QEC_DECODING_SERVER_RING decoder=%lld dispatched=%lld",
                      &id, &dispatched) == 2)
        ring_dispatched[id] = dispatched;
    }
    if (readLineWithPrefix("QEC_DECODING_SERVER_DISPATCHED", 10000, line)) {
      long long parsed = -1;
      if (std::sscanf(line.c_str(), "QEC_DECODING_SERVER_DISPATCHED count=%lld",
                      &parsed) == 1)
        count = parsed;
    }
    if (readLineWithPrefix("QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS", 5000,
                           line)) {
      long long parsed = -1;
      if (std::sscanf(line.c_str(),
                      "QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count=%lld",
                      &parsed) == 1)
        max_concurrent_decoders = parsed;
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    pid = -1;
    if (outFd >= 0) {
      ::close(outFd);
      outFd = -1;
    }
    return count;
  }

  // Reap the child and return its exit code (-1: no child / signal exit).
  // Only meaningful when the server exited on its own (e.g. rejected its
  // command line); the READY-path tests use stopAndGetDispatchCount.
  int exitCode() {
    if (pid <= 0)
      return -1;
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid)
      return -1;
    pid = -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  }

  ~ServerProcess() {
    if (pid > 0) {
      ::kill(pid, SIGKILL);
      int status = 0;
      ::waitpid(pid, &status, 0);
    }
    if (outFd >= 0)
      ::close(outFd);
  }

  std::uint16_t port = 0;
  std::map<long long, std::uint16_t> ring_ports;
  std::map<long long, std::int64_t> ring_dispatched;
  std::string captured;
  std::int64_t max_concurrent_decoders = -1;

private:
  bool readLineWithPrefix(const char *prefix, int timeout_ms,
                          std::string &line_out) {
    std::string line;
    const auto deadline_ms = timeout_ms;
    int waited_ms = 0;
    while (waited_ms < deadline_ms) {
      pollfd pfd{outFd, POLLIN, 0};
      const int ready = ::poll(&pfd, 1, 100);
      waited_ms += 100;
      if (ready <= 0)
        continue;
      char c = 0;
      ssize_t n = 0;
      while ((n = ::read(outFd, &c, 1)) == 1) {
        captured.push_back(c);
        if (c == '\n') {
          if (line.rfind(prefix, 0) == 0) {
            line_out = line;
            return true;
          }
          line.clear();
        } else {
          line.push_back(c);
        }
        pollfd inner{outFd, POLLIN, 0};
        if (::poll(&inner, 1, 0) <= 0)
          break;
      }
      if (n == 0)
        return false; // EOF: the server exited without printing the prefix
    }
    return false;
  }

  pid_t pid = -1;
  int outFd = -1;
};

struct RealtimeGuard {
  bool armed = false;
  ~RealtimeGuard() {
    if (armed)
      cudaq::realtime::finalize();
  }
};

} // namespace

// Caller-side device_call channel arguments for the selected transport.
std::vector<std::string> channel_arguments(std::uint16_t server_port) {
  const std::string transport = env_or("QEC_DECODING_SERVER_TRANSPORT", "udp");
  if (transport == "cpu_roce") {
    // The server's READY port is its TCP rendezvous port; the RDMA topology
    // comes from the same env vars CUDA-Q's CpuRoceChannelTester uses.
    // Unlike udp, the RDMA ring geometry is part of the wire contract: the
    // channel writes requests directly into the server's rings, so slots /
    // slot-size must match the server's --num-slots / --slot-size defaults
    // (8 x 256, see decoding_server.cpp ServerConfig).
    return {"--cudaq-device-call=cpu_roce",
            "--cudaq-device-call-slots=8",
            "--cudaq-device-call-slot-size=256",
            "ib-device=" +
                env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE", "mlx5_0"),
            "local-ip=" + env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_IP", "10.0.0.1"),
            "rendezvous-host=" +
                env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_IP", "10.0.0.2"),
            "rendezvous-port=" + std::to_string(server_port)};
  }
  return {"--cudaq-device-call=udp", "udp-host=127.0.0.1",
          "udp-port=" + std::to_string(server_port)};
}

// Runs the full two-process round-trip against a server configured with
// `config_file`. Each invocation spawns a fresh server on an ephemeral port.
void run_two_process_decode_test(const std::string &config_file) {
  ServerProcess server;
  std::string error;
  ASSERT_TRUE(server.start(config_file, error)) << error;

  std::vector<std::string> args = {"test_decoding_server"};
  for (auto &arg : channel_arguments(server.port))
    args.push_back(std::move(arg));
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, decoding_server_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], kExpectedCorrection);

  // Two-process self-verification: the decode can only have happened in the
  // server (this process configured no decoders), and the server's dispatch
  // counter proves the device_calls crossed the selected transport. Three
  // calls: reset_decoder, enqueue_syndromes, get_corrections.
  const std::int64_t dispatched = server.stopAndGetDispatchCount();
  EXPECT_GE(dispatched, 3) << "server output:\n" << server.captured;
}

TEST(DecodingServerTwoProcess, TwoProcessHostDispatch) {
  run_two_process_decode_test("decoding_server_config.yaml");
}

TEST(DecodingServerTwoProcess, TwoProcessHostDispatchMultiErrorLut) {
  run_two_process_decode_test("decoding_server_config_multi_error_lut.yaml");
}

// ---------------------------------------------------------------------------
// Two decoders (two logical qubits) in ONE server, driven from ONE __qpu__
// kernel: qubit A uses decoder 0 and qubit B uses decoder 1, with different
// active syndrome bits. On the server each decoder executes on its own
// per-decoder worker thread; the server's shutdown
// QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS line reports the busy
// high-water mark of those workers.
// ---------------------------------------------------------------------------

__qpu__ std::int64_t dual_decoding_server_kernel() {
  constexpr std::uint64_t kDecoderA = 0;
  constexpr std::uint64_t kDecoderB = 1;
  constexpr std::uint64_t kBlockSize = 3;
  constexpr std::uint64_t kSyndromeSize = 3;
  constexpr std::size_t kActiveA = 1;
  constexpr std::size_t kActiveB = 2;

  cudaq::qec::decoding::reset_decoder(kDecoderA);
  cudaq::qec::decoding::reset_decoder(kDecoderB);

  std::vector<bool> syndrome_a(kSyndromeSize);
  std::vector<bool> syndrome_b(kSyndromeSize);
  for (std::size_t i = 0; i < kSyndromeSize; ++i) {
    syndrome_a[i] = false;
    syndrome_b[i] = false;
  }
  syndrome_a[kActiveA] = true;
  syndrome_b[kActiveB] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(kDecoderA, syndrome_a,
                                               /*tag=*/1);
  cudaq::qec::decoding::enqueue_syndromes_test(kDecoderB, syndrome_b,
                                               /*tag=*/1);

  auto corr_a = cudaq::qec::decoding::get_corrections(kDecoderA, kBlockSize,
                                                      /*reset=*/true);
  auto corr_b = cudaq::qec::decoding::get_corrections(kDecoderB, kBlockSize,
                                                      /*reset=*/true);
  std::int64_t out = 0;
  if (corr_a[kActiveA])
    out = out + 1; // bit 0: decoder 0 corrected its active bit
  if (corr_b[kActiveB])
    out = out + 2; // bit 1: decoder 1 corrected its active bit
  if (corr_a[kActiveB])
    out = out + 4; // bit 2 set = cross-talk (decoder 0 saw B's syndrome)
  if (corr_b[kActiveA])
    out = out + 8; // bit 3 set = cross-talk (decoder 1 saw A's syndrome)
  return out;
}

TEST(DecodingServerTwoProcess, TwoProcessHostDispatchDualDecoders) {
  // Two identical 3-bit-identity pymatching decoders (ids 0 and 1) in one
  // server -- one per logical qubit.
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_dual_config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "decoders:\n";
    for (int id = 0; id < 2; ++id) {
      config_file << "  - id: " << id << "\n"
                  << "    type: pymatching\n"
                  << "    block_size: 3\n"
                  << "    syndrome_size: 3\n"
                  << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    D_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    error_rate_vec: [0.1, 0.1, 0.1]\n"
                  << "    decoder_custom_args:\n"
                  << "      merge_strategy: smallest_weight\n";
    }
  }

  ServerProcess server;
  std::string error;
  ASSERT_TRUE(server.start(config_path, error)) << error;

  std::vector<std::string> args = {"test_decoding_server"};
  for (auto &arg : channel_arguments(server.port))
    args.push_back(std::move(arg));
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, dual_decoding_server_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], 3);

  // Six calls crossed the wire (reset/enqueue/get per decoder), and both
  // decoders' execution workers ran (high-water mark >= 1; == 2 when the
  // decodes genuinely overlapped, which tiny identity decodes need not).
  const std::int64_t dispatched = server.stopAndGetDispatchCount();
  EXPECT_GE(dispatched, 6) << "server output:\n" << server.captured;
  EXPECT_GE(server.max_concurrent_decoders, 1) << "server output:\n"
                                               << server.captured;
}

// The wire named by the YAML transport section, no --transport on the
// command line: the deployment file is the single source of truth for the
// provider (and its args), and the two-process decode still round-trips.
TEST(DecodingServerTwoProcess, TwoProcessHostDispatchYamlTransportSection) {
  if (env_or("QEC_DECODING_SERVER_TRANSPORT", "udp") != "udp")
    GTEST_SKIP() << "YAML-section test pins provider udp";
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_yaml_transport_config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "transport:\n"
                << "  provider: udp\n"
                << "  args: [--num-slots=8, --port=0]\n"
                << "decoders:\n";
    for (int id = 0; id < 2; ++id) {
      config_file << "  - id: " << id << "\n"
                  << "    type: pymatching\n"
                  << "    block_size: 3\n"
                  << "    syndrome_size: 3\n"
                  << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    D_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    error_rate_vec: [0.1, 0.1, 0.1]\n"
                  << "    decoder_custom_args:\n"
                  << "      merge_strategy: smallest_weight\n";
    }
  }

  ServerProcess server;
  std::string error;
  ASSERT_TRUE(server.start(config_path, error, 15000, /*transport_cli=*/false))
      << error;

  std::vector<std::string> args = {"test_decoding_server"};
  for (auto &arg : channel_arguments(server.port))
    args.push_back(std::move(arg));
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, dual_decoding_server_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], 3);

  const std::int64_t dispatched = server.stopAndGetDispatchCount();
  EXPECT_GE(dispatched, 6) << "server output:\n" << server.captured;
}

// Provider parsers start at argv[1]. A malformed first YAML argument must be
// observed and rejected rather than silently becoming the program name.
TEST(DecodingServerTwoProcess, YamlProviderParsesFirstArgument) {
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_yaml_first_arg.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "transport:\n"
                << "  provider: udp\n"
                << "  args: [--port=not-a-port]\n"
                << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/false,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
}

// A YAML that names its provider cannot be contradicted from the command
// line: --transport alongside a transport section is a startup error, not a
// silent precedence decision.
TEST(DecodingServerTwoProcess, TransportCliConflictsWithYamlSection) {
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_conflict_config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "transport:\n"
                << "  provider: udp\n"
                << "decoders:\n"
                << "  - id: 0\n"
                << "    type: pymatching\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    error_rate_vec: [0.1, 0.1, 0.1]\n"
                << "    decoder_custom_args:\n"
                << "      merge_strategy: smallest_weight\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000, /*transport_cli=*/true,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("conflicts with the transport section"),
            std::string::npos)
      << server.captured;
}

TEST(DecodingServerTwoProcess, DeviceGraphRequiresYamlTransportProvider) {
  ScopedEnv legacy_provider("CUDAQ_REALTIME_BRIDGE_LIB",
                            "/tmp/legacy-provider.so");
  ScopedEnv legacy_device("QEC_DEVICE_GRAPH_DEVICE", "mlx5_legacy");
  ScopedEnv legacy_peer_ip("QEC_DEVICE_GRAPH_PEER_IP", "192.0.2.1");
  ScopedEnv legacy_remote_qp("QEC_DEVICE_GRAPH_REMOTE_QP", "17");
  ScopedEnv legacy_frame_size("QEC_DEVICE_GRAPH_FRAME_SIZE", "384");
  ScopedEnv legacy_num_pages("QEC_DEVICE_GRAPH_NUM_PAGES", "64");

  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_device_graph_no_transport.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    dispatch: device_graph\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/false,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("must name its transport provider in the "
                                 "YAML transport section"),
            std::string::npos)
      << server.captured;
}

// A mixed-dispatch topology must not obtain its device_graph provider from the
// generic CLI fallback. This is the composed per-ring path rather than the
// standalone all-device_graph path covered above.
TEST(DecodingServerTwoProcess,
     MixedDispatchRequiresYamlDeviceGraphTransportProvider) {
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_mixed_no_transport.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "  - id: 1\n"
                << "    type: single_error_lut\n"
                << "    dispatch: device_graph\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/true,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("device_graph dispatch must name its "
                                 "transport provider in the YAML"),
            std::string::npos)
      << server.captured;
}

// Even when YAML names gpu_roce, generic provider arguments must not create a
// second configuration path into a device_graph ring.
TEST(DecodingServerTwoProcess, DeviceGraphRejectsCliProviderArguments) {
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_device_graph_cli_args.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "transport:\n"
                << "  provider: gpu_roce\n"
                << "  args: [--device=mlx5_yaml]\n"
                << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    dispatch: device_graph\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/false,
                            /*capture_stderr=*/true, {"--device=mlx5_cli"}))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("provider arguments for device_graph "
                                 "or gpu_roce transport must be set in the "
                                 "YAML"),
            std::string::npos)
      << server.captured;
}

// GPU placement belongs to decoder.cuda_device_id. Allowing --gpu in the
// transport arguments would put two contradictory settings in the same YAML.
TEST(DecodingServerTwoProcess, DeviceGraphRejectsTransportGpuArgument) {
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_device_graph_gpu_arg.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "transport:\n"
                << "  provider: gpu_roce\n"
                << "  device_graph:\n"
                << "    args: [--gpu=1]\n"
                << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    dispatch: device_graph\n"
                << "    cuda_device_id: 1\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/false,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("must not set --gpu; set decoder "
                                 "cuda_device_id in YAML instead"),
            std::string::npos)
      << server.captured;
}

// Host dispatch over gpu_roce has the same single source of GPU placement:
// decoder.cuda_device_id. Reject the duplicate transport setting before the
// provider is loaded so this regression needs no GPU or RoCE device.
TEST(DecodingServerTwoProcess, HostGpuRoceRejectsTransportGpuArgument) {
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_host_gpu_roce_gpu_arg.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "transport:\n"
                << "  provider: gpu_roce\n"
                << "  args: [--gpu=1]\n"
                << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    cuda_device_id: 1\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/false,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("gpu_roce transport arguments must not set "
                                 "--gpu; set decoder cuda_device_id in YAML "
                                 "instead"),
            std::string::npos)
      << server.captured;
}

// GPU RoCE remains YAML-only even if a caller tries to select it for the
// generic host-dispatch path.
TEST(DecodingServerTwoProcess, HostDispatchRejectsCliGpuRoceTransport) {
  ScopedEnv transport("QEC_DECODING_SERVER_TRANSPORT", "gpu_roce");
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_host_cli_gpu_roce.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "decoders:\n"
                << "  - id: 0\n"
                << "    type: single_error_lut\n"
                << "    block_size: 3\n"
                << "    syndrome_size: 3\n"
                << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                << "    D_sparse: [0, -1, 1, -1, 2, -1]\n";
  }

  ServerProcess server;
  std::string error;
  EXPECT_FALSE(server.start(config_path, error, 8000,
                            /*transport_cli=*/true,
                            /*capture_stderr=*/true))
      << "server unexpectedly reached READY: " << server.captured;
  EXPECT_NE(0, server.exitCode()) << server.captured;
  EXPECT_NE(server.captured.find("gpu_roce transport must be selected and "
                                 "configured in the YAML"),
            std::string::npos)
      << server.captured;
}

// ---------------------------------------------------------------------------
// ONE RING PER DECODER, TWO PROCESSES: the server opens one provider
// instance (one udp endpoint, one ring, one dispatcher) per decoder and
// publishes ring<id>=<port> tokens on its READY line; the caller wires each
// decoder's device_call session to its own endpoint via device-scoped
// channel arguments (udp-port.<id>=).  The per-ring dispatched counts on
// shutdown prove BOTH rings carried this test's traffic -- the property the
// shared-ring DualDecoders test cannot show.
// ---------------------------------------------------------------------------

TEST(DecodingServerTwoProcess, TwoProcessPerDecoderRings) {
  if (env_or("QEC_DECODING_SERVER_TRANSPORT", "udp") != "udp")
    GTEST_SKIP() << "per-decoder ring endpoints exercised over udp only";

  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_per_ring_config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "decoders:\n";
    for (int id = 0; id < 2; ++id) {
      config_file << "  - id: " << id << "\n"
                  << "    type: pymatching\n"
                  << "    block_size: 3\n"
                  << "    syndrome_size: 3\n"
                  << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    D_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    error_rate_vec: [0.1, 0.1, 0.1]\n"
                  << "    decoder_custom_args:\n"
                  << "      merge_strategy: smallest_weight\n";
    }
  }

  ServerProcess server;
  std::string error;
  ASSERT_TRUE(server.start(config_path, error)) << error;
  ASSERT_EQ(server.ring_ports.size(), 2u)
      << "server did not publish one ring endpoint per decoder";
  ASSERT_NE(server.ring_ports[0], server.ring_ports[1])
      << "decoders share one endpoint; rings are not per-decoder";

  // Device-scoped endpoints: decoder 0's session dials ring0, decoder 1's
  // session dials ring1 (device_id == decoder_id in the QEC wrappers).
  std::vector<std::string> args = {
      "test_decoding_server", "--cudaq-device-call=udp", "udp-host=127.0.0.1",
      "udp-port=" + std::to_string(server.ring_ports[0]),
      "udp-port.1=" + std::to_string(server.ring_ports[1])};
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, dual_decoding_server_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], 3);

  // The decisive assertions: EACH ring dispatched this test's three RPCs
  // (reset/enqueue/get per decoder) -- traffic was per-ring, not payload-
  // demuxed over one wire.
  const std::int64_t dispatched = server.stopAndGetDispatchCount(2);
  EXPECT_GE(dispatched, 6) << "server output:\n" << server.captured;
  ASSERT_EQ(server.ring_dispatched.size(), 2u) << "server output:\n"
                                               << server.captured;
  EXPECT_GE(server.ring_dispatched[0], 3) << "ring 0 idle; server output:\n"
                                          << server.captured;
  EXPECT_GE(server.ring_dispatched[1], 3) << "ring 1 idle; server output:\n"
                                          << server.captured;
}

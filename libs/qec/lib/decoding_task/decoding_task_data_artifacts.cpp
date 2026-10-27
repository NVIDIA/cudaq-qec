/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The data artifact store: SHA-256 (FIPS 180-4), the canonical bytes that
// identify a template, a window request and a window variant, and the store
// that keys its entries by them. Runs at configuration time and may
// allocate; nothing here is on the hot path.

#include "decoding_task_data_artifacts.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace cudaq::qec {
namespace {

// SHA-256 round constants: the first 32 bits of the fractional parts of the
// cube roots of the first 64 primes (FIPS 180-4, section 4.2.2).
constexpr std::array<std::uint32_t, 64> sha256_k{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Mixes one 64-byte block into the hash state (FIPS 180-4, section 6.2.2).
void sha256_block(std::array<std::uint32_t, 8> &h, const unsigned char *b) {
  std::array<std::uint32_t, 64> w{};
  auto s = h;
  for (std::size_t t = 0; t < 16; ++t)
    w[t] = std::uint32_t(b[4 * t]) << 24 | std::uint32_t(b[4 * t + 1]) << 16 |
           std::uint32_t(b[4 * t + 2]) << 8 | std::uint32_t(b[4 * t + 3]);
  for (std::size_t t = 16; t < 64; ++t) {
    const auto s0 =
        std::rotr(w[t - 15], 7) ^ std::rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
    const auto s1 =
        std::rotr(w[t - 2], 17) ^ std::rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
    w[t] = w[t - 16] + s0 + w[t - 7] + s1;
  }
  for (std::size_t t = 0; t < 64; ++t) {
    const auto e1 =
        std::rotr(s[4], 6) ^ std::rotr(s[4], 11) ^ std::rotr(s[4], 25);
    const auto ch = (s[4] & s[5]) ^ (~s[4] & s[6]);
    const auto t1 = s[7] + e1 + ch + sha256_k[t] + w[t];
    const auto a0 =
        std::rotr(s[0], 2) ^ std::rotr(s[0], 13) ^ std::rotr(s[0], 22);
    const auto maj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
    const auto t2 = a0 + maj;
    s = {t1 + t2, s[0], s[1], s[2], s[3] + t1, s[4], s[5], s[6]};
  }
  for (std::size_t i = 0; i < 8; ++i)
    h[i] += s[i];
} // end - sha256_block()

// Appends a field's name and value on a line of its own, so no two fields
// or values can run together.
void put(std::string &out, const char *name, const std::string &value) {
  out += name;
  out += '=';
  out += value;
  out += '\n';
}

// An integer list, comma-separated.
template <typename T>
std::string list(const std::vector<T> &values) {
  std::string out;
  for (const auto v : values)
    out += std::to_string(v) + ',';
  return out;
}

// A double by its bit pattern, so equal values give equal bytes. Zero is
// written as +0.0, since -0.0 compares equal to it.
std::string bits(double value) {
  return std::to_string(
      std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
}

// An optional value, or "-" when unset.
template <typename T, typename F>
std::string optional_of(const std::optional<T> &value, F &&format) {
  return value ? format(*value) : std::string("-");
}

// A round range as begin:end.
std::string range(experimental::round_range r) {
  return std::to_string(r.begin) + ':' + std::to_string(r.end);
}

// A string by its length and bytes, so it cannot be confused with the
// separators.
std::string text(const std::string &value) {
  return std::to_string(value.size()) + ':' + value;
}

// The canonical bytes of a window variant: its template's hash, its
// request's bytes and its variant key.
std::string window_bytes(const std::string &template_sha256,
                         const window_request &request,
                         const variant_key &key) {
  std::string out = "window\n";
  const auto index = [](std::int64_t v) { return std::to_string(v); };
  put(out, "template", template_sha256);
  out += canonical_bytes(request);
  put(out, "position", optional_of(key.position, index));
  put(out, "terminal", optional_of(key.terminal, index));
  return out;
}

} // namespace

// Hashes each whole 64-byte block where it lies, then pads the rest with a 1
// bit, zeros, and the message length in bits (FIPS 180-4, sections 5.1.1 and
// 6.2). Only the rest is copied: one tail block, or two when the 8-byte
// length does not fit after it.
std::string sha256_hex(std::string_view bytes) {
  std::array<std::uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                 0xa54ff53a, 0x510e527f, 0x9b05688c,
                                 0x1f83d9ab, 0x5be0cd19};
  const auto *data = reinterpret_cast<const unsigned char *>(bytes.data());
  const std::size_t whole = bytes.size() - bytes.size() % 64;
  const std::size_t rest = bytes.size() - whole;
  const std::size_t tail_size = rest < 56 ? 64 : 128;
  const std::uint64_t length = std::uint64_t(bytes.size()) * 8;
  std::array<unsigned char, 128> tail{};
  std::string out;
  char digit[9] = {};
  for (std::size_t i = 0; i < whole; i += 64)
    sha256_block(h, data + i);
  std::copy(data + whole, data + bytes.size(), tail.begin());
  tail[rest] = 0x80;
  for (std::size_t i = 0; i < 8; ++i)
    tail[tail_size - 1 - i] = static_cast<unsigned char>(length >> (8 * i));
  for (std::size_t i = 0; i < tail_size; i += 64)
    sha256_block(h, tail.data() + i);
  for (const auto word : h) {
    std::snprintf(digit, sizeof(digit), "%08x", word);
    out += digit;
  }
  return out;
} // end - sha256_hex()

// Every field of dem_chunks_spec and dem_chunk_spec, in declaration order.
std::string canonical_bytes(const dem_chunks_spec &spec) {
  std::string out = "dem_chunks\n";
  const auto number = [](std::uint64_t v) { return std::to_string(v); };
  for (const auto &entry : spec.phases) {
    const auto &p = entry.spec;
    put(out, "phase", std::to_string(entry.id.value));
    put(out, "num_faults", std::to_string(p.num_faults));
    put(out, "H_sparse", list(p.H_sparse));
    for (const auto &seam : p.seam_specs) {
      put(out, "seam_spec", std::to_string(seam.id.value));
      put(out, "seam_H_sparse", list(seam.spec.H_sparse));
      put(out, "seam_O_sparse", list(seam.spec.O_sparse));
    }
    put(out, "O_sparse", list(p.O_sparse));
    std::string rates;
    for (const auto rate : p.error_rates)
      rates += bits(rate) + ',';
    put(out, "error_rates", rates);
    put(out, "D_sparse", list(p.D_sparse));
    put(out, "num_measurements", optional_of(p.num_measurements, number));
    put(out, "m2o_sparse", list(p.m2o_sparse));
  } // end - for(entry)
  for (const auto &c : spec.connections)
    put(out, "connection",
        std::to_string(c.from_phase.value) + ',' +
            std::to_string(c.to_phase.value));
  put(out, "seam",
      std::to_string(spec.seam.from_seam.value) + ',' +
          std::to_string(spec.seam.to_seam.value));
  put(out, "num_rounds", optional_of(spec.num_rounds, number));
  put(out, "measurements_per_round",
      optional_of(spec.measurements_per_round, number));
  put(out, "phases_supply_D_sparse",
      optional_of(spec.phases_supply_D_sparse,
                  [](bool v) { return std::string(v ? "1" : "0"); }));
  for (const auto &name : spec.observable_names)
    put(out, "observable_name", text(name));
  return out;
} // end - canonical_bytes()

// Every field of window_request, in declaration order.
std::string canonical_bytes(const window_request &request) {
  std::string out = "window_request\n";
  const auto enumerator = [](auto v) {
    return std::to_string(static_cast<int>(v));
  };
  put(out, "period", std::to_string(request.period));
  put(out, "offset", std::to_string(request.offset));
  put(out, "commit", optional_of(request.commit, range));
  for (const auto &buffer : request.buffers)
    put(out, "buffer", range(buffer));
  put(out, "input_disposition",
      optional_of(request.input.disposition, enumerator));
  put(out, "input_prior", optional_of(request.input.prior, bits));
  put(out, "input_boundary", optional_of(request.input.boundary, enumerator));
  put(out, "output_disposition",
      optional_of(request.output.disposition, enumerator));
  put(out, "output_imports", optional_of(request.output.imports, range));
  for (const auto &send : request.sends) {
    std::string rounds;
    for (const auto &r : send.rounds)
      rounds += range(r) + ',';
    put(out, "send", text(send.id));
    put(out, "send_mode", optional_of(send.mode, enumerator));
    put(out, "send_rounds", rounds);
  }
  put(out, "merge_identical", request.columns.merge_identical ? "1" : "0");
  put(out, "conflicting_parallel",
      enumerator(request.columns.conflicting_parallel));
  return out;
} // end - canonical_bytes()

// A hash hit whose spec differs means canonical_bytes() missed a field.
std::shared_ptr<const experiment_template>
data_artifact_store::add_template(const dem_chunks_spec &spec) {
  const auto bytes = canonical_bytes(spec);
  auto hash = sha256_hex(bytes);
  const auto it = templates_.find(hash);
  if (it != templates_.end()) {
    if (it->second.spec != spec)
      throw std::logic_error(
          "data_artifact_store: two templates share sha256 " + hash +
          " but differ");
    return it->second.handle;
  }
  auto handle = std::make_shared<const experiment_template>(spec);
  records_.push_back({"dem_chunks", bytes.size(), hash});
  templates_.emplace(std::move(hash), template_entry{spec, handle});
  return handle;
}

void data_artifact_store::add_stim_dem(std::string_view text) {
  add_stim_dem(text, sha256_hex(text));
}

void data_artifact_store::add_stim_dem(std::string_view text,
                                       std::string sha256) {
  if (!stim_dems_.insert(sha256).second)
    return;
  records_.push_back({"stim_dem", text.size(), std::move(sha256)});
}

// Builds each variant key on the store's template, at the first shot length
// and instance that has it; variant_probe_rounds() bounds the lengths that
// hold every key. A hash hit whose request or key differs means the window
// bytes missed a field, as for templates. A window holding the readout round
// gets m2o when the template has a readout map: the key fixes where a window
// ends, so every instance of a variant agrees.
void data_artifact_store::add_windows(
    const std::shared_ptr<const experiment_template> &context,
    const window_request &request) {
  const auto hash = template_hash(context->spec(), "add_windows");
  const auto &t = templates_.at(hash).handle;
  const bool readout_map = !t->readout().empty();
  for (auto R = t->min_rounds(); R < variant_probe_rounds(*t, request); ++R) {
    const auto instances = window_instances(request, R);
    for (auto k = instances.begin; k < instances.end; ++k) {
      const auto key = window_variant(*t, request, k, R);
      const auto bytes = window_bytes(hash, request, key);
      auto window_hash = sha256_hex(bytes);
      if (const auto hit = windows_.find(window_hash); hit != windows_.end()) {
        if (hit->second.request != request || hit->second.key != key)
          throw std::logic_error("data_artifact_store: two window variants "
                                 "share sha256 " +
                                 window_hash + " but differ");
        continue;
      }
      const bool with_m2o =
          readout_map && place_window(t, request, k, R).rounds.end ==
                             static_cast<std::int64_t>(R);
      auto handle = std::make_shared<const decoding_window>(
          window_from_dem_chunks(t, request, k, R, with_m2o));
      records_.push_back({"window", bytes.size(), window_hash});
      windows_.emplace(std::move(window_hash),
                       window_entry{request, key, std::move(handle)});
    }
  } // end - for(R)
} // end - data_artifact_store::add_windows()

std::shared_ptr<const experiment_template>
data_artifact_store::find_template(const dem_chunks_spec &spec) const {
  return templates_.at(template_hash(spec, "find_template")).handle;
}

// Never builds: an absent variant is a configuration-time omission.
std::shared_ptr<const decoding_window>
data_artifact_store::find_window(const experiment_template &context,
                                 const window_request &request,
                                 const variant_key &key) const {
  const auto hash = template_hash(context.spec(), "find_window");
  const auto it = windows_.find(sha256_hex(window_bytes(hash, request, key)));
  if (it == windows_.end() || it->second.request != request ||
      it->second.key != key)
    throw std::logic_error("data_artifact_store: the window variant was not "
                           "built at configuration time");
  return it->second.handle;
}

// By content, as add_template() keys it, so an equal template built
// elsewhere names the same entry. Every lookup of a template comes through
// here.
std::string data_artifact_store::template_hash(const dem_chunks_spec &spec,
                                               const std::string &what) const {
  auto hash = sha256_hex(canonical_bytes(spec));
  const auto it = templates_.find(hash);
  if (it == templates_.end() || it->second.spec != spec)
    throw std::logic_error("data_artifact_store::" + what +
                           ": no template with this content was added");
  return hash;
}

} // namespace cudaq::qec

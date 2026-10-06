/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Build the plugin:
// g++ -std=c++17 -shared -fPIC per_round_decoder.cpp -I<cudaq-qec install>/include -L<cudaq-qec install>/lib -lcudaq-qec-decoders -o libper_round_decoder.so
// clang-format on

// A decoder for Streaming Interleaved Feed-forward Latency (SIFL) shots, whose
// number of stabilizer rounds is only known when the shot ends. It holds one
// sub-decoder per round count r, each built from a full DEM of an r-round
// circuit (read from `dem_dir/r<r>.txt`), and hands every shot to the one
// matching its length. A `round_width`-bit syndrome is one stabilizer round; a
// `terminal_width`-bit syndrome is the data readout that ends the shot and
// triggers the decode.

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/pcm_utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cudaq::qec {

namespace {
template <typename T>
std::vector<T> read_line(std::istream &in) {
  std::string line;
  std::getline(in, line);
  std::istringstream ss(line);
  std::vector<T> out;
  for (T v; ss >> v;)
    out.push_back(v);
  return out;
}

// Splits a -1 terminated flat sparse vector into its rows.
std::vector<std::vector<std::uint32_t>>
split_rows(const std::vector<std::int64_t> &flat) {
  std::vector<std::vector<std::uint32_t>> rows(1);
  for (auto v : flat)
    if (v == -1)
      rows.emplace_back();
    else
      rows.back().push_back(v);
  rows.pop_back();
  return rows;
}
} // namespace

class per_round_decoder : public decoder {
public:
  per_round_decoder(decoder_init inputs, decode_result_type output,
                    const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), output) {
    const auto dir = params.get<std::string>("dem_dir");
    round_width_ = params.get<std::uint64_t>("round_width");
    terminal_width_ = params.get<std::uint64_t>("terminal_width");
    const auto max_rounds = params.get<std::uint64_t>("max_rounds");
    const auto delegate =
        params.get<std::string>("delegate_type", "pymatching");

    // Each file holds four lines: H, O, D (-1 terminated rows) and the rates.
    for (std::uint64_t r = 1; r <= max_rounds; ++r) {
      std::ifstream f(dir + "/r" + std::to_string(r) + ".txt");
      if (!f)
        throw std::runtime_error("per_round_decoder: missing DEM for " +
                                 std::to_string(r) + " rounds");
      auto H = read_line<std::int64_t>(f);
      auto O = read_line<std::int64_t>(f);
      auto D = split_rows(read_line<std::int64_t>(f));
      auto rates = read_line<double>(f);
      const auto num_detectors = D.size();
      decoder_init sub(
          sparse_binary_matrix(
              pcm_from_sparse_vec(H, num_detectors, rates.size())),
          sparse_binary_matrix(pcm_from_sparse_vec(O, 1, rates.size())), rates,
          sparse_binary_matrix::from_nested_csr(
              num_detectors, r * round_width_ + terminal_width_, D));
      sub_decoders_.push_back(get_decoder(delegate, std::move(sub)));
    }
  }

  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t len) override {
    buffer_.insert(buffer_.end(), syndrome, syndrome + len);
    if (len == round_width_) {
      if (++rounds_ > sub_decoders_.size())
        throw std::runtime_error("per_round_decoder: too many rounds");
      return false;
    }
    if (len != terminal_width_ || rounds_ == 0)
      throw std::runtime_error("per_round_decoder: unexpected syndrome size " +
                               std::to_string(len));
    // Decode inline so the result is ready when this call returns.
    auto &sub = *sub_decoders_[rounds_ - 1];
    sub.enqueue_syndrome(buffer_.data(), buffer_.size());
    correction_ = *sub.get_obs_corrections();
    buffer_.clear();
    rounds_ = 0;
    return true;
  }

  void reset_decoder() override {
    decoder::reset_decoder();
    buffer_.clear();
    rounds_ = 0;
    correction_ = 0;
  }
  const uint8_t *get_obs_corrections() const override { return &correction_; }
  void clear_corrections() override { correction_ = 0; }
  // Decoding happens in enqueue_syndrome.
  decoder_result decode(const std::vector<float_t> &) override {
    return {true, std::vector<float_t>(block_size, 0.0f)};
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      per_round_decoder,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decode_result_type> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<per_round_decoder>(
            std::move(inputs), output.value_or(decode_result_type::errors),
            params);
      })

private:
  std::uint64_t round_width_ = 0, terminal_width_ = 0, rounds_ = 0;
  std::vector<std::unique_ptr<decoder>> sub_decoders_;
  std::vector<uint8_t> buffer_;
  uint8_t correction_ = 0;
};

CUDAQ_EXT_PT_REGISTER_TYPE(per_round_decoder)

namespace {
struct schema_registration {
  schema_registration() {
    using k = decoding::config::param_kind;
    decoding::config::register_decoder_schema({"per_round_decoder",
                                               {{"dem_dir", k::string},
                                                {"round_width", k::uint64},
                                                {"terminal_width", k::uint64},
                                                {"max_rounds", k::uint64},
                                                {"delegate_type", k::string}}});
  }
} registration;
} // namespace

} // namespace cudaq::qec

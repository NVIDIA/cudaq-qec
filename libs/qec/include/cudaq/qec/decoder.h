/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/extension_point.h"
#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/qec/cancellation.h"
#include "cudaq/qec/decoder_init.h"
#include <algorithm>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace cudaq::qec {

#if defined(CUDAQX_QEC_FLOAT_TYPE)
using float_t = CUDAQX_QEC_FLOAT_TYPE;
#else
using float_t = double;
#endif

/// @brief The requested decoder output contract.
enum class decode_result_type : std::uint8_t {
  errors,      ///< Primary result is an error frame.
  observables, ///< Primary result is an observable frame.
};

/// @brief Standardized auxiliary outputs a decoder may be requested to return.
/// Values are independent flags and may be combined.
enum class decoder_auxiliary_result_type : std::uint8_t {
  none = 0,
  /// Residual detector values in `opt_results["residual_detectors"]`.
  residual_detectors = 1u << 0,
};

constexpr decoder_auxiliary_result_type
operator|(decoder_auxiliary_result_type lhs,
          decoder_auxiliary_result_type rhs) noexcept {
  return static_cast<decoder_auxiliary_result_type>(
      static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

constexpr decoder_auxiliary_result_type
operator&(decoder_auxiliary_result_type lhs,
          decoder_auxiliary_result_type rhs) noexcept {
  return static_cast<decoder_auxiliary_result_type>(
      static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
}

constexpr decoder_auxiliary_result_type
operator~(decoder_auxiliary_result_type value) noexcept {
  return static_cast<decoder_auxiliary_result_type>(
      ~static_cast<std::uint8_t>(value));
}

/// @brief Immutable primary and auxiliary output contract for a decoder.
struct decoder_output_request {
  constexpr decoder_output_request(
      decode_result_type primary,
      decoder_auxiliary_result_type auxiliary =
          decoder_auxiliary_result_type::none) noexcept
      : primary(primary), auxiliary(auxiliary) {}

  decode_result_type primary;
  decoder_auxiliary_result_type auxiliary;

  constexpr bool requests(decoder_auxiliary_result_type output) const noexcept {
    return output != decoder_auxiliary_result_type::none &&
           (auxiliary & output) == output;
  }

  constexpr bool has_auxiliary_outputs() const noexcept {
    return auxiliary != decoder_auxiliary_result_type::none;
  }
};

/// @brief Validates that all keys in a heterogeneous map are found in a list of
/// acceptable types
/// @param config The heterogeneous map to validate
/// @param acceptable_types List of acceptable key types
/// @return Vector of invalid keys (empty if all keys are valid)
inline std::vector<std::string>
validate_config_parameters(const cudaqx::heterogeneous_map &config,
                           const std::vector<std::string> &acceptable_types) {
  std::vector<std::string> invalid_types;
  for (const auto &[key, _] : config) {
    if (std::find(acceptable_types.begin(), acceptable_types.end(), key) ==
        acceptable_types.end()) {
      invalid_types.push_back(key);
    }
  }
  return invalid_types;
}

/// @brief Decoder results
struct decoder_result {
  /// @brief Whether or not the decoder converged.
  bool converged = false;

  /// @brief Decoder values in the instance's construction-time output basis.
  /// Error results have length `block_size`; observable results have length
  /// `get_num_observables()`.
  std::vector<float_t> result;

  /// @brief Optional additional results from the decoder stored in a
  /// heterogeneous map. For equality comparison, this field is treated as a
  /// boolean flag - two decoder_results are considered equal only if both have
  /// empty opt_results (either std::nullopt or an empty map). If either result
  /// has non-empty opt_results, they are considered not equal.
  std::optional<cudaqx::heterogeneous_map> opt_results;

  // Manually define the equality operator
  bool operator==(const decoder_result &other) const {
    // First compare the non-optional fields
    if (std::tie(converged, result) !=
        std::tie(other.converged, other.result)) {
      return false;
    }
    // If both do not have opt_results or both have empty maps, then they are
    // equal
    bool this_empty = !opt_results.has_value() || opt_results->size() == 0;
    bool other_empty =
        !other.opt_results.has_value() || other.opt_results->size() == 0;
    if (this_empty && other_empty) {
      return true;
    }
    // Otherwise, they are not equal
    return false;
  }

  // Manually define the inequality operator
  bool operator!=(const decoder_result &other) const {
    return !(*this == other);
  }
};

/// @brief Return type for asynchronous decoding results
class async_decoder_result {
public:
  std::future<cudaq::qec::decoder_result> fut;

  /// @brief Construct an async_decoder_result from a std::future.
  /// @param f A rvalue reference to a std::future containing a decoder_result.
  async_decoder_result(std::future<cudaq::qec::decoder_result> &&f)
      : fut(std::move(f)) {}

  async_decoder_result(async_decoder_result &&other) noexcept
      : fut(std::move(other.fut)) {}

  async_decoder_result &operator=(async_decoder_result &&other) noexcept {
    if (this != &other) {
      fut = std::move(other.fut);
    }

    return *this;
  }

  /// @brief Block until the decoder result is ready and retrieve it.
  /// Wait until the underlying future is ready and then
  /// return the stored decoder result.
  /// @return The decoder_result obtained from the asynchronous operation.
  cudaq::qec::decoder_result get() { return fut.get(); }

  /// @brief Check if the asynchronous result is ready.
  /// @return `true` if the future is ready, `false` otherwise.
  bool ready() {
    return fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }
};

/// @brief The `decoder` base class should be subclassed by specific decoder
/// implementations. The `heterogeneous_map` provides a placeholder for
/// arbitrary constructor parameters that can be unique to each specific
/// decoder.
class decoder
    : public cudaqx::extension_point<decoder, decoder_init,
                                     std::optional<decoder_output_request>,
                                     const cudaqx::heterogeneous_map &> {
private:
  struct rt_impl;
  struct rt_impl_deleter {
    void operator()(rt_impl *p) const;
  };
  std::unique_ptr<rt_impl, rt_impl_deleter> pimpl;

public:
  decoder() = delete;

  /// @brief Constructor
  /// @param inputs Stable model and measurement inputs. Taken by value so the
  /// factory can move its immutable handle into the decoder.
  /// @param requested_output The result basis this instance produces, fixed
  /// for its lifetime.
  decoder(decoder_init inputs,
          decoder_output_request requested_output = decoder_output_request{
              decode_result_type::errors});

  /// @brief Decode a single syndrome
  /// @param syndrome A vector of syndrome measurements where the floating point
  /// value is the probability that the syndrome measurement is a |1>. The
  /// length of the syndrome vector should be equal to `syndrome_size`.
  /// @returns A result in the form this instance was constructed for. A
  /// decoder that cannot produce that form rejects construction, so this never
  /// negotiates the form per call.
  virtual decoder_result decode(const std::vector<float_t> &syndrome) = 0;

  /// @brief Decode a single syndrome
  /// @param syndrome An order-1 tensor of syndrome measurements where a 1 bit
  /// represents that the syndrome measurement is a |1>. The
  /// length of the syndrome vector should be equal to `syndrome_size`.
  /// @returns A result in the instance's constructed output form.
  virtual decoder_result decode(const cudaqx::tensor<uint8_t> &syndrome);

  /// @brief Decode a single syndrome (tensor form), cooperatively
  /// cancellable via `tok`. Converts and delegates to
  /// `decode(vector, cancellation_token)`.
  /// @param syndrome An order-1 tensor of syndrome measurements where a 1 bit
  /// represents that the syndrome measurement is a |1>. The
  /// length of the syndrome vector should be equal to `syndrome_size`.
  /// @param tok The cancellation token to use.
  /// @returns decoder_result, or std::nullopt if a stop was honored before
  /// any result was available.
  virtual std::optional<decoder_result>
  decode(const cudaqx::tensor<uint8_t> &syndrome, cancellation_token tok);

  /// @brief Decode a single syndrome
  /// @param syndrome A vector of syndrome measurements where the floating point
  /// value is the probability that the syndrome measurement is a |1>.
  /// @returns A future containing a result in the instance's constructed
  /// output form.
  virtual std::future<decoder_result>
  decode_async(const std::vector<float_t> &syndrome);

  /// @brief Decode a single syndrome, cooperatively cancellable via `tok`.
  /// The default implementation ignores `tok` and delegates to
  /// decode(syndrome), i.e. a decoder is non-cancellable unless it overrides
  /// this overload. A decoder that can poll `tok` should override this and
  /// return std::nullopt once `tok.stop_requested()` answers true (see
  /// `cancellation_token`).
  /// @param syndrome A vector of syndrome measurements where the floating point
  /// value is the probability that the syndrome measurement is a |1>.
  /// @param tok The cancellation token to use.
  /// @returns decoder_result, or std::nullopt if a stop was honored before
  /// any result was available.
  virtual std::optional<decoder_result>
  decode(const std::vector<float_t> &syndrome, cancellation_token tok) {
    return decode(syndrome);
  }

  /// @brief Decode multiple independent syndromes (may be done in serial or
  /// parallel depending on the specific implementation)
  /// @param syndrome A vector of `N` syndrome measurements where the floating
  /// point value is the probability that the syndrome measurement is a |1>.
  /// @returns One result per input in the instance's constructed output form.
  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndrome);

  /// @brief Decode multiple independent syndromes, additionally returning
  /// results that describe the batch as a whole rather than any single shot.
  ///
  /// Per-shot data belongs in `decoder_result::opt_results`. This overload
  /// exists for data that is naturally batch-wide: returning it once as flat
  /// batch-indexed arrays is more efficient to produce and to consume than
  /// splitting it into one small map entry per shot.
  ///
  /// The default implementation forwards to the single-argument overload and
  /// leaves @p batch_opt_results empty, so decoders that have no batch-level
  /// data need not override this. Decoders that override either overload
  /// should pull both into scope with `using decoder::decode_batch;` to avoid
  /// overload hiding.
  ///
  /// @param syndrome A vector of `N` syndrome measurements where the floating
  /// point value is the probability that the syndrome measurement is a |1>.
  /// @param batch_opt_results Output parameter receiving batch-level results,
  /// or left unset if the decoder produces none.
  /// @returns 2-D vector of size `N` x `block_size` with soft probabilities of
  /// errors in each index.
  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndrome,
               std::optional<cudaqx::heterogeneous_map> &batch_opt_results);

  /// @brief Decode multiple independent syndromes, cooperatively cancellable
  /// via `tok`. The default implementation ignores `tok` and delegates to
  /// decode_batch(syndrome), mirroring the single-syndrome default above.
  /// @param syndrome A vector of `N` syndrome measurements where the floating
  /// point value is the probability that the syndrome measurement is a |1>.
  /// @param tok The cancellation token to use.
  /// @returns One result per shot, or std::nullopt if a stop was honored
  /// before the batch completed.
  virtual std::optional<std::vector<decoder_result>>
  decode_batch(const std::vector<std::vector<float_t>> &syndrome,
               cancellation_token tok);

  /// @brief Construct a registered decoder by name.
  /// @param name The registered decoder name.
  /// @param inputs Stable decoder model inputs. Supplying an observable model
  /// does not select observable output.
  /// @param param_map Optional decoder-specific parameters.
  /// @note When no result type is supplied, the plugin chooses its documented
  /// default. PyMatching defaults to error-frame output.
  static std::unique_ptr<decoder>
  get(const std::string &name, decoder_init inputs,
      const cudaqx::heterogeneous_map &param_map = cudaqx::heterogeneous_map());

  /// @brief Construct a registered decoder with an explicit output contract.
  static std::unique_ptr<decoder>
  get(const std::string &name, decoder_init inputs,
      decoder_output_request output,
      const cudaqx::heterogeneous_map &param_map = cudaqx::heterogeneous_map());

  static std::unique_ptr<decoder>
  get(const std::string &name, const cudaq::qec::sparse_binary_matrix &H,
      const cudaqx::heterogeneous_map &param_map =
          cudaqx::heterogeneous_map()) {
    return get(name, decoder_init{H}, param_map);
  }

  static std::unique_ptr<decoder>
  get(const std::string &name, const cudaqx::tensor<uint8_t> &H,
      const cudaqx::heterogeneous_map &param_map =
          cudaqx::heterogeneous_map()) {
    return get(name, cudaq::qec::sparse_binary_matrix(H), param_map);
  }

  static std::unique_ptr<decoder>
  get(const std::string &name, const std::string &stim_dem_text,
      const cudaqx::heterogeneous_map &param_map =
          cudaqx::heterogeneous_map()) {
    return get(name, decoder_init::from_stim_dem(stim_dem_text), param_map);
  }

  static std::unique_ptr<decoder>
  get(const std::string &name, const char *stim_dem_text,
      const cudaqx::heterogeneous_map &param_map =
          cudaqx::heterogeneous_map()) {
    return get(name, decoder_init::from_stim_dem(stim_dem_text), param_map);
  }

  static std::unique_ptr<decoder>
  get(const std::string &name, std::string_view stim_dem_text,
      const cudaqx::heterogeneous_map &param_map =
          cudaqx::heterogeneous_map()) {
    return get(name, decoder_init::from_stim_dem(std::string{stim_dem_text}),
               param_map);
  }

  std::size_t get_block_size() { return block_size; }
  std::size_t get_syndrome_size() { return syndrome_size; }

  /// @brief The result form this instance was constructed to produce. Fixed at
  /// construction; every decode operation returns this form.
  decode_result_type get_result_type() const noexcept {
    return output_request_.primary;
  }

  decoder_auxiliary_result_type get_auxiliary_result_type() const noexcept {
    return output_request_.auxiliary;
  }

  const decoder_output_request &get_output_request() const noexcept {
    return output_request_;
  }

  // -- Begin realtime decoding API --

  // Note: all of the current realtime decoding API is designed to be used with
  // hard syndromes.

  /// @brief Get the number of measurement syndromes per decode call, i.e. the
  /// measurement count of the model's measurement-to-detector map. Zero when
  /// the model supplies no such map, because its syndromes are already
  /// detectors.
  uint32_t get_num_msyn_per_decode() const;

  /// @brief The CUDA device this decoder was pinned to at construction via
  /// the "cuda_device_id" parameter, or -1 when no pin was requested.
  /// Construction pins the constructing thread persistently (the thread that
  /// creates a decoder is the thread expected to drive its decode calls).
  int get_cuda_device_id() const { return cuda_device_id_; }

  /// @brief Set the decoder id.
  void set_decoder_id(uint32_t decoder_id);

  /// @brief Get the decoder id.
  uint32_t get_decoder_id() const;

  /// @brief Enqueue a syndrome for decoding (pointer version)
  /// @return True if enough syndromes have been enqueued to trigger a decode.
  virtual bool enqueue_syndrome(const uint8_t *syndrome,
                                std::size_t syndrome_length);

  /// @brief Enqueue a syndrome for decoding (vector version)
  /// @return True if enough syndromes have been enqueued to trigger a decode.
  virtual bool enqueue_syndrome(const std::vector<uint8_t> &syndrome);

  /// @brief Get the current observable corrections.
  virtual const uint8_t *get_obs_corrections() const;

  /// @brief Get the number of observables.
  std::size_t get_num_observables() const;

  /// @brief Clear any stored corrections.
  virtual void clear_corrections();

  /// @brief Reset the decoder, clearing all per-shot memory and corrections.
  virtual void reset_decoder();

  // -- End realtime decoding API --

  // -- Begin realtime graph dispatch API --

  /// @brief Returns true if this decoder supports graph-based realtime
  /// dispatch via capture_decode_graph().
  virtual bool supports_graph_dispatch() const { return false; }

  /// @brief Capture a CUDA graph for realtime dispatch.
  ///
  /// Returns a pointer to a cudaq::qec::realtime::graph_resources struct
  /// (caller must include realtime/graph_resources.h to interpret it).
  /// Returns nullptr if graph dispatch is not supported.
  /// The decoder retains ownership of the returned pointer.
  virtual void *capture_decode_graph(int reserved_sms = 0) {
    (void)reserved_sms;
    return nullptr;
  }

  /// @brief Release graph resources previously returned by
  /// capture_decode_graph().
  virtual void release_decode_graph(void *graph_resources) {
    (void)graph_resources;
  }

  // -- End realtime graph dispatch API --

  /// @brief Destructor
  virtual ~decoder() = default;

  /// @brief Get the version of the decoder. Subclasses that are not part of the
  /// standard GitHub repo should override this to provide a more tailored
  /// version string.
  /// @return A string containing the version of the decoder
  virtual std::string get_version() const;

protected:
  /// @brief The immutable construction inputs owned by this decoder.
  const decoder_init &get_inputs() const noexcept { return inputs_; }

  /// @brief Project an error frame onto observables through the model's O.
  ///
  /// A decoder that internally computes an error frame but was constructed for
  /// observable output calls this before returning. The projection lives here
  /// so it is implemented once rather than per plugin.
  /// @throws std::runtime_error if no observable mapping is available.
  void project_errors_to_observables(const float_t *errors,
                                     float_t *observables,
                                     std::size_t observables_size) const;

  /// @brief Declare that this decoder consumes its realtime input as a stream
  /// of detector layers rather than one full syndrome per decode.
  ///
  /// Everything the realtime path can derive from the model -- D, the
  /// measurement buffer, the detector buffers, the corrections buffer -- is
  /// sized by the base constructor from `decoder_init`. Layer geometry is
  /// the exception: it is a property of how the decoder consumes rounds, not
  /// of the model, and the base cannot ask a subclass for it while the
  /// subclass is still being constructed. A streaming decoder therefore hands
  /// it over here, from its own constructor.
  ///
  /// @param num_syndromes_per_round Width of the widest detector layer, which
  /// bounds the per-layer buffers.
  /// @param detector_layer_offsets Offsets `[0, w0, w0+w1, ...]`; `back()`
  /// must equal the model's detector count.
  /// @throws std::logic_error if called more than once. This is construction
  /// state, not a reconfiguration point: re-entering it on a live decoder
  /// would reset its buffers mid-stream.
  void
  initialize_streaming_layout(std::size_t num_syndromes_per_round,
                              std::vector<std::size_t> detector_layer_offsets);

  /// @brief For a classical `[n,k]` code, this is `n`.
  std::size_t block_size = 0;

  /// @brief For a classical `[n,k]` code, this is `n-k`
  std::size_t syndrome_size = 0;

  /// @brief CUDA device id consumed from the construction parameters by
  /// decoder::get(); -1 = unpinned. See get_cuda_device_id().
  int cuda_device_id_ = -1;

private:
  static std::unique_ptr<decoder>
  get_impl(const std::string &name, decoder_init inputs,
           std::optional<decoder_output_request> output,
           const cudaqx::heterogeneous_map &param_map);
  /// @brief The decoder's immutable construction inputs.
  const decoder_init inputs_;
  const decoder_output_request output_request_;
};

/// @brief Convert a single soft probability to a hard 0/1 decision.
/// @param in Soft probability input in range [0.0, 1.0]
/// @param thresh Values >= thresh return true; all others return false.
template <typename t_soft,
          typename std::enable_if<std::is_floating_point<t_soft>::value,
                                  int>::type = 0>
constexpr inline bool convert_soft_to_hard(t_soft in, t_soft thresh = 0.5) {
  return in >= thresh;
}

/// @brief Convert a vector of soft probabilities to a vector of hard
/// probabilities.
/// @param in Soft probability input vector in range [0.0, 1.0]
/// @param out Hard probability output vector containing only 0/false or 1/true.
/// @param thresh Values >= thresh are assigned 1/true and all others are
/// assigned 0/false.
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_soft_to_hard(const std::vector<t_soft> &in,
                                     std::vector<t_hard> &out,
                                     t_soft thresh = 0.5) {
  out.resize(in.size());
  for (std::size_t i = 0; i < in.size(); i++)
    out[i] = static_cast<t_hard>(convert_soft_to_hard(in[i], thresh));
}

/// @brief Convert a vector of soft probabilities to a tensor<uint8_t> of hard
/// probabilities. Tensor must be uninitialized, or initialized to a rank-1
/// tensor for equal dim as the vector.
/// @param in Soft probability input vector in range [0.0, 1.0]
/// @param out Hard probability output tensor containing only 0/false or 1/true.
/// @param thresh Values >= thresh are assigned 1/true and all others are
/// assigned 0/false.
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_soft_to_tensor_hard(const std::vector<t_soft> &in,
                                            cudaqx::tensor<t_hard> &out,
                                            t_soft thresh = 0.5) {
  if (out.shape().empty())
    out = cudaqx::tensor<t_hard>({in.size()});
  if (out.rank() != 1)
    throw std::runtime_error(
        "Vector to tensor conversion requires rank-1 tensor");
  if (out.shape()[0] != in.size())
    throw std::runtime_error(
        "Vector to tensor conversion requires tensor dim == vector length");
  auto raw_ptr = out.data();
  for (size_t i = 0; i < in.size(); ++i)
    raw_ptr[i] = static_cast<t_hard>(convert_soft_to_hard(in[i], thresh));
}

/// @brief Convert a vector of hard probabilities to a vector of soft
/// probabilities.
/// @param in Hard probability input vector containing only 0/false or 1/true.
/// @param in_size The size of the input vector (in elements)
/// @param out Soft probability output vector in the range [0.0, 1.0]
/// @param true_val The soft probability value assigned when the input is 1
/// (default to 1.0)
/// @param false_val The soft probability value assigned when the input is 0
/// (default to 0.0)
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_hard_to_soft(const t_hard *in, std::size_t in_size,
                                     std::vector<t_soft> &out,
                                     const t_soft true_val = 1.0,
                                     const t_soft false_val = 0.0) {
  out.resize(in_size);
  for (std::size_t i = 0; i < in_size; i++)
    out[i] = static_cast<t_soft>(in[i] ? true_val : false_val);
}

/// @brief Convert a vector of hard probabilities to a vector of soft
/// probabilities.
/// @param in Hard probability input vector containing only 0/false or 1/true.
/// @param out Soft probability output vector in the range [0.0, 1.0]
/// @param true_val The soft probability value assigned when the input is 1
/// (default to 1.0)
/// @param false_val The soft probability value assigned when the input is 0
/// (default to 0.0)
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_hard_to_soft(const std::vector<t_hard> &in,
                                     std::vector<t_soft> &out,
                                     const t_soft true_val = 1.0,
                                     const t_soft false_val = 0.0) {
  convert_vec_hard_to_soft(in.data(), in.size(), out, true_val, false_val);
}

/// @brief Convert a 2D vector of soft probabilities to a 2D vector of hard
/// probabilities.
/// @param in Soft probability input vector in range [0.0, 1.0]
/// @param out Hard probability output vector containing only 0/false or 1/true.
/// @param thresh Values >= thresh are assigned 1/true and all others are
/// assigned 0/false.
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_soft_to_hard(const std::vector<std::vector<t_soft>> &in,
                                     std::vector<std::vector<t_hard>> &out,
                                     t_soft thresh = 0.5) {
  std::size_t row_index = 0;
  out.resize(in.size());
  for (auto &r : in) {
    auto &out_row = out[row_index++];
    out_row.resize(r.size());
    for (std::size_t c = 0; c < r.size(); c++)
      out_row[c] = static_cast<t_hard>(convert_soft_to_hard(r[c], thresh));
  }
}

/// @brief Convert a 2D vector of hard probabilities to a 2D vector of soft
/// probabilities.
/// @param in Hard probability input vector containing only 0/false or 1/true.
/// @param out Soft probability output vector in the range [0.0, 1.0]
/// @param true_val The soft probability value assigned when the input is 1
/// (default to 1.0)
/// @param false_val The soft probability value assigned when the input is 0
/// (default to 0.0)
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_hard_to_soft(const std::vector<std::vector<t_hard>> &in,
                                     std::vector<std::vector<t_soft>> &out,
                                     const t_soft true_val = 1.0,
                                     const t_soft false_val = 0.0) {
  out.clear();
  out.reserve(in.size());
  for (auto &r : in) {
    std::vector<t_soft> out_row;
    out_row.reserve(r.size());
    for (auto c : r)
      out_row.push_back(static_cast<t_soft>(c ? true_val : false_val));
    out.push_back(std::move(out_row));
  }
}

/// Construct a registered decoder using its documented default result type.
/// Model inputs, including an observable matrix, do not select that type.
std::unique_ptr<decoder>
get_decoder(const std::string &name, decoder_init inputs,
            const cudaqx::heterogeneous_map options = {});

std::unique_ptr<decoder>
get_decoder(const std::string &name, decoder_init inputs,
            decoder_output_request output,
            const cudaqx::heterogeneous_map options = {});

inline std::unique_ptr<decoder>
get_decoder(const std::string &name, const cudaq::qec::sparse_binary_matrix &H,
            const cudaqx::heterogeneous_map options = {}) {
  return get_decoder(name, decoder_init{H}, options);
}

inline std::unique_ptr<decoder>
get_decoder(const std::string &name, const cudaqx::tensor<uint8_t> &H,
            const cudaqx::heterogeneous_map options = {}) {
  return get_decoder(name, cudaq::qec::sparse_binary_matrix(H), options);
}

inline std::unique_ptr<decoder>
get_decoder(const std::string &name, const std::string &stim_dem_text,
            const cudaqx::heterogeneous_map options = {}) {
  return get_decoder(name, decoder_init::from_stim_dem(stim_dem_text), options);
}

/// Raw-DEM convenience overloads use the decoder's documented default output.
/// To select output explicitly, construct decoder_init with from_stim_dem()
/// and call the generic decoder_init overload.
inline std::unique_ptr<decoder>
get_decoder(const std::string &name, const char *stim_dem_text,
            const cudaqx::heterogeneous_map options = {}) {
  return get_decoder(name, decoder_init::from_stim_dem(stim_dem_text), options);
}

inline std::unique_ptr<decoder>
get_decoder(const std::string &name, std::string_view stim_dem_text,
            const cudaqx::heterogeneous_map options = {}) {
  return get_decoder(
      name, decoder_init::from_stim_dem(std::string{stim_dem_text}), options);
}

namespace details {
/// DEM-derived defaults; pointers alias into the source `dem`.
struct dem_default_values {
  const cudaqx::tensor<uint8_t> *O = nullptr;
  const std::vector<double> *error_rate_vec = nullptr;
};

/// Return DEM defaults for keys not already supplied by the user.
dem_default_values dem_defaults_for_missing_keys(
    const std::function<bool(const std::string &)> &contains_user_key,
    const detector_error_model &dem);
} // namespace details

} // namespace cudaq::qec

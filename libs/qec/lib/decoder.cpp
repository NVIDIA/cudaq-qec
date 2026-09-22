/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cuda-qx/core/library_utils.h"
#include "hardware_guards.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/version.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fmt/ranges.h>
#include <span>
#include <vector>

INSTANTIATE_REGISTRY(cudaq::qec::decoder, cudaq::qec::decoder_init,
                     std::optional<cudaq::qec::decoder_output_request>,
                     const cudaqx::heterogeneous_map &)

// Include decoder implementations AFTER registry instantiation
#include "decoders/sliding_window.h"

namespace cudaq::qec {

struct decoder::rt_impl {
  /// The number of measurement syndromes to be decoded per decode call (i.e.
  /// the number of columns in the D_sparse matrix)
  uint32_t num_msyn_per_decode = 0;

  /// The index of the next syndrome to be written in the msyn_buffer
  uint32_t msyn_buffer_index = 0;

  /// The buffer of measurement syndromes received from the client. Length is
  /// num_msyn_per_decode.
  std::vector<uint8_t> msyn_buffer;

  /// The current observable corrections. The length of this vector is the
  /// number of rows in the O_sparse matrix.
  std::vector<uint8_t> corrections;

  /// Persistent buffers to avoid dynamic memory allocation.
  std::vector<uint8_t> persistent_detector_buffer;
  std::vector<float_t> persistent_soft_detector_buffer;

  /// Detector rows paired with the number of leading msyn_buffer columns they
  /// need, i.e. one past the last column they read, sorted by that count. A
  /// detector is resolvable once msyn_buffer_index reaches its count, so
  /// enqueue_syndrome() can fold detectors in as their measurements arrive
  /// instead of reducing the whole buffer on the final round. Only used when
  /// no streaming layout is installed; the sliding-window path emits a layer
  /// per round and is already incremental.
  std::vector<std::pair<uint32_t, uint32_t>> detector_ready_order;

  /// How far through detector_ready_order the current shot has progressed.
  std::size_t detector_ready_cursor = 0;

  /// Whether to log decoder stats.
  bool should_log = false;

  /// A simple counter to distinguish log messages.
  uint32_t log_counter = 0;

  /// The id of the decoder (for instrumentation)
  uint32_t decoder_id = 0;

  /// Written last by initialize_streaming_layout(), so per-round streaming
  /// never activates on incomplete geometry. Also the one-shot construction
  /// latch: a second call is rejected rather than resetting a live decoder.
  bool round_streaming_initialized = false;

  /// The model's measurement-to-detector map, by detector row. Empty when the
  /// model supplies none, i.e. the decoder is handed detectors directly.
  std::vector<std::vector<uint32_t>> measurement_to_detectors;

  /// The number of syndromes per round.  Only used for sliding window decoder.
  size_t num_syndromes_per_round = 0;

  /// Whether the first round detectors are included.  Only used for sliding
  /// window decoder.
  bool has_first_round_detectors = false;

  /// The current round.  Only used for sliding window decoder.
  uint32_t current_round = 0;

  /// Detector-layer offsets [0, w0, w0+w1, ...] for the [B | S...S | B] layout;
  /// back() == total detectors. Only used for sliding window decoder.
  std::vector<std::size_t> detector_layer_offsets;

  /// Index of the next detector layer to emit. Only used for sliding window.
  std::size_t detector_layer_index = 0;
};

void decoder::rt_impl_deleter::operator()(rt_impl *p) const { delete p; }

decoder::decoder(decoder_init inputs, decoder_output_request requested_output)
    : pimpl(std::unique_ptr<rt_impl, rt_impl_deleter>(new rt_impl())),
      inputs_(std::move(inputs)), output_request_(requested_output) {
  syndrome_size = inputs_.num_detectors();
  block_size = inputs_.num_error_mechanisms();

  // Everything the realtime path needs that the model determines is sized
  // here, from the model. Nothing arrives later: a decoder is usable as soon
  // as it is constructed.
  if (const auto *D = inputs_.measurement_to_detectors()) {
    pimpl->measurement_to_detectors = D->to_nested_csr();
    pimpl->num_msyn_per_decode = D->num_cols();
    pimpl->detector_ready_order.reserve(pimpl->measurement_to_detectors.size());
    for (std::size_t detector = 0;
         detector < pimpl->measurement_to_detectors.size(); ++detector) {
      std::uint32_t columns_needed = 0;
      for (const auto column : pimpl->measurement_to_detectors[detector])
        columns_needed = std::max(columns_needed, column + 1);
      pimpl->detector_ready_order.emplace_back(
          columns_needed, static_cast<std::uint32_t>(detector));
    }
    std::sort(pimpl->detector_ready_order.begin(),
              pimpl->detector_ready_order.end());
  }
  pimpl->persistent_detector_buffer.resize(this->syndrome_size);
  pimpl->persistent_soft_detector_buffer.resize(this->syndrome_size);
  reset_decoder();

  // We allow detailed logging of decoder stats via the CUDAQ_QEC_DEBUG_DECODER
  // environment variable or the CUDAQ_LOG_LEVEL=info environment variable. If
  // it is set with CUDAQ_LOG_LEVEL, it will be instrumented at the info level
  // just like any other message, but if it is set with CUDAQ_QEC_DEBUG_DECODER,
  // it will be instrumented as a simple printf.
  if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
    pimpl->should_log = ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y';
}

void decoder::project_errors_to_observables(
    const float_t *errors, float_t *observables,
    std::size_t observables_size) const {
  // Hot path: one call per shot on the realtime path. Sizes and O-row counts
  // are fixed by construction, so they are not re-checked here. There is one
  // observable model -- the one this decoder was constructed with -- so there
  // is no second source to fall back to.
  if (!inputs_.has_observable_model())
    throw std::runtime_error("decoder was asked to project an error frame onto "
                             "observables but its model supplies no observable "
                             "mapping");
  if (observables_size > 0)
    std::fill(observables, observables + observables_size, float_t{0});

  const auto &O = inputs_.observable_flips_matrix();
  assert(O.layout() == sparse_binary_matrix_layout::csr);
  const auto &ptr = O.ptr();
  const auto &indices = O.indices();
  for (std::size_t row = 0; row < O.num_rows(); ++row) {
    bool parity = false;
    for (auto pos = ptr[row]; pos < ptr[row + 1]; ++pos)
      parity ^= convert_soft_to_hard(errors[indices[pos]]);
    observables[row] = static_cast<float_t>(parity);
  }
}

// Provide a trivial implementation of for tensor<uint8_t> decode call. Child
// classes should override this if they never want to pass through floats.
decoder_result decoder::decode(const cudaqx::tensor<uint8_t> &syndrome) {
  // Check tensor is of order-1
  // If order >1, we could check that other modes are of dim = 1 such that
  // n x 1, or 1 x n tensors are still valid.
  if (syndrome.rank() != 1) {
    throw std::runtime_error("Decode requires rank-1 tensors");
  }
  std::vector<float_t> soft_syndrome(syndrome.shape()[0]);
  std::vector<uint8_t> vec_cast(syndrome.data(),
                                syndrome.data() + syndrome.shape()[0]);
  convert_vec_hard_to_soft(vec_cast, soft_syndrome);
  return decode(soft_syndrome);
}

// Provide a trivial implementation of the multi-syndrome decoder. Child classes
// should override this if they can do it more efficiently than this.
std::vector<decoder_result>
decoder::decode_batch(const std::vector<std::vector<float_t>> &syndrome) {
  std::vector<decoder_result> result;
  result.reserve(syndrome.size());
  for (auto &s : syndrome)
    result.push_back(decode(s));
  return result;
}

// Decoders with no batch-level results inherit this: run the ordinary batch
// decode and leave the out-param unset.
std::vector<decoder_result> decoder::decode_batch(
    const std::vector<std::vector<float_t>> &syndrome,
    std::optional<cudaqx::heterogeneous_map> &batch_opt_results) {
  batch_opt_results.reset();
  return decode_batch(syndrome);
}

std::string decoder::get_version() const {
  std::stringstream ss;
  ss << "CUDA-Q QEC Base Decoder Interface " << cudaq::qec::getVersion() << " ("
     << cudaq::qec::getFullRepositoryVersion() << ")";
  return ss.str();
}

std::future<decoder_result>
decoder::decode_async(const std::vector<float_t> &syndrome) {
  // Captured by value: the worker must not dereference decoder members to
  // find its device. The std::async thread is brand-new and unpinned, so it
  // guards itself for the duration of the call (the one exception to the
  // one-thread-owns-one-decoder persistent pin).
  const int cuda_id = cuda_device_id_;
  return std::async(std::launch::async, [this, syndrome, cuda_id] {
    cudaq::qec::detail_affinity::CudaDeviceGuard dev(cuda_id);
    return this->decode(syndrome);
  });
}

/// Reads "cuda_device_id" from the construction parameters. Absent -> -1.
/// Negative or >= cudaGetDeviceCount() -> std::runtime_error (fail fast:
/// never silently decode on the wrong GPU).
static int read_cuda_device_id(const cudaqx::heterogeneous_map &params) {
  if (!params.contains("cuda_device_id"))
    return -1;
  const int value = params.get<int>("cuda_device_id");
  if (value < 0)
    throw std::runtime_error("cuda_device_id must be >= 0 (got " +
                             std::to_string(value) + ")");
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || value >= count)
    throw std::runtime_error("cuda_device_id " + std::to_string(value) +
                             " is out of range: " + std::to_string(count) +
                             " CUDA device(s) visible");
  return value;
}

/// Selects the construction device and restores the previous device unless
/// commit() is called. Makes failed plugin construction transactional: if the
/// plugin ctor throws, the calling thread is left on its original device
/// instead of leaking the pin to whichever device was selected for the attempt.
class ConstructionDevicePin {
public:
  explicit ConstructionDevicePin(int target) : target_(target) {
    cudaError_t err = cudaGetDevice(&previous_);
    if (err != cudaSuccess)
      throw std::runtime_error(
          "cuda_device_id " + std::to_string(target_) +
          " could not be selected because cudaGetDevice() failed: " +
          cudaGetErrorString(err));
    err = cudaSetDevice(target_);
    if (err != cudaSuccess)
      throw std::runtime_error("cudaSetDevice(" + std::to_string(target_) +
                               ") failed: " + cudaGetErrorString(err));
    selected_ = true;
  }

  ~ConstructionDevicePin() {
    if (selected_ && !committed_ && previous_ != target_)
      (void)cudaSetDevice(previous_);
  }

  ConstructionDevicePin(const ConstructionDevicePin &) = delete;
  ConstructionDevicePin &operator=(const ConstructionDevicePin &) = delete;

  // Keep the pin: one thread owns one decoder, so leaving the constructing
  // thread on the target device lets later allocations and kernel launches --
  // including lazy ones inside decode() -- land there with no per-call
  // machinery.
  void commit() { committed_ = true; }

private:
  int target_ = -1;
  int previous_ = -1;
  bool selected_ = false;
  bool committed_ = false;
};

std::unique_ptr<decoder>
decoder::get(const std::string &name, decoder_init inputs,
             const cudaqx::heterogeneous_map &param_map) {
  return get_impl(name, std::move(inputs), std::nullopt, param_map);
}

std::unique_ptr<decoder>
decoder::get(const std::string &name, decoder_init inputs,
             decoder_output_request output,
             const cudaqx::heterogeneous_map &param_map) {
  return get_impl(name, std::move(inputs), output, param_map);
}

std::unique_ptr<decoder>
decoder::get_impl(const std::string &name, decoder_init inputs,
                  std::optional<decoder_output_request> output,
                  const cudaqx::heterogeneous_map &param_map) {
  for (const char *reserved : {"H", "O", "D", "error_rate_vec"})
    if (param_map.contains(reserved))
      throw std::runtime_error(
          fmt::format("'{}' is framework model data; provide it through "
                      "decoder_init instead of decoder custom parameters",
                      reserved));
  auto [mutex, registry] = get_registry();
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error(
        "invalid decoder requested: " + name +
        ". Run with CUDAQ_LOG_LEVEL=info (environment variable) to see "
        "additional plugin diagnostics at startup.");
  const int cuda_device_id = read_cuda_device_id(param_map);
  if (cuda_device_id < 0) {
    // The plugin validates the requested form against its model during
    // construction; there is nothing left for the factory to re-check.
    return iter->second(std::move(inputs), output, param_map);
  }
  ConstructionDevicePin device_pin(cuda_device_id);
  // The key is consumed here; strip it so plugins that strictly validate
  // their parameter keys do not reject it.
  cudaqx::heterogeneous_map plugin_params;
  for (const auto &kv : param_map)
    if (kv.first != "cuda_device_id")
      plugin_params.insert(kv.first, kv.second);
  auto d = iter->second(std::move(inputs), output, plugin_params);
  d->cuda_device_id_ = cuda_device_id;
  device_pin.commit();
  return d;
}

namespace details {

dem_default_values dem_defaults_for_missing_keys(
    const std::function<bool(const std::string &)> &contains_user_key,
    const detector_error_model &dem) {
  dem_default_values out;
  if (!contains_user_key("O") && dem.num_observables() > 0)
    out.O = &dem.observables_flips_matrix;
  if (!contains_user_key("error_rate_vec"))
    out.error_rate_vec = &dem.error_rates;
  return out;
}

} // namespace details

uint32_t decoder::get_num_msyn_per_decode() const {
  return pimpl->num_msyn_per_decode;
}

void decoder::set_decoder_id(uint32_t decoder_id) {
  pimpl->decoder_id = decoder_id;
}

uint32_t decoder::get_decoder_id() const { return pimpl->decoder_id; }

void decoder::initialize_streaming_layout(
    std::size_t num_syndromes_per_round,
    std::vector<std::size_t> detector_layer_offsets) {
  if (pimpl->round_streaming_initialized)
    throw std::logic_error(
        "initialize_streaming_layout() is construction state and may be called "
        "only once");
  if (detector_layer_offsets.empty())
    throw std::invalid_argument(
        "initialize_streaming_layout() requires at least one detector layer");
  if (detector_layer_offsets.back() != syndrome_size)
    throw std::invalid_argument(fmt::format(
        "detector layer offsets end at {} but the model has {} detectors",
        detector_layer_offsets.back(), syndrome_size));

  pimpl->num_syndromes_per_round = num_syndromes_per_round;
  // A first-round detector layer references a single measurement per detector.
  pimpl->has_first_round_detectors =
      !pimpl->measurement_to_detectors.empty() &&
      pimpl->measurement_to_detectors[0].size() == 1;
  pimpl->detector_layer_offsets = std::move(detector_layer_offsets);
  pimpl->detector_layer_index = 0;
  pimpl->current_round = 0;
  // Layers are emitted one at a time, so the widest layer bounds the buffers
  // rather than the full detector count.
  pimpl->persistent_detector_buffer.resize(num_syndromes_per_round);
  pimpl->persistent_soft_detector_buffer.resize(num_syndromes_per_round);
  pimpl->detector_ready_cursor = 0;
  pimpl->round_streaming_initialized = true;
}

// Fold in every detector whose measurements have all arrived, i.e. that needs
// no more than \p columns_available leading columns of msyn_buffer.
// detector_ready_order is sorted by that requirement, so this walks a cursor
// forward and reduces each detector exactly once per shot. Not used by the
// sliding window path, which emits one detector layer per round and is already
// incremental.
template <typename PimplType>
static void
resolve_ready_detectors(const std::vector<std::vector<uint32_t>> &D_sparse,
                        PimplType *pimpl, uint32_t columns_available) {
  const auto &order = pimpl->detector_ready_order;
  while (pimpl->detector_ready_cursor < order.size() &&
         order[pimpl->detector_ready_cursor].first <= columns_available) {
    const uint32_t detector = order[pimpl->detector_ready_cursor].second;
    uint8_t value = 0;
    for (const auto col : D_sparse[detector])
      value ^= pimpl->msyn_buffer[col];
    pimpl->persistent_detector_buffer[detector] = value;
    pimpl->detector_ready_cursor++;
  }
}

bool decoder::enqueue_syndrome(const uint8_t *syndrome,
                               std::size_t syndrome_length) {
  if (pimpl->msyn_buffer_index + syndrome_length > pimpl->msyn_buffer.size()) {
    // CUDA_QEC_WARN("Syndrome buffer overflow. Syndrome will be ignored.");
    printf("Syndrome buffer overflow. Syndrome will be ignored.\n");
    return false;
  }

  pimpl->current_round++;
  bool did_decode = false;
  if (syndrome_length > 0) {
    std::memcpy(pimpl->msyn_buffer.data() + pimpl->msyn_buffer_index, syndrome,
                syndrome_length);
    pimpl->msyn_buffer_index += syndrome_length;
  }

  // Resolve every detector whose measurements have now all arrived. Doing this
  // per round instead of in one pass at the end leaves the final round with
  // only the detectors that genuinely depend on it, and reads the measurements
  // while they are still hot from the copy above.
  if (!pimpl->round_streaming_initialized)
    resolve_ready_detectors(pimpl->measurement_to_detectors, pimpl.get(),
                            pimpl->msyn_buffer_index);

  bool should_decode = false;
  if (!pimpl->round_streaming_initialized) {
    should_decode = (pimpl->msyn_buffer_index == pimpl->msyn_buffer.size());
  } else {
    should_decode =
        (pimpl->current_round >= 2) ||
        (pimpl->current_round == 1 && pimpl->has_first_round_detectors);
  }
  if (should_decode) {
    // These are just for logging. They are initialized in such a way to avoid
    // dynamic memory allocation if logging is disabled.
    std::vector<uint32_t> log_msyn;
    std::vector<uint32_t> log_detectors;
    std::vector<uint32_t> log_errors;
    std::vector<uint32_t> log_observables;
    std::vector<uint8_t> log_observable_corrections;
    // The four time points are used to measure the duration of each of 3 steps.
    std::chrono::time_point<std::chrono::high_resolution_clock> log_t0, log_t1,
        log_t2, log_t3;
    std::chrono::duration<double> log_dur1, log_dur2, log_dur3;

    const bool log_due_to_log_level =
        cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info);
    const bool should_log = pimpl->should_log || log_due_to_log_level;

    if (should_log) {
      log_t0 = std::chrono::high_resolution_clock::now();
      log_errors.reserve(syndrome_length);
      log_observables.reserve(get_num_observables());
      log_observable_corrections.resize(get_num_observables());
    }

    // Decode now. A full buffer makes every remaining detector resolvable, so
    // the per-round pass above will normally have finished all of them and this
    // drain costs one comparison. It is kept so that a complete detector buffer
    // is guaranteed by construction rather than inferred from when
    // should_decode fires; a partially reduced buffer would silently decode
    // stale values left over from the previous shot.
    if (!pimpl->round_streaming_initialized) {
      resolve_ready_detectors(pimpl->measurement_to_detectors, pimpl.get(),
                              pimpl->num_msyn_per_decode);
    } else {
      const std::size_t k = pimpl->detector_layer_index++;
      const std::size_t off = pimpl->detector_layer_offsets[k];
      const std::size_t width = pimpl->detector_layer_offsets[k + 1] - off;
      pimpl->persistent_detector_buffer.resize(width);
      for (std::size_t j = 0; j < width; j++) {
        uint8_t v = 0;
        for (auto col : pimpl->measurement_to_detectors[off + j])
          v ^= pimpl->msyn_buffer[col];
        pimpl->persistent_detector_buffer[j] = v;
      }
    }

    if (should_log) {
      log_msyn.reserve(pimpl->msyn_buffer.size());
      for (std::size_t d = 0, D = pimpl->msyn_buffer.size(); d < D; d++) {
        if (pimpl->msyn_buffer[d])
          log_msyn.push_back(d);
      }
      log_detectors.reserve(pimpl->persistent_detector_buffer.size());
      for (std::size_t d = 0, D = pimpl->persistent_detector_buffer.size();
           d < D; d++) {
        if (pimpl->persistent_detector_buffer[d])
          log_detectors.push_back(d);
      }
      log_t1 = std::chrono::high_resolution_clock::now();
    }
    // Send the data to the decoder.
    convert_vec_hard_to_soft(pimpl->persistent_detector_buffer,
                             pimpl->persistent_soft_detector_buffer);
    auto decoded_result = decode(pimpl->persistent_soft_detector_buffer);
    std::span<const float_t> decoded_values = decoded_result.result;

    // If we didn't get a decoded result, just return
    if (pimpl->round_streaming_initialized) {
      if (decoded_values.empty()) {
        return false;
      }
    }
    // Process the results.
    // TODO - should this interrogate the decoded_result.converged flag?
    const auto num_observables = get_num_observables();
    const char *result_type_str = nullptr;
    const char *result_type_name = nullptr;
    std::size_t expected_result_size = 0;
    switch (output_request_.primary) {
    case decode_result_type::errors:
      result_type_str = "errs";
      result_type_name = "errors";
      expected_result_size = block_size;
      break;
    case decode_result_type::observables:
      result_type_str = "obs";
      result_type_name = "observables";
      expected_result_size = num_observables;
      break;
    }
    if ((!pimpl->round_streaming_initialized &&
         decoded_values.size() != expected_result_size) ||
        (pimpl->round_streaming_initialized && !decoded_values.empty() &&
         decoded_values.size() != expected_result_size)) {
      throw std::runtime_error(fmt::format(
          "Decoder result size ({}) does not match expected size ({}) for "
          "result type {}",
          decoded_values.size(), expected_result_size, result_type_name));
    }

    // Flip an observable correction and mirror it into the per-call log so the
    // logged flips stay faithful to the applied corrections.
    auto flip_correction = [&](std::size_t i) {
      pimpl->corrections[i] ^= 1;
      if (should_log)
        log_observable_corrections[i] ^= 1;
    };

    if (should_log)
      log_t2 = std::chrono::high_resolution_clock::now();

    switch (output_request_.primary) {
    case decode_result_type::observables:
      // Observable-frame path: decoder already projected to observables via its
      // internal "O" matrix; use the result directly.
      for (std::size_t i = 0; i < num_observables; i++)
        if (decoded_values[i]) {
          if (should_log)
            log_observables.push_back(i);
          flip_correction(i);
        }
      break;
    case decode_result_type::errors:
      // Error-frame path: decoder returns a block-sized error vector; project
      // to observables via O_sparse.
      if (!inputs_.has_observable_model())
        throw std::runtime_error(
            "Error-frame decoders need an observable model to project through; "
            "this one was constructed without one");
      if (should_log)
        for (std::size_t e = 0, E = decoded_values.size(); e < E; e++)
          if (decoded_values[e])
            log_errors.push_back(e);
      // For each observable, flip its correction once for each predicted error
      // that flips it (net parity over O_sparse[i]).
      {
        const auto &O = inputs_.observable_flips_matrix();
        const auto &ptr = O.ptr();
        const auto &indices = O.indices();
        for (std::size_t i = 0; i < num_observables; i++)
          for (auto k = ptr[i]; k < ptr[i + 1]; ++k)
            if (decoded_values[indices[k]])
              flip_correction(i);
      }
      break;
    }
    if (should_log) {
      log_t3 = std::chrono::high_resolution_clock::now();
      log_dur1 = log_t1 - log_t0;
      log_dur2 = log_t2 - log_t1;
      log_dur3 = log_t3 - log_t2;
      pimpl->log_counter++;
      auto s = fmt::format(
          "[DecoderStats][{}] Counter:{} DecoderId:{} InputMsyn:{} "
          "InputDetectors:{} Converged:{} ResultType:{} Errors:{} "
          "Observables:{} "
          "ObservableCorrectionsThisCall:{} ObservableCorrectionsTotal:{} "
          "Dur1:{:.1f}us Dur2:{:.1f}us Dur3:{:.1f}us",
          static_cast<const void *>(this), pimpl->log_counter,
          pimpl->decoder_id, fmt::join(log_msyn, ","),
          fmt::join(log_detectors, ","), decoded_result.converged ? 1 : 0,
          result_type_str, fmt::join(log_errors, ","),
          fmt::join(log_observables, ","),
          fmt::join(log_observable_corrections, ","),
          fmt::join(std::vector<uint8_t>(pimpl->corrections.begin(),
                                         pimpl->corrections.end()),
                    ","),
          log_dur1.count() * 1e6, log_dur2.count() * 1e6,
          log_dur3.count() * 1e6);
      if (log_due_to_log_level)
        CUDA_QEC_INFO("{}", s);
      else
        printf("%s\n", s.c_str());
    }
    did_decode = true;
    // Prepare for more data.
    pimpl->msyn_buffer_index = 0;
    pimpl->current_round = 0;
    pimpl->detector_layer_index = 0;
    pimpl->detector_ready_cursor = 0;
  }
  return did_decode;
}

bool decoder::enqueue_syndrome(const std::vector<uint8_t> &syndrome) {
  return enqueue_syndrome(syndrome.data(), syndrome.size());
}

void decoder::clear_corrections() {
  pimpl->corrections.clear();
  pimpl->corrections.resize(get_num_observables());
  const bool log_due_to_log_level =
      cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info);
  const bool should_log = pimpl->should_log || log_due_to_log_level;
  if (should_log) {
    pimpl->log_counter++;
    std::string s =
        fmt::format("[DecoderStats][{}] Counter:{} clear_corrections called",
                    static_cast<const void *>(this), pimpl->log_counter);
    if (log_due_to_log_level)
      CUDA_QEC_INFO("{}", s);
    else
      printf("%s\n", s.c_str());
  }
}

const uint8_t *decoder::get_obs_corrections() const {
  const bool log_due_to_log_level =
      cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info);
  const bool should_log = pimpl->should_log || log_due_to_log_level;
  if (should_log) {
    pimpl->log_counter++;
    std::string s =
        fmt::format("[DecoderStats][{}] Counter:{} get_obs_corrections called",
                    static_cast<const void *>(this), pimpl->log_counter);
    if (log_due_to_log_level)
      CUDA_QEC_INFO("{}", s);
    else
      printf("%s\n", s.c_str());
  }
  return pimpl->corrections.data();
}

std::size_t decoder::get_num_observables() const {
  return inputs_.num_observables();
}

void decoder::reset_decoder() {
  // Zero out all data that is considered "per-shot" memory.
  pimpl->msyn_buffer_index = 0;
  pimpl->current_round = 0;
  pimpl->detector_layer_index = 0;
  pimpl->detector_ready_cursor = 0;
  pimpl->msyn_buffer.clear();
  pimpl->msyn_buffer.resize(pimpl->num_msyn_per_decode);
  pimpl->corrections.clear();
  pimpl->corrections.resize(get_num_observables());
  const bool log_due_to_log_level =
      cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info);
  const bool should_log = pimpl->should_log || log_due_to_log_level;
  if (should_log) {
    pimpl->log_counter++;
    std::string s =
        fmt::format("[DecoderStats][{}] Counter:{} reset_decoder called",
                    static_cast<const void *>(this), pimpl->log_counter);
    if (log_due_to_log_level)
      CUDA_QEC_INFO("{}", s);
    else
      printf("%s\n", s.c_str());
  }
}

std::unique_ptr<decoder> get_decoder(const std::string &name,
                                     decoder_init inputs,
                                     const cudaqx::heterogeneous_map options) {
  return decoder::get(name, std::move(inputs), options);
}

std::unique_ptr<decoder> get_decoder(const std::string &name,
                                     decoder_init inputs,
                                     decoder_output_request output,
                                     const cudaqx::heterogeneous_map options) {
  return decoder::get(name, std::move(inputs), output, options);
}

// Constructor function for auto-loading plugins
__attribute__((constructor)) void load_decoder_plugins() {
  // Load plugins from the decoder-specific plugin directory
  std::filesystem::path libPath{cudaqx::__internal__::getCUDAQXLibraryPath(
      cudaqx::__internal__::CUDAQXLibraryType::QECDecoders)};
  auto pluginPath = libPath.parent_path() / "decoder-plugins";
  load_plugins(pluginPath.string(), PluginType::DECODER);
}

// Destructor function to clean up only decoder plugins
__attribute__((destructor)) void cleanup_decoder_plugins() {
  // Clean up decoder-specific plugins
  cleanup_plugins(PluginType::DECODER);
}
} // namespace cudaq::qec

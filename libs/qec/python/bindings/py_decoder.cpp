/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuda-qx/core/kwargs_utils.h"
#include "cuda-qx/core/library_utils.h"
#include "sparse_matrix_casters.h"
#include "type_casters.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fmt/core.h>
#include <functional>
#include <limits>
#include <link.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <unordered_map>

namespace nb = nanobind;
using namespace cudaqx;

namespace cudaq::qec {

class PyDecoder : public decoder {
public:
  NB_TRAMPOLINE(decoder, 1);

  /// @brief Construct from a scipy sparse matrix (CSR, CSC, COO, ...) or a
  ///        dense numpy array of any numeric dtype.
  ///
  /// Sparse storage is built directly from either form, so
  /// qec.Decoder.__init__(self, H) has the same memory behavior as native
  /// get_decoder(..., H) (no intermediate dense tensor copy).
  PyDecoder(nb::object mat)
      : decoder(decoder_init(sparse_binary_matrix_from_python(mat))) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    NB_OVERRIDE_PURE_NAME("decode", decode, syndrome);
  }
};

// Registry to store decoder factory functions
class PyDecoderRegistry {
private:
  static std::unordered_map<std::string,
                            std::function<nb::object(nb::object, nb::kwargs)>>
      registry;

public:
  static void
  register_decoder(const std::string &name,
                   std::function<nb::object(nb::object, nb::kwargs)> factory) {
    cudaq::qec::info("Registering Pythonic Decoder with name {}", name);
    registry[name] = factory;
  }

  static nb::object get_decoder(const std::string &name, nb::object H,
                                nb::kwargs options) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      throw std::runtime_error("Unknown decoder: " + name);
    }

    return it->second(H, options);
  }

  static bool contains(const std::string &name) {
    return registry.find(name) != registry.end();
  }
};

std::unordered_map<std::string,
                   std::function<nb::object(nb::object, nb::kwargs)>>
    PyDecoderRegistry::registry;

namespace {

struct batch_decoder_result {
  // Python-facing constructor for decoder plugin authors. nanobind enforces
  // ndim on the typed array arguments and rejects wrong-rank input with
  // TypeError, but it does not enforce dtype strictly (int32 input is
  // silently coerced to float) and does not enforce C-contiguity. We
  // therefore re-coerce through np.ascontiguousarray below to guarantee
  // both invariants — batch[i] reads rows by raw pointer and relies on
  // them. Cross-array invariants (row count, opt_results length) are also
  // checked here since nanobind cannot see them.
  batch_decoder_result(nb::ndarray<nb::numpy, float_t, nb::ndim<2>> result_arr,
                       nb::ndarray<nb::numpy, bool, nb::ndim<1>> converged_arr,
                       nb::object opt_results, nb::object batch_opt_results)
      : batch_opt_results(batch_opt_results) {
    size = converged_arr.shape(0);
    if (result_arr.shape(0) != size)
      throw std::runtime_error(
          "BatchDecoderResult result row count must match the number of "
          "convergence flags.");

    if (opt_results.is_none()) {
      nb::list lst;
      for (std::size_t i = 0; i < size; ++i)
        lst.append(nb::none());
      this->opt_results = lst;
    } else {
      this->opt_results = nb::cast<nb::list>(opt_results);
      if (nb::len(this->opt_results) != size)
        throw std::runtime_error(
            "BatchDecoderResult opt_results length must match the number of "
            "convergence flags.");
    }

    nb::module_ np = nb::module_::import_("numpy");
    nb::object float_dtype = sizeof(float_t) == sizeof(float)
                                 ? np.attr("float32")
                                 : np.attr("float64");
    this->result = np.attr("ascontiguousarray")(nb::cast(result_arr),
                                                nb::arg("dtype") = float_dtype);
    this->converged = np.attr("ascontiguousarray")(
        nb::cast(converged_arr), nb::arg("dtype") = np.attr("bool_"));
  }

  // Trusted internal constructor: callers guarantee shape/dtype invariants.
  // Used by makeBatchDecoderResult and batchSliceToBatchDecoderResult.
  batch_decoder_result(nb::object result, nb::object converged,
                       nb::list opt_results, std::size_t size,
                       nb::object batch_opt_results = nb::none())
      : result(result), converged(converged), opt_results(opt_results),
        batch_opt_results(batch_opt_results), size(size) {}

  nb::object result;
  nb::object converged;
  nb::list opt_results;
  // Batch-level results, or None. Describes the batch as a whole, so it is
  // deliberately not sliced or indexed alongside the per-shot arrays.
  nb::object batch_opt_results;
  std::size_t size = 0;
};

Py_ssize_t normalizeBatchIndex(Py_ssize_t index, std::size_t size) {
  const auto signed_size = static_cast<Py_ssize_t>(size);
  if (index < 0)
    index += signed_size;
  if (index < 0 || index >= signed_size)
    throw nb::index_error();
  return index;
}

decoder_result batchItemToDecoderResult(const batch_decoder_result &batch,
                                        Py_ssize_t index) {
  const auto normalized_index = normalizeBatchIndex(index, batch.size);
  nb::object py_index = nb::int_(normalized_index);
  auto row = nb::cast<nb::ndarray<nb::numpy, float_t>>(
      batch.result.attr("__getitem__")(py_index));

  decoder_result result;
  nb::object converged = batch.converged.attr("__getitem__")(py_index);
  const int is_converged = PyObject_IsTrue(converged.ptr());
  if (is_converged < 0) {
    PyErr_Clear();
    throw nb::type_error(
        "BatchDecoderResult converged entries must be bool-like.");
  }
  result.converged = is_converged != 0;
  const auto result_size = row.shape(0);
  const auto *data = static_cast<const float_t *>(row.data());
  result.result.assign(data, data + result_size);

  nb::object opt_results = batch.opt_results.attr("__getitem__")(py_index);
  if (!opt_results.is_none())
    result.opt_results = nb::cast<cudaqx::heterogeneous_map>(opt_results);
  return result;
}

batch_decoder_result
batchSliceToBatchDecoderResult(const batch_decoder_result &batch,
                               const nb::slice &slice) {
  nb::object result = batch.result.attr("__getitem__")(slice);
  nb::object converged = batch.converged.attr("__getitem__")(slice);
  nb::list opt_results =
      nb::cast<nb::list>(batch.opt_results.attr("__getitem__")(slice));
  // batch_opt_results is dropped: its arrays are indexed by position in the
  // full batch, so carrying them onto a slice unchanged would silently
  // misalign them with the sliced per-shot rows.
  return batch_decoder_result(result, converged, opt_results,
                              nb::len(converged));
}

// Wrap a heap-allocated buffer (owned via unique_ptr) in a NumPy ndarray that
// will free the buffer when garbage-collected. The unique_ptr argument keeps
// the buffer alive until ownership is transferred to the capsule; if any step
// inside this function throws, the buffer is freed by the unique_ptr
// destructor on unwind.
template <typename T, std::size_t Rank>
nb::object makeOwnedNdarray(std::unique_ptr<T[]> data,
                            const std::size_t (&shape)[Rank]) {
  auto *raw_data = data.get();
  nb::capsule owner(raw_data,
                    [](void *p) noexcept { delete[] static_cast<T *>(p); });
  data.release();
  return nb::cast(
      nb::ndarray<nb::numpy, T>(raw_data, Rank, shape, std::move(owner)));
}

nb::object decoderResultToNumpy(const std::vector<float_t> &result) {
  const auto num_elements = result.size();
  auto data = std::make_unique<float_t[]>(num_elements);
  std::copy(result.begin(), result.end(), data.get());

  size_t shape[1] = {num_elements};
  return makeOwnedNdarray<float_t>(std::move(data), shape);
}

nb::object decoderResultsToNumpy(const std::vector<decoder_result> &results) {
  const auto num_results = results.size();
  // Empty batch yields shape (0, 0). The per-shot width is unknown without
  // running a decode (and depends on decoder mode — e.g. decode_to_observables
  // produces num_observables-wide rows, not block_size-wide). Callers should
  // not rely on result.shape[1] when the batch is empty.
  const auto result_size = num_results == 0 ? 0 : results.front().result.size();

  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto actual_size = results[i].result.size();
    if (actual_size != result_size) {
      throw std::runtime_error(fmt::format(
          "Cannot return decode_batch results as a NumPy array because result "
          "vectors have inconsistent sizes: expected row width {}, but row {} "
          "has width {}.",
          result_size, i, actual_size));
    }
  }

  const auto num_elements = num_results * result_size;
  auto data = std::make_unique<float_t[]>(num_elements);

  auto *out = data.get();
  for (const auto &result : results) {
    out = std::copy(result.result.begin(), result.result.end(), out);
  }

  size_t shape[2] = {num_results, result_size};
  return makeOwnedNdarray<float_t>(std::move(data), shape);
}

nb::object
decoderResultsConvergedToNumpy(const std::vector<decoder_result> &results) {
  const auto num_results = results.size();
  auto data = std::make_unique<bool[]>(num_results);
  for (std::size_t i = 0; i < num_results; ++i)
    data[i] = results[i].converged;

  size_t shape[1] = {num_results};
  return makeOwnedNdarray<bool>(std::move(data), shape);
}

nb::list
decoderResultsOptResultsToList(const std::vector<decoder_result> &results) {
  nb::list opt_results;
  for (const auto &result : results) {
    if (result.opt_results.has_value()) {
      opt_results.append(nb::cast(result.opt_results));
    } else {
      opt_results.append(nb::none());
    }
  }
  return opt_results;
}

batch_decoder_result makeBatchDecoderResult(
    const std::vector<decoder_result> &results,
    const std::optional<cudaqx::heterogeneous_map> &batch_opt_results) {
  return batch_decoder_result{
      decoderResultsToNumpy(results),
      decoderResultsConvergedToNumpy(results),
      decoderResultsOptResultsToList(results),
      results.size(),
      batch_opt_results.has_value() ? nb::cast(*batch_opt_results) : nb::none(),
  };
}

} // namespace

namespace {
/// Read and remove the "output" keyword, which selects the result basis.
std::optional<decode_result_type> pop_requested_output(nb::kwargs &options) {
  if (!options.contains("output"))
    return std::nullopt;
  const auto value = nb::cast<std::string>(options["output"]);
  options.attr("pop")("output");
  if (value == "errors")
    return decode_result_type::errors;
  if (value == "observables")
    return decode_result_type::observables;
  throw std::runtime_error("output must be 'errors' or 'observables'");
}
} // namespace

void bindDecoder(nb::module_ &mod) {
  // Store a sentinel (non-null pointer required by PyCapsule_New) and invoke
  // plugin cleanup when the module is garbage-collected.
  static const int sentinel = 0;
  mod.attr("_cleanup") = nb::capsule(
      &sentinel, [](void *) noexcept { cleanup_plugins(PluginType::DECODER); });

  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  nb::class_<decoder_result>(qecmod, "DecoderResult", R"pbdoc(
    Single-shot decoder result.

    Returned by `decoder.decode(...)`. Carries the convergence flag, the
    decoded correction chain, and optional decoder-specific metadata.

    Like `BatchDecoderResult`, this is conceptually output-only — user code
    should not need to construct or mutate one. Unlike `BatchDecoderResult`,
    the no-arg constructor and writable fields are preserved here because
    Python decoder plugins implementing a `decode` override use the
    construct-then-mutate pattern:

        res = DecoderResult()
        res.converged = True
        res.result = np.arange(...)
        return res

    Tightening this construction surface would break every existing Python
    decoder plugin and is deferred to a future change.
)pbdoc")
      .def(nb::init<>(), R"pbdoc(
        Default constructor for DecoderResult.

        Creates a new DecoderResult instance with default values.
    )pbdoc")
      .def_rw("converged", &decoder_result::converged, R"pbdoc(
        Boolean flag indicating if the decoder converged to a solution.

        True if the decoder successfully found a valid correction chain,
        False if the decoder failed to converge or exceeded iteration limits.
    )pbdoc")
      .def_prop_rw(
          "result",
          [](const decoder_result &self) {
            return decoderResultToNumpy(self.result);
          },
          [](decoder_result &self, const std::vector<float_t> &value) {
            self.result = value;
          },
          R"pbdoc(
        The decoded correction chain or recovery operation.

        Contains the sequence of corrections that should be applied to recover
        the original quantum state. The format depends on the specific decoder
        implementation.

        Returns a 1-D NumPy array of the configured QEC floating point dtype
        (float64 in standard wheels). A fresh array is allocated per access —
        the underlying storage is a `std::vector<float_t>` and the data is
        copied out on read.

        Accepts any sequence of floats on assignment; this is the path Python
        decoder plugins use in their `decode` overrides (see class docstring).
    )pbdoc")
      .def_rw("opt_results", &decoder_result::opt_results, R"pbdoc(
        Optional additional results from the decoder stored in a heterogeneous map.

        This field may be empty if no additional results are available.
    )pbdoc")
      .def("__len__", [](const decoder_result &) { return 3; })
      .def("__getitem__",
           [](const decoder_result &r, size_t i) -> nb::object {
             switch (i) {
             case 0:
               return nb::cast(r.converged);
             case 1:
               return decoderResultToNumpy(r.result);
             case 2:
               return nb::cast(r.opt_results);
             default:
               throw nb::index_error();
             }
           })
      // Enable iteration protocol
      .def("__iter__", [](const decoder_result &r) -> nb::object {
        return nb::make_tuple(r.converged, decoderResultToNumpy(r.result),
                              r.opt_results)
            .attr("__iter__")();
      });

  nb::class_<batch_decoder_result>(qecmod, "BatchDecoderResult", R"pbdoc(
    Batched decoder result.

    Produced by `decoder.decode_batch(...)`. This type is output-only: it
    carries decoder output back to the caller and is not parsed by decoders.
    User code should not need to construct one — call
    `decoder.decode_batch(...)` and read the result.

    Python decoder plugins implementing a `decode_batch` override may use the
    constructor to produce one. `result` should be a 2-D NumPy array of the
    configured QEC floating point dtype (float64 in standard wheels);
    `converged` should be a 1-D NumPy bool array; `opt_results` is a list of
    per-shot dicts or None entries; `batch_opt_results` is an optional dict of
    batch-level results. The constructor coerces `result` and
    `converged` to C-contiguous storage of the expected dtype (via
    `np.ascontiguousarray`), copying when the input doesn't already satisfy
    those invariants. Wrong rank (e.g. 1-D `result`) is rejected with
    TypeError.

    An empty batch (zero syndromes) yields `result.shape == (0, 0)` and
    `converged.shape == (0,)`. The per-shot width is unknown without running
    a decode and depends on decoder mode, so `result.shape[1]` is only
    meaningful when the batch is non-empty.

    Access patterns, fastest to slowest:

      1. Vectorized: read `result`, `converged`, or `opt_results` directly.
         The properties return the underlying NumPy arrays and Python list
         with no copy. This is the recommended path for batch processing.

      2. Slicing: `batch[a:b]` returns another BatchDecoderResult that shares
         data with the parent. NumPy basic slicing — including stepped slices
         like `batch[::2]` — returns views, so no data is copied. The opt
         results list slice creates a new Python list, but its entries are
         shared references.

      3. Integer indexing / iteration: `batch[i]` or `for r in batch:` yields
         a DecoderResult copy of one shot. This compatibility surface exists
         for code written against the previous `list[DecoderResult]` return
         type. Each access copies the row out into a fresh per-shot buffer
         because DecoderResult's underlying storage (`std::vector<float_t>`)
         cannot alias into the batch's packed NumPy array — the layouts are
         incompatible. Avoid in hot loops; prefer pattern 1.
)pbdoc")
      .def(nb::init<nb::ndarray<nb::numpy, float_t, nb::ndim<2>>,
                    nb::ndarray<nb::numpy, bool, nb::ndim<1>>, nb::object,
                    nb::object>(),
           nb::arg("result"), nb::arg("converged"),
           nb::arg("opt_results") = nb::none(),
           nb::arg("batch_opt_results") = nb::none())
      .def_prop_ro(
          "result",
          [](const batch_decoder_result &self) { return self.result; },
          R"pbdoc(
        A two-dimensional NumPy array of decoder outputs, with one row per shot.

        This is the fast path for batch consumers. Its dtype is the configured
        QEC floating point type.
    )pbdoc")
      .def_prop_ro(
          "converged",
          [](const batch_decoder_result &self) { return self.converged; },
          R"pbdoc(
        A one-dimensional NumPy bool array indicating convergence per shot.
    )pbdoc")
      .def_prop_ro(
          "opt_results",
          [](const batch_decoder_result &self) { return self.opt_results; },
          R"pbdoc(
        A list of per-shot optional result dictionaries, or None entries.
    )pbdoc")
      .def_prop_ro(
          "batch_opt_results",
          [](const batch_decoder_result &self) {
            return self.batch_opt_results;
          },
          R"pbdoc(
        A dict of batch-level optional results, or None.

        Unlike `opt_results`, this describes the batch as a whole rather than
        any single shot: its arrays are indexed by position in the batch. Most
        decoders produce None here. It is the fast path for data that would
        otherwise cost one Python dict and several small arrays per shot.

        Because its arrays are indexed by position in the full batch, slicing a
        BatchDecoderResult drops this field rather than carrying it through
        misaligned.
    )pbdoc")
      .def(
          "__getitem__",
          [](const batch_decoder_result &self, nb::handle key) -> nb::object {
            if (PyIndex_Check(key.ptr())) {
              Py_ssize_t index =
                  PyNumber_AsSsize_t(key.ptr(), PyExc_IndexError);
              if (index == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                throw nb::index_error();
              }
              return nb::cast(batchItemToDecoderResult(self, index));
            }
            if (nb::isinstance<nb::slice>(key)) {
              return nb::cast(batchSliceToBatchDecoderResult(
                  self, nb::borrow<nb::slice>(key)));
            }
            throw nb::type_error(
                "BatchDecoderResult indices must be integers or slices.");
          },
          R"pbdoc(
        Return one or more batch entries.

        Integer indexing materializes a DecoderResult copy of one shot. This
        path exists for compatibility with code written against the previous
        `list[DecoderResult]` return type. Each access copies the row out
        per shot because DecoderResult's storage (`std::vector<float_t>`)
        cannot alias into the batch's packed NumPy array. Prefer reading
        `result`, `converged`, and `opt_results` directly for batch
        workflows.

        Slicing returns another BatchDecoderResult whose `result` and
        `converged` arrays are NumPy views into the parent (basic slicing,
        including stepped slices, does not copy). `opt_results` is a new list
        with shared element references. No per-shot work.
    )pbdoc")
      .def(
          "__len__", [](const batch_decoder_result &self) { return self.size; },
          R"pbdoc(
        Return the number of shots in the batch.
    )pbdoc");

  nb::class_<async_decoder_result>(qecmod, "AsyncDecoderResult",
                                   R"pbdoc(
      A future-like object that holds the result of an asynchronous decoder call.
      Call get() to block until the result is available.
    )pbdoc")
      .def("get", &async_decoder_result::get,
           nb::call_guard<nb::gil_scoped_release>(),
           "Return the decoder result (blocking until ready)")
      .def("ready", &async_decoder_result::ready,
           nb::call_guard<nb::gil_scoped_release>(),
           "Return True if the asynchronous decoder result is ready, False "
           "otherwise");

  nb::class_<decoder, PyDecoder>(
      qecmod, "Decoder", "Represents a decoder for quantum error correction")
      .def(nb::init<nb::object>(),
           R"pbdoc(
        Construct from a scipy sparse matrix (CSR, CSC, COO or any other
        ``scipy.sparse`` format).  For bring-your-own-decoder classes that
        call ``qec.Decoder.__init__(self, H)`` where ``H`` is a scipy sparse
        matrix passed to ``get_decoder``.
      )pbdoc")
      .def(
          "decode",
          [](decoder &decoder, const std::vector<float_t> &syndrome) {
            return decoder.decode(syndrome);
          },
          "Decode the given syndrome to determine the error correction",
          nb::arg("syndrome"))
      .def(
          "decode_async",
          [](decoder &dec,
             const std::vector<float_t> &syndrome) -> async_decoder_result {
            // Release the GIL while launching asynchronous work.
            nb::gil_scoped_release release;
            return async_decoder_result(dec.decode_async(syndrome));
          },
          "Asynchronously decode the given syndrome", nb::arg("syndrome"),
          // The worker dereferences the decoder, and ~async_decoder_result
          // joins it before nanobind drops this reference.
          nb::keep_alive<0, 1>())
      .def(
          "decode_batch",
          [](decoder &decoder,
             const std::vector<std::vector<float_t>> &syndrome) {
            // Always the two-argument overload: decoders with no batch-level
            // results inherit the base implementation and leave it unset.
            std::optional<cudaqx::heterogeneous_map> batch_opt_results;
            auto results = decoder.decode_batch(syndrome, batch_opt_results);
            return makeBatchDecoderResult(results, batch_opt_results);
          },
          "Decode multiple syndromes and return the results",
          nb::arg("syndrome"))
      .def("get_block_size", &decoder::get_block_size,
           "Get the size of the code block")
      .def("get_syndrome_size", &decoder::get_syndrome_size,
           "Get the size of the syndrome")
      .def("get_version", &decoder::get_version,
           "Get the version of the decoder");

  nb::class_<detector_error_model>(qecmod, "DetectorErrorModel",
                                   R"pbdoc(
      A detector error model (DEM) for a quantum error correction circuit. A
      DEM can be created from a QEC circuit and a noise model. It contains
      information about which errors flip which detectors. This is used by the
      decoder to help make predictions about observables flips.
    )pbdoc")
      .def(nb::init<>())
      .def_prop_rw(
          "detector_error_matrix",
          [](const detector_error_model &self) -> nb::object {
            const auto &t = self.detector_error_matrix;
            auto rows = t.shape()[0];
            auto cols = t.shape()[1];
            size_t shape[2] = {rows, cols};
            auto arr = nb::ndarray<nb::numpy, uint8_t>(
                const_cast<uint8_t *>(t.data()), 2, shape, nb::none());
            return nb::cast(arr).attr("copy")();
          },
          [](detector_error_model &self,
             const nb::ndarray<nb::numpy, uint8_t> &a) {
            auto borrow = toTensor(a);
            // Make sure that we own the data within the `detector_error_model`
            // as the input array may go out of scope.
            self.detector_error_matrix.copy(borrow.data(), borrow.shape());
          },
          R"pbdoc(
            The detector error matrix is a specific kind of circuit-level parity-check
            matrix where each row represents a detector, and each column represents
            an error mechanism. The entries of this matrix are H[i,j] = 1 if detector
            i is triggered by error mechanism j, and 0 otherwise.
          )pbdoc")
      .def_rw("error_rates", &detector_error_model::error_rates,
              R"pbdoc(
      The list of weights has length equal to the number of columns of the
      detector error matrix, which assigns a likelihood to each error mechanism.
    )pbdoc")
      .def_rw("error_ids", &detector_error_model::error_ids, R"pbdoc(
       Error mechanism ID. From a probability perspective, each error mechanism
       ID is independent of all other error mechanism ID. For all errors with
       the *same* ID, only one of them can happen. That is - the errors
       containing the same ID are correlated with each other.
    )pbdoc")
      .def_prop_rw(
          "observables_flips_matrix",
          [](const detector_error_model &self) -> nb::object {
            const auto &t = self.observables_flips_matrix;
            auto rows = t.shape()[0];
            auto cols = t.shape()[1];
            size_t shape[2] = {rows, cols};
            auto arr = nb::ndarray<nb::numpy, uint8_t>(
                const_cast<uint8_t *>(t.data()), 2, shape, nb::none());
            return nb::cast(arr).attr("copy")();
          },
          [](detector_error_model &self,
             const nb::ndarray<nb::numpy, uint8_t> &a) {
            auto borrow = toTensor(a);
            // Make sure that we own the data within the `detector_error_model`
            // as the input array may go out of scope.
            self.observables_flips_matrix.copy(borrow.data(), borrow.shape());
          },
          R"pbdoc(
            The observables flips matrix is a specific kind of circuit-level parity-
            check matrix where each row represents a Pauli observable, and each
            column represents an error mechanism. The entries of this matrix are
            O[i,j] = 1 if Pauli observable i is flipped by error mechanism j, and 0
            otherwise.
          )pbdoc")
      .def("num_detectors", &detector_error_model::num_detectors,
           R"pbdoc(
            The number of detectors in the detector error model
          )pbdoc")
      .def("num_error_mechanisms", &detector_error_model::num_error_mechanisms,
           R"pbdoc(
            The number of error mechanisms in the detector error model
          )pbdoc")
      .def("num_observables", &detector_error_model::num_observables,
           R"pbdoc(
            The number of observables in the detector error model
          )pbdoc")
      .def("canonicalize_for_rounds",
           static_cast<void (detector_error_model::*)(uint32_t, bool)>(
               &detector_error_model::canonicalize_for_rounds),
           R"pbdoc(
            Canonicalize the detector error model for a given number of rounds.

            Columns sharing the same detector and observable signature are
            merged, with rates composed to match the input model. By default,
            zero-syndrome columns that still flip an observable (undetectable
            logical errors) are retained so the model's observable-flip
            probability is preserved. Set ``remove_zero_syndrome_errors=True``
            to drop all columns with no detector signature, which is
            appropriate when the canonicalized DEM is consumed only for
            round-based decoding.

            Canonicalization does not preserve cross-column exclusivity
            structure: each output column is given a fresh unique error id and
            treated as independent of every other column, so any ``error_ids``
            correlation in the input model is discarded.
          )pbdoc",
           nb::arg("num_syndromes_per_round"),
           nb::arg("remove_zero_syndrome_errors") = false)
      .def(
          "canonicalize_for_rounds_with_boundary",
          static_cast<void (detector_error_model::*)(uint32_t, uint32_t, bool)>(
              &detector_error_model::canonicalize_for_rounds_with_boundary),
          R"pbdoc(
            Boundary-aware canonicalization for memory-experiment DEMs whose
            first and last detector layers (the boundaries) are narrower than
            the interior layers. The first ``num_boundary_syndromes`` detector
            rows form the initial round, each subsequent block of
            ``num_syndromes_per_round`` rows is an interior round, and the
            trailing ``num_boundary_syndromes`` rows form the final round.
            This makes the round-based column ordering respect the true rounds
            even when the boundary width differs from the interior width.

            ``remove_zero_syndrome_errors`` behaves as in
            ``canonicalize_for_rounds``. Raises ``ValueError`` if
            ``num_syndromes_per_round`` is
            zero or ``num_boundary_syndromes`` exceeds
            ``num_syndromes_per_round``.
          )pbdoc",
          nb::arg("num_syndromes_per_round"), nb::arg("num_boundary_syndromes"),
          nb::arg("remove_zero_syndrome_errors") = false);

  qecmod.def("dem_from_stim_text", &dem_from_stim_text,
             R"pbdoc(
        Parse a Stim detector error model string into a DetectorErrorModel.

        Args:
            dem_text: A Stim detector error model string.
            use_decomp_suggestions: If error mechanism separated by ``^`` are decomposed
      )pbdoc",
             nb::arg("dem_text"), nb::arg("use_decomp_suggestions") = false);

  // Expose decorator function that handles inheritance
  qecmod.def("decoder", [&](const std::string &name) {
    return nb::cpp_function([name](nb::object decoder_class) -> nb::object {
      // Create new class that inherits from both Decoder and the original
      nb::object base_decoder =
          nb::module_::import_("cudaq_qec").attr("Decoder");
      // Create new type using Python's type() function
      nb::tuple bases = nb::make_tuple(base_decoder);
      // __dict__ is a read-only mappingproxy; copy to a real dict for
      // PyType_Type.tp_new
      nb::dict namespace_dict;
      namespace_dict.update(decoder_class.attr("__dict__"));

      if (!nb::hasattr(decoder_class, "decode"))
        throw std::runtime_error("Decoder class must implement decode method");

      // Use Python's type() so the correct metaclass (nanobind's) is resolved
      nb::object type_fn = nb::module_::import_("builtins").attr("type");
      nb::object new_class =
          type_fn(decoder_class.attr("__name__"), bases, namespace_dict);

      // Register the new class in the decoder registry
      PyDecoderRegistry::register_decoder(
          name, [new_class](nb::object H, nb::kwargs options) {
            nb::object instance = new_class(H, **options);
            return instance;
          });
      return new_class;
    });
  });

  auto get_decoder_from_dem_text = [](const std::string &name,
                                      const std::string &dem_text,
                                      nb::kwargs options)
      -> std::variant<nb::object, std::unique_ptr<decoder>> {
    if (PyDecoderRegistry::contains(name)) {
      auto dem = dem_from_stim_text(dem_text);

      auto defaults = details::dem_defaults_for_missing_keys(
          [&](const std::string &key) { return options.contains(key); }, dem);
      if (defaults.O)
        options["O"] = cudaq::python::copyCUDAQXTensorToPyArray(*defaults.O);
      if (defaults.error_rate_vec)
        options["error_rate_vec"] =
            vector_to_numpy_1d(*defaults.error_rate_vec);

      nb::object H_obj = nb::cast(
          cudaq::python::copyCUDAQXTensorToPyArray(dem.detector_error_matrix));
      return PyDecoderRegistry::get_decoder(name, H_obj, options);
    }

    const auto output = pop_requested_output(options);
    auto inputs = decoder_init::from_stim_dem(dem_text);
    return output ? get_decoder(name, std::move(inputs), *output,
                                hetMapFromKwargs(options))
                  : get_decoder(name, std::move(inputs),
                                hetMapFromKwargs(options));
  };

  qecmod.def(
      "get_decoder",
      [get_decoder_from_dem_text](const std::string &name, nb::object H,
                                  nb::kwargs options)
          -> std::variant<nb::object, std::unique_ptr<decoder>> {
        if (nb::isinstance<nb::str>(H)) {
          return get_decoder_from_dem_text(name, nb::cast<std::string>(H),
                                           options);
        }

        if (PyDecoderRegistry::contains(name)) {
          return PyDecoderRegistry::get_decoder(name, H, options);
        }

        cudaq::qec::sparse_binary_matrix H_sparse =
            sparse_binary_matrix_from_python(nb::cast<nb::object>(H));

        const auto output = pop_requested_output(options);

        // Absent O means no observable model, not a zero-row one. Fabricating
        // an empty O here would make an H-only construction look like a
        // supplied model and defeat the construction-time validation.
        std::optional<sparse_binary_matrix> O_sparse;
        if (options.contains("O")) {
          O_sparse = sparse_binary_matrix_from_python(
              nb::cast<nb::object>(options["O"]));
          options.attr("pop")("O");
        }

        std::vector<double> error_rates;
        if (options.contains("error_rate_vec")) {
          error_rates =
              nb::cast<std::vector<double>>(options["error_rate_vec"]);
          options.attr("pop")("error_rate_vec");
        }

        if (name == "tensor_network_decoder") {
          throw std::runtime_error(
              "Decoder 'tensor_network_decoder' is not available. "
              "To enable it, install the python module's dependencies via:\n\n"
              "    pip install cudaq-qec[tensor-network-decoder]\n");
        }

        decoder_init inputs(std::move(H_sparse), std::move(O_sparse),
                            std::move(error_rates));
        return output ? get_decoder(name, std::move(inputs), *output,
                                    hetMapFromKwargs(options))
                      : get_decoder(name, std::move(inputs),
                                    hetMapFromKwargs(options));
      },
      R"pbdoc(
        Get a decoder by name.

        ``H`` may be:

        - A scipy sparse matrix (CSR, CSC, COO, or any ``scipy.sparse`` format):
          the preferred input — no dense allocation occurs, and any format is
          normalised to CSR internally before building the C++ sparse storage.
        - A dense 2D NumPy ``uint8`` array in row-major order: a full dense
          ``cudaqx::tensor`` is built first, then converted to CSC sparse storage.
          For large PCMs this can allocate as much memory as ``rows * cols``.
        - A Stim detector error model string: native C++ decoders receive the
          raw DEM text via ``decoder_init``; Python-registered decoders receive
          the DEM-derived PCM plus ``O`` and ``error_rate_vec`` defaults.

        Native decoders select their result basis with ``output="errors"`` or
        ``output="observables"``. When omitted, the plugin's documented default
        applies; PyMatching defaults to errors. Matrix ``O`` and
        ``error_rate_vec`` keyword adapters are normalized into decoder_init.
        Supplying ``O`` never selects the output mode.

        For Python-registered decoders (``cudaq.qec.decoder`` decorator), ``H``
        is passed through to ``__init__`` unchanged (NumPy array or scipy sparse
        matrix). DEM string inputs are parsed first as described above. Call
        ``Decoder.__init__(self, H)`` so nanobind can store the PCM internally
        without building a dense ``rows x cols`` allocation.
      )pbdoc");

  qecmod.def(
      "get_sorted_pcm_column_indices",
      [](const nb::ndarray<nb::numpy, uint8_t> &H,
         std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);

        return cudaq::qec::get_sorted_pcm_column_indices(
            tensor_H, num_syndromes_per_round);
      },
      R"pbdoc(
        Get the sorted column indices of a parity check matrix.

        This function returns the column indices of a parity check matrix in
        topological order.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A NumPy array containing the sorted column indices

        See Also:
            :cpp:func:`cudaq::qec::get_sorted_pcm_column_indices`: The
            underlying C++ implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "pcm_is_sorted",
      [](const nb::ndarray<nb::numpy, uint8_t> &H,
         std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        return cudaq::qec::pcm_is_sorted(tensor_H, num_syndromes_per_round);
      },
      R"pbdoc(
        Check if a parity check matrix is sorted.

        This function checks if a parity check matrix is sorted in topological
        order.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A boolean indicating if the parity check matrix is sorted

        See Also:
            :cpp:func:`cudaq::qec::pcm_is_sorted`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "reorder_pcm_columns",
      [](nb::object H,
         const std::vector<std::uint32_t> &column_order) -> nb::object {
        if (nb::hasattr(H, "tocsr")) {
          auto H_sparse = sparse_binary_matrix_from_scipy(H);
          auto H_new = cudaq::qec::reorder_pcm_columns(H_sparse, column_order);
          return sparse_binary_matrix_to_scipy_csc(H_new);
        }

        auto tensor_H =
            pcmToTensor(nb::cast<nb::ndarray<nb::numpy, uint8_t>>(H));
        auto H_new = cudaq::qec::reorder_pcm_columns(tensor_H, column_order);

        // Construct a new ndarray from H_new (deep copy)
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape[2] = {rows, cols};
        auto arr = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape, nb::none());
        return nb::cast(arr).attr("copy")();
      },
      R"pbdoc(
        Reorder the columns of a parity check matrix.

        This function reorders the columns of a parity check matrix according to
        the given column order.

        Args:
            H: A NumPy array or scipy sparse matrix
            column_order: A NumPy array containing the column order

        Returns:
            A NumPy array, or a scipy CSC matrix when H is scipy sparse.

        See Also:
            :cpp:func:`cudaq::qec::reorder_pcm_columns`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("column_order"));

  qecmod.def(
      "sort_pcm_columns",
      [](nb::ndarray<nb::numpy, uint8_t> &H,
         std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        auto H_new =
            cudaq::qec::sort_pcm_columns(tensor_H, num_syndromes_per_round);

        // Construct a new ndarray from H_new (deep copy)
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape[2] = {rows, cols};
        auto arr = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape, nb::none());
        return nb::cast(arr).attr("copy")();
      },
      R"pbdoc(
        Sort the columns of a parity check matrix.

        This function sorts the columns of a parity check matrix in topological
        order.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A NumPy array containing the sorted parity check matrix

        See Also:
            :cpp:func:`cudaq::qec::sort_pcm_columns`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "dump_pcm",
      [](const nb::ndarray<nb::numpy, uint8_t> &H) {
        auto tensor_H = pcmToTensor(H);
        tensor_H.dump_bits();
        printf("\n");
        fflush(stdout);
      },
      R"pbdoc(
        Dump the parity check matrix to stdout.

        This function dumps the parity check matrix to stdout.
      )pbdoc",
      nb::arg("H"));

  qecmod.def(
      "generate_random_pcm",
      // Signed `weight` so the C++ guard reports negatives instead of nanobind
      // rejecting at the marshalling boundary.
      [](std::uint32_t n_rounds, std::uint32_t n_errs_per_round,
         std::uint32_t n_syndromes_per_round, int weight, std::uint32_t seed) {
        std::mt19937_64 rng(seed);
        if (seed == 0)
          rng = std::mt19937_64(std::random_device()());

        auto H_new = cudaq::qec::generate_random_pcm(n_rounds, n_errs_per_round,
                                                     n_syndromes_per_round,
                                                     weight, std::move(rng));
        // Construct a new ndarray from H_new (deep copy)
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape[2] = {rows, cols};
        auto arr = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape, nb::none());
        return nb::cast(arr).attr("copy")();
      },
      R"pbdoc(
        Generate a random parity check matrix.

        This function creates a random parity check matrix for quantum error correction
        with specified parameters controlling the structure and randomness.

        Args:
            n_rounds: Number of measurement rounds in the error correction protocol
            n_errs_per_round: Number of error mechanisms per round
            n_syndromes_per_round: Number of syndrome measurements per round
            weight: The weight parameter controlling the sparsity of the matrix
            seed: Random seed for reproducibility (0 for random seed)

        See Also:
            :cpp:func:`cudaq::qec::generate_random_pcm`: The underlying C++
            implementation of this function.

        Returns:
            A NumPy array containing the generated parity check matrix
      )pbdoc",
      nb::arg("n_rounds"), nb::arg("n_errs_per_round"),
      nb::arg("n_syndromes_per_round"), nb::arg("weight"), nb::arg("seed") = 0);

  qecmod.def(
      "get_pcm_for_rounds",
      [](const nb::ndarray<nb::numpy, uint8_t> &H,
         std::uint32_t num_syndromes_per_round, std::uint32_t start_round,
         std::uint32_t end_round, bool straddle_start_round,
         bool straddle_end_round, std::uint32_t num_boundary_syndromes) {
        auto tensor_H = pcmToTensor(H);

        auto [H_new, first_column, last_column] =
            cudaq::qec::get_pcm_for_rounds(
                tensor_H, num_syndromes_per_round, start_round, end_round,
                straddle_start_round, straddle_end_round,
                num_boundary_syndromes);

        // Construct a new ndarray from H_new (deep copy)
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape[2] = {rows, cols};
        auto arr = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape, nb::none());
        return nb::make_tuple(nb::cast(arr).attr("copy")(), first_column,
                              last_column);
      },
      R"pbdoc(
        Get a sub-PCM for a range of rounds.

        This function returns a sub-parity check matrix for a range of rounds.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round
            start_round: The starting round
            end_round: The ending round
            straddle_start_round: Whether to allow error mechanisms that
              straddle the start round (i.e. include prior rounds, too). This
              defaults to false.
            straddle_end_round: Whether to allow error mechanisms that straddle
              the end round (i.e. include future rounds, too). This defaults to
              false.
            num_boundary_syndromes: The number of syndrome measurements in the boundary layers

        Returns:
            A tuple containing the sub-parity check matrix and the first and last
            column indices of the sub-PCM relative to the original PCM.

        See Also:
            :cpp:func:`cudaq::qec::get_pcm_for_rounds`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("num_syndromes_per_round"), nb::arg("start_round"),
      nb::arg("end_round"), nb::arg("straddle_start_round") = false,
      nb::arg("straddle_end_round") = false,
      nb::arg("num_boundary_syndromes") = 0);

  qecmod.def(
      "pcm_extend_to_n_rounds",
      [](const nb::ndarray<nb::numpy, uint8_t> &H,
         std::uint32_t num_syndromes_per_round, std::uint32_t n_rounds) {
        auto tensor_H = pcmToTensor(H);
        auto [H_new, column_list] = cudaq::qec::pcm_extend_to_n_rounds(
            tensor_H, num_syndromes_per_round, n_rounds);
        // Construct a new ndarray from H_new.
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape[2] = {rows, cols};
        auto arr = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape, nb::none());
        return nb::make_tuple(nb::cast(arr).attr("copy")(), column_list);
      },
      R"pbdoc(
        Extend a parity check matrix to a given number of rounds.

        This function extends a parity check matrix to a given number of rounds.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round
            n_rounds: The number of rounds to extend the parity check matrix to

        Returns:
            A tuple containing the extended parity check matrix and the list of
            column indices from the original PCM that were used to form the new
            PCM.

        See Also:
            :cpp:func:`cudaq::qec::pcm_extend_to_n_rounds`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("num_syndromes_per_round"), nb::arg("n_rounds"));

  qecmod.def(
      "shuffle_pcm_columns",
      [](nb::object H, std::uint32_t seed) -> nb::object {
        std::mt19937_64 rng(seed);
        if (seed == 0)
          rng = std::mt19937_64(std::random_device()());

        if (nb::hasattr(H, "tocsr")) {
          auto H_sparse = sparse_binary_matrix_from_scipy(H);
          auto H_new =
              cudaq::qec::shuffle_pcm_columns(H_sparse, std::move(rng));
          return sparse_binary_matrix_to_scipy_csc(H_new);
        }

        auto tensor_H =
            pcmToTensor(nb::cast<nb::ndarray<nb::numpy, uint8_t>>(H));
        auto H_new = cudaq::qec::shuffle_pcm_columns(tensor_H, std::move(rng));
        // Construct a new ndarray from H_new (deep copy)
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape[2] = {rows, cols};
        auto arr = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape, nb::none());
        return nb::cast(arr).attr("copy")();
      },
      R"pbdoc(
        Shuffle the columns of a parity check matrix.

        This function shuffles the columns of a parity check matrix.

        Args:
            H: A NumPy array or scipy sparse matrix
            seed: Random seed for reproducibility (0 for random seed)

        Returns:
            A NumPy array, or a scipy CSC matrix when H is scipy sparse.

        See Also:
            :cpp:func:`cudaq::qec::shuffle_pcm_columns`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("seed") = 0);

  qecmod.def(
      "simplify_pcm",
      [](const nb::ndarray<nb::numpy, uint8_t> &H,
         const nb::ndarray<nb::numpy, double> &weights,
         std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        const auto *w_ptr = static_cast<const double *>(weights.data());
        auto weights_vec = std::vector<double>(w_ptr, w_ptr + weights.size());
        auto [H_new, weights_new] = cudaq::qec::simplify_pcm(
            tensor_H, weights_vec, num_syndromes_per_round);
        // Construct a new ndarray from H_new.
        auto rows = H_new.shape()[0];
        auto cols = H_new.shape()[1];
        size_t shape_h[2] = {rows, cols};
        auto arr_h = nb::ndarray<nb::numpy, uint8_t>(
            const_cast<uint8_t *>(H_new.data()), 2, shape_h, nb::none());
        // Construct a new ndarray from weights_new.
        size_t shape_w[1] = {weights_new.size()};
        auto arr_w = nb::ndarray<nb::numpy, double>(weights_new.data(), 1,
                                                    shape_w, nb::none());
        return nb::make_tuple(nb::cast(arr_h).attr("copy")(),
                              nb::cast(arr_w).attr("copy")());
      },
      R"pbdoc(
        Simplify a parity check matrix.

        This function simplifies a parity check matrix by removing duplicate
        columns and 0-weight columns.

        Args:
            H: A NumPy array representing the parity check matrix
            weights: A NumPy array containing the weights of the columns
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A tuple containing the simplified parity check matrix and the weights

        See Also:
            :cpp:func:`cudaq::qec::simplify_pcm`: The underlying C++
            implementation of this function.
      )pbdoc",
      nb::arg("H"), nb::arg("weights"), nb::arg("num_syndromes_per_round"));

  const auto loadDecoderLibrary = [](const std::string &path) {
    static void *decoderLibHandle = nullptr;

    // Unload if previously-loaded
    // Note: dlclose makes no guarantees about unloading. Hence, we need to
    // manually execute `init_func` of the kernels (should be executed as static
    // initializers in those libs) to make sure they are registered again.
    if (decoderLibHandle)
      dlclose(decoderLibHandle);

    decoderLibHandle = dlopen(path.c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (!decoderLibHandle) {
      // Retrieve the error message
      char *error_msg = dlerror();
      throw std::runtime_error(
          fmt::format("Failed to load decoder library at '{}': {}", path,
                      (error_msg ? std::string(error_msg) : "unknown error.")));
    }

    // List of `init_func` names to call for decoder registration.
    //
    // The `enqueue_syndromes` mangled string encodes the public API's
    // parameter type. Under CUDA-Q alias `using measure_result =
    // measure_handle;`, C++ mangling resolves the typedef to its underlying
    // type, so `INS_14measure_handleESaIS3_EE` is the inner-vector mangling for
    // `std::vector<cudaq::measure_handle>`. If CUDA-Q ever renames the
    // underlying handle type or changes the alias direction, this string must
    // be re-derived from `nm` on the per-target device .o.
    //
    // `enqueue_syndromes_test` is the bool-typed counterpart and stays bound
    // to `INS_t6vectorIbSaIbEE`; the Python frontend hands it pre-discriminated
    // bits via `cudaq.to_bools(...)`.
    // clang-format off
    static const std::vector<std::string> initFuncNames = {
        "function_enqueue_syndromes._ZN5cudaq3qec8decoding17enqueue_syndromesEmRKSt6vectorINS_14measure_handleESaIS3_EEm.init_func",
        "function_enqueue_syndromes_test._ZN5cudaq3qec8decoding22enqueue_syndromes_testEmRKSt6vectorIbSaIbEEm.init_func",
        "function_get_corrections._ZN5cudaq3qec8decoding15get_correctionsEmmb.init_func",
        "function_reset_decoder._ZN5cudaq3qec8decoding13reset_decoderEm.init_func"};
    // clang-format on
    for (const auto &funcName : initFuncNames) {
      // Use dlsym to get the function pointer
      using InitFuncType = void (*)();
      InitFuncType initFunc = reinterpret_cast<InitFuncType>(
          dlsym(decoderLibHandle, funcName.c_str()));
      if (!initFunc) {
        char *error_msg = dlerror();
        throw std::runtime_error(fmt::format(
            "Failed to locate init function '{}' in decoder library at '{}': "
            "{}",
            funcName, path,
            (error_msg ? std::string(error_msg) : "unknown error.")));
      }
      // Call the init function to register/update the decoder quake code
      initFunc();
    }
  };

  qecmod.def(
      "load_quantinuum_realtime_decoding",
      [&]() {
        const std::filesystem::path path{
            cudaqx::__internal__::getCUDAQXLibraryPath(
                cudaqx::__internal__::CUDAQXLibraryType::QEC)};
        const auto quantinuumLibPath =
            path.parent_path() / "libcudaq-qec-realtime-decoding-quantinuum.so";
        loadDecoderLibrary(quantinuumLibPath);
      },
      R"pbdoc(
        [Internal] Load Quantinuum realtime decoder library.
      )pbdoc");

  qecmod.def(
      "load_simulation_realtime_decoding",
      [&]() {
        const std::filesystem::path path{
            cudaqx::__internal__::getCUDAQXLibraryPath(
                cudaqx::__internal__::CUDAQXLibraryType::QEC)};
        const auto simulationLibPath =
            path.parent_path() / "libcudaq-qec-realtime-decoding-simulation.so";
        loadDecoderLibrary(simulationLibPath);
      },
      R"pbdoc(
        [Internal] Load local simulation realtime decoder library.
      )pbdoc");

  qecmod.def(
      "generate_timelike_sparse_detector_matrix",
      [](std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds,
         bool include_first_round) {
        return cudaq::qec::generate_timelike_sparse_detector_matrix(
            num_syndromes_per_round, num_rounds, include_first_round);
      },
      R"pbdoc(
        Generate a sparse detector matrix for a given number of syndromes per round
        and number of rounds. Time-like here means that each round of syndrome measurement
        bits are xor'd against the preceding round.

        Args:
            num_syndromes_per_round: The number of syndrome measurements per round
            num_rounds: The number of rounds to generate the sparse detector matrix for
            include_first_round: Whether to include the first round of syndrome measurements

        Returns:
            The detector matrix format is CSR-like, with -1 values indicating the end of each row.
      )pbdoc",
      nb::arg("num_syndromes_per_round"), nb::arg("num_rounds"),
      nb::arg("include_first_round"));

  qecmod.def(
      "pcm_to_sparse_vec",
      [](nb::object pcm) {
        if (nb::hasattr(pcm, "tocsr")) {
          return cudaq::qec::pcm_to_sparse_vec(
              sparse_binary_matrix_from_scipy(pcm));
        }

        auto tensor_pcm =
            pcmToTensor(nb::cast<nb::ndarray<nb::numpy, uint8_t>>(pcm));
        return cudaq::qec::pcm_to_sparse_vec(tensor_pcm);
      },
      R"pbdoc(
        Return a sparse representation of the PCM.

        Args:
            pcm: A NumPy array or scipy sparse matrix.
      )pbdoc",
      nb::arg("pcm"));
}

} // namespace cudaq::qec

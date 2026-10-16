/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include "cudaq/qec/logger.h"

#include "common/DeviceCodeRegistry.h"
#include "cudaq/python/PythonCppInterop.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/version.h"

#include "cuda-qx/core/kwargs_utils.h"
#include "cudaq_type_casters.h"
#include "type_casters.h"

namespace nb = nanobind;
using namespace cudaqx;

namespace cudaq::qec {

class PyCode : public qec::code {
public:
  NB_TRAMPOLINE(qec::code, 6);

protected:
  // Trampoline methods for pure virtual functions
  std::size_t get_num_data_qubits() const override {
    NB_OVERRIDE_PURE(get_num_data_qubits);
  }

  std::size_t get_num_ancilla_qubits() const override {
    NB_OVERRIDE_PURE(get_num_ancilla_qubits);
  }

  std::size_t get_num_ancilla_x_qubits() const override {
    NB_OVERRIDE_PURE(get_num_ancilla_x_qubits);
  }

  std::size_t get_num_ancilla_z_qubits() const override {
    NB_OVERRIDE_PURE(get_num_ancilla_z_qubits);
  }

  std::size_t get_num_x_stabilizers() const override {
    NB_OVERRIDE_PURE(get_num_x_stabilizers);
  }

  std::size_t get_num_z_stabilizers() const override {
    NB_OVERRIDE_PURE(get_num_z_stabilizers);
  }
};

/// @brief A wrapper class that handles Python-defined quantum error correction
/// codes
/// @details This class serves as a bridge between Python-defined QEC codes and
/// the C++ implementation, managing the conversion of Python QEC code
/// definitions to their C++ counterparts.
class PyCodeHandle : public qec::code {
protected:
  /// @brief Python object representing the registered QEC code
  nb::object pyCode;

  /// @brief Keep the original Python kernels so that we can hand them back to
  /// Python.
  std::unordered_map<qec::operation, nb::object> m_py_operation_encodings;

  // Jitted python kernels are only alive as long as the
  // CppPyKernelDecorator they were compiled from, so we must cache them here
  std::vector<cudaq::python::CppPyKernelDecorator *> cachedDecorators;

public:
  ~PyCodeHandle() {
    for (auto decorator : cachedDecorators)
      delete decorator;
  }

  /// @brief Constructs a PyCodeHandle from a Python QEC code object
  /// @param registeredCode Python object containing the QEC code definition
  /// @throw std::runtime_error if the Python code lacks required attributes
  /// (stabilizers or operation_encodings)
  /// @details Initializes the handle by:
  /// - Validating the presence of required attributes
  /// - Converting Python stabilizers to C++ representation
  /// - Processing operation encodings and registering CUDA-Q kernels
  PyCodeHandle(nb::object registeredCode) : pyCode(registeredCode) {
    if (!nb::hasattr(registeredCode, "stabilizers"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.stabilizers = "
          "[cudaq.SpinOperator(...)] (qec.Stabilizers(...)). Please provide "
          "the stabilizers.");
    if (!nb::hasattr(registeredCode, "pauli_observables"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.pauli_observables = "
          "[cudaq.SpinOperator(...)]. Please provide the observables.");
    if (!nb::hasattr(registeredCode, "operation_encodings"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.operation_encodings = "
          "{...}. Please provide the CUDA-Q kernels for the operation "
          "encodings.");

    if (nb::hasattr(registeredCode, "pauli_observables")) {
      auto obs_terms = nb::cast<std::vector<cudaq::spin_op_term>>(
          registeredCode.attr("pauli_observables"));
      m_pauli_observables.reserve(obs_terms.size());
      for (auto &&term : obs_terms)
        m_pauli_observables.emplace_back(std::move(term));
    }
    // Get the stabilizers. First convert to spin_op_term's and then convert to
    // spin_op's.
    auto stab_terms = nb::cast<std::vector<cudaq::spin_op_term>>(
        registeredCode.attr("stabilizers"));
    m_stabilizers.reserve(stab_terms.size());
    for (auto &term : stab_terms)
      m_stabilizers.emplace_back(std::move(term));

    // Get the CUDA-Q kernels for the operation encodings
    auto opsDict =
        nb::cast<nb::dict>(registeredCode.attr("operation_encodings"));

    // For each CUDA-Q kernel, extract the JIT-ed function pointer
    for (auto [opKey, kernelHandle] : opsDict) {
      nb::object kernel = nb::borrow<nb::object>(kernelHandle);
      auto opKeyEnum = nb::cast<qec::operation>(opKey);

      // Save the original Python kernel object so that we can return a valid
      // CUDA-Q kernel back to Python later.
      m_py_operation_encodings.emplace(opKeyEnum, kernel);

      // Create the kernel interop object
      auto kernInterop = new cudaq::python::CppPyKernelDecorator(kernel);

      // Make sure we cast the function pointer correctly
      if (opKeyEnum == operation::stabilizer_round) {
        // The kernel must return list[cudaq.measure_handle]; a list[bool]
        // return discriminates measurements to bits and drops the record
        // indices that cudaq.detector needs. Validate the lowered return type.
        nb::object retTyObj = nb::hasattr(kernel, "return_type")
                                  ? kernel.attr("return_type")
                                  : nb::none();
        bool returnsHandleVector = false;
        if (!retTyObj.is_none()) {
          nb::object cc =
              nb::module_::import_("cudaq.mlir.dialects").attr("cc");
          // CUDA-Q renamed cc.StdvecType to cc.SequenceType (cuda-quantum
          // #5089). Accept either name so this keeps working across both the
          // pinned CUDA-Q version and main.
          nb::object seqTy = nb::hasattr(cc, "SequenceType")
                                 ? cc.attr("SequenceType")
                                 : cc.attr("StdvecType");
          if (nb::cast<bool>(seqTy.attr("isinstance")(retTyObj))) {
            nb::object eleTy = seqTy.attr("getElementType")(retTyObj);
            returnsHandleVector = nb::cast<bool>(
                cc.attr("MeasureHandleType").attr("isinstance")(eleTy));
          }
        }
        if (!returnsHandleVector)
          throw std::runtime_error(
              "Invalid stabilizer_round kernel: it must return "
              "list[cudaq.measure_handle]. Change the kernel's return "
              "annotation to '-> list[cudaq.measure_handle]'.");

        encoding fptr = kernInterop->getDirectKernelCall<qkernel<
            std::vector<measure_result>(patch, const std::vector<std::size_t> &,
                                        const std::vector<std::size_t> &)>>();
        operation_encodings.insert({opKeyEnum, std::move(fptr)});
        cachedDecorators.emplace_back(kernInterop);
        continue;
      }

      // FIXME handle other signatures later... this assumes single patch
      // signatures
      encoding fptr = kernInterop->getDirectKernelCall<qkernel<void(patch)>>();
      operation_encodings.insert({opKeyEnum, std::move(fptr)});
      cachedDecorators.emplace_back(kernInterop);
    }
  }

  /// @brief Expose read-only access to stored Python kernels.
  const std::unordered_map<qec::operation, nb::object> &
  get_py_operation_encodings() const {
    return m_py_operation_encodings;
  }

protected:
  // Trampoline methods for pure virtual functions
  std::size_t get_num_data_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_data_qubits")());
  }

  std::size_t get_num_ancilla_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_ancilla_qubits")());
  }

  std::size_t get_num_ancilla_x_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_ancilla_x_qubits")());
  }

  std::size_t get_num_ancilla_z_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_ancilla_z_qubits")());
  }

  std::size_t get_num_x_stabilizers() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_x_stabilizers")());
  }

  std::size_t get_num_z_stabilizers() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_z_stabilizers")());
  }

  /// @brief True if the Python class itself defines this method. A plain
  /// hasattr won't do: the @qec.code decorator makes the class inherit from
  /// the bound Code class, which exposes same-named read-only getters.
  bool has_py_override(const char *methodName) const {
    nb::handle base = nb::type<qec::code>();
    for (nb::handle cls : nb::cast<nb::tuple>(pyCode.type().attr("__mro__"))) {
      if (cls.is(base))
        break;
      if (nb::cast<bool>(cls.attr("__dict__").attr("__contains__")(methodName)))
        return true;
    }
    return false;
  }

  /// @brief Convert a Python-provided stabilizer schedule matrix (any 2D
  /// sequence of integers in [0, 255]) into a tensor, validating it against
  /// the corresponding parity matrix.
  cudaqx::tensor<uint8_t>
  py_stabilizer_schedule(const char *methodName,
                         const cudaqx::tensor<uint8_t> &parity) const {
    std::vector<std::vector<uint8_t>> rows;
    try {
      rows = nb::cast<std::vector<std::vector<uint8_t>>>(
          pyCode.attr(methodName)());
    } catch (...) {
      throw std::runtime_error(
          std::string(methodName) +
          " must return a 2D array/list of integers in [0, 255].");
    }
    if (rows.empty()) {
      if (parity.size() != 0)
        throw std::runtime_error(std::string(methodName) +
                                 " shape does not match the parity matrix.");
      return parity;
    }
    const std::size_t num_cols = rows[0].size();
    if (parity.rank() != 2 || parity.shape()[0] != rows.size() ||
        parity.shape()[1] != num_cols)
      throw std::runtime_error(std::string(methodName) +
                               " shape does not match the parity matrix.");
    cudaqx::tensor<uint8_t> schedule({rows.size(), num_cols});
    for (std::size_t r = 0; r < rows.size(); ++r) {
      if (rows[r].size() != num_cols)
        throw std::runtime_error(std::string(methodName) +
                                 " rows must all have the same length.");
      for (std::size_t c = 0; c < num_cols; ++c) {
        // A schedule may reorder interactions but not change which data
        // qubits each stabilizer touches.
        if ((rows[r][c] != 0) != (parity.at({r, c}) != 0))
          throw std::runtime_error(
              std::string(methodName) +
              " support pattern does not match the parity matrix.");
        schedule.at({r, c}) = rows[r][c];
      }
    }
    return schedule;
  }

public:
  cudaqx::tensor<uint8_t> get_stabilizer_schedule_x() const override {
    if (!has_py_override("get_stabilizer_schedule_x"))
      return code::get_stabilizer_schedule_x();
    return py_stabilizer_schedule("get_stabilizer_schedule_x", get_parity_x());
  }

  cudaqx::tensor<uint8_t> get_stabilizer_schedule_z() const override {
    if (!has_py_override("get_stabilizer_schedule_z"))
      return code::get_stabilizer_schedule_z();
    return py_stabilizer_schedule("get_stabilizer_schedule_z", get_parity_z());
  }
};

namespace {
static nb::object get_python_kernel_or_throw(const code &self, operation op) {
  auto *pyHandle = dynamic_cast<const PyCodeHandle *>(&self);
  if (!pyHandle)
    throw std::runtime_error("This code was not registered from Python; no "
                             "Python kernel is available.");

  const auto &pyKernels = pyHandle->get_py_operation_encodings();
  auto it = pyKernels.find(op);
  if (it == pyKernels.end())
    throw std::runtime_error("No Python kernel registered for requested op.");

  return it->second;
}

template <typename T>
static auto copyCodeMatrixToPyArray(const cudaqx::tensor<T> &tensor,
                                    std::size_t emptyCols) {
  if (tensor.rank() == 0 && tensor.size() == 0) {
    T *data_copy = new T[0];
    size_t arr_shape[] = {0, emptyCols};
    return nb::ndarray<nb::numpy, T>(
        data_copy, 2, arr_shape, nb::capsule(data_copy, [](void *p) noexcept {
          delete[] static_cast<T *>(p);
        }));
  }

  return cudaq::python::copyCUDAQXTensorToPyArray(tensor);
}
} // namespace

// Registry to store code factory functions
class PyCodeRegistry {
private:
  static std::unordered_map<std::string, std::function<nb::object(nb::kwargs)>>
      registry;

public:
  static std::vector<std::string> get_keys() {
    std::vector<std::string> keys;
    for (const auto &pair : registry) {
      keys.push_back(pair.first);
    }
    return keys;
  }

  static void register_code(const std::string &name,
                            std::function<nb::object(nb::kwargs)> factory) {
    cudaq::qec::info("Registering Pythonic QEC Code with name {}", name);
    registry[name] = factory;
  }

  static nb::object get_code(const std::string &name, nb::kwargs options) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      throw std::runtime_error("Unknown code: " + name);
    }

    // Constructs the Python type with kwargs...
    return it->second(options);
  }

  static bool contains(const std::string &name) {
    return registry.find(name) != registry.end();
  }
};

std::unordered_map<std::string, std::function<nb::object(nb::kwargs)>>
    PyCodeRegistry::registry;

void bindCode(nb::module_ &mod) {

  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  nb::class_<qec::two_qubit_depolarization, cudaq::kraus_channel>(
      qecmod, "TwoQubitDepolarization",
      R"#(Models the decoherence of the each qubit independently in a two-qubit operation into a mixture "
      of the computational basis states, `|0>` and `|1>`.)#")
      .def(nb::init<double>(), nb::arg("probability"),
           "Initialize the `TwoQubitDepolarizationChannel` with the provided "
           "`probability`.");

  nb::class_<qec::two_qubit_bitflip, cudaq::kraus_channel>(
      qecmod, "TwoQubitBitFlip",
      R"#(Models independent bit flip errors after a two-qubit operation.)#")
      .def(nb::init<double>(), nb::arg("probability"),
           "Initialize the `TwoQubitBitFlip` with the provided "
           "`probability`.");

  nb::enum_<operation>(
      qecmod, "operation",
      "Enumeration of quantum operations for state preparation")
      .value("prep0", operation::prep0, "Prepare qubit in |0⟩ state")
      .value("prep1", operation::prep1, "Prepare qubit in |1⟩ state")
      .value("prepp", operation::prepp, "Prepare qubit in |+⟩ state")
      .value("prepm", operation::prepm, "Prepare qubit in |-⟩ state")
      .value("x", operation::x, "Apply the logical X operation")
      .value("y", operation::y, "Apply the logical Y operation")
      .value("z", operation::z, "Apply the logical Z operation")
      .value("h", operation::h, "Apply the logical H operation")
      .value("s", operation::s, "Apply the logical S operation")
      .value("cx", operation::cx, "Apply the logical CX operation")
      .value("cy", operation::cy, "Apply the logical CY operation")
      .value("cz", operation::cz, "Apply the logical CZ operation")
      .value("stabilizer_round", operation::stabilizer_round,
             "Apply the stabilizer round operation.");

  qecmod.def(
      "get_code",
      [](const std::string &name, nb::kwargs options) -> std::unique_ptr<code> {
        if (PyCodeRegistry::contains(name))
          return std::make_unique<PyCodeHandle>(
              PyCodeRegistry::get_code(name, options));

        if (options.contains("stabilizers")) {
          auto obj = options["stabilizers"];
          if (!nb::isinstance<nb::list>(obj))
            throw std::runtime_error(
                "invalid stabilizers passed to get_code, must be a list of "
                "string pauli words or list of cudaq.SpinOperator.");

          if (nb::isinstance<nb::str>(nb::cast<nb::list>(obj)[0])) {
            options.attr("pop")("stabilizers");
            auto words = nb::cast<std::vector<std::string>>(obj);
            std::vector<cudaq::spin_op_term> ops;
            for (auto &os : words)
              ops.emplace_back(cudaq::spin_op::from_word(os));
            sortStabilizerOps(ops);
            return get_code(name, ops, hetMapFromKwargs(options));
          }

          if (nb::isinstance<cudaq::spin_op_term>(nb::cast<nb::list>(obj)[0])) {
            options.attr("pop")("stabilizers");
            return get_code(name,
                            nb::cast<std::vector<cudaq::spin_op_term>>(obj),
                            hetMapFromKwargs(options));
          }

          throw std::runtime_error(
              "get_code error - invalid stabilizers element type.");
        }

        return get_code(name, hetMapFromKwargs(options));
      },
      "Retrieve a quantum error correction code by name with optional "
      "parameters");

  qecmod.def(
      "get_available_codes",
      []() {
        auto codes = cudaq::qec::get_available_codes();
        auto py_codes = PyCodeRegistry::get_keys();
        codes.insert(codes.end(), py_codes.begin(), py_codes.end());
        return codes;
      },
      "Get a list of all available quantum error correction codes (C++ and "
      "Python).");

  nb::class_<code, PyCode>(qecmod, "Code",
                           "Represents a quantum error correction code")
      .def(nb::init<>())
      .def(
          "get_parity",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_parity(),
                                           2 * code.get_num_data_qubits());
          },
          "Get the parity check matrix of the code")
      .def(
          "get_parity_x",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_parity_x(),
                                           code.get_num_data_qubits());
          },
          "Get the X-type parity check matrix of the code")
      .def(
          "get_parity_z",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_parity_z(),
                                           code.get_num_data_qubits());
          },
          "Get the Z-type parity check matrix of the code")
      .def(
          "get_stabilizer_schedule_x",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_stabilizer_schedule_x(),
                                           code.get_num_data_qubits());
          },
          "Get the X-stabilizer schedule matrix passed to the code's "
          "stabilizer_round kernel. Entry 0 = no support, entry k >= 1 = "
          "interaction at timestep k; defaults to the X-type parity check "
          "matrix (every interaction at timestep 1).")
      .def(
          "get_stabilizer_schedule_z",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_stabilizer_schedule_z(),
                                           code.get_num_data_qubits());
          },
          "Get the Z-stabilizer schedule matrix passed to the code's "
          "stabilizer_round kernel. Entry 0 = no support, entry k >= 1 = "
          "interaction at timestep k; defaults to the Z-type parity check "
          "matrix (every interaction at timestep 1).")
      .def(
          "get_pauli_observables_matrix",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_pauli_observables_matrix(),
                                           2 * code.get_num_data_qubits());
          },
          "Get a matrix of the Pauli observables of the code")
      .def(
          "get_observables_x",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_observables_x(),
                                           code.get_num_data_qubits());
          },
          "Get the Pauli X observables of the code")
      .def(
          "get_observables_z",
          [](code &code) {
            return copyCodeMatrixToPyArray(code.get_observables_z(),
                                           code.get_num_data_qubits());
          },
          "Get the Pauli Z observables of the code")
      .def("get_stabilizers", &code::get_stabilizers,
           "Get the stabilizer generators of the code")
      .def("contains_operation", &code::contains_operation, nb::arg("op"),
           "Return true if this code contains the given operation encoding")
      .def(
          "get_operation_one_qubit",
          [](const code &self, operation op) -> nb::object {
            if (!self.contains_operation(op))
              throw std::runtime_error(
                  "No encoding registered for requested op.");
            return get_python_kernel_or_throw(self, op);
          },
          nb::arg("op"),
          R"pbdoc(
              Get a CUDA-Q Python kernel for a one-qubit logical operation.

              Returns:
                A valid CUDA-Q Python kernel object.
            )pbdoc")
      .def(
          "get_operation_two_qubit",
          [](const code &self, operation op) -> nb::object {
            if (!self.contains_operation(op))
              throw std::runtime_error(
                  "No encoding registered for requested op.");
            return get_python_kernel_or_throw(self, op);
          },
          nb::arg("op"),
          R"pbdoc(
              Get a CUDA-Q Python kernel for a two-qubit logical operation.

              Returns:
                A valid CUDA-Q Python kernel object.
            )pbdoc")
      .def(
          "get_stabilizer_round",
          [](const code &self) -> nb::object {
            if (!self.contains_operation(operation::stabilizer_round))
              throw std::runtime_error(
                  "No stabilizer_round encoding is registered.");
            return get_python_kernel_or_throw(self,
                                              operation::stabilizer_round);
          },
          R"pbdoc(
              Get a CUDA-Q Python kernel for a stabilizer round logical operation.

              Returns:
                A valid CUDA-Q Python kernel object.
            )pbdoc")
      .def("get_num_data_qubits", &code::get_num_data_qubits,
           "Total number of physical data qubits required by the code.")
      .def("get_num_ancilla_qubits", &code::get_num_ancilla_qubits,
           "Total number of ancilla qubits required by the code.")
      .def(
          "get_num_ancilla_x_qubits", &code::get_num_ancilla_x_qubits,
          "Number of X-type ancilla qubits used for X stabilizer measurements.")
      .def(
          "get_num_ancilla_z_qubits", &code::get_num_ancilla_z_qubits,
          "Number of Z-type ancilla qubits used for Z stabilizer measurements.")
      .def("get_num_x_stabilizers", &code::get_num_x_stabilizers,
           "Number of X-type stabilizers.")
      .def("get_num_z_stabilizers", &code::get_num_z_stabilizers,
           "Number of Z-type stabilizers.");

  qecmod.def("code", [&](const std::string &name) {
    auto cppCodes = qec::get_available_codes();
    if (std::find(cppCodes.begin(), cppCodes.end(), name) != cppCodes.end())
      throw std::runtime_error("Invalid Python QEC Code name. " + name +
                               " is already used in the C++ Code registry.");

    return nb::cpp_function([name](nb::object code_class) -> nb::object {
      // Create new class that inherits from both Code and the original
      nb::object base_code = nb::module_::import_("cudaq_qec").attr("Code");
      // Create new type using Python's type() function
      nb::tuple bases = nb::make_tuple(base_code);
      // __dict__ is a read-only mappingproxy; copy to a real dict for
      // PyType_Type.tp_new
      nb::dict namespace_dict;
      namespace_dict.update(code_class.attr("__dict__"));

      if (!nb::hasattr(code_class, "get_num_data_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_data_qubits method");

      if (!nb::hasattr(code_class, "get_num_ancilla_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_qubits method");

      if (!nb::hasattr(code_class, "get_num_ancilla_x_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_x_qubits method");

      if (!nb::hasattr(code_class, "get_num_ancilla_z_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_z_qubits method");

      if (!nb::hasattr(code_class, "get_num_x_stabilizers"))
        throw std::runtime_error(
            "Code class must implement get_num_x_stabilizers method");

      if (!nb::hasattr(code_class, "get_num_z_stabilizers"))
        throw std::runtime_error(
            "Code class must implement get_num_z_stabilizers method");

      // Use Python's type() so the correct metaclass (nanobind's) is resolved
      nb::object type_fn = nb::module_::import_("builtins").attr("type");
      nb::object new_class =
          type_fn(code_class.attr("__name__"), bases, namespace_dict);

      // Register the new class in the code registry
      PyCodeRegistry::register_code(name, [new_class](nb::kwargs options) {
        nb::object instance = new_class(**options);
        return instance;
      });
      return new_class;
    });
  });

  qecmod.def(
      "generate_random_bit_flips",
      [](std::size_t numBits, double error_probability) {
        auto data = generate_random_bit_flips(numBits, error_probability);
        return cudaq::python::copy1DCUDAQXTensorToPyArray(data);
      },
      "Generate a rank-1 tensor for random bits", nb::arg("numBits"),
      nb::arg("error_probability"));
  qecmod.def(
      "sample_memory_circuit",
      [](code &code, std::size_t numShots, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        auto [synd, dataRes] =
            noise ? sample_memory_circuit(code, numShots, numRounds, *noise)
                  : sample_memory_circuit(code, numShots, numRounds);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(synd),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample the memory circuit of the code", nb::arg("code"),
      nb::arg("numShots"), nb::arg("numRounds"), nb::arg("noise") = nb::none());
  qecmod.def(
      "sample_memory_circuit",
      [](code &code, operation op, std::size_t numShots, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        auto [synd, dataRes] =
            noise ? sample_memory_circuit(code, op, numShots, numRounds, *noise)
                  : sample_memory_circuit(code, op, numShots, numRounds);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(synd),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample the memory circuit of the code with a specific initial "
      "operation. Returns (syndromes, data): syndromes has numFixed "
      "boundary detectors (basis matching op), then one block per "
      "inter-round transition, then numFixed more boundary detectors.",
      nb::arg("code"), nb::arg("op"), nb::arg("numShots"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  auto make_directional_sample_memory_circuit = [](auto samplingFn) {
    return [samplingFn](code &code, operation op, std::size_t numShots,
                        std::size_t numRounds,
                        std::optional<cudaq::noise_model> noise =
                            std::nullopt) {
      cudaq::noise_model emptyNoise;
      auto [synd, dataRes] = samplingFn(code, op, numShots, numRounds,
                                        noise ? *noise : emptyNoise);
      return nb::make_tuple(cudaq::python::copyCUDAQXTensorToPyArray(synd),
                            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
    };
  };

  qecmod.def(
      "x_sample_memory_circuit",
      make_directional_sample_memory_circuit(x_sample_memory_circuit),
      "Sample the memory circuit of the code with a specific initial "
      "operation, keeping only the X stabilizer syndromes (same detector "
      "layout as sample_memory_circuit, restricted to X stabilizers).",
      nb::arg("code"), nb::arg("op"), nb::arg("numShots"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  qecmod.def(
      "z_sample_memory_circuit",
      make_directional_sample_memory_circuit(z_sample_memory_circuit),
      "Sample the memory circuit of the code with a specific initial "
      "operation, keeping only the Z stabilizer syndromes (same detector "
      "layout as sample_memory_circuit, restricted to Z stabilizers).",
      nb::arg("code"), nb::arg("op"), nb::arg("numShots"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  nb::class_<decoder_context>(qecmod, "DecoderContext",
                              R"pbdoc(
      Lazy handle returned by ``decoder_context_from_memory_circuit``.

      Stores the raw circuit analysis. Each ``*_component`` method canonicalizes
      exactly the requested stabilizer type and returns a ``(dem, m2d, m2o)``
      tuple.
    )pbdoc")
      .def_prop_ro("num_measurements", &decoder_context::num_measurements,
                   "Total number of measurements per shot.")
      .def(
          "x_component",
          [](const decoder_context &h) {
            auto ctx = h.x_component();
            return nb::make_tuple(ctx.dem, ctx.m2d.rows, ctx.m2o.rows);
          },
          R"pbdoc(
       Canonicalize X-stabilizer detectors; return (dem, m2d, m2o).

       ``dem`` is the canonicalized DetectorErrorModel, ``m2d`` and ``m2o``
       are lists of lists of measurement indices.
    )pbdoc")
      .def(
          "z_component",
          [](const decoder_context &h) {
            auto ctx = h.z_component();
            return nb::make_tuple(ctx.dem, ctx.m2d.rows, ctx.m2o.rows);
          },
          R"pbdoc(
       Canonicalize Z-stabilizer detectors; return (dem, m2d, m2o).

       ``dem`` is the canonicalized DetectorErrorModel, ``m2d`` and ``m2o``
       are lists of lists of measurement indices.
    )pbdoc")
      .def(
          "full_component",
          [](const decoder_context &h) {
            auto ctx = h.full_component();
            return nb::make_tuple(ctx.dem, ctx.m2d.rows, ctx.m2o.rows);
          },
          R"pbdoc(
       Canonicalize both stabilizer types with boundary awareness;
       return (dem, m2d, m2o).

       ``dem`` is the canonicalized DetectorErrorModel, ``m2d`` and ``m2o``
       are lists of lists of measurement indices.
    )pbdoc");

  qecmod.def(
      "d_sparse",
      [](const std::vector<std::vector<std::size_t>> &rows) {
        cudaq::M2DSparseMatrix m2d;
        m2d.rows = rows;
        return cudaq::qec::d_sparse(m2d);
      },
      R"pbdoc(
       Flatten a measurement-to-detector map into the -1-terminated sparse
       vector a realtime decoder config expects for its D_sparse.

       ``m2d`` is a list of lists: ``m2d[d]`` contains the measurement indices
       whose XOR forms detector ``d``. Each detector's indices are emitted in
       order, followed by -1.

       This is a standalone helper for consumers who hold an m2d map (e.g.,
       extracted from a ``decoder_inputs`` returned by a component method)
       and want to build the D_sparse vector without going through a
       ``DecoderContext``.
    )pbdoc",
      nb::arg("m2d"));

  qecmod.def(
      "dem_from_memory_circuit",
      [](code &code, operation op, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt,
         bool decompose_errors = false) {
        if (!noise)
          throw std::runtime_error(
              "dem_from_memory_circuit requires a noise model; noise=None is "
              "not supported.");
        return dem_from_memory_circuit(code, op, numRounds, *noise,
                                       decompose_errors);
      },
      R"pbdoc(
        Generate a detector error model from a memory circuit.

        This function generates a detector error model from a memory circuit.
        The memory circuit is specified by the code, the initial state preparation
        operation, and the number of stabilizer measurement rounds. The noise
        model is required.

        Args:
            code: The code to generate the detector error model for.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.
            decompose_errors: If True, hyperedge error mechanisms are decomposed
                into pairs of two-detector edges by Stim before returning.

        Returns:
            A detector error model.
      )pbdoc",
      nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none(), nb::arg("decompose_errors") = false);

  auto make_directional_dem_from_memory_circuit = [](auto demFn,
                                                     const char *name) {
    return [demFn, name](code &code, operation op, std::size_t numRounds,
                         std::optional<cudaq::noise_model> noise = std::nullopt,
                         bool decompose_errors = false) {
      if (!noise)
        throw std::runtime_error(std::string(name) +
                                 " requires a noise model; noise=None is "
                                 "not supported.");
      return demFn(code, op, numRounds, *noise, decompose_errors);
    };
  };

  qecmod.def("x_dem_from_memory_circuit",
             make_directional_dem_from_memory_circuit(
                 x_dem_from_memory_circuit, "x_dem_from_memory_circuit"),
             R"pbdoc(
        Generate a detector error model from a memory circuit in the X basis.

        This function generates a detector error model from a memory circuit in
        the X basis. The memory circuit is specified by the code, the initial
        state preparation operation, and the number of stabilizer measurement
        rounds. The noise model is required.

        Args:
            code: The code to generate the detector error model for.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.
            decompose_errors: If True, hyperedge error mechanisms are decomposed
                into pairs of two-detector edges by Stim before returning.

        Returns:
            A detector error model.
      )pbdoc",
             nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
             nb::arg("noise") = nb::none(),
             nb::arg("decompose_errors") = false);

  qecmod.def("z_dem_from_memory_circuit",
             make_directional_dem_from_memory_circuit(
                 z_dem_from_memory_circuit, "z_dem_from_memory_circuit"),
             R"pbdoc(
        Generate a detector error model from a memory circuit in the Z basis.

        This function generates a detector error model from a memory circuit in
        the Z basis. The memory circuit is specified by the code, the initial
        state preparation operation, and the number of stabilizer measurement
        rounds. The noise model is required.

        Args:
            code: The code to generate the detector error model for.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.
            decompose_errors: If True, hyperedge error mechanisms are decomposed
                into pairs of two-detector edges by Stim before returning.

        Returns:
            A detector error model.
      )pbdoc",
             nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
             nb::arg("noise") = nb::none(),
             nb::arg("decompose_errors") = false);

  qecmod.def(
      "decoder_context_from_memory_circuit",
      [](code &code, operation op, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt,
         bool decompose_errors = false) {
        if (!noise)
          throw std::runtime_error(
              "decoder_context_from_memory_circuit requires a noise model; "
              "noise=None is not supported.");
        return decoder_context_from_memory_circuit(code, op, numRounds, *noise,
                                                   decompose_errors);
      },
      R"pbdoc(
        Run a memory-circuit analysis and return a lazy DecoderContext handle.

        Executes ``dem_from_kernel`` once and stores the raw result.
        Canonicalization is deferred: call ``x_component()``,
        ``z_component()``, or ``full_component()`` on the returned handle to
        canonicalize exactly the stabilizer type needed.

        Args:
            code: The code to characterize.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.
            decompose_errors: If True, hyperedge error mechanisms are decomposed
                into pairs of two-detector edges by Stim before returning.

        Returns:
            A DecoderContext handle; call a component method to obtain the
            canonicalized ``(dem, m2d, m2o)`` tuple.
      )pbdoc",
      nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none(), nb::arg("decompose_errors") = false);

  qecmod.def(
      "sample_code_capacity",
      [](code &code, std::size_t numShots, double errorProb,
         std::optional<int> seed = std::nullopt) {
        if (seed.has_value()) {
          auto [syndromes, dataRes] =
              sample_code_capacity(code, numShots, errorProb, seed.value());
          return nb::make_tuple(
              cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
              cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
        }

        auto [syndromes, dataRes] =
            sample_code_capacity(code, numShots, errorProb);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample syndrome measurements with code capacity noise.", nb::arg("code"),
      nb::arg("numShots"), nb::arg("errorProb"), nb::arg("seed") = nb::none());
  qecmod.def(
      "sample_code_capacity",
      [](const nb::ndarray<nb::numpy, uint8_t> H, std::size_t numShots,
         double errorProb, std::optional<std::size_t> seed = std::nullopt) {
        if (seed.has_value()) {
          auto [syndromes, dataRes] = sample_code_capacity(
              toTensor(H), numShots, errorProb, seed.value());
          return nb::make_tuple(
              cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
              cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
        }

        auto [syndromes, dataRes] =
            sample_code_capacity(toTensor(H), numShots, errorProb);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample syndrome measurements with code capacity noise.", nb::arg("H"),
      nb::arg("numShots"), nb::arg("errorProb"), nb::arg("seed") = nb::none());

  std::stringstream ss;
  ss << "CUDA-Q QEC " << cudaq::qec::getVersion() << " ("
     << cudaq::qec::getFullRepositoryVersion() << ")";
  qecmod.attr("__version__") = ss.str();
}
} // namespace cudaq::qec

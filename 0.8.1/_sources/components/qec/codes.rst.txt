QEC Codes
=========

The ``cudaq-qec`` code interface (:code:`cudaq::qec::code`) defines what a quantum error correcting code is: a mapping from logical operations to their physical CUDA-Q kernel implementations. This page covers the framework — the class structure and how to define or extend a code — together with the codes that ship with the library. Read it to understand the model or to implement your own code. For a runnable, end-to-end program, see the :doc:`Creating New QEC Codes example </examples_rst/qec/creating_qec_codes>`.

QEC Code Framework :code:`cudaq::qec::code`
-------------------------------------------

The :code:`cudaq::qec::code` class serves as the base class for all quantum error correcting codes in CUDA-Q QEC. It provides
a flexible extension point for implementing new codes and defines the core interface that all QEC codes must support.

The core abstraction here is that of a mapping or dictionary of logical operations to their
corresponding physical implementation in the error correcting code as CUDA-Q quantum kernels.

Class Structure
^^^^^^^^^^^^^^^

The code base class provides:

1. **Operation Enumeration**: Defines supported logical operations

   .. code-block:: cpp

       enum class operation {
           x,     // Logical X gate
           y,     // Logical Y gate
           z,     // Logical Z gate
           h,     // Logical Hadamard gate
           s,     // Logical S gate
           cx,    // Logical CNOT gate
           cy,    // Logical CY gate
           cz,    // Logical CZ gate
           stabilizer_round,  // Stabilizer measurement round
           prep0, // Prepare |0⟩ state
           prep1, // Prepare |1⟩ state
           prepp, // Prepare |+⟩ state
           prepm  // Prepare |-⟩ state
       };


2. **Patch Type**: Defines the structure of a logical qubit patch

   .. code-block:: cpp

       struct patch {
           cudaq::qview<> data;  // View of data qubits
           cudaq::qview<> ancx;  // View of X stabilizer ancilla qubits
           cudaq::qview<> ancz;  // View of Z stabilizer ancilla qubits
       };

   The `patch` type represents a logical qubit in quantum error correction codes. It contains:

   - `data`: A view of the data qubits in the patch
   - `ancx`: A view of the ancilla qubits used for X stabilizer measurements
   - `ancz`: A view of the ancilla qubits used for Z stabilizer measurements

   This structure is designed for use within CUDA-Q kernel code and provides a
   convenient way to access different qubit subsets within a logical qubit patch.


3. **Kernel Type Aliases**: Defines quantum kernel signatures

   .. code-block:: cpp

       using one_qubit_encoding = cudaq::qkernel<void(patch)>;
       using two_qubit_encoding = cudaq::qkernel<void(patch, patch)>;
       using stabilizer_round = cudaq::qkernel<std::vector<cudaq::measure_result>(
           patch, const std::vector<std::size_t>&, const std::vector<std::size_t>&)>;

   The two vector arguments of :code:`stabilizer_round` are the flattened X and
   Z stabilizer *schedule* matrices, which can encode an optimized gate order
   on top of the parity-check support. See
   :cpp:func:`cudaq::qec::code::get_stabilizer_schedule_x` for the encoding
   and the default (the plain parity matrices).

4. **Protected Members**:

   - :code:`operation_encodings`: Maps operations to their quantum kernel implementations. The key is the ``operation`` enum and the value is a variant on the above kernel type aliases.
   - :code:`m_stabilizers`: Stores the code's stabilizer generators

Implementing a New Code
^^^^^^^^^^^^^^^^^^^^^^^

To implement a new quantum error correcting code:

1. **Create a New Class**:

   .. code-block:: cpp

       class my_code : public qec::code {
       protected:
           // Implement required virtual methods
       public:
           my_code(const heterogeneous_map& options);
       };

2. **Implement Required Virtual Methods**:

   .. code-block:: cpp

       // Number of physical data qubits
       std::size_t get_num_data_qubits() const override;

       // Total number of ancilla qubits
       std::size_t get_num_ancilla_qubits() const override;

       // Number of X-type ancilla qubits
       std::size_t get_num_ancilla_x_qubits() const override;

       // Number of Z-type ancilla qubits
       std::size_t get_num_ancilla_z_qubits() const override;

3. **Define Quantum Kernels**:

   Create CUDA-Q kernels for each logical operation:

   .. code-block:: cpp

       __qpu__ void x(patch p) {
           // Implement logical X
       }

       __qpu__ std::vector<cudaq::measure_result> stabilizer(patch p,
           const std::vector<std::size_t>& x_stabs,
           const std::vector<std::size_t>& z_stabs) {
           // Implement stabilizer measurements
       }

   .. note::

      The two vector arguments passed to the :code:`stabilizer_round` kernel
      are the flattened X and Z stabilizer *schedule* matrices returned by
      :cpp:func:`cudaq::qec::code::get_stabilizer_schedule_x` and
      :cpp:func:`cudaq::qec::code::get_stabilizer_schedule_z`. By default
      these equal the plain parity-check matrices (every entry 0 or 1), but a
      code can override the :code:`get_stabilizer_schedule_*` methods to
      encode a gate order, in which case entry :code:`k >= 1` means the
      interaction executes at timestep :code:`k` (the built-in
      :code:`surface_code` does this to avoid hook errors). A kernel that only
      needs the support pattern should therefore test entries for
      :code:`!= 0` rather than :code:`== 1`.

4. **Register Operations**:

   In the constructor, register quantum kernels for each operation:

   .. code-block:: cpp

        my_code::my_code(const heterogeneous_map& options) : code() {
            // Register operations
            operation_encodings.insert(
               std::make_pair(operation::x, x));
            operation_encodings.insert(
               std::make_pair(operation::stabilizer_round, stabilizer));

            // Define stabilizer generators
            m_stabilizers = fromPauliWords({"XXXX", "ZZZZ"});
        }


   Note that in your constructor, you have access to user-provided ``options``. For
   example, if your code depends on an integer parameter called ``distance``, you can
   retrieve that from the user via

   .. code-block:: cpp

        my_code::my_code(const heterogeneous_map& options) : code() {
            // ... fill the map and stabilizers ...

            // Get the user-provided distance, or just
            // set to 3 if user did not provide one
            this->distance = options.get<int>("distance", /*defaultValue*/ 3);
        }

5. **Register Extension Point**:

   Add extension point registration.

   .. code-block:: cpp

       class my_code : public qec::code {
           // ... members from above ...

           CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
               my_code,
               static std::unique_ptr<qec::code> create(
                   const heterogeneous_map &options) {
                   return std::make_unique<my_code>(options);
               })
       };

       CUDAQ_EXT_PT_REGISTER_TYPE(my_code)

Example: Steane Code
^^^^^^^^^^^^^^^^^^^^^

The Steane [[7,1,3]] code provides a complete example implementation:

1. **Header Definition**:

   - Declares quantum kernels for all logical operations
   - Defines the code class with required virtual methods
   - Specifies 7 data qubits and 6 ancilla qubits (3 X-type, 3 Z-type)

2. **Implementation**:

   .. code-block:: cpp

       steane::steane(const heterogeneous_map &options) : code() {
           // Register all logical operations
           operation_encodings.insert(
               std::make_pair(operation::x, x));
           // ... register other operations ...

           // Define stabilizer generators
           m_stabilizers = fromPauliWords({
               "XXXXIII", "IXXIXXI", "IIXXIXX",
               "ZZZZIII", "IZZIZZI", "IIZZIZZ"
           });
       }

3. **Quantum Kernels**:

   Implements fault-tolerant logical operations:

   .. code-block:: cpp

       __qpu__ void x(patch logicalQubit) {
           // Apply logical X to specific data qubits
           x(logicalQubit.data[4], logicalQubit.data[5],
             logicalQubit.data[6]);
       }

       __qpu__ std::vector<cudaq::measure_result> stabilizer(patch logicalQubit,
           const std::vector<std::size_t>& x_stabilizers,
           const std::vector<std::size_t>& z_stabilizers) {
           // Measure X stabilizers
           h(logicalQubit.ancx);
           // ... apply controlled-X gates ...
           h(logicalQubit.ancx);

           // Measure Z stabilizers
           // ... apply controlled-X gates ...

           // Return measurement results
           return mz(logicalQubit.ancz, logicalQubit.ancx);
       }

Implementing a New Code in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-Q QEC supports implementing quantum error correction codes in Python
using the :code:`@qec.code` decorator. This provides a more accessible way
to prototype and develop new codes.

1. **Create a New Python File**:

   Create a new file (e.g., :code:`my_steane.py`) with your code implementation:

   .. literalinclude:: ../../examples/qec/python/my_steane.py
      :language: python
      :start-after: [Begin Documentation1]
      :end-before: [End Documentation1]

2. **Define Quantum Kernels**:

   Implement the required quantum kernels using the :code:`@cudaq.kernel` decorator:

   .. literalinclude:: ../../examples/qec/python/my_steane.py
      :language: python
      :start-after: [Begin Documentation2]
      :end-before: [End Documentation2]

   .. note::

      The kernel registered for :code:`stabilizer_round` must be annotated to
      return :code:`list[cudaq.measure_handle]`.

   .. note::

      As in C++, the two list arguments passed to the
      :code:`stabilizer_round` kernel are the flattened X and Z stabilizer
      schedule matrices, which default to the parity-check matrices. A Python
      code can optionally define :code:`get_stabilizer_schedule_x` /
      :code:`get_stabilizer_schedule_z` methods returning a 2D array with the
      same shape and support pattern as the corresponding parity-check
      matrix, where entry :code:`k >= 1` schedules that interaction at
      timestep :code:`k`.

3. **Implement the Code Class**:

   Create a class decorated with :code:`@qec.code` that implements the required interface:

   .. literalinclude:: ../../examples/qec/python/my_steane.py
      :language: python
      :start-after: [Begin Documentation3]
      :end-before: [End Documentation3]

4. **Using the Code**:

   The code can now be used like any other CUDA-Q QEC code:

   .. literalinclude:: ../../examples/qec/python/my_steane_test.py
      :language: python
      :start-after: [Begin Documentation]

Key Points
^^^^^^^^^^^

* The :code:`@qec.code` decorator takes the name of the code as an argument
* Operation encodings are registered via the :code:`operation_encodings` dictionary
* Stabilizer generators are defined as a list of :code:`cudaq.SpinOperator`
* The code must implement all required methods from the base class interface


Using the Code Framework
^^^^^^^^^^^^^^^^^^^^^^^^^

To use an implemented code:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Create a code instance
        code = qec.get_code("steane")

        # Access stabilizer information
        stabilizers = code.get_stabilizers()
        parity = code.get_parity()

        # The code can now be used for various numerical
        # experiments - see section below.

.. tab:: C++

    .. code-block:: cpp

        // Create a code instance
        auto code = cudaq::qec::get_code("steane");

        // Access stabilizer information
        auto stabilizers = code->get_stabilizers();
        auto parity = code->get_parity();

        // The code can now be used for various numerical
        // experiments - see section below.


Pre-built QEC Codes
-------------------

CUDA-Q QEC provides several well-studied quantum error correction codes out of the box. Here's a detailed overview of each:

Steane Code
^^^^^^^^^^^

The Steane code is a ``[[7,1,3]]`` CSS (Calderbank-Shor-Steane) code that encodes
one logical qubit into seven physical qubits with a code distance of 3.

**Key Properties**:

* Data qubits: 7
* Encoded qubits: 1
* Code distance: 3
* Ancilla qubits: 6 (3 for X stabilizers, 3 for Z stabilizers)

**Stabilizer Generators**:

* X-type: ``["XXXXIII", "IXXIXXI", "IIXXIXX"]``
* Z-type: ``["ZZZZIII", "IZZIZZI", "IIZZIZZ"]``

The Steane code can correct any single-qubit error and detect up to two errors.
It is particularly notable for being the smallest CSS code that can implement a universal set of transversal gates.

Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Create Steane code instance
        steane = qec.get_code("steane")

.. tab:: C++

    .. code-block:: cpp

        auto steane = cudaq::qec::get_code("steane");

Repetition Code
^^^^^^^^^^^^^^^
The repetition code is a simple [[n,1,n]] code that protects against
bit-flip (X) errors by encoding one logical qubit into n physical qubits, where n is the code distance.

**Key Properties**:

* Data qubits: n (distance)
* Encoded qubits: 1
* Code distance: n
* Ancilla qubits: n-1 (all for Z stabilizers)

**Stabilizer Generators**:

* For distance 3: ``["ZZI", "IZZ"]``
* For distance 5: ``["ZZIII", "IZZII", "IIZZI", "IIIZZ"]``

The repetition code is primarily educational as it can only correct
X errors. However, it serves as an excellent introduction to QEC concepts.

Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Create distance-3 repetition code
        code = qec.get_code('repetition', distance=3)

        # Access stabilizers
        stabilizers = code.get_stabilizers()  # Returns ["ZZI", "IZZ"]

.. tab:: C++

    .. code-block:: cpp

        auto code = qec::get_code("repetition", {{"distance", 3}});

        // Access stabilizers
        auto stabilizers = code->get_stabilizers();

Surface Code
^^^^^^^^^^^^

The library provides a **rotated surface code** on a two-dimensional qubit
layout with **open boundaries** (a single patch). It is a **CSS** code—:math:`X`
and :math:`Z` errors are handled in separate CSS sectors—encoding **one logical
qubit** into :math:`d^2` data qubits with code distance :math:`d` in this
layout. Stabilizers have weight four in the bulk and weight two on the boundary, following the grid convention
described in `Towards a Standardized Definition of Quantum Circuits for Quantum
Error Correction with Rotated Surface Codes
<https://arxiv.org/abs/2311.10687>`__.

**Key Properties** (distance :math:`d` in this implementation):

* Data qubits: :math:`d^2`
* Encoded logical qubits: 1
* Code distance: :math:`d`
* Stabilizers: :math:`d^2 - 1` total—:math:`(d^2 - 1) / 2` :math:`X`-type and
  :math:`(d^2 - 1) / 2` :math:`Z`-type. The :code:`patch` type assigns **one
  ancilla per stabilizer** measurement, so the ancilla count matches the
  stabilizer count here; other hardware layouts could fold or share ancillas
  differently.

**Stabilizer Generators** (example: ``distance`` :math:`= 3`)

Data qubits are indexed in row-major order (left to right, top to bottom); the
leftmost character of each Pauli string is qubit ``0``, matching the rest of this
document. For :math:`d = 3` there are nine data qubits:

::

   d0  d1  d2
   d3  d4  d5
   d6  d7  d8

* X-type (weight 2 on the left and right boundaries, weight 4 in the bulk):

  * ``XIIXIIIII``
  * ``IXXIXXIII``
  * ``IIIXXIXXI``
  * ``IIIIIXIIX``

* Z-type (weight 2 on the top and bottom boundaries, weight 4 in the bulk):

  * ``IZZIIIIII``
  * ``ZZIZZIIII``
  * ``IIIIZZIZZ``
  * ``IIIIIIZZI``

These Pauli words are exactly those used internally for :math:`d=3`;
:code:`get_stabilizers()` returns the same generators in a canonical sorted order
(rather than grouped as X-type then Z-type).

For other distances, stabilizer supports are generated from the same rotated
grid; use :ref:`stabilizer_grid <qec_stabilizer_grid_python>` or
:code:`get_stabilizers()` to inspect them.

You must pass ``distance`` when constructing the code; there is no default.

**Orientation**

The surface code accepts an optional ``orientation`` string that selects which
Pauli type occupies the bulk checkerboard and which boundary pair carries the
X- versus Z-type stabilizers. The first character (``X`` or ``Z``) sets the bulk
type; the second character (``H`` or ``V``) sets the boundary placement. Valid
values are ``"XV"``, ``"XH"``, ``"ZV"``, and ``"ZH"`` (aliases ``"O1"``, ``"O2"``,
``"O3"``, and ``"O4"`` respectively; case-insensitive). The default is ``"ZH"``,
which reproduces the layout described above. The logical observables and the CNOT
extraction schedule are orientation-aware, so changing the orientation changes the
returned stabilizers, observables, and measurement schedule consistently.

The :ref:`stabilizer_grid <qec_stabilizer_grid_python>` helper documents how
stabilizers and data qubits are indexed on the grid and provides helpers to
print the layout. **Python:** :ref:`cudaq_qec.stabilizer_grid <qec_stabilizer_grid_python>` — **C++:**
:cpp:class:`cudaq::qec::surface_code::stabilizer_grid` — see
:ref:`API <qec_stabilizer_grid_cpp>`. The header :file:`cudaq/qec/codes/surface_code.h`
contains the full declaration.

**Stabilizer measurement schedule**

The surface code's :code:`stabilizer_round` kernel executes one depth-4
extraction round: the X- and Z-check CNOTs are interleaved over four shared
timesteps. Within each plaquette the CNOT order follows the standard zigzag
schedule for the rotated surface code (`Tomita & Svore
<https://arxiv.org/abs/1404.3747>`__): the X and Z plaquettes traverse their
corners in transposed orders, selected per orientation so that mid-round
ancilla faults ("hook errors", `Dennis et al.
<https://arxiv.org/abs/quant-ph/0110143>`__) propagate onto data-qubit pairs
perpendicular to the same-type logical operator. This preserves the full code
distance :math:`d` under circuit-level noise; a naive schedule (both plaquette
types in ascending qubit-index order) halves the effective distance of one
memory basis. The schedule is available from the :code:`stabilizer_grid`
helper via
:cpp:func:`~cudaq::qec::surface_code::stabilizer_grid::get_cnot_schedule_x` /
:code:`get_cnot_schedule_z` (matrix form) and
:code:`get_cnot_schedule_pairs_x` / :code:`get_cnot_schedule_pairs_z` (flat
pair-list form).

Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Rotated surface code; distance is required
        code = qec.get_code('surface_code', distance=3)  # default orientation "ZH"

        # Optionally select an orientation (one of "XV", "XH", "ZV", "ZH")
        code_xh = qec.get_code('surface_code', distance=3, orientation='XH')

        stabilizers = code.get_stabilizers()
        parity = code.get_parity()

.. tab:: C++

    .. code-block:: cpp

        auto code = cudaq::qec::get_code(
            "surface_code", cudaqx::heterogeneous_map{{"distance", 3}});

        // Optionally select an orientation (one of "XV", "XH", "ZV", "ZH")
        auto code_xh = cudaq::qec::get_code(
            "surface_code", cudaqx::heterogeneous_map{
                                {"distance", 3},
                                {"orientation", std::string("XH")}});

        auto stabilizers = code->get_stabilizers();
        auto parity = code->get_parity();



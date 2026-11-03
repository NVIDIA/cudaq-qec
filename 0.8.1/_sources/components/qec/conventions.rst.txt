Conventions
===========

The pre-built ``cudaq-qec`` codes and decoders follow a common set of conventions for how errors, syndromes, and logical observables are laid out. This page documents them; the decoders, examples, and API reference all build on these conventions.

To address vectors of qubits (`cudaq::qvector`), CUDA-Q indexing starts from 0, and 0 corresponds
to the leftmost position when working with Pauli strings (`cudaq::spin_op`). For example, applying a Pauli X operator
to qubit 1 out of 7 would be `X_1 = IXIIIII`.

While implementing your own codes and decoders, you are free to follow any convention that is convenient to you. However,
to interact with the pre-built QEC codes and decoders within this library, the following conventions are used. All of these codes
are CSS codes, and so we separate :math:`X`-type and :math:`Z`-type errors. For example, an error vector for 3 qubits will
have 6 entries, 3 bits representing the presence of a bit-flip on each qubit, and 3 bits representing a phase-flip on each qubit.
An error vector representing a bit-flip on qubit 0, and a phase-flip on qubit 1 would look like `E = 100010`. This means that this
error vector is just two error vectors (`E_X, E_Z`) concatenated together (`E = E_X | E_Z`).

These errors are detected by stabilizers. :math:`Z`-stabilizers detect :math:`X`-type errors and vice versa. Thus we write our
CSS parity check matrices as

.. math::
  H_{CSS} = \begin{pmatrix}
   H_Z & 0 \\
   0 & H_X
   \end{pmatrix},

so that when we generate a syndrome vector by multiplying the parity check matrix by an error vector we get

.. math::
   \begin{align}
  S &= H \cdot E\\
  S_X &= H_Z \cdot E_x\\
  S_Z &= H_X \cdot E_Z.
  \end{align}

This means that for the concatenated syndrome vector `S = S_X | S_Z`, the first part, `S_X`, are syndrome bits triggered by `Z`
stabilizers detecting `X` errors. This is because the `Z` stabilizers like `ZZI` and `IZZ` anti-commute with `X` errors like
`IXI`.

The decoder prediction as to what error happened is `D = D_X | D_Z`. A successful error decoding does not require that `D = E`,
but that `D + E` is not a logical operator. There are a couple ways to check this.
For bitflip errors, we check that the residual error `R = D_X + E_X` is not `L_X`. Since `X` anticommutes
with `Z`, we can check that `L_Z(D_X + E_X) = 0`. This is because we just need to check if they have mutual support on an even
or odd number of qubits. We could also check that `R` is not a stabilizer.

Similar to the parity check matrix, the logical observables are also stored in a matrix as

.. math::
  L = \begin{pmatrix}
   L_Z & 0 \\
   0 & L_X
   \end{pmatrix},

so that when determining logical errors, we can do matrix multiplication

.. math::
   \begin{align}
  P &= L \cdot R\\
  P_X &= L_Z \cdot R_x\\
  P_Z &= L_X \cdot R_Z.
  \end{align}

Here we're using `P` as this can be stored in a Pauli frame tracker to track observable flips.

Each logical qubit has logical observables associated with it. Depending on what basis the data qubits are measured in, either the
`X` or `Z` logical observables can be measured. The data qubits which support the logical observables are contained in the `qec::code` class as well.

To do a logical `Z(X)` measurement, measure out all of the data qubits in the `Z(X)` basis. Then check support on the appropriate
`Z(x)` observable.

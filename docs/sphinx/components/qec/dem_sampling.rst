.. _dem_sampling:

DEM Sampling
============

The ``dem_sampling`` function samples errors and syndromes from a detector
error model (DEM), which is useful for generating synthetic syndrome data to
exercise a decoder. Given a binary check matrix :math:`H` of shape
``[num_checks x num_error_mechanisms]`` and a vector of per-mechanism Bernoulli
probabilities, it generates random error vectors and computes
:math:`\text{syndromes} = \text{errors} \cdot H^T \pmod{2}`.

In Python, the ``backend`` parameter (``"auto"``, ``"gpu"``, or ``"cpu"``)
controls whether sampling runs on the GPU via cuStabilizer or on the CPU.
The function accepts NumPy arrays and PyTorch CUDA tensors. In C++ the CPU
and GPU paths live in separate namespaces (``cudaq::qec::dem_sampler::cpu``
and ``cudaq::qec::dem_sampler::gpu``).

For a complete, runnable walkthrough — including GPU acceleration and
input-type handling — see the :ref:`DEM Sampling Example <dem_sampling_example>`.

# Welcome to the CUDA-QX repository

This repository contains a set of libraries that build on
NVIDIA CUDA-Q. These libraries enable the rapid development of hybrid quantum-classical
application code leveraging state-of-the-art CPUs, GPUs, and QPUs.

## Getting Started

To learn more about how to work with the CUDA-QX libraries, please take a look at the
[CUDA-QX Documentation][cudaqx_docs]. The page contains detailed
[installation instructions][official_install] for officially released packages.

[cudaqx_docs]: https://nvidia.github.io/cudaqx
[official_install]: https://nvidia.github.io/cudaqx/quickstart/installation.html

## Looking for CUDA-Q Solvers?

The CUDA-Q Solvers library has been removed from this repository; version 0.6.0
was its final planned release. Development continues in
[CUDA-Q Algorithms][cudaq_algorithms_github], which supersedes CUDA-Q Solvers
and expands on it:

```bash
pip install cudaq-algorithms
```

See the [CUDA-Q Algorithms documentation][cudaq_algorithms_docs] for tutorials
and examples, [cudaq-algorithms on PyPI][cudaq_algorithms_pypi] for the package,
and [NVIDIA/cudaq-algorithms on GitHub][cudaq_algorithms_github] for the source
code. The CUDA-Q QEC library continues to be developed in this repository.

[cudaq_algorithms_docs]: https://nvidia.github.io/cudaq-algorithms/
[cudaq_algorithms_pypi]: https://pypi.org/project/cudaq-algorithms/
[cudaq_algorithms_github]: https://github.com/NVIDIA/cudaq-algorithms

## Contributing

There are many ways in which you can get involved with CUDA-QX. If you are
interested in developing quantum applications with the CUDA-QX libraries,
this repository is a great place to get started! For more information about
contributing to the CUDA-QX platform, please take a look at
[Contributing.md](./Contributing.md).

## License

The code in this repository is licensed under [Apache License 2.0](./LICENSE).

When distributed via PyPI, GHCR, or NGC, the binaries generated from this source
code are also distributed under the Apache License 2.0; however, the
`libcudaq-qec-nv-qldpc-decoder.so` library is closed source and is subject to
the [NVIDIA Software License Agreement][github_qec_license]

[github_qec_license]: https://github.com/NVIDIA/cudaqx/blob/main/libs/qec/LICENSE

**NOTICE AND DISCLAIMER:** This software automatically retrieves, accesses or
interacts with external materials. Those retrieved materials are not
distributed with this software and are governed solely by separate terms,
conditions and licenses. You are solely responsible for finding, reviewing
and complying with all applicable terms, conditions, and licenses, and for
verifying the security, integrity and suitability of any retrieved materials
for your specific use case. This software is provided "AS IS", without
warranty of any kind. The author makes no representations or warranties
regarding any retrieved materials, and assumes no liability for any losses,
damages, liabilities or legal consequences from your use or inability to use
this software or any retrieved materials. Use this software and the
retrieved materials at your own risk.

Contributions to this repository are accepted under the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/).
Every commit in a pull request must be signed off (`git commit -s`) to certify
that you have the right to submit it under the Apache License 2.0. See
[Contributing.md](./Contributing.md#commit-sign-off) for details.

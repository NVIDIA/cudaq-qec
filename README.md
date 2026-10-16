# Welcome to the CUDA-Q QEC repository

This repository contains the CUDA-Q QEC library, which builds on
NVIDIA CUDA-Q to enable the rapid development of quantum error correction
applications leveraging state-of-the-art CPUs, GPUs, and QPUs.

## Getting Started

To learn more about how to work with the CUDA-Q QEC library, please take a look at the
[CUDA-Q QEC Documentation][cudaq_qec_docs]. The page contains detailed
[installation instructions][official_install] for officially released packages.

[cudaq_qec_docs]: https://nvidia.github.io/cudaq-qec
[official_install]: https://nvidia.github.io/cudaq-qec/quickstart/installation.html

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

There are many ways in which you can get involved with CUDA-Q QEC. If you are
interested in developing quantum applications with the CUDA-Q QEC library,
this repository is a great place to get started! For more information about
contributing to CUDA-Q QEC, please take a look at
[Contributing.md](./Contributing.md).

## License

The code in this repository is licensed under [Apache License 2.0](./LICENSE).

When distributed via PyPI, GHCR, or NGC, the binaries generated from this source
code are also distributed under the Apache License 2.0; however, the
`libcudaq-qec-nv-qldpc-decoder.so` library is closed source and is subject to
the [NVIDIA Software License Agreement][github_qec_license]

[github_qec_license]: https://github.com/NVIDIA/cudaq-qec/blob/main/libs/qec/LICENSE

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

Contributing a pull request to this repository requires accepting the
Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. A CLA-bot will
automatically determine whether you need to provide a CLA and decorate the PR
appropriately. Simply follow the instructions provided by the bot. You will only
need to do this once.

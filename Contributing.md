# Contributing

Thank you for your interest in contributing to CUDA-Q QEC! Based on the type of
contribution, it will fall into three categories:

1. Report a bug, feature request, or documentation issue:

    File an [issue][cudaq_qec_issues] describing what you encountered or what
    you want to see changed. The NVIDIA team will evaluate the issues and triage
    them, scheduling them for a release. If you believe the issue needs priority
    attention comment on the issue to notify the team.

1. Share your work built upon CUDA-Q QEC:

    We would love to hear more about your work! Please share with us on
    [NVIDIA/cudaq-qec GitHub
    Discussions](https://github.com/NVIDIA/cudaq-qec/discussions) or consider
    contributing to our [examples](./docs/sphinx/examples/)! We also take any
    CUDA-Q QEC related questions on this forum.

1. Implement a feature or bug-fix:

    Please file an [issue][cudaq_qec_issues] on the repository and express
    your interest in contributing to its implementation. Someone from the CUDA-Q QEC
    team will respond on the issue to discuss how to best proceed with the
    suggestion. 

When you contribute code to this repository, whether be it an example, bug fix,
or feature, make sure that you can contribute your work under the used
[open-source license](./LICENSE), that is make sure no license and/or patent
conflict is introduced by your pull-request. To confirm this, you will need to
[sign off on your commits](#commit-sign-off) as described below. Thanks in advance
for your patience as we review your contributions; we do appreciate them!

[cudaq_qec_issues]: https://github.com/NVIDIA/cudaq-qec/issues

## Pull Request Guidelines

These guidelines apply to every PR. A condensed checklist is also rendered into
each new PR via [`.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md);
this section is the authoritative source.

### Before requesting review

- **Self-review first.** Before asking others to review your code, read through
  the full PR diff yourself in GitHub or your editor.
- **Use Draft PRs.** Keep a PR in "Draft" state to signal it is not ready for
  review while still exercising CI.
- **Back out unrelated changes** that were just temporary scaffolding or are no
  longer needed.
- **Review logs from local runs.** Make sure there are no unexplained warnings
  or error messages.
- **Review logs from CI runs.** Same standard as local runs.
- **Run full CI as early as possible** to limit accumulation of latent bugs
  during development.

### PR scope and size

- **Keep PRs under ~1000 lines** of diff when possible. Exceptions can be made,
  but they should be the exception, not the rule.
- **Break large changes into easy-to-review PRs.**
- **Refactoring goes in its own PR.** Changes that move a lot of code but
  require no test changes should not be mixed with behavior changes.
- **Refactors should not modify or disable existing tests.** If they need to,
  raise it as an issue and discuss before proceeding.

### Tests

- **New functionality must add new tests.**
- **Tests must actually fail when the functionality is broken.** Ask yourself:
  if someone else changes this code and breaks it, will my tests catch it
  (including program crashes)?
- **Add negative tests** where appropriate, to confirm that exceptions are
  raised and caught under the correct conditions.
- **Use truth data** when algorithmic correctness cannot be checked with simple
  `EXPECT_*` / `assert` statements.
- **CI runtime.** If new tests significantly increase CI time, notify the team
  and/or ask for suggestions.

### Documentation

- **Public-facing APIs *must* have docs**, written as Doxygen comments so they
  flow into the generated reference.
- **Doxygen comments are encouraged** for non-public functions but not
  required.
- If public docs were not created for a user-visible change, **report and track
  that they still need to be written.**

### Code style

- **Follow the existing naming conventions** in the area you are modifying
  (`snake_case` vs `camelCase` depends on the surrounding code).

### Performance

- When appropriate, **add timing or benchmarking scripts** to help measure
  performance.
- **Put runtime performance numbers in the PR description** when the change
  affects performance.

### Dependencies

- **Raise new dependencies to the team** for awareness and potential
  objections, and **file the required OSRB tickets** before merging.

## Commit Sign-off

We require that all contributors "sign-off" on their commits. This certifies
that the contribution is your original work, or you have rights to submit it
under the same license, or a compatible license. Any contribution which contains
commits that are not signed off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```txt
Signed-off-by: Your Name <your@email.com>
```

By signing off on your commits you attest to the [Developer Certificate of Origin
(DCO)](https://developercertificate.org/). Full text of the DCO:

```txt
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
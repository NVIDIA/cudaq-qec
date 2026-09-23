<!--
Thanks for contributing to CUDA-Q Libraries!

Please read the full Pull Request Guidelines in Contributing.md:
https://github.com/NVIDIA/cudaq-qec/blob/main/Contributing.md#pull-request-guidelines
-->

## Description

<!-- What does this PR change, and why? Link related issues. -->

## Runtime / performance impact

<!--
If this change affects performance, paste benchmark numbers here.
Otherwise write "N/A".
-->

## Self-review checklist

Please confirm each item before requesting review. Check `[x]` or strike
through and explain.

### Before requesting review
- [ ] I reviewed my own full diff in GitHub or my editor.
- [ ] PR is in Draft if it is not yet ready for review.
- [ ] Temporary / debugging changes have been removed.
- [ ] Local test logs reviewed; no unexplained warnings or errors.
- [ ] CI logs reviewed; no unexplained warnings or errors.
- [ ] Full CI has been run.

### Scope and size
- [ ] PR is under ~1000 lines, or an exception is justified in the description.
- [ ] Refactoring-only changes are isolated in their own PR(s).
- [ ] No existing tests were disabled or modified just to make this PR pass
      (if so, an issue has been raised).

### Tests
- [ ] New functionality has new tests.
- [ ] Tests fail if the new functionality is broken (including crashes), not
      just when it is missing.
- [ ] Negative tests added where exceptions are expected.
- [ ] Truth data added where simple `EXPECT_*` / `assert` checks are
      insufficient for algorithmic correctness.
- [ ] CI runtime impact considered; team notified if significant.

### Documentation
- [ ] Public-facing APIs have Doxygen docs.
- [ ] User-visible behavior changes have public docs, or a follow-up is
      tracked.

### Code style
- [ ] Naming follows the existing convention (`snake_case` vs `camelCase`) for
      the area being modified.

### Dependencies
- [ ] No new third-party dependencies, **or** the team has been notified and
      OSRB tickets filed.

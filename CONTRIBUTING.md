# Contributing to ENPIRE

Thank you for your interest in contributing to ENPIRE.

**This project will only accept contributions under Apache-2.0.**

---

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/):

```
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

### How to sign off

Add a `Signed-off-by` trailer to every commit message using your real name
and email address:

```
git commit -s -m "Brief description of the change"
```

This appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Commits without a valid `Signed-off-by` line will not be accepted.

---

## Contribution workflow

1. Fork the repository and create a feature branch from `main`.
2. Follow the code style enforced by `ruff` (`uv run ruff check enpire tests/enpire`).
3. Add or update tests for any changed behavior (`uv run pytest -q tests/enpire`).
4. Consult `enpire/env/docs/source_provenance.yaml` before moving code
   migrated from upstream Forge branches.
5. Add a characterization test before refactoring existing behavior.
6. Never commit credentials, device serials, private paths, calibration
   results, or licensed model assets — see `SECURITY.md`.
7. Open a pull request against `main` with a clear description of the change
   and why it is needed.

---

## Code of Conduct

Contributors are expected to act professionally and respectfully.  Harassment
or discrimination of any kind will not be tolerated.

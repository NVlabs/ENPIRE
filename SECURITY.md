# Reporting Security Vulnerabilities

NVIDIA is committed to the security and trust of our software products and
services, including all source code repositories managed through our
open-source GitHub organizations.

**If you discover a security vulnerability in this repository, please report
it privately to the NVIDIA Security Team.**

Do **not** create a public GitHub issue for security vulnerabilities.

To report a vulnerability, please visit:
**https://www.nvidia.com/en-us/security/**

NVIDIA will acknowledge receipt of your report, investigate the issue, and
work with you on a coordinated disclosure timeline.

---

## Credential and hardware safety

In addition to software vulnerabilities, the following operational security
rules apply to all contributors and operators of ENPIRE:

- Never commit API keys, access tokens, device serial numbers, private
  endpoints, calibration results, datasets, checkpoints, or
  workstation-specific paths to Git.
- Credentials must be injected through the process environment or an external
  secret manager. ENPIRE artifacts redact values stored under credential-like
  keys, but redaction is a final safeguard — never rely on it as permission to
  pass secrets through configuration files or command-line arguments.
- If a credential is accidentally pasted into a prompt, terminal, issue, or
  log, revoke and rotate it immediately. Removing the text from the latest
  commit is not sufficient because the value may remain in history, caches, or
  external systems.
- Real-robot commands require explicit operator intent, a verified station and
  calibration profile, and a physical emergency stop within reach.

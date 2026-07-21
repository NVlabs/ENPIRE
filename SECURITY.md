# Security

Do not commit credentials, private endpoints, device serial numbers, calibration
results, datasets, checkpoints, or workstation-specific paths.

Credentials must be injected through the process environment or an external
secret manager. ENPIRE artifacts redact values stored under credential-like
keys, but redaction is a final safeguard, not permission to pass secrets through
configuration files or command-line arguments.

If a credential is pasted into a prompt, terminal, issue, or log, revoke and
rotate it. Removing the text from the latest commit is not sufficient because
the value may remain in history, caches, or external systems.

Real-robot commands require explicit operator intent, a verified station and
calibration profile, and an available physical emergency stop.

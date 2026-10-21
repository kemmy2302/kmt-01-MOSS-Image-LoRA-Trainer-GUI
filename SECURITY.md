# Security Policy

## Intended environment

This application is intended for one user on localhost or a trusted private LAN. It has no authentication, TLS, user separation, or internet-facing hardening.

- Keep `listen_host` at `127.0.0.1` unless LAN access is required.
- Do not expose the configured port to the internet.
- Do not run it on an untrusted shared network.
- Only select datasets and model files you trust.

## Reporting a vulnerability

Please report security issues privately to the repository owner when possible. Do not include model files, training images, access tokens, full local paths, or unredacted logs in a public issue.

## Sensitive local files

`config.json`, `runtime/`, `configs/`, `.venv/`, datasets, and model files must not be committed. Review logs for usernames, paths, filenames, and other private data before sharing them.

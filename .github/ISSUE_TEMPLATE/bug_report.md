---
name: Bug report
about: Something is broken or behaves unexpectedly
title: "bug: "
labels: ["bug", "needs-triage"]
---

## What happened

A clear, concise description of the bug.

## How to reproduce

Minimum steps to reproduce. Ideally a self-contained snippet:

```rust
// or python / shell — whichever is shortest
```

## What you expected

What you thought would happen.

## What actually happened

What did happen (logs, error messages, panic backtrace).

## Environment

- NamiDB version (`cargo pkgid namidb` or `python -c "import namidb; print(namidb.__version__)"`):
- OS + arch:
- Rust toolchain (`rustc --version`):
- Python version (if relevant):
- Object store (`memory://`, `s3://...`, LocalStack, R2, etc.):

## Additional context

Anything else worth knowing — links to RFCs, related issues, related crates.

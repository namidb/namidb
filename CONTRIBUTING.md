# Contributing to NamiDB

We develop in the open. This document explains how to engage.

## TL;DR

1. **Read the RFCs** — [`docs/rfc/`](./docs/rfc/). They are the canonical
   source for design decisions on the storage engine, query engine and
   surrounding subsystems.
2. **Open issues to discuss** before sending large PRs.
3. **Small PRs are welcome any time** — typo fixes, docs improvements,
   perf tweaks, test additions.

## Workflow

- `main` is the development branch. Releases are tagged.
- Every PR runs `cargo fmt --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test --workspace`.
- Commits should be signed (`git commit -S`).
- PR titles follow Conventional Commits: `feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `bench:`, `chore:`.

## RFC process

For anything bigger than a bug fix or a few-line refactor, write an
RFC:

1. Copy `docs/rfc/_template.md` to `docs/rfc/NNN-short-name.md` (next
   free `NNN`).
2. Open a PR with **only the RFC**, in `Draft` state.
3. Solicit feedback. Iterate.
4. Maintainers mark `accepted` or `rejected`. Implementation PRs
   reference the RFC number.

## Coding standards

- **Rust edition 2021**, MSRV 1.85 (kept current).
- `unsafe` only in hot paths with documented invariants and `// SAFETY:`
  comments.
- All public APIs documented (`cargo doc --workspace --no-deps` must
  succeed).
- Errors via `thiserror`; avoid `anyhow` in library crates (OK in
  binaries / tests).
- Tracing instrumentation on `pub` async functions
  (`#[tracing::instrument]`).
- Tests live next to code (`#[cfg(test)] mod tests`) for unit;
  integration tests in the crate's `tests/` directory.

## Testing

- Property tests with `proptest` for invariants.
- Loom for concurrency-critical code paths where appropriate.
- Local integration with an S3-compatible endpoint via
  `docker compose -f tests/docker-compose.s3.yml up` (LocalStack).
- Benchmarks with `criterion`; results expected to be reproducible.

## Code of Conduct

Be kind, be specific, assume good faith. Personal attacks are not
tolerated.

## Licensing

NamiDB is licensed under the Business Source License 1.1 (BSL 1.1),
with automatic conversion to Apache License 2.0 three years after each
release. A separate commercial license is available for teams that want
to embed or redistribute NamiDB outside the bounds of BSL.

By contributing you agree to license your contribution under the same
BSL 1.1 the rest of the project uses, and to dual-license it under the
commercial license offered by the Licensor. A formal CLA will be
requested for code contributions before they land in `main`.

For commercial / proprietary embedding contact
[`info@namidb.com`](mailto:info@namidb.com).

## Communication

- Email: [`hello@namidb.com`](mailto:hello@namidb.com)
- Security: [`security@namidb.com`](mailto:security@namidb.com)

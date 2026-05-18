## Summary

What does this change and why? One or two sentences.

## Type

- [ ] Bug fix (`fix:`)
- [ ] New feature or capability (`feat:`)
- [ ] Refactor with no behaviour change (`refactor:`)
- [ ] Performance improvement (`perf:`)
- [ ] Documentation (`docs:`)
- [ ] Test only (`test:`)
- [ ] Build / CI / chore (`chore:` / `ci:` / `build:`)
- [ ] Bench / benchmark (`bench:`)

## RFC reference

If this implements an RFC, link it. If this is a non-trivial design
change without an RFC yet, open one before sending the implementation.

- RFC: (none)

## Test plan

How did you verify this works? What edge cases did you cover?

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace --exclude namidb-py`
- [ ] (if Python bindings touched) `maturin develop --release && pytest crates/namidb-py/tests`
- [ ] (if storage layer touched) integration tests against LocalStack
- [ ] (if perf-sensitive) bench harness numbers attached

## Breaking changes

Does this break any public surface (Cypher syntax, Rust API, Python
API, CLI flags, on-disk format, env vars)? If yes, describe.

## Checklist

- [ ] I have read [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- [ ] I have added tests covering the change
- [ ] I have updated relevant documentation (RFCs, README, doc comments)
- [ ] I have signed my commits (`git commit -S`)

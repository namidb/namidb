# Security Policy

We take the security of NamiDB seriously. If you believe you have found
a vulnerability — particularly one that affects data integrity,
durability, or coordination guarantees on object storage — please
report it privately so we can fix it before public disclosure.

## Reporting a vulnerability

**Please do not open a public GitHub issue.**

Email reports to **[`security@namidb.com`](mailto:security@namidb.com)**
with as much of the following as you can provide:

- A description of the vulnerability and its impact.
- A minimal reproduction — code, query, or workflow — that exhibits
  the issue.
- The version of NamiDB you tested against (`cargo pkgid namidb` or
  the wheel version).
- The object-store backend (`memory://`, `s3://`, R2, GCS, Azure,
  LocalStack, MinIO, Tigris) and the configuration that is relevant.
- Whether you intend to disclose the issue publicly, and the timeline
  you have in mind.

You can alternatively use GitHub's **[private security advisory](https://github.com/namidb/namidb/security/advisories/new)**
flow, which lets us collaborate on a fix in a private fork before
publishing.

## What to expect

- **Within 3 business days**: an acknowledgement that we received the
  report and have a maintainer assigned.
- **Within 14 days**: a triage update — whether the report is
  confirmed, the affected versions, and an initial remediation plan.
- **Within 90 days** (or sooner): a fix is shipped and a coordinated
  public disclosure is published. We may extend this window in
  agreement with the reporter for unusually complex issues; we will
  not let it extend silently.

We do not currently run a paid bug bounty, but we are happy to credit
reporters in the advisory and in the release notes if they wish.

## Supported versions

NamiDB's public API has been stable since 1.0. Security fixes target
`main` and the most recent published version. Older versions are not
supported.

| Version                                                       | Supported          |
|---------------------------------------------------------------|--------------------|
| `main` (HEAD)                                                 | :white_check_mark: |
| Most recent tagged release                                    | :white_check_mark: |
| Older tagged releases                                         | :x:                |

## Scope

In scope:

- The NamiDB engine crates under [`crates/`](./crates/), the CLI, and
  the Python bindings.
- The integration helpers under [`tests/`](./tests/) when used as
  documented (LocalStack docker-compose, R2 wrapper).
- The CI configuration under [`.github/workflows/`](./.github/workflows/).

Out of scope:

- Third-party dependencies. Please report those upstream and let us
  know so we can plan a version bump.
- Vulnerabilities that require an attacker to already control the
  object-store credentials, signing keys, or the writer process.
- Social engineering of project maintainers.

## Coordinated disclosure

We follow standard responsible-disclosure practice. Public disclosure
happens through a [GitHub Security
Advisory](https://github.com/namidb/namidb/security/advisories) once a
fix has been merged into `main` and is available in a published
release. We may also publish a blog post on
[`namidb.com`](https://namidb.com) for issues that materially affect
deployment guidance.

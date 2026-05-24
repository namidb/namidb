# RFC 022: Bolt protocol — wire compatibility with Neo4j drivers

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-05-24
**Updated:** 2026-05-24
**Implements:** (pending)
**Supersedes:** none

## Summary

Add a Bolt v4.4 + v5.0 + v5.4 wire-protocol listener to `namidb-server`
so every published Neo4j driver (Python, Java, JavaScript, .NET, Go,
Rust) can connect with `bolt://host:7687` without any code change. The
implementation lives in a new `namidb-bolt` crate that owns the
PackStream codec, the handshake, the message set and the per-connection
state machine. The server crate adds a second TCP listener that hands
each accepted socket to a Bolt session that drives the same
`WriterSession` as the REST handler does today.

Bolt is the cheapest unlock the project has: zero driver work, zero
client SDK to maintain, and direct compatibility with every tool that
already speaks Neo4j (cypher-shell, Spring Data Neo4j, LangChain,
LlamaIndex, neo4j-browser, Apache Hop, Memgraph Lab, the JDBC bridge).

## Motivation

Today the server exposes one route, `POST /v0/cypher`, that takes JSON
in and emits JSON out. Every client needs to wrap that endpoint by
hand. The Python binding does it (`crates/namidb-py`), the CLI does it
(`crates/namidb-cli`), but anybody coming from Neo4j has to throw their
existing driver away.

Bolt was designed for exactly this problem. It is a small binary
protocol (PackStream values + chunked framing + a 9-message vocabulary
in v5.4) that the Neo4j drivers have spoken since 2016. The protocol
itself is permissively documented and stable: the
[Bolt specification](https://neo4j.com/docs/bolt/current/) covers every
byte on the wire. Memgraph adopted it for the same reason in 2020;
Apache AGE and KuzuDB never did, which is part of why neither of them
ever shipped a credible Neo4j-replacement story.

Adding Bolt to NamiDB does three things:

1. **Lets every Neo4j tool work.** `pip install neo4j; driver =
   GraphDatabase.driver("bolt://localhost:7687")` is the entire
   onboarding for a Python user, no `pip install namidb` required.
2. **Closes the JDBC and BI gap.** The Neo4j JDBC driver and Tableau /
   Metabase / Grafana connectors all sit on Bolt; they get NamiDB
   support for free.
3. **Establishes the protocol surface for Cloud.** When the Cloud
   beta opens up Bolt is what the routing tier needs to speak, so
   building it now avoids a second protocol migration later.

## Design

### Scope (v0)

Supported Bolt versions:

- **Bolt 4.4** — last v4 protocol, widely deployed. Drivers 4.x and
  5.x both negotiate down to it.
- **Bolt 5.0** — split HELLO/LOGON, new structure tags for nodes /
  relationships / dates, `id`/`element_id`.
- **Bolt 5.4** — `notifications_minimum_severity`, telemetry. Default
  version a 2026 driver picks.

Out-of-scope for v0:

- Routing tier (`ROUTE` message returns a one-server cluster). Multi-
  server routing lives with Cloud.
- Multidatabase (`db` field in HELLO). One namespace per session, the
  field is accepted and ignored.
- TLS termination inside `namidb-bolt`. Terminate at a reverse proxy
  (caddy / nginx / envoy) for v0; native TLS lands with RFC-024
  alongside JWT auth.
- Bookmarks beyond echo-back. Snapshot isolation is per-session
  today, so the bookmark we emit is `namidb:manifest:<version>` and we
  reject anything we did not emit.

### Layering

```
┌──────────────────────────────────────────────────────────────────┐
│  namidb-server (TCP :7687)                                       │
│    accept → namidb_bolt::Session::run(socket, state)             │
├──────────────────────────────────────────────────────────────────┤
│  namidb-bolt                                                     │
│   ├── codec.rs     PackStream encode/decode                      │
│   ├── handshake.rs magic + version negotiation                   │
│   ├── chunk.rs     2-byte framing, 0x0000 terminator             │
│   ├── message.rs   request/response message types                │
│   ├── state.rs     state machine                                 │
│   └── session.rs   one async task per connection                 │
├──────────────────────────────────────────────────────────────────┤
│  namidb-query (existing)  parse → plan → execute / execute_write │
│  namidb-storage (existing) WriterSession, Snapshot               │
└──────────────────────────────────────────────────────────────────┘
```

`namidb-bolt` has no dependency on `namidb-server`. The server crate
wires the session against its `AppState`, the same state used by the
REST router, so both protocols share one `WriterSession` per process
and inherit the single-writer invariant from RFC-001.

### Handshake

Every Bolt connection starts with a 20-byte handshake:

```
client → server:
  0x60 0x60 B0 17                  (4 bytes — protocol magic)
  <ver1> <ver2> <ver3> <ver4>      (4 × 4 bytes — preferred versions)

server → client:
  <chosen-version>                 (4 bytes — MSB minor.major.0.0 or
                                    0x00000000 if none acceptable)
```

A version is `major.minor` with `major`/`minor` in the two lowest
bytes:

```
0x00 0x00 0x04 0x04  →  Bolt 4.4
0x00 0x00 0x00 0x05  →  Bolt 5.0
0x00 0x00 0x04 0x05  →  Bolt 5.4
```

The server compares the four client offers in order and emits the
first one it knows. If none match it emits `0x00 0x00 0x00 0x00` and
closes the socket.

### Chunked framing

Every Bolt message rides inside one or more chunks. A chunk is:

```
<length: u16 big-endian> <length bytes of body>
```

Length 0 marks end-of-message. So a small message looks like

```
00 0E  B1 01 A1 84 75 73 65 72 84 6E 65 6F 34 6A     ← chunk
00 00                                                ← terminator
```

Chunks let big messages stream without us needing to know the total
length up front. The codec keeps reading chunks until it sees a 0-len
terminator, then hands the concatenated body to the message decoder.

### PackStream

PackStream is a tagged binary format. Every value starts with one or
more marker bytes, then payload. The marker encoding from the spec:

| Type | Marker | Notes |
|---|---|---|
| `Null` | `0xC0` | |
| `Bool` | `0xC2` / `0xC3` | false / true |
| `TinyInt` | `0x00..0x7F` and `0xF0..0xFF` | i8 inlined in marker |
| `Int8/16/32/64` | `0xC8`/`0xC9`/`0xCA`/`0xCB` | big-endian, 2's complement |
| `Float64` | `0xC1` | IEEE 754, big-endian |
| `Bytes8/16/32` | `0xCC`/`0xCD`/`0xCE` | length prefix |
| `String` | `0x80+len` (tiny ≤ 15), `0xD0`/`0xD1`/`0xD2` | UTF-8 |
| `List` | `0x90+len` (tiny), `0xD4`/`0xD5`/`0xD6` | |
| `Map` | `0xA0+len` (tiny), `0xD8`/`0xD9`/`0xDA` | flat key,value pairs |
| `Struct` | `0xB0+fields` (tiny), `0xDC`/`0xDD` | followed by 1-byte tag + N fields |

NamiDB only needs to encode struct tags it cares about:

| Tag | Struct | Used by |
|---|---|---|
| `0x4E` | `Node` | result values |
| `0x52` | `Relationship` | result values |
| `0x72` | `UnboundRelationship` | path segments |
| `0x50` | `Path` | result values |
| `0x44` | `Date` (5.0+) | |
| `0x49` | `LocalDateTime` (5.0+) | |
| `0x46` | `DateTime` (5.0+, zoned) | |
| `0x45` | `Duration` (5.0+) | |
| `0x10..0x14` | Request messages | HELLO, GOODBYE, RESET, RUN, DISCARD |
| `0x3F` | `PULL` | |
| `0x11` | `BEGIN` | |
| `0x12` | `COMMIT` | |
| `0x13` | `ROLLBACK` | |
| `0x6A`/`0x6B` | `LOGON` / `LOGOFF` (5.1+) | |
| `0x66` | `ROUTE` (4.3+) | |
| `0x70` | `SUCCESS` | server response |
| `0x71` | `RECORD` | one result row |
| `0x7E` | `IGNORED` | |
| `0x7F` | `FAILURE` | |
| `0x54` | `TELEMETRY` (5.4+) | |

### Message vocabulary (v5.4 surface)

Client requests:

```
HELLO      {user_agent, …, notifications_minimum_severity?}
LOGON      {scheme, principal?, credentials?}
LOGOFF
GOODBYE
RESET
RUN        cypher, params, {bookmarks?, tx_timeout?, mode?, db?, …}
PULL       {n, qid?}
DISCARD    {n, qid?}
BEGIN      {bookmarks?, tx_timeout?, mode?, db?, …}
COMMIT
ROLLBACK
ROUTE      {routing_context, bookmarks?, {db}?}
TELEMETRY  {api}                              ← v5.4, no-op for us
```

Server responses:

```
SUCCESS   {metadata}    end of stream, includes fields/qid/has_more/etc.
RECORD    [values]      one row, repeated per result
FAILURE   {code, message}
IGNORED                 sent for every request after a FAILURE until RESET
```

### State machine

A single-database, single-tenant Bolt session walks the states below.
We track the state on `Session::state` and reject messages that do not
match the current state with `FAILURE{code: "Neo.ClientError.Request.Invalid", ...}`.

```
                       NEGOTIATION
                            │
                       HELLO + LOGON
                            ▼
                       READY ◄────────────────┐
                  RUN /     \ BEGIN           │
                    ▼        ▼                │
                STREAMING  TX_READY           │
                PULL/        │                │ COMMIT or
                DISCARD      │ RUN            │ ROLLBACK
                    │        ▼                │
                    │     TX_STREAMING        │
                    │     PULL/DISCARD        │
                    │        │                │
                    └────────┴──────► READY ──┘
                            │
                       (any FAILURE)
                            ▼
                         FAILED ──── RESET ──► READY
```

`GOODBYE` is legal in every state and closes the connection cleanly.
`INTERRUPTED` is the transient state between receiving a request and
processing it; the spec models it explicitly because of how RESET
interrupts pipelined messages. We collapse it onto `FAILED` for v0
since we do not pipeline.

### RuntimeValue ↔ Bolt mapping

Our `namidb_query::RuntimeValue` covers exactly the value types the
spec defines. The conversion is total in both directions:

| `RuntimeValue` | Bolt encoding |
|---|---|
| `Null` | `0xC0` |
| `Bool(b)` | `0xC2` / `0xC3` |
| `Integer(i64)` | TinyInt / Int8 / Int16 / Int32 / Int64 |
| `Float(f64)` | Float64 |
| `String(s)` | String (UTF-8) |
| `Bytes(b)` | Bytes |
| `Vector(Vec<f32>)` | List of Float64 |
| `List(items)` | List |
| `Map(m)` | Map |
| `Date(d)` | Struct `0x44 D` |
| `DateTime(micros)` | Struct `0x49 I` (LocalDateTime, UTC) |
| `Node` | Struct `0x4E { id: i64, labels: [string], properties: map, element_id: string }` |
| `Rel` | Struct `0x52 { id, start_id, end_id, type, properties, element_id, start_element_id, end_element_id }` |
| `Path` | Struct `0x50 { nodes: [Node], rels: [UnboundRelationship], sequence: [i64] }` |

NodeId today is a `u128` UUIDv7. Bolt wants `i64` for the legacy `id`
field, so we hash the lower 64 bits (truncate after the variant bits)
into the legacy slot and carry the full UUID as the `element_id`
string. Drivers 5.x prefer `element_id`; 4.x clients see a stable
truncated id that round-trips through the same session.

### Errors

Bolt errors are `FAILURE { code, message }`. The code uses Neo4j's
dotted category scheme; we map our error families:

| NamiDB error | Bolt code |
|---|---|
| Parse error | `Neo.ClientError.Statement.SyntaxError` |
| Planner / lower error | `Neo.ClientError.Statement.SemanticError` |
| Unsupported feature (v0) | `Neo.ClientError.Statement.NotSupported` |
| Runtime eval error | `Neo.ClientError.Statement.ArgumentError` |
| Storage error | `Neo.TransientError.General.DatabaseUnavailable` |
| Auth failure | `Neo.ClientError.Security.Unauthorized` |

`message` carries our existing error text verbatim. Clients can match
the dotted code to bucket retryable vs fatal.

### Server integration

`namidb-server` gains one new flag set:

```
--bolt-listen 0.0.0.0:7687   (default off — opt-in for v0)
--bolt-disabled              (kill switch for the rare HTTP-only deployment)
NAMIDB_BOLT_LISTEN=...       (env var equivalent)
```

When bolt is enabled the server spawns a second `TcpListener` and
hands each accepted socket to a per-connection task:

```rust
tokio::spawn(async move {
    let session = namidb_bolt::Session::new(socket, state.clone());
    if let Err(e) = session.run().await {
        tracing::warn!(error = %e, "bolt session ended");
    }
});
```

The session holds a `Weak<AppState>` and reacquires the writer lock
the same way `cypher` does, so reads and writes serialize through the
existing `tokio::Mutex`. RFC-021 (concurrent reads) lifts that
serialization off the read path; once it lands Bolt reads run
parallel without any change in this crate.

### Cypher value lifetimes

Bolt `RECORD` messages stream a row at a time. Today `execute()`
returns `Vec<Row>` (materialised). The session iterates that vector
and emits a `RECORD` per row, which is correct but not streaming.
RFC-026 (streaming executor, the dependency for `/v0/cypher/stream`)
lets us emit rows lazily; the Bolt code is structured around an
iterator from day one so the swap is a one-line change.

### Auth

`LOGON` carries `{scheme: "basic" | "bearer" | "none", principal?,
credentials?}`. v0 maps:

- `none` — accepted iff `--auth-token` is unset (parity with REST).
- `basic` — `principal` is ignored, `credentials` is checked against
  `--auth-token`. Constant-time comparison, same as `require_auth`.
- `bearer` — `credentials` is the token, same comparison.

Other schemes (`kerberos`, custom) return
`Neo.ClientError.Security.Unauthorized`. JWT lands with RFC-024.

## Alternatives considered

### A. Stay HTTP-only, ship a Python driver wrapper

The Python binding already works; we could ship matching JS / Java /
Go bindings. Each costs months of CI + packaging + maintenance and
gives us no leverage over the existing Neo4j ecosystem (LangChain,
Spring Data, etc. would not pick up a NamiDB driver). Rejected: Bolt
gives every client for the cost of one wire format.

### B. PostgreSQL wire protocol

Materialize, RisingWave and CockroachDB all chose Postgres wire. The
upside is gigantic ecosystem support for `psql` and BI tools. The
downside is exactly the friction we are avoiding with Bolt: every
client is built around tabular SQL semantics and stumbles on graph
shapes (Node / Rel / Path do not have first-class Postgres
equivalents). Rejected for v0; can land later as a second listener
without affecting Bolt.

### C. Apache Arrow Flight SQL

Modern columnar protocol with great DataFrame support (Polars-Flight,
DuckDB, DataFusion). Adds zero-copy result streaming, which is
attractive. Downside: zero overlap with Neo4j-shaped clients —
LangChain and the Neo4j drivers do not speak Arrow Flight. Deferred
to a parallel RFC; it complements Bolt rather than replacing it.

### D. Implement Bolt via an external proxy

A proxy translates `bolt://` to `POST /v0/cypher`. Memgraph briefly
shipped this as `mgconsole-http`. It works but pays a triple
serialization tax (PackStream → JSON → planner → JSON → PackStream)
and never reaches the streaming semantics Bolt clients expect.
Rejected: native implementation costs ~1 month and is the correct
long-term shape anyway.

### E. Hand-roll vs reuse a crate

`bolt-proto` (0.13, 2023) covers PackStream and most messages but is
unmaintained, depends on outdated tokio, and the message vocabulary
covers Bolt 3 only. Forking it costs more than starting clean given
how small the surface is (one codec, one state machine,
~2000 LoC total). Rejected: implement in-tree.

## Drawbacks

1. **Protocol surface area** — Bolt has its own bug class: clients
   that pipeline RUN+PULL aggressively, drivers that send invalid
   structs, version-skew between v4.4 and v5.x. Mitigation: golden
   test vectors from the official Neo4j testkit, plus a real-driver
   integration test in CI.

2. **State machine duplicate** — we now have one Cypher entry point
   over HTTP (request-scoped, no transactions) and one over Bolt
   (session-scoped, transactions). Bugs in transaction semantics will
   surface on Bolt first. Mitigation: every transaction-shaped test
   runs through both protocols.

3. **Two listeners means two attack surfaces** — Bolt accepts
   binary frames pre-auth, so a malformed PackStream value could
   crash the codec before LOGON. Mitigation: `proptest`-style fuzzing
   of the decoder, and a hard limit (1 MiB) on pre-auth message
   size.

4. **Bookmarks lie if the user expects multi-statement causal
   consistency** — we only emit one bookmark per session (the
   manifest version at COMMIT). Cross-session "wait for bookmark X"
   semantics need a writer-side ratchet (RFC-027). Mitigation:
   document the v0 limit and reject foreign bookmarks with a clear
   error.

## Open questions

- **Q1: TELEMETRY (v5.4) ack** — do we record it for our own metrics
  or echo it back as a no-op SUCCESS? Leaning no-op; revisit when
  RFC-023 (metrics) lands and we know what to do with the data.

- **Q2: ROUTE behaviour** — return a single-server cluster
  (`{servers: [{addresses: [self], role: "WRITE"}, {addresses:
  [self], role: "READ"}]}`) or fail with
  `Neo.ClientError.General.ForbiddenOnReadOnlyDatabase`? Leaning
  single-server: that is what a Memgraph single-node returns and what
  drivers expect from an embedded deployment.

- **Q3: Transactional READ mode** — `mode: "r"` in BEGIN /  RUN
  promises the server will refuse writes. Easy to enforce
  (`plan.contains_write()` already exists), open question is whether
  to also refuse `MERGE` (it can write or not depending on data).
  Leaning yes-refuse-merge in read mode — drivers send `mode: "r"`
  explicitly when they want read replicas.

- **Q4: Connection limits** — uncapped today via `tokio::spawn`. A
  rogue client can spawn 100K tasks. Likely
  `--bolt-max-connections=1024` default before we ship the listener
  on by default.

## References

- Bolt protocol specification (bolt 5.4) — https://neo4j.com/docs/bolt/current/
- PackStream specification — https://neo4j.com/docs/bolt/current/packstream/
- Memgraph Bolt implementation —
  https://github.com/memgraph/memgraph/tree/master/src/communication/bolt
- Neo4j testkit — https://github.com/neo4j-drivers/testkit
- `bolt-proto` crate (reference only) — https://crates.io/crates/bolt-proto
- RFC-001 (single-writer invariant)
- RFC-009 (write clauses + transaction semantics)
- RFC-021 (concurrent reads — unlocks Bolt read parallelism)
- RFC-024 (JWT auth — replaces basic over Bolt)

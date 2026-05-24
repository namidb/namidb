# namidb-bolt

Bolt wire protocol implementation for NamiDB.

The crate owns the PackStream codec, the connection handshake, the
message vocabulary and the per-session state machine. It does not own
the TCP listener: `namidb-server` wires the session against a writer
session and an auth token.

Negotiated Bolt versions: **4.4**, **5.0**, **5.4**.

See [RFC-022](../../docs/rfc/022-bolt-protocol.md) for the design.

## Layout

```
src/
├── lib.rs        public surface
├── codec.rs      PackStream encode/decode
├── chunk.rs      framing (2-byte length, 0x0000 terminator)
├── handshake.rs  magic + version negotiation
├── message.rs    request/response message types
├── state.rs      state machine
├── session.rs    per-connection async task
├── value.rs      Bolt Value + Node/Rel/Path structs
└── error.rs      crate-local error enum
```

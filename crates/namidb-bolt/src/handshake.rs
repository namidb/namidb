//! Bolt connection handshake.
//!
//! Every Bolt connection begins with a 20-byte handshake, before any
//! framed messages flow:
//!
//! ```text
//! client → server   60 60 B0 17                          (magic)
//!                   <v1>  <v2>  <v3>  <v4>               (4 × 4 bytes)
//! server → client   <chosen>                             (4 bytes)
//! ```
//!
//! Each version is `[unused, range, minor, major]` big-endian. A
//! non-zero `range` means "any minor from `minor - range` up to
//! `minor`, of the same `major`". Drivers send their preferred
//! versions in descending order; the server picks the first offer it
//! supports. When no offer matches the server replies with four zero
//! bytes and the client is expected to close.
//!
//! v0 supports Bolt 4.4, 5.0 and 5.4 only. Other versions can land by
//! extending [`SUPPORTED_VERSIONS`].

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{BoltError, Result};

/// Magic preamble that every Bolt client sends before its version
/// list.
pub const MAGIC: [u8; 4] = [0x60, 0x60, 0xB0, 0x17];

/// Compact version representation. `Version { major, minor }` is what
/// the rest of the crate keys behaviour off of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
}

impl Version {
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Encode as the 4-byte wire form a server returns: `[0, 0,
    /// minor, major]`.
    pub fn to_wire(self) -> [u8; 4] {
        [0, 0, self.minor, self.major]
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Versions the server is willing to negotiate, highest preference
/// first. Insertion order maps to preference order during matching.
pub const SUPPORTED_VERSIONS: &[Version] =
    &[Version::new(5, 4), Version::new(5, 0), Version::new(4, 4)];

/// One client offer. `range` lets a client say "any minor between
/// `minor - range` and `minor`" of `major`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offer {
    pub range: u8,
    pub minor: u8,
    pub major: u8,
}

impl Offer {
    fn parse(bytes: [u8; 4]) -> Self {
        // bytes[0] is the reserved high byte; spec mandates zero, but
        // we don't reject non-zero. Future versions may carve it up.
        Self {
            range: bytes[1],
            minor: bytes[2],
            major: bytes[3],
        }
    }

    fn matches(&self, version: Version) -> bool {
        if self.major != version.major {
            return false;
        }
        let low = self.minor.saturating_sub(self.range);
        (low..=self.minor).contains(&version.minor)
    }
}

/// Read the 20-byte handshake from `r` and return the negotiated
/// version, or `None` if no offer matches a supported version. In the
/// latter case the caller must still write `[0; 4]` and close the
/// socket — [`write_response`] does that.
pub async fn read_offers<R>(r: &mut R) -> Result<[Offer; 4]>
where
    R: AsyncReadExt + Unpin,
{
    let mut buf = [0u8; 20];
    r.read_exact(&mut buf).await?;
    if buf[..4] != MAGIC {
        return Err(BoltError::Handshake(format!(
            "expected magic 0x6060B017, got 0x{:02X}{:02X}{:02X}{:02X}",
            buf[0], buf[1], buf[2], buf[3]
        )));
    }
    Ok([
        Offer::parse([buf[4], buf[5], buf[6], buf[7]]),
        Offer::parse([buf[8], buf[9], buf[10], buf[11]]),
        Offer::parse([buf[12], buf[13], buf[14], buf[15]]),
        Offer::parse([buf[16], buf[17], buf[18], buf[19]]),
    ])
}

/// Pick the highest-preference version we support that any of the
/// four offers covers.
pub fn negotiate(offers: &[Offer; 4]) -> Option<Version> {
    for &candidate in SUPPORTED_VERSIONS {
        for offer in offers {
            if offer.matches(candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Write the 4-byte handshake reply. `None` means "no acceptable
/// version" and the spec mandates four zero bytes.
pub async fn write_response<W>(w: &mut W, version: Option<Version>) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let bytes = match version {
        Some(v) => v.to_wire(),
        None => [0, 0, 0, 0],
    };
    w.write_all(&bytes).await?;
    w.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    fn handshake_bytes(versions: [[u8; 4]; 4]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        buf.extend_from_slice(&MAGIC);
        for v in versions {
            buf.extend_from_slice(&v);
        }
        buf
    }

    #[tokio::test]
    async fn negotiates_5_4_when_offered_first() {
        let bytes = handshake_bytes([
            [0, 0, 4, 5], // 5.4
            [0, 0, 0, 5], // 5.0
            [0, 0, 4, 4], // 4.4
            [0, 0, 0, 0],
        ]);
        let (mut client, mut server) = duplex(64);
        client.write_all(&bytes).await.unwrap();
        let offers = read_offers(&mut server).await.unwrap();
        let v = negotiate(&offers).unwrap();
        assert_eq!(v, Version::new(5, 4));
    }

    #[tokio::test]
    async fn falls_back_to_4_4_for_old_driver() {
        // An old driver might offer 4.4 only.
        let bytes = handshake_bytes([[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        let (mut client, mut server) = duplex(64);
        client.write_all(&bytes).await.unwrap();
        let offers = read_offers(&mut server).await.unwrap();
        let v = negotiate(&offers).unwrap();
        assert_eq!(v, Version::new(4, 4));
    }

    #[tokio::test]
    async fn no_match_returns_none() {
        // Bolt 3 only, which we don't support.
        let bytes = handshake_bytes([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        let (mut client, mut server) = duplex(64);
        client.write_all(&bytes).await.unwrap();
        let offers = read_offers(&mut server).await.unwrap();
        assert!(negotiate(&offers).is_none());
    }

    #[tokio::test]
    async fn version_range_matches() {
        // Driver says "any minor between 0 and 4 of major 5". We
        // pick 5.4 (highest minor in range).
        let bytes = handshake_bytes([
            [0, 4, 4, 5], // range 4, minor 4, major 5 → 5.0..=5.4
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);
        let (mut client, mut server) = duplex(64);
        client.write_all(&bytes).await.unwrap();
        let offers = read_offers(&mut server).await.unwrap();
        let v = negotiate(&offers).unwrap();
        assert_eq!(v, Version::new(5, 4));
    }

    #[tokio::test]
    async fn bad_magic_rejected() {
        let mut bytes = vec![0x00, 0x00, 0x00, 0x01];
        bytes.extend_from_slice(&[0; 16]);
        let (mut client, mut server) = duplex(64);
        client.write_all(&bytes).await.unwrap();
        let err = read_offers(&mut server).await.unwrap_err();
        assert!(matches!(err, BoltError::Handshake(_)));
    }

    #[tokio::test]
    async fn write_response_emits_4_bytes() {
        let (mut server_side, mut client_side) = duplex(8);
        write_response(&mut server_side, Some(Version::new(5, 4)))
            .await
            .unwrap();
        let mut buf = [0u8; 4];
        client_side.read_exact(&mut buf).await.unwrap();
        assert_eq!(buf, [0, 0, 4, 5]);

        let (mut server_side, mut client_side) = duplex(8);
        write_response(&mut server_side, None).await.unwrap();
        let mut buf = [0u8; 4];
        client_side.read_exact(&mut buf).await.unwrap();
        assert_eq!(buf, [0, 0, 0, 0]);
    }
}

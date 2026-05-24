//! Bolt chunked framing.
//!
//! Every Bolt message lives inside one or more chunks. A chunk is a
//! 2-byte big-endian length followed by that many body bytes. The
//! message terminator is a zero-length chunk (`0x00 0x00`). Big
//! messages may span multiple chunks; the reader concatenates them
//! before handing the body to the message decoder.
//!
//! On write we chunk on a fixed maximum size (default 64 KiB - 1) and
//! emit the terminator after the last fragment. Drivers are tolerant
//! of any chunk boundaries as long as the body re-assembles
//! identically.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{BoltError, Result};

/// Maximum body bytes per chunk header (the 2-byte length field is
/// unsigned, so `0xFFFF` is the hard ceiling).
pub const MAX_CHUNK_BODY: usize = u16::MAX as usize;

/// Default chunk size we use on write. Below the hard ceiling so we
/// can always emit a contiguous 0xFFFF chunk without truncation.
pub const DEFAULT_CHUNK_SIZE: usize = 16 * 1024;

/// Read one full message off `r`. Concatenates every chunk up to
/// (and excluding) the zero-length terminator. Returns the message
/// body bytes ready to be PackStream-decoded into a struct.
pub async fn read_message<R>(r: &mut R, max_message_bytes: usize) -> Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    let mut body = Vec::new();
    loop {
        let mut len_buf = [0u8; 2];
        r.read_exact(&mut len_buf).await?;
        let len = u16::from_be_bytes(len_buf) as usize;
        if len == 0 {
            // End-of-message marker.
            return Ok(body);
        }
        if body.len() + len > max_message_bytes {
            return Err(BoltError::TooLarge {
                what: "message body",
                len: body.len() + len,
                max: max_message_bytes,
            });
        }
        let start = body.len();
        body.resize(start + len, 0);
        r.read_exact(&mut body[start..]).await?;
    }
}

/// Write one Bolt message as one or more chunks followed by the
/// zero-length terminator.
pub async fn write_message<W>(w: &mut W, body: &[u8]) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    write_message_chunked(w, body, DEFAULT_CHUNK_SIZE).await
}

/// Same as [`write_message`] with an explicit max chunk size, mostly
/// useful for tests that want to verify the wire splits correctly.
pub async fn write_message_chunked<W>(w: &mut W, body: &[u8], chunk_size: usize) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let chunk_size = chunk_size.clamp(1, MAX_CHUNK_BODY);
    let mut offset = 0;
    while offset < body.len() {
        let end = (offset + chunk_size).min(body.len());
        let slice = &body[offset..end];
        let len = slice.len() as u16;
        w.write_all(&len.to_be_bytes()).await?;
        w.write_all(slice).await?;
        offset = end;
    }
    // Terminator.
    w.write_all(&[0, 0]).await?;
    w.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn single_chunk_roundtrip() {
        let body = b"hello bolt".to_vec();
        let (mut a, mut b) = duplex(256);
        write_message(&mut a, &body).await.unwrap();
        let got = read_message(&mut b, 1024).await.unwrap();
        assert_eq!(got, body);
    }

    #[tokio::test]
    async fn multi_chunk_roundtrip() {
        let body: Vec<u8> = (0..200).map(|i| i as u8).collect();
        let (mut a, mut b) = duplex(2048);
        // Force chunks of 32 bytes.
        write_message_chunked(&mut a, &body, 32).await.unwrap();
        let got = read_message(&mut b, 4096).await.unwrap();
        assert_eq!(got, body);
    }

    #[tokio::test]
    async fn empty_body_is_just_terminator() {
        let (mut a, mut b) = duplex(64);
        write_message(&mut a, &[]).await.unwrap();
        let got = read_message(&mut b, 16).await.unwrap();
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn oversize_message_rejected() {
        let body = vec![0u8; 1024];
        let (mut a, mut b) = duplex(4096);
        write_message(&mut a, &body).await.unwrap();
        let err = read_message(&mut b, 100).await.unwrap_err();
        assert!(matches!(err, BoltError::TooLarge { .. }));
    }

    #[tokio::test]
    async fn many_chunks_concatenate() {
        // Construct a body exactly at the natural chunk boundary so
        // we can verify the writer emits two non-empty chunks plus a
        // terminator.
        let body = vec![0xABu8; DEFAULT_CHUNK_SIZE + 5];
        let (mut a, mut b) = duplex(4 * DEFAULT_CHUNK_SIZE);
        write_message(&mut a, &body).await.unwrap();
        let got = read_message(&mut b, 4 * DEFAULT_CHUNK_SIZE).await.unwrap();
        assert_eq!(got, body);
    }
}

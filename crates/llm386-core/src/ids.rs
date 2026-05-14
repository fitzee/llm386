//! Identifier and primitive newtypes shared across LLM386.
//!
//! All ids are `Copy` and have a stable on-disk encoding independent
//! of host pointer width.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Time-ordered 128-bit identifier for a context block.
///
/// Layout (high → low bits):
/// - 48 bits: milliseconds since the Unix epoch
/// - 80 bits: cryptographic randomness
///
/// The natural `Ord` impl is therefore chronological, which lets the
/// LMDB `blocks_by_id` table double as a time index.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u128);

impl BlockId {
    /// Build a `BlockId` from a millisecond timestamp and a 128-bit
    /// random value (only the low 80 bits of the random value are
    /// used; only the low 48 bits of the timestamp are used).
    #[must_use]
    pub const fn from_parts(timestamp_ms: u64, random_bits: u128) -> Self {
        let ts_low_48 = (timestamp_ms & ((1u64 << 48) - 1)) as u128;
        let rnd_low_80 = random_bits & ((1u128 << 80) - 1);
        Self((ts_low_48 << 80) | rnd_low_80)
    }

    /// Recover the embedded millisecond timestamp.
    #[must_use]
    pub fn timestamp_ms(self) -> u64 {
        let masked = (self.0 >> 80) & ((1u128 << 48) - 1);
        u64::try_from(masked).expect("48-bit value always fits in u64")
    }
}

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockId({:032x})", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Error returned when a string cannot be parsed as a [`BlockId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBlockIdError(String);

impl fmt::Display for ParseBlockIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid BlockId `{}`: expected 32 hex characters", self.0)
    }
}

impl std::error::Error for ParseBlockIdError {}

impl FromStr for BlockId {
    type Err = ParseBlockIdError;

    /// Parse a `BlockId` from its 32-hex-char [`Display`] form.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 32 {
            return Err(ParseBlockIdError(s.to_string()));
        }
        u128::from_str_radix(s, 16)
            .map(BlockId)
            .map_err(|_| ParseBlockIdError(s.to_string()))
    }
}

/// Identifier for a session — an isolated conversation or agent run.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct SessionId(pub u128);

impl fmt::Debug for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SessionId({:032x})", self.0)
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Identifier for a single page+pack invocation, used by the trace layer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct CallId(pub u128);

impl fmt::Debug for CallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CallId({:032x})", self.0)
    }
}

impl fmt::Display for CallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Blake3 content hash of a block's raw bytes — used for dedup and
/// for caching tokenizer counts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Hash arbitrary bytes with blake3.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash(")?;
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

/// Milliseconds since the Unix epoch.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug, Serialize, Deserialize,
)]
pub struct Timestamp(pub u64);

/// A precomputed token count for a particular tokenizer.
///
/// `u32` is wide enough for any realistic context window (~4 billion
/// tokens) and gives a stable on-disk encoding regardless of host
/// pointer width — `usize` would vary between 32- and 64-bit targets.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug, Serialize, Deserialize,
)]
pub struct TokenCount(pub u32);

impl TokenCount {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_id_roundtrips_timestamp() {
        let id = BlockId::from_parts(1_700_000_000_000, 0xdead_beef_cafe_babe_f00d);
        assert_eq!(id.timestamp_ms(), 1_700_000_000_000);
    }

    #[test]
    fn block_id_truncates_excess_timestamp_bits() {
        let id = BlockId::from_parts(u64::MAX, 0);
        assert_eq!(id.timestamp_ms(), (1u64 << 48) - 1);
    }

    #[test]
    fn block_ids_sort_chronologically() {
        let earlier = BlockId::from_parts(1_000, u128::MAX);
        let later = BlockId::from_parts(2_000, 0);
        assert!(earlier < later);
    }

    #[test]
    fn content_hash_is_deterministic() {
        let a = ContentHash::of(b"hello world");
        let b = ContentHash::of(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn content_hash_differs_for_different_input() {
        let a = ContentHash::of(b"hello world");
        let b = ContentHash::of(b"hello WORLD");
        assert_ne!(a, b);
    }

    #[test]
    fn block_id_roundtrips_through_string() {
        let id = BlockId::from_parts(1_700_000_000_000, 0xdead_beef_cafe_babe_f00d);
        let s = id.to_string();
        assert_eq!(s.len(), 32);
        let parsed: BlockId = s.parse().unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn block_id_parse_rejects_wrong_length() {
        assert!("abc".parse::<BlockId>().is_err());
        assert!("0".repeat(31).parse::<BlockId>().is_err());
        assert!("0".repeat(33).parse::<BlockId>().is_err());
    }

    #[test]
    fn block_id_parse_rejects_non_hex() {
        assert!("g".repeat(32).parse::<BlockId>().is_err());
    }

    #[test]
    fn token_count_arithmetic_saturates() {
        assert_eq!(
            TokenCount(u32::MAX).saturating_add(TokenCount(1)),
            TokenCount(u32::MAX),
        );
        assert_eq!(TokenCount(0).saturating_sub(TokenCount(5)), TokenCount(0));
    }
}

// The canonical envelope every tailer emits and `ingest` reads: source, kind,
// session, timestamp, and a kind-specific JSON payload.
//
// event_id is a ULID whose 80-bit entropy is sha256(key)[..10], so re-parsing a
// source line (e.g. after a lost cursor) mints the same id and downstream dedup
// recognizes the re-emission.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Canonical source identifiers. The full set is declared up front; later
/// tailers use the rest.
#[allow(dead_code)]
pub mod source {
    pub const CLAUDE_CODE: &str = "claude_code";
    pub const CODEX_CLI: &str = "codex_cli";
    pub const COWORK: &str = "cowork";
    pub const GITHUB: &str = "github";
    pub const SYNTY: &str = "synty";
}

/// Canonical payload kinds.
#[allow(dead_code)]
pub mod kind {
    pub const USER_PROMPT: &str = "user_prompt";
    pub const ASSISTANT_MESSAGE: &str = "assistant_message";
    pub const TOOL_CALL: &str = "tool_call";
    pub const TOOL_RESULT: &str = "tool_result";
    pub const THINKING: &str = "thinking";
    pub const ATTACHMENT_REF: &str = "attachment_ref";
    pub const SESSION_START: &str = "session_start";
    pub const SESSION_END: &str = "session_end";
    pub const AGENT_META: &str = "agent_meta";
}

/// The canonical envelope.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Event {
    pub event_id: String,
    pub stream: String,
    pub seq: i64,
    pub ts: String, // RFC3339 (UTC, 'Z')
    pub source: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub session_id: String,
    pub kind: String,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub rollup_dim: String,
}

/// Per-stream monotonically increasing sequence numbers. In-memory: they reset
/// on restart; ordering across restarts is reconciled via event_id, not seq.
#[derive(Default)]
pub struct Sequencer {
    streams: HashMap<String, i64>,
}

impl Sequencer {
    pub fn new() -> Self {
        Self::default()
    }
    /// Next sequence for `stream`; the first call for a stream returns 0.
    pub fn next(&mut self, stream: &str) -> i64 {
        let n = self.streams.entry(stream.to_string()).or_insert(0);
        let cur = *n;
        *n += 1;
        cur
    }
}

const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// A deterministic ULID: 48-bit millisecond timestamp + 80-bit entropy taken
/// from sha256(key)[..10]. The same (ts_ms, key) always yields the same id.
pub fn deterministic_ulid(ts_ms: u64, key: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(key.as_bytes());
    let mut entropy = [0u8; 10];
    entropy.copy_from_slice(&digest[..10]);
    ulid_string(ts_ms, &entropy)
}

/// Encode (48-bit time, 80-bit entropy) into the canonical 26-char Crockford
/// base32 ULID string (the oklog/ulid byte packing).
fn ulid_string(ts_ms: u64, entropy: &[u8; 10]) -> String {
    let mut b = [0u8; 16];
    b[0] = (ts_ms >> 40) as u8;
    b[1] = (ts_ms >> 32) as u8;
    b[2] = (ts_ms >> 24) as u8;
    b[3] = (ts_ms >> 16) as u8;
    b[4] = (ts_ms >> 8) as u8;
    b[5] = ts_ms as u8;
    b[6..16].copy_from_slice(entropy);

    let e = |i: u8| CROCKFORD[(i & 0x1f) as usize] as char;
    let mut s = String::with_capacity(26);
    // Timestamp: 48 bits → 10 chars (top char carries only 2 significant bits).
    s.push(e(b[0] >> 5));
    s.push(e(b[0]));
    s.push(e(b[1] >> 3));
    s.push(e((b[1] << 2) | (b[2] >> 6)));
    s.push(e(b[2] >> 1));
    s.push(e((b[2] << 4) | (b[3] >> 4)));
    s.push(e((b[3] << 1) | (b[4] >> 7)));
    s.push(e(b[4] >> 2));
    s.push(e((b[4] << 3) | (b[5] >> 5)));
    s.push(e(b[5]));
    // Entropy: 80 bits → 16 chars.
    s.push(e(b[6] >> 3));
    s.push(e((b[6] << 2) | (b[7] >> 6)));
    s.push(e(b[7] >> 1));
    s.push(e((b[7] << 4) | (b[8] >> 4)));
    s.push(e((b[8] << 1) | (b[9] >> 7)));
    s.push(e(b[9] >> 2));
    s.push(e((b[9] << 3) | (b[10] >> 5)));
    s.push(e(b[10]));
    s.push(e(b[11] >> 3));
    s.push(e((b[11] << 2) | (b[12] >> 6)));
    s.push(e(b[12] >> 1));
    s.push(e((b[12] << 4) | (b[13] >> 4)));
    s.push(e((b[13] << 1) | (b[14] >> 7)));
    s.push(e(b[14] >> 2));
    s.push(e((b[14] << 3) | (b[15] >> 5)));
    s.push(e(b[15]));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequencer_is_per_stream_from_zero() {
        let mut s = Sequencer::new();
        assert_eq!(s.next("a"), 0);
        assert_eq!(s.next("a"), 1);
        assert_eq!(s.next("b"), 0); // independent stream
        assert_eq!(s.next("a"), 2);
    }

    #[test]
    fn ulid_is_26_crockford_chars() {
        let id = deterministic_ulid(1_700_000_000_000, "k");
        assert_eq!(id.len(), 26);
        assert!(id.chars().all(|c| CROCKFORD.contains(&(c as u8))));
    }

    // Determinism: same (ts, key) → same id; either differing → different id.
    #[test]
    fn ulid_is_deterministic() {
        let a = deterministic_ulid(1_700_000_000_000, "source\x00file\x000\x000");
        let b = deterministic_ulid(1_700_000_000_000, "source\x00file\x000\x000");
        let c = deterministic_ulid(1_700_000_000_000, "source\x00file\x000\x001");
        let d = deterministic_ulid(1_700_000_000_001, "source\x00file\x000\x000");
        assert_eq!(a, b);
        assert_ne!(a, c); // different key
        assert_ne!(a, d); // different timestamp
    }

    // The first 10 chars encode the timestamp and sort lexicographically with
    // it (the ULID ordering property).
    #[test]
    fn ulid_timestamp_prefix_is_monotone() {
        let early = deterministic_ulid(1_700_000_000_000, "k");
        let late = deterministic_ulid(1_700_000_001_000, "k");
        assert!(early[..10] < late[..10]);
    }

    // The envelope round-trips and matches the wire field names ingest reads.
    #[test]
    fn event_json_uses_canonical_field_names() {
        let e = Event {
            event_id: "E".into(),
            stream: "edge-x-claudecode".into(),
            seq: 3,
            ts: "2026-05-31T20:00:00Z".into(),
            source: source::CLAUDE_CODE.into(),
            session_id: "S1".into(),
            kind: kind::USER_PROMPT.into(),
            payload: serde_json::json!({"text": "hi"}),
            rollup_dim: String::new(),
        };
        let j = serde_json::to_string(&e).unwrap();
        assert!(j.contains(r#""kind":"user_prompt""#));
        assert!(j.contains(r#""session_id":"S1""#));
        assert!(j.contains(r#""source":"claude_code""#));
        assert!(!j.contains("rollup_dim")); // empty → omitted
        let back: Event = serde_json::from_str(&j).unwrap();
        assert_eq!(back.seq, 3);
        assert_eq!(back.payload["text"], "hi");
    }
}

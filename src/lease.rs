// A soft TTL lease over a bucket object — elects one builder per epoch so a
// fleet of viewers doesn't duplicate the index-build+publish work. It is an
// OPTIMIZATION, not a correctness mechanism: the read-model publishes are
// immutable builds behind a pointer swap, so a double-builder (clock skew, the
// narrow takeover race below) only wastes compute, never corrupts.
//
// Protocol: acquire = conditional create of `lease/<scope>`; an expired lease
// is deleted and re-created, then VERIFIED by reading back — which collapses
// the delete/create interleavings to one winner in everything but pathological
// timing. Holders refresh (overwrite their own lease) periodically and must
// abort publish if a refresh finds the lease no longer theirs.

use crate::bucket::Bucket;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const TTL_MS: i64 = 15 * 60 * 1000;

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct Body {
    holder: String,
    expires_ms: i64,
}

fn key(scope: &str) -> String {
    format!("lease/{scope}")
}

fn read(b: &dyn Bucket, scope: &str) -> Result<Option<Body>> {
    Ok(b.get(&key(scope))?.and_then(|raw| serde_json::from_slice(&raw).ok()))
}

fn write_body(holder: &str, now_ms: i64, ttl_ms: i64) -> (Body, Vec<u8>) {
    let body = Body { holder: holder.to_string(), expires_ms: now_ms + ttl_ms };
    let raw = serde_json::to_vec(&body).expect("lease body serializes");
    (body, raw)
}

/// Try to take the lease. Ok(true) = held (go build); Ok(false) = someone else
/// holds a live lease (skip the build, pull their output when it lands).
pub fn acquire(b: &dyn Bucket, scope: &str, holder: &str, now_ms: i64, ttl_ms: i64) -> Result<bool> {
    let (body, raw) = write_body(holder, now_ms, ttl_ms);
    if b.put_if_absent(&key(scope), &raw)? {
        return Ok(read(b, scope)?.as_ref() == Some(&body));
    }
    match read(b, scope)? {
        // Our own live lease: re-acquiring just extends it (idempotent).
        Some(cur) if cur.holder == holder && cur.expires_ms > now_ms => {
            b.put(&key(scope), &raw)?;
            Ok(true)
        }
        Some(cur) if cur.expires_ms > now_ms => Ok(false),
        _ => {
            // Expired (or unreadable): take over, then verify we won — two
            // takers can interleave delete/create, so trust only the read-back.
            b.delete(&key(scope))?;
            if !b.put_if_absent(&key(scope), &raw)? {
                return Ok(false);
            }
            Ok(read(b, scope)?.as_ref() == Some(&body))
        }
    }
}

/// Extend our own lease. Ok(false) = it is no longer ours — the caller must
/// abort before publishing.
pub fn refresh(b: &dyn Bucket, scope: &str, holder: &str, now_ms: i64, ttl_ms: i64) -> Result<bool> {
    match read(b, scope)? {
        Some(cur) if cur.holder == holder => {
            let (_, raw) = write_body(holder, now_ms, ttl_ms);
            b.put(&key(scope), &raw)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Drop the lease if it is still ours (best-effort; expiry covers the rest).
pub fn release(b: &dyn Bucket, scope: &str, holder: &str) -> Result<()> {
    if let Some(cur) = read(b, scope)? {
        if cur.holder == holder {
            b.delete(&key(scope))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::LocalFs;

    fn bucket() -> (LocalFs, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "synty-lease-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t").replace("::", "-")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        (LocalFs::new(&dir), dir)
    }

    // One builder wins; a second acquire against a live lease is refused; the
    // lease frees on release and on expiry.
    #[test]
    fn lease_elects_one_builder() {
        let (b, dir) = bucket();
        assert!(acquire(&b, "build", "mac-a", 1_000, 60_000).unwrap());
        assert!(!acquire(&b, "build", "mac-b", 2_000, 60_000).unwrap(), "live lease refuses");
        assert!(acquire(&b, "build", "mac-a", 3_000, 60_000).unwrap(), "holder re-acquires");

        release(&b, "build", "mac-a").unwrap();
        assert!(acquire(&b, "build", "mac-b", 4_000, 60_000).unwrap(), "released lease is free");

        // mac-b's lease expires at 64_000 — mac-a takes over after that.
        assert!(!acquire(&b, "build", "mac-a", 50_000, 60_000).unwrap());
        assert!(acquire(&b, "build", "mac-a", 70_000, 60_000).unwrap(), "expired lease is taken");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Refresh extends only the holder's lease; once lost, refresh says so and
    // the loser must abort before publishing.
    #[test]
    fn refresh_only_works_for_the_holder() {
        let (b, dir) = bucket();
        assert!(acquire(&b, "build", "mac-a", 1_000, 10_000).unwrap());
        assert!(refresh(&b, "build", "mac-a", 5_000, 10_000).unwrap(), "holder refreshes");
        assert!(!refresh(&b, "build", "mac-b", 5_000, 10_000).unwrap(), "non-holder cannot");
        // a's refresh pushed expiry to 15_000; b takes over after that.
        assert!(acquire(&b, "build", "mac-b", 20_000, 10_000).unwrap());
        assert!(!refresh(&b, "build", "mac-a", 21_000, 10_000).unwrap(), "lost lease refuses refresh");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // The primitive under it all: N threads race put_if_absent on one key —
    // exactly one creation succeeds.
    #[test]
    fn put_if_absent_is_atomic_across_threads() {
        let (b, dir) = bucket();
        let wins: usize = std::thread::scope(|s| {
            let b = &b;
            let hs: Vec<_> = (0..16)
                .map(|i| s.spawn(move || b.put_if_absent("k", format!("w{i}").as_bytes()).unwrap()))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap() as usize).sum()
        });
        assert_eq!(wins, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

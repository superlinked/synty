// Content-addressed embedding store over a `Bucket`. Each per-document ColBERT
// embedding is keyed by a hash of its text, so a message is encoded exactly once
// no matter how many trackers, devices, or re-index runs see it — the encode
// (which dominates cost) is shared across the whole fleet through the bucket.
//
// On-disk value is a compact little-endian blob: rows u32, cols u32, then the
// matrix as f16 (half the bytes of f32; the precision loss is well below what
// MaxSim over normalized ColBERT vectors notices). The store is write-once: a
// hash that exists is never re-put.

use crate::bucket::{self, Bucket};
use anyhow::{anyhow, Result};
use half::f16;
use ndarray::Array2;
use std::collections::HashSet;

pub struct EmbStore {
    bucket: Box<dyn Bucket>,
    /// Model namespace: vectors from different encoders are incompatible, so
    /// every model gets its own prefix — a mixed-model fleet can never
    /// silently share wrong vectors.
    ns: String,
}

impl EmbStore {
    /// Open the store at `uri` (a bucket URI or local path) for `model`. Solo
    /// runs default to a local dir so re-indexing reuses embeddings even
    /// without a bucket.
    pub fn open(uri: &str, model: &str) -> Result<Self> {
        Ok(Self { bucket: bucket::open(uri)?, ns: model_ns(model) })
    }

    pub fn get(&self, hash: u64) -> Result<Option<Array2<f32>>> {
        match self.bucket.get(&self.key(hash))? {
            // An unreadable blob is treated as a miss, so corruption just
            // re-encodes rather than failing the build.
            Some(bytes) => Ok(decode(&bytes).ok()),
            None => Ok(None),
        }
    }

    /// Fetch an explicit batch in order. This is used for small known sets
    /// such as a topic's representative members, where every hash is expected
    /// to exist and cloud request latency should overlap.
    pub fn get_many(&self, hashes: &[u64]) -> Result<Vec<Option<Array2<f32>>>> {
        let keys: Vec<String> = hashes.iter().map(|&hash| self.key(hash)).collect();
        decode_many(self.bucket.get_many(&keys)?)
    }

    /// Resolve a corpus-sized batch from one listing, then fetch only objects
    /// that exist. A missing hash is an encode candidate, not 100 ms of remote
    /// negative lookup repeated for every new document.
    pub fn get_known(&self, hashes: &[u64]) -> Result<Vec<Option<Array2<f32>>>> {
        let known: HashSet<String> = self.bucket.list(&format!("embeddings/{}", self.ns))?.into_iter().collect();
        let mut positions = Vec::new();
        let mut keys = Vec::new();
        for (i, &hash) in hashes.iter().enumerate() {
            let key = self.key(hash);
            if known.contains(&key) {
                positions.push(i);
                keys.push(key);
            }
        }
        let fetched = decode_many(self.bucket.get_many(&keys)?)?;
        let mut out = vec![None; hashes.len()];
        for (i, value) in positions.into_iter().zip(fetched) {
            out[i] = value;
        }
        Ok(out)
    }

    /// Store an embedding if absent (write-once; another device may have raced
    /// to write the same content, which is fine — the bytes are identical).
    pub fn put(&self, hash: u64, emb: &Array2<f32>) -> Result<()> {
        self.bucket.put_if_absent(&self.key(hash), &encode(emb))?;
        Ok(())
    }

    /// Share one encoder batch with bounded cloud concurrency. The entries are
    /// still conditional write-once objects, so fleet races remain harmless.
    pub fn put_many(&self, entries: &[(u64, Array2<f32>)]) -> Result<()> {
        let objects: Vec<(String, Vec<u8>)> =
            entries.iter().map(|(hash, emb)| (self.key(*hash), encode(emb))).collect();
        self.bucket.put_many_if_absent(&objects)?;
        Ok(())
    }

    /// Sharded key so a directory/prefix never holds the whole fleet's vectors.
    fn key(&self, hash: u64) -> String {
        format!("embeddings/{}{:02x}/{:016x}.emb", self.ns, hash & 0xff, hash)
    }
}

fn model_ns(model: &str) -> String {
    format!("m{:08x}/", crate::index::fnv1a(model.as_bytes()) as u32)
}

/// Content-addressed summary store over a `Bucket`: one write-once object per
/// (unit key, input hash) — the first viewer to need a summary generates it
/// for the whole fleet, exactly like embeddings. The body carries the unit key
/// in clear (the object name only has its hash), so a viewer can materialize
/// its local cache from the bucket without knowing the keys in advance. An
/// empty summary is also a result worth sharing: it records "generated and
/// gate-rejected", which stops every other viewer from retrying.
pub struct SummaryStore {
    bucket: Box<dyn Bucket>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SummaryBody {
    key: String,
    hash: String,
    summary: String,
}

impl SummaryStore {
    pub fn open(uri: &str) -> Result<Self> {
        Ok(Self { bucket: bucket::open(uri)? })
    }

    /// The fleet's summary for (key, input-hash), if any viewer generated it.
    pub fn get(&self, key: &str, hash: &str) -> Result<Option<String>> {
        match self.bucket.get(&skey(key, hash))? {
            Some(raw) => Ok(serde_json::from_slice::<SummaryBody>(&raw).ok().map(|b| b.summary)),
            None => Ok(None),
        }
    }

    /// Share a generated summary (write-once; a racing viewer's identical work
    /// simply loses the race, which is fine).
    pub fn put(&self, key: &str, hash: &str, summary: &str) -> Result<()> {
        let body = SummaryBody { key: key.into(), hash: hash.into(), summary: summary.into() };
        self.bucket.put_if_absent(&skey(key, hash), &serde_json::to_vec(&body)?)?;
        Ok(())
    }

    /// Two-way sync between the local cache and the fleet store, off one
    /// listing: pull objects this machine can't account for (never overwriting
    /// an existing local entry — its hash may be newer; the pending pass
    /// resolves that per job), and push local entries the fleet lacks (e.g. a
    /// cache built before this machine joined the bucket). `keep` filters
    /// pulls — without it, orphaned topic summaries from superseded
    /// clusterings would cycle forever through pull → prune → pull. Returns
    /// (pulled, pushed).
    pub fn sync_cache(
        &self,
        cache: &mut crate::units::SummaryCache,
        keep: &dyn Fn(&str) -> bool,
    ) -> Result<(usize, usize)> {
        let listed: std::collections::HashSet<String> =
            self.bucket.list("summaries")?.into_iter().collect();
        let have: std::collections::HashSet<String> =
            cache.iter().map(|(k, v)| skey(k, &v.hash)).collect();

        let mut pulled = 0;
        for obj in &listed {
            if have.contains(obj) {
                continue;
            }
            let Some(raw) = self.bucket.get(obj)? else { continue };
            let Ok(b) = serde_json::from_slice::<SummaryBody>(&raw) else { continue };
            if !cache.contains_key(&b.key) && keep(&b.key) {
                cache.insert(b.key, crate::units::CachedSummary { hash: b.hash, summary: b.summary });
                pulled += 1;
            }
        }
        let mut pushed = 0;
        for (k, v) in cache.iter() {
            if !listed.contains(&skey(k, &v.hash)) {
                self.put(k, &v.hash, &v.summary)?;
                pushed += 1;
            }
        }
        Ok((pulled, pushed))
    }
}

/// Sharded by the unit key's hash; the input hash makes the object immutable —
/// changed inputs are a NEW object, never an overwrite.
fn skey(key: &str, hash: &str) -> String {
    let kf = crate::index::fnv1a(key.as_bytes());
    format!("summaries/{:02x}/{:016x}-{hash}.json", kf & 0xff, kf)
}

fn encode(a: &Array2<f32>) -> Vec<u8> {
    let (rows, cols) = (a.nrows() as u32, a.ncols() as u32);
    let mut out = Vec::with_capacity(8 + a.len() * 2);
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&cols.to_le_bytes());
    for &v in a.iter() {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn decode(b: &[u8]) -> Result<Array2<f32>> {
    if b.len() < 8 {
        return Err(anyhow!("embedding blob too short"));
    }
    let rows = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let want = 8 + rows * cols * 2;
    if b.len() != want {
        return Err(anyhow!("embedding blob size {} != {want}", b.len()));
    }
    let mut data = Vec::with_capacity(rows * cols);
    for chunk in b[8..].chunks_exact(2) {
        data.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    Array2::from_shape_vec((rows, cols), data).map_err(|e| anyhow!("embedding shape: {e}"))
}

fn decode_many(values: Vec<Option<Vec<u8>>>) -> Result<Vec<Option<Array2<f32>>>> {
    Ok(values
        .into_iter()
        // Corruption remains a cache miss, matching `get`.
        .map(|value| value.and_then(|bytes| decode(&bytes).ok()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrips() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = decode(&encode(&a)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn store_is_write_once_and_shared() {
        let dir = std::env::temp_dir().join(format!("synty-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = EmbStore::open(dir.to_str().unwrap(), crate::DEFAULT_MODEL).unwrap();
        let a = Array2::from_shape_vec((1, 2), vec![7.0, 8.0]).unwrap();
        assert!(s.get(42).unwrap().is_none());
        s.put(42, &a).unwrap();
        // A second "device" opening the same store reads the cached vector.
        let s2 = EmbStore::open(dir.to_str().unwrap(), crate::DEFAULT_MODEL).unwrap();
        assert_eq!(s2.get(42).unwrap().unwrap(), a);
        // A different encoder's vectors are incompatible — its store is a
        // separate namespace, never a silent hit on the default's.
        let other = EmbStore::open(dir.to_str().unwrap(), "some/other-encoder").unwrap();
        assert!(other.get(42).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corpus_batch_reuses_known_vectors_and_leaves_misses_for_encoding() {
        let dir = std::env::temp_dir().join(format!("synty-store-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = EmbStore::open(dir.to_str().unwrap(), crate::DEFAULT_MODEL).unwrap();
        let a = Array2::from_shape_vec((1, 2), vec![7.0, 8.0]).unwrap();
        let b = Array2::from_shape_vec((1, 2), vec![9.0, 10.0]).unwrap();
        s.put_many(&[(42, a.clone()), (7, b.clone())]).unwrap();

        let found = s.get_known(&[7, 99, 42]).unwrap();
        assert_eq!(found[0].as_ref(), Some(&b));
        assert!(found[1].is_none(), "an absent corpus hash remains an encode candidate");
        assert_eq!(found[2].as_ref(), Some(&a));

        // Conditional batch writes preserve the first fleet writer.
        let replacement = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        s.put_many(&[(42, replacement)]).unwrap();
        assert_eq!(s.get(42).unwrap().as_ref(), Some(&a));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Sharded, content-addressed, model-namespaced keys: every encoder gets
    // its own prefix, so vectors from different models can never mix.
    #[test]
    fn key_is_sharded_and_model_namespaced() {
        let def = EmbStore { bucket: bucket::open("/tmp").unwrap(), ns: model_ns(crate::DEFAULT_MODEL) };
        assert!(def.key(0xABCD).starts_with("embeddings/m"));
        assert!(def.key(0xABCD).ends_with("/cd/000000000000abcd.emb"));
        let other = EmbStore { bucket: bucket::open("/tmp").unwrap(), ns: model_ns("some/other-encoder") };
        assert_ne!(def.key(0xABCD), other.key(0xABCD), "models never share a namespace");
    }

    // The collaboration contract: the first viewer's summary serves the fleet —
    // a second store sees it by (key, hash) and via blind pull, and a changed
    // input is a different object, never an overwrite.
    #[test]
    fn summary_store_is_write_once_and_pullable() {
        let dir = std::env::temp_dir().join(format!("synty-sumstore-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = SummaryStore::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(a.get("S1", "h1").unwrap(), None);
        a.put("S1", "h1", "Fixed the login redirect.").unwrap();
        a.put("S1", "h1", "A racing different text").unwrap(); // loses: write-once
        let b = SummaryStore::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(b.get("S1", "h1").unwrap().as_deref(), Some("Fixed the login redirect."));
        assert_eq!(b.get("S1", "h2").unwrap(), None, "changed input = different object");

        // A fresh viewer materializes its local cache without knowing any keys.
        let mut cache = crate::units::SummaryCache::default();
        assert_eq!(b.sync_cache(&mut cache, &|_: &str| true).unwrap(), (1, 0));
        assert_eq!(cache["S1"].summary, "Fixed the login redirect.");
        // A second sync is a no-op.
        assert_eq!(b.sync_cache(&mut cache, &|_: &str| true).unwrap(), (0, 0));
        // A machine with a pre-fleet cache shares it on its first sync.
        cache.insert("S2".into(), crate::units::CachedSummary { hash: "h9".into(), summary: "Added keyset pagination.".into() });
        assert_eq!(b.sync_cache(&mut cache, &|_: &str| true).unwrap(), (0, 1));
        let fresh = SummaryStore::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(fresh.get("S2", "h9").unwrap().as_deref(), Some("Added keyset pagination."));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

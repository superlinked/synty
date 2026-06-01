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

pub struct EmbStore {
    bucket: Box<dyn Bucket>,
}

impl EmbStore {
    /// Open the store at `uri` (a bucket URI or local path). Solo runs default
    /// to a local dir so re-indexing reuses embeddings even without a bucket.
    pub fn open(uri: &str) -> Result<Self> {
        Ok(Self { bucket: bucket::open(uri)? })
    }

    pub fn get(&self, hash: u64) -> Result<Option<Array2<f32>>> {
        match self.bucket.get(&key(hash))? {
            // An unreadable blob (e.g. an older f32 entry) is treated as a miss,
            // so a format change just re-encodes rather than failing the build.
            Some(bytes) => Ok(decode(&bytes).ok()),
            None => Ok(None),
        }
    }

    /// Store an embedding if absent (write-once; another device may have raced
    /// to write the same content, which is fine — the bytes are identical).
    pub fn put(&self, hash: u64, emb: &Array2<f32>) -> Result<()> {
        let k = key(hash);
        if self.bucket.exists(&k)? {
            return Ok(());
        }
        self.bucket.put(&k, &encode(emb))
    }
}

/// Sharded key so a directory/prefix never holds the whole fleet's vectors.
fn key(hash: u64) -> String {
    format!("embeddings/{:02x}/{:016x}.emb", hash & 0xff, hash)
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
        let s = EmbStore::open(dir.to_str().unwrap()).unwrap();
        let a = Array2::from_shape_vec((1, 2), vec![7.0, 8.0]).unwrap();
        assert!(s.get(42).unwrap().is_none());
        s.put(42, &a).unwrap();
        // A second "device" opening the same store reads the cached vector.
        let s2 = EmbStore::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(s2.get(42).unwrap().unwrap(), a);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Sharded, content-addressed keys.
    #[test]
    fn key_is_sharded() {
        assert_eq!(key(0xABCD), "embeddings/cd/000000000000abcd.emb");
    }
}

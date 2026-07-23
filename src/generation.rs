// Coordinate corpus derivation and indexing so an immutable read-model build
// always consumes one complete docs/analysis/trace generation.

use anyhow::{Context, Result};
use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Exclusive guard shared by ingest writers and index snapshot consumers.
pub(crate) struct Guard(File);

/// Lock the generation containing `docs_path` until the returned guard drops.
pub(crate) fn exclusive(docs_path: &str) -> Result<Guard> {
    let docs = Path::new(docs_path);
    let parent = docs
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let lock_path = parent.join(".ingest-index.lock");
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open generation lock {}", lock_path.display()))?;
    FileExt::lock_exclusive(&file)
        .with_context(|| format!("lock generation {}", lock_path.display()))?;
    Ok(Guard(file))
}

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    // An index reader arriving between the three ingest writes must wait and
    // then observe one complete generation, never a docs/projection mixture.
    #[test]
    fn index_waits_for_the_complete_ingest_generation() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("synty-generation-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let docs = dir.join("docs.jsonl");
        let analysis = dir.join("analysis.json");
        let trace = dir.join("trace.json");
        let docs_s = docs.to_string_lossy().into_owned();

        let (partial_tx, partial_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let producer_docs = docs_s.clone();
        let producer = std::thread::spawn(move || {
            let _guard = exclusive(&producer_docs).unwrap();
            std::fs::write(&producer_docs, "generation-b\n").unwrap();
            partial_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            std::fs::write(analysis, r#"{"generation":"b"}"#).unwrap();
            std::fs::write(trace, r#"{"generation":"b"}"#).unwrap();
        });
        partial_rx.recv().unwrap();

        let (attempt_tx, attempt_rx) = mpsc::channel();
        let (read_tx, read_rx) = mpsc::channel();
        let reader_docs = docs_s.clone();
        let reader_dir = dir.clone();
        let reader = std::thread::spawn(move || {
            attempt_tx.send(()).unwrap();
            let _guard = exclusive(&reader_docs).unwrap();
            read_tx
                .send((
                    std::fs::read_to_string(&reader_docs).unwrap(),
                    std::fs::read_to_string(reader_dir.join("analysis.json")).unwrap(),
                    std::fs::read_to_string(reader_dir.join("trace.json")).unwrap(),
                ))
                .unwrap();
        });
        attempt_rx.recv().unwrap();
        assert!(
            read_rx.try_recv().is_err(),
            "the index reader must block while ingest has published only docs"
        );
        finish_tx.send(()).unwrap();
        let generation = read_rx.recv().unwrap();
        assert_eq!(generation.0, "generation-b\n");
        assert!(generation.1.contains(r#""generation":"b""#));
        assert!(generation.2.contains(r#""generation":"b""#));

        producer.join().unwrap();
        reader.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}

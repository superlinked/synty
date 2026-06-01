// S3 and GCS bucket backends via object_store, behind the `s3` / `gcs` cargo
// features so the default build stays local-only and dependency-light. A small
// blocking shim runs object_store's async calls on a current-thread runtime;
// keys are namespaced under an optional in-bucket prefix and returned relative
// to it, so the rest of the code treats cloud and local buckets identically.

use crate::bucket::Bucket;
use anyhow::{anyhow, bail, Result};
use futures::StreamExt;
use object_store::{path::Path as OPath, ObjectStore};
use std::sync::Arc;
use tokio::runtime::Runtime;

struct Cloud {
    store: Arc<dyn ObjectStore>,
    rt: Runtime,
    prefix: String,
}

pub fn open(kind: &str, rest: &str) -> Result<Box<dyn Bucket>> {
    let (bucket, prefix) = match rest.split_once('/') {
        Some((b, p)) => (b.to_string(), p.trim_end_matches('/').to_string()),
        None => (rest.to_string(), String::new()),
    };
    let store: Arc<dyn ObjectStore> = match kind {
        #[cfg(feature = "s3")]
        "s3" => Arc::new(
            object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(&bucket)
                .build()
                .map_err(|e| anyhow!("s3 init: {e}"))?,
        ),
        #[cfg(feature = "gcs")]
        "gcs" => Arc::new(
            object_store::gcp::GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(&bucket)
                .build()
                .map_err(|e| anyhow!("gcs init: {e}"))?,
        ),
        _ => bail!("{kind} backend not built in (rebuild with --features {kind})"),
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("runtime: {e}"))?;
    Ok(Box::new(Cloud { store, rt, prefix }))
}

impl Cloud {
    fn full(&self, key: &str) -> OPath {
        if self.prefix.is_empty() {
            OPath::from(key)
        } else {
            OPath::from(format!("{}/{key}", self.prefix))
        }
    }
    /// Drop the in-bucket prefix so listed keys are relative (as LocalFs does).
    fn relative(&self, loc: &str) -> String {
        if self.prefix.is_empty() {
            loc.to_string()
        } else {
            loc.strip_prefix(&format!("{}/", self.prefix)).unwrap_or(loc).to_string()
        }
    }
}

impl Bucket for Cloud {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let p = self.full(key);
        let payload = bytes.to_vec().into();
        self.rt.block_on(self.store.put(&p, payload)).map_err(|e| anyhow!("put {key}: {e}"))?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let p = self.full(key);
        self.rt
            .block_on(async {
                match self.store.get(&p).await {
                    Ok(res) => Ok(Some(res.bytes().await?.to_vec())),
                    Err(object_store::Error::NotFound { .. }) => Ok(None),
                    Err(e) => Err(e),
                }
            })
            .map_err(|e| anyhow!("get {key}: {e}"))
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let p = self.full(key);
        match self.rt.block_on(self.store.head(&p)) {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(anyhow!("head {key}: {e}")),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let p = self.full(prefix);
        let keys = self
            .rt
            .block_on(async {
                let mut stream = self.store.list(Some(&p));
                let mut out = Vec::new();
                while let Some(item) = stream.next().await {
                    out.push(item?.location.to_string());
                }
                Ok::<_, object_store::Error>(out)
            })
            .map_err(|e| anyhow!("list {prefix}: {e}"))?;
        let mut rel: Vec<String> = keys.iter().map(|k| self.relative(k)).collect();
        rel.sort();
        Ok(rel)
    }
}

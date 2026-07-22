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

/// Enough overlap to hide object-store request latency without turning a
/// workstation build into an unbounded connection or memory spike.
const OBJECT_CONCURRENCY: usize = 32;

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
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("runtime: {e}"))?;
    let store: Arc<dyn ObjectStore> = match kind {
        #[cfg(feature = "s3")]
        "s3" => {
            let cfg = crate::config::load();
            let mut builder = object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(&bucket)
                // S3 supports If-None-Match natively (since 2024), but
                // object_store 0.11 returns NotImplemented for PutMode::Create
                // unless conditional put is switched on explicitly — and
                // put_if_absent is what leases and write-once stores run on.
                .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch);
            if let Some(profile) = cfg.aws_profile.as_deref() {
                // An explicit profile opts into shared-config loading. This is
                // Roles Anywhere and other credential_process providers
                // refresh without persisting returned keys. With no profile,
                // object_store keeps its native env/web-identity/container/IMDS
                // provider chain.
                let credentials = aws_config::profile::ProfileFileCredentialsProvider::builder()
                    .profile_name(profile)
                    .build();
                let region_provider =
                    aws_config::default_provider::region::DefaultRegionChain::builder()
                    .profile_name(profile)
                    .build();
                if let Some(region) = rt.block_on(region_provider.region()) {
                    builder = builder.with_region(region.as_ref());
                }
                builder = builder.with_credentials(Arc::new(SdkCredentials {
                    inner: aws_credential_types::provider::SharedCredentialsProvider::new(
                        credentials,
                    ),
                    cache_key: profile.to_string(),
                }));
            } else {
                // object_store's instance-role provider handles credentials,
                // but otherwise defaults the signing region to us-east-1.
                // The SDK region chain adds shared config and EC2 IMDSv2, so a
                // role-backed instance works without a temporary shell env.
                let region_provider =
                    aws_config::default_provider::region::DefaultRegionChain::builder().build();
                if let Some(region) = rt.block_on(region_provider.region()) {
                    builder = builder.with_region(region.as_ref());
                }
            }
            Arc::new(builder.build().map_err(|e| anyhow!("s3 init: {e}"))?)
        }
        #[cfg(feature = "gcs")]
        "gcs" => Arc::new(
            object_store::gcp::GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(&bucket)
                .build()
                .map_err(|e| anyhow!("gcs init: {e}"))?,
        ),
        _ => bail!("{kind} backend not built in (rebuild with --features {kind})"),
    };
    Ok(Box::new(Cloud { store, rt, prefix }))
}

/// Bridge an AWS rotating provider into object_store's signer. Cache by profile
/// until five minutes before expiry so repeated upload batches do not repeatedly
/// invoke credential_process, then refresh without persisting returned keys.
#[cfg(feature = "s3")]
#[derive(Debug)]
struct SdkCredentials {
    inner: aws_credential_types::provider::SharedCredentialsProvider,
    cache_key: String,
}

#[cfg(feature = "s3")]
static PROFILE_CREDENTIALS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, aws_credential_types::Credentials>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

#[cfg(feature = "s3")]
#[async_trait::async_trait]
impl object_store::CredentialProvider for SdkCredentials {
    type Credential = object_store::aws::AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        use aws_credential_types::provider::ProvideCredentials;
        let cached = PROFILE_CREDENTIALS
            .lock()
            .map_err(|_| credential_error("credential cache lock poisoned"))?
            .get(&self.cache_key)
            .filter(|c| credential_fresh(c))
            .cloned();
        let c = match cached {
            Some(c) => c,
            None => {
                let c = self.inner.provide_credentials().await.map_err(|e| {
                    object_store::Error::Generic {
                        store: "S3",
                        source: Box::new(e),
                    }
                })?;
                PROFILE_CREDENTIALS
                    .lock()
                    .map_err(|_| credential_error("credential cache lock poisoned"))?
                    .insert(self.cache_key.clone(), c.clone());
                c
            }
        };
        Ok(Arc::new(object_store::aws::AwsCredential {
            key_id: c.access_key_id().to_string(),
            secret_key: c.secret_access_key().to_string(),
            token: c.session_token().map(ToString::to_string),
        }))
    }
}

#[cfg(feature = "s3")]
fn credential_fresh(c: &aws_credential_types::Credentials) -> bool {
    c.expiry().is_none_or(|expiry| {
        expiry > std::time::SystemTime::now() + std::time::Duration::from_secs(5 * 60)
    })
}

#[cfg(feature = "s3")]
fn credential_error(message: &str) -> object_store::Error {
    object_store::Error::Generic {
        store: "S3",
        source: Box::new(std::io::Error::other(message.to_string())),
    }
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

    fn get_many(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>> {
        let requests = futures::stream::iter(keys.iter().cloned().enumerate()).map(|(i, key)| {
            let store = self.store.clone();
            let path = self.full(&key);
            async move {
                let value = match store.get(&path).await {
                    Ok(res) => res.bytes().await.map(|b| Some(b.to_vec())),
                    Err(object_store::Error::NotFound { .. }) => Ok(None),
                    Err(e) => Err(e),
                };
                (i, key, value)
            }
        });
        let results = self.rt.block_on(requests.buffer_unordered(OBJECT_CONCURRENCY).collect::<Vec<_>>());
        let mut out = vec![None; keys.len()];
        for (i, key, value) in results {
            out[i] = value.map_err(|e| anyhow!("get {key}: {e}"))?;
        }
        Ok(out)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.size(key)?.is_some())
    }

    fn size(&self, key: &str) -> Result<Option<u64>> {
        let p = self.full(key);
        match self.rt.block_on(self.store.head(&p)) {
            Ok(meta) => Ok(Some(meta.size as u64)),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
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

    fn list_after(&self, prefix: &str, offset: &str) -> Result<Vec<String>> {
        let p = self.full(prefix);
        let offset = self.full(offset);
        let keys = self
            .rt
            .block_on(async {
                let mut stream = self.store.list_with_offset(Some(&p), &offset);
                let mut out = Vec::new();
                while let Some(item) = stream.next().await {
                    out.push(item?.location.to_string());
                }
                Ok::<_, object_store::Error>(out)
            })
            .map_err(|e| anyhow!("list {prefix} after {offset}: {e}"))?;
        let mut rel: Vec<String> = keys.iter().map(|k| self.relative(k)).collect();
        rel.sort();
        Ok(rel)
    }

    fn put_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool> {
        use object_store::{PutMode, PutOptions};
        let p = self.full(key);
        let payload = bytes.to_vec().into();
        let opts = PutOptions::from(PutMode::Create);
        match self.rt.block_on(self.store.put_opts(&p, payload, opts)) {
            Ok(_) => Ok(true),
            Err(object_store::Error::AlreadyExists { .. }) => Ok(false),
            Err(e) => Err(anyhow!("put_if_absent {key}: {e}")),
        }
    }

    fn put_many_if_absent(&self, objects: &[(String, Vec<u8>)]) -> Result<usize> {
        use object_store::{PutMode, PutOptions};
        let requests = futures::stream::iter(objects.iter()).map(|(key, bytes)| {
            let store = self.store.clone();
            let path = self.full(key);
            let key = key.clone();
            let payload = bytes.clone().into();
            async move {
                let result = match store.put_opts(&path, payload, PutOptions::from(PutMode::Create)).await {
                    Ok(_) => Ok(true),
                    Err(object_store::Error::AlreadyExists { .. }) => Ok(false),
                    Err(e) => Err(e),
                };
                (key, result)
            }
        });
        let results = self.rt.block_on(requests.buffer_unordered(OBJECT_CONCURRENCY).collect::<Vec<_>>());
        let mut created = 0;
        for (key, result) in results {
            created += usize::from(result.map_err(|e| anyhow!("put_if_absent {key}: {e}"))?);
        }
        Ok(created)
    }

    fn delete(&self, key: &str) -> Result<()> {
        let p = self.full(key);
        match self.rt.block_on(self.store.delete(&p)) {
            Ok(()) => Ok(()),
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(anyhow!("delete {key}: {e}")),
        }
    }
}

#[cfg(all(test, feature = "s3"))]
mod tests {
    use super::*;
    use aws_credential_types::provider::{ProvideCredentials, future};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct CountingProvider(Arc<AtomicUsize>);

    impl ProvideCredentials for CountingProvider {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            self.0.fetch_add(1, Ordering::SeqCst);
            future::ProvideCredentials::ready(Ok(aws_credential_types::Credentials::new(
                "key",
                "secret",
                Some("token".into()),
                Some(std::time::SystemTime::now() + std::time::Duration::from_secs(3600)),
                "test",
            )))
        }
    }

    #[test]
    fn rotating_credentials_are_cached_across_bucket_sessions() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cache_key = format!("test-profile-{}", std::process::id());
        let provider = SdkCredentials {
            inner: aws_credential_types::provider::SharedCredentialsProvider::new(
                CountingProvider(calls.clone()),
            ),
            cache_key: cache_key.clone(),
        };
        let reopened = SdkCredentials {
            inner: aws_credential_types::provider::SharedCredentialsProvider::new(
                CountingProvider(calls.clone()),
            ),
            cache_key,
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        use object_store::CredentialProvider;
        let first = rt.block_on(provider.get_credential()).unwrap();
        let second = rt.block_on(reopened.get_credential()).unwrap();
        assert_eq!(first.key_id, "key");
        assert_eq!(second.token.as_deref(), Some("token"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "one helper invocation serves repeated batches"
        );
    }
}

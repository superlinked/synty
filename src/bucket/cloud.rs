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
        summarize_batch_puts(results)
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

fn summarize_batch_puts(
    results: Vec<(String, object_store::Result<bool>)>,
) -> Result<usize> {
    let mut created = 0;
    let mut first_error = None;
    for (key, result) in results {
        match result {
            Ok(was_created) => created += usize::from(was_created),
            Err(error) if first_error.is_none() => first_error = Some((key, error)),
            Err(_) => {}
        }
    }
    if let Some((key, error)) = first_error {
        bail!(
            "put_many_if_absent partially completed: {created} object(s) created; first failure at {key}: {error}"
        );
    }
    Ok(created)
}

#[cfg(all(test, feature = "s3"))]
mod tests {
    use super::*;
    use aws_credential_types::provider::{ProvideCredentials, future};
    use futures::stream::BoxStream;
    use object_store::{
        GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, PutMultipartOpts,
        PutOptions, PutPayload, PutResult,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_cloud(store: Arc<dyn ObjectStore>) -> Cloud {
        Cloud {
            store,
            rt: tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap(),
            prefix: String::new(),
        }
    }

    #[derive(Debug)]
    struct FailingGet(object_store::memory::InMemory);

    impl std::fmt::Display for FailingGet {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("failing-get")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FailingGet {
        async fn put_opts(&self, location: &OPath, payload: PutPayload, opts: PutOptions) -> object_store::Result<PutResult> {
            self.0.put_opts(location, payload, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &OPath,
            opts: PutMultipartOpts,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.0.put_multipart_opts(location, opts).await
        }
        async fn get_opts(&self, _location: &OPath, _options: GetOptions) -> object_store::Result<GetResult> {
            Err(object_store::Error::Generic {
                store: "scenario",
                source: Box::new(std::io::Error::other("injected get failure")),
            })
        }
        async fn delete(&self, location: &OPath) -> object_store::Result<()> {
            self.0.delete(location).await
        }
        fn list(&self, prefix: Option<&OPath>) -> BoxStream<'_, object_store::Result<ObjectMeta>> {
            self.0.list(prefix)
        }
        async fn list_with_delimiter(&self, prefix: Option<&OPath>) -> object_store::Result<ListResult> {
            self.0.list_with_delimiter(prefix).await
        }
        async fn copy(&self, from: &OPath, to: &OPath) -> object_store::Result<()> {
            self.0.copy(from, to).await
        }
        async fn copy_if_not_exists(&self, from: &OPath, to: &OPath) -> object_store::Result<()> {
            self.0.copy_if_not_exists(from, to).await
        }
    }

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

    #[test]
    fn cloud_batches_preserve_order_missing_values_and_create_only_counts() {
        let cloud = test_cloud(Arc::new(object_store::memory::InMemory::new()));
        cloud.put("a", b"first").unwrap();
        cloud.put("c", b"third").unwrap();

        let values = cloud.get_many(&["c".into(), "missing".into(), "a".into()]).unwrap();
        assert_eq!(values, vec![Some(b"third".to_vec()), None, Some(b"first".to_vec())]);

        let created = cloud
            .put_many_if_absent(&[
                ("a".into(), b"replacement".to_vec()),
                ("b".into(), b"second".to_vec()),
                ("c".into(), b"replacement".to_vec()),
            ])
            .unwrap();
        assert_eq!(created, 1, "AlreadyExists objects do not inflate the creation count");
        assert_eq!(cloud.get("a").unwrap().as_deref(), Some(&b"first"[..]));
        assert_eq!(cloud.get("b").unwrap().as_deref(), Some(&b"second"[..]));
        assert_eq!(cloud.get("c").unwrap().as_deref(), Some(&b"third"[..]));
    }

    #[test]
    fn cloud_batch_get_propagates_non_not_found_errors() {
        let cloud = test_cloud(Arc::new(FailingGet(object_store::memory::InMemory::new())));
        let error = cloud.get_many(&["broken".into()]).unwrap_err().to_string();
        assert!(error.contains("injected get failure"), "backend error must remain actionable: {error}");
    }

    #[test]
    fn cloud_batch_put_error_reports_every_successful_create() {
        let failure = object_store::Error::Generic {
            store: "scenario",
            source: Box::new(std::io::Error::other("injected put failure")),
        };
        let error = summarize_batch_puts(vec![
            ("already-there".into(), Ok(false)),
            ("broken".into(), Err(failure)),
            ("created-after-error".into(), Ok(true)),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("1 object(s) created"), "{error}");
        assert!(error.contains("broken") && error.contains("injected put failure"), "{error}");
    }
}

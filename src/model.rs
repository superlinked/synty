// Resolve a model to a local directory, downloading from HuggingFace on first
// use (a directory spec is used as-is). Avoids hf_hub's no-timeout hang by
// fetching each file with ureq under connect/read timeouts and retry.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

const FILES: &[&str] = &[
    "tokenizer.json",
    "config.json",
    "config_sentence_transformers.json",
    "special_tokens_map.json",
    "1_Dense/config.json",
    "1_Dense/model.safetensors",
    "model.safetensors",
];

/// The ColBERT encoder model (the retrieval model).
pub fn ensure_model(spec: &str) -> Result<PathBuf> {
    ensure_repo(spec, FILES)
}

/// A directory spec is used verbatim; an HF repo id is cached under
/// `SYNTY_MODEL_DIR` (default `~/.cache/synty/models`) and the listed files are
/// downloaded if absent. Shared by the encoder and the Qwen summarizer.
pub fn ensure_repo(spec: &str, files: &[&str]) -> Result<PathBuf> {
    if Path::new(spec).is_dir() {
        return Ok(PathBuf::from(spec));
    }
    let dir = cache_path(&cache_root(), spec);
    let missing: Vec<&str> = files.iter().copied().filter(|f| !dir.join(f).exists()).collect();
    if !missing.is_empty() {
        eprintln!("fetching {spec} → {} ({} files)", dir.display(), missing.len());
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(20))
            .timeout_read(Duration::from_secs(300))
            .build();
        for f in missing {
            download(&agent, spec, f, &dir.join(f)).with_context(|| format!("download {f}"))?;
        }
    }
    Ok(dir)
}

fn cache_root() -> PathBuf {
    if let Ok(d) = std::env::var("SYNTY_MODEL_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("synty").join("models")
}

fn cache_path(root: &Path, spec: &str) -> PathBuf {
    root.join(spec.replace('/', "_"))
}

fn download(agent: &ureq::Agent, repo: &str, file: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let mut last = String::new();
    for attempt in 1..=5 {
        match agent.get(&url).call() {
            Ok(resp) => {
                let tmp = dest.with_extension("part");
                let mut w = std::fs::File::create(&tmp)?;
                std::io::copy(&mut resp.into_reader(), &mut w)?;
                std::fs::rename(&tmp, dest)?;
                return Ok(());
            }
            Err(e) => {
                last = e.to_string();
                eprintln!("  {file}: attempt {attempt}/5 failed: {last}");
            }
        }
    }
    Err(anyhow!("{url}: {last}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pointing at a local model dir uses it verbatim, never the network.
    #[test]
    fn local_dir_used_as_is() {
        let d = std::env::temp_dir();
        assert_eq!(ensure_model(d.to_str().unwrap()).unwrap(), d);
    }

    // An HF id becomes a flat cache subdir.
    #[test]
    fn cache_path_flattens_repo_id() {
        assert_eq!(
            cache_path(Path::new("/m"), "mixedbread-ai/mxbai-edge-colbert-v0-32m"),
            PathBuf::from("/m/mixedbread-ai_mxbai-edge-colbert-v0-32m")
        );
    }
}

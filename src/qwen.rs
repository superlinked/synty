// Local abstractive session summaries via Qwen3-0.6B on candle. One concise
// sentence per session, generated offline and cached by an input hash so the
// reader never runs the model at view time. Greedy decode (temperature 0) keeps
// it reproducible. This is the ONLY place a generative model is used — retrieval,
// clustering, and keyphrases stay LLM-free, and nothing leaves the machine.

use crate::units::{self, CachedSummary, DocInput, SessionInput};
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const REPO: &str = "Qwen/Qwen3-0.6B";
const FILES: &[&str] = &["tokenizer.json", "config.json", "model.safetensors"];
const MAX_NEW: usize = 64;
// Bump when the prompt changes so cached summaries regenerate.
const PROMPT_VERSION: &str = "v5";

struct Summarizer {
    model: ModelForCausalLM,
    tok: Tokenizer,
    device: Device,
    im_end: u32,
    eos: u32,
}

impl Summarizer {
    fn load() -> Result<Self> {
        let spec = std::env::var("SYNTY_LLM").unwrap_or_else(|_| REPO.to_string());
        let dir = crate::model::ensure_repo(&spec, FILES)?;
        let device = select_device();
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let cfg: Config = serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)
            .context("parse qwen config")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)
                .map_err(|e| anyhow!("safetensors: {e}"))?
        };
        let model = ModelForCausalLM::new(&cfg, vb).map_err(|e| anyhow!("load qwen: {e}"))?;
        let im_end = tok.token_to_id("<|im_end|>").unwrap_or(151645);
        let eos = tok.token_to_id("<|endoftext|>").unwrap_or(151643);
        Ok(Self { model, tok, device, im_end, eos })
    }

    /// Greedy-decode a single completion for `prompt`, stopping at the chat
    /// end-of-turn token. Returns the cleaned text and (prompt, output) token counts.
    fn generate(&mut self, prompt: &str) -> Result<(String, usize, usize)> {
        self.model.clear_kv_cache();
        let enc = self.tok.encode(prompt, false).map_err(|e| anyhow!("encode: {e}"))?;
        let ids = enc.get_ids();
        if ids.is_empty() {
            return Ok((String::new(), 0, 0));
        }
        let mut input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mut pos = 0usize;
        let mut out: Vec<u32> = Vec::new();
        for step in 0..MAX_NEW {
            let logits = self.model.forward(&input, pos).map_err(|e| anyhow!("forward: {e}"))?;
            let next = logits.squeeze(0)?.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if next == self.im_end || next == self.eos {
                break;
            }
            out.push(next);
            pos = if step == 0 { ids.len() } else { pos + 1 };
            input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
        }
        let text = self.tok.decode(&out, true).map_err(|e| anyhow!("decode: {e}"))?;
        Ok((clean(&text), ids.len(), out.len()))
    }
}

/// Run the summarizer on the Apple GPU when built with `--features metal`,
/// falling back to CPU; otherwise CPU. Mirrors the encoder's device choice.
fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                eprintln!("summarize: device = Metal (Apple GPU)");
                return d;
            }
            Err(e) => eprintln!("summarize: Metal unavailable ({e}); using CPU"),
        }
    }
    eprintln!("summarize: device = CPU");
    Device::Cpu
}

/// One-line instruction prompt in Qwen's chat format, fed the extractive signals
/// (ask, keyphrases, on-topic turns) for the model to synthesize.
fn prompt_for(s: &SessionInput) -> String {
    let mut turns = String::new();
    for (i, t) in s.turns.iter().enumerate() {
        turns.push_str(&format!("{}. {}\n", i + 1, t));
    }
    let files = if s.files.is_empty() { "(none recorded)".into() } else { s.files.join(", ") };
    format!(
        "<|im_start|>user\nYou are writing a one-line memory of a developer's coding session for a searchable index. \
Write ONE self-contained past-tense sentence (max 26 words) that a teammate with NO prior context can fully understand. \
Name the concrete subject — the feature, file, component, repo, or system worked on — instead of vague references like \"the slide\", \"the model\", or \"it\". \
Say what was built, changed, investigated, or decided, with the key specifics. \
Skip greetings, status preambles, and meta-commentary. \
Never echo a field label or output the repository name by itself. No preamble, no quotes, no lists.\n\n\
Repo: {}\nFiles changed: {}\nInitial request: {}\nKey terms: {}\nMessages (chronological):\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        s.repo,
        files,
        s.ask,
        s.keyphrases.join(", "),
        turns,
    )
}

/// One-line instruction prompt for a GitHub PR/issue (title + body).
fn prompt_for_doc(d: &DocInput) -> String {
    format!(
        "<|im_start|>user\nYou are writing a one-line memory of a GitHub {} for a searchable index. \
Write ONE self-contained past-tense sentence (max 26 words) a teammate with no prior context can understand. \
Name the concrete subject and the key change or decision. \
Skip boilerplate and templates. Never echo a field label. No preamble, no quotes, no lists.\n\n\
Repo: {}\nTitle: {}\nBody:\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        d.kind, d.repo, d.title, d.text,
    )
}

/// Strip any reasoning block, surrounding quotes, and extra lines; collapse to
/// one capped line.
fn clean(s: &str) -> String {
    let s = s.rsplit("</think>").next().unwrap_or(s);
    let s = s.trim().trim_matches('"').trim();
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or(s);
    let line = crate::excerpt(line, 220);
    // Reject degenerate outputs that just echo a prompt field; the caller then
    // falls back to the extractive line.
    let low = line.to_lowercase();
    let echo = ["repo:", "files changed:", "initial request:", "key terms:", "messages"]
        .iter()
        .any(|p| low.starts_with(p));
    if line.len() < 15 || echo {
        return String::new();
    }
    line
}

/// 8-byte content hash of arbitrary parts, salted by the prompt version so a
/// prompt change invalidates the cache.
fn hash_parts(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    h.update(PROMPT_VERSION.as_bytes());
    for p in parts {
        h.update(b"\0");
        h.update(p.as_bytes());
    }
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn input_hash(s: &SessionInput) -> String {
    hash_parts(&[&s.ask, &s.keyphrases.join(","), &s.files.join(","), &s.turns.join("\n")])
}

fn doc_hash(d: &DocInput) -> String {
    hash_parts(&[d.kind, &d.title, &d.text])
}

/// One unit of summarization work: a cache key, its input hash, and the prompt.
struct Job {
    key: String,
    hash: String,
    prompt: String,
    label: String,
}

/// Generate and cache one-line summaries for every work unit (session, PR, or
/// issue) whose inputs changed. `SYNTY_LLM_ONLY=<substr,...>` restricts to
/// matching keys and `SYNTY_LLM_LIMIT=N` caps the count — both for quick checks.
pub fn summarize_all() -> Result<()> {
    let sessions = units::session_inputs()?;
    let docs = units::doc_inputs()?;
    let mut jobs: Vec<Job> = Vec::with_capacity(sessions.len() + docs.len());
    for s in &sessions {
        jobs.push(Job { key: s.id.clone(), hash: input_hash(s), prompt: prompt_for(s), label: crate::short(&s.id) });
    }
    for d in &docs {
        jobs.push(Job { key: d.key.clone(), hash: doc_hash(d), prompt: prompt_for_doc(d), label: d.key.clone() });
    }

    let mut cache = units::load_summary_cache();
    let mut todo: Vec<&Job> = jobs
        .iter()
        .filter(|j| cache.get(&j.key).map(|c| c.hash != j.hash).unwrap_or(true))
        .collect();
    if let Ok(only) = std::env::var("SYNTY_LLM_ONLY") {
        let want: Vec<String> = only.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        todo.retain(|j| want.iter().any(|w| j.key.contains(w.as_str())));
    }
    if let Ok(n) = std::env::var("SYNTY_LLM_LIMIT") {
        if let Ok(n) = n.parse::<usize>() {
            todo.truncate(n);
        }
    }
    if todo.is_empty() {
        eprintln!("summaries up to date ({} items)", jobs.len());
        return Ok(());
    }
    eprintln!("summarizing {} item(s) with {REPO}…", todo.len());
    let t_load = std::time::Instant::now();
    let mut llm = Summarizer::load()?;
    eprintln!("model loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    let t_gen = std::time::Instant::now();
    let (mut in_tok, mut out_tok) = (0usize, 0usize);
    for (n, j) in todo.iter().enumerate() {
        let (summary, pt, ot) = match llm.generate(&j.prompt) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  {} failed: {e}", j.label);
                (String::new(), 0, 0)
            }
        };
        in_tok += pt;
        out_tok += ot;
        eprintln!("  [{}/{}] {} — {}", n + 1, todo.len(), j.label, summary);
        cache.insert(j.key.clone(), CachedSummary { hash: j.hash.clone(), summary });
        // Persist periodically so a long pass is resumable.
        if (n + 1) % 10 == 0 {
            units::save_summary_cache(&cache)?;
        }
    }
    units::save_summary_cache(&cache)?;

    let secs = t_gen.elapsed().as_secs_f64();
    eprintln!(
        "done: {} items in {:.1}s — {:.2} s/item, {:.1} output tok/s ({} prompt + {} output tok)",
        todo.len(),
        secs,
        secs / todo.len() as f64,
        out_tok as f64 / secs,
        in_tok,
        out_tok,
    );
    Ok(())
}

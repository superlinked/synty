// Local abstractive session summaries via Qwen3-0.6B on candle. One concise
// sentence per session, generated offline and cached by an input hash so the
// reader never runs the model at view time. Greedy decode (temperature 0) keeps
// it reproducible. This is the ONLY place a generative model is used — retrieval,
// clustering, and keyphrases stay LLM-free, and nothing leaves the machine.

use crate::units::{self, CachedSummary, SessionInput};
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
const PROMPT_VERSION: &str = "v2";

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
        let device = Device::Cpu;
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
    /// end-of-turn token.
    fn generate(&mut self, prompt: &str) -> Result<String> {
        self.model.clear_kv_cache();
        let enc = self.tok.encode(prompt, false).map_err(|e| anyhow!("encode: {e}"))?;
        let ids = enc.get_ids();
        if ids.is_empty() {
            return Ok(String::new());
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
        Ok(clean(&text))
    }
}

/// One-line instruction prompt in Qwen's chat format, fed the extractive signals
/// (ask, keyphrases, on-topic turns) for the model to synthesize.
fn prompt_for(s: &SessionInput) -> String {
    let mut turns = String::new();
    for t in &s.turns {
        turns.push_str("- ");
        turns.push_str(t);
        turns.push('\n');
    }
    format!(
        "<|im_start|>user\nYou are labeling a developer's coding session for a work-memory index. \
Write ONE concise past-tense sentence (max 22 words) describing what was done. \
No preamble, no quotes, no lists.\n\n\
Repo: {}\nInitial request: {}\nKey terms: {}\nMessages:\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        s.repo,
        s.ask,
        s.keyphrases.join(", "),
        turns,
    )
}

/// Strip any reasoning block, surrounding quotes, and extra lines; collapse to
/// one capped line.
fn clean(s: &str) -> String {
    let s = s.rsplit("</think>").next().unwrap_or(s);
    let s = s.trim().trim_matches('"').trim();
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or(s);
    crate::excerpt(line, 200)
}

fn input_hash(s: &SessionInput) -> String {
    let mut h = Sha256::new();
    h.update(PROMPT_VERSION.as_bytes());
    h.update(b"\0");
    h.update(s.ask.as_bytes());
    h.update(b"\0");
    h.update(s.keyphrases.join(",").as_bytes());
    h.update(b"\0");
    for t in &s.turns {
        h.update(t.as_bytes());
        h.update(b"\0");
    }
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Generate and cache one-line summaries for every session whose inputs changed.
/// `SYNTY_LLM_LIMIT` caps how many are generated in one run (for quick checks).
pub fn summarize_all() -> Result<()> {
    let inputs = units::session_inputs()?;
    let mut cache = units::load_summary_cache();
    let mut todo: Vec<&SessionInput> = inputs
        .iter()
        .filter(|s| cache.get(&s.id).map(|c| c.hash != input_hash(s)).unwrap_or(true))
        .collect();
    if let Ok(n) = std::env::var("SYNTY_LLM_LIMIT") {
        if let Ok(n) = n.parse::<usize>() {
            todo.truncate(n);
        }
    }
    if todo.is_empty() {
        eprintln!("summaries up to date ({} sessions)", inputs.len());
        return Ok(());
    }
    eprintln!("summarizing {} session(s) with {REPO} (CPU)…", todo.len());
    let mut llm = Summarizer::load()?;
    for (n, s) in todo.iter().enumerate() {
        let summary = match llm.generate(&prompt_for(s)) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  {} failed: {e}", crate::short(&s.id));
                String::new()
            }
        };
        eprintln!("  [{}/{}] {} — {}", n + 1, todo.len(), crate::short(&s.id), summary);
        cache.insert(s.id.clone(), CachedSummary { hash: input_hash(s), summary });
        // Persist periodically so a long CPU pass is resumable.
        if (n + 1) % 5 == 0 {
            units::save_summary_cache(&cache)?;
        }
    }
    units::save_summary_cache(&cache)?;
    Ok(())
}

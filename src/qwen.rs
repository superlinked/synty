// Local abstractive session summaries via Qwen3-0.6B on candle. One concise
// sentence per session, generated offline and cached by an input hash so the
// reader never runs the model at view time. Greedy decode (temperature 0) keeps
// it reproducible. This is the ONLY place a generative model is used — retrieval
// and clustering stay LLM-free, and nothing leaves the machine.

use crate::units::{self, CachedSummary, DocInput, SessionInput};
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use ndarray::Array2;
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
        Ok((text, ids.len(), out.len())) // raw; the caller cleans per job type
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
/// (ask + the longest turns) for the model to synthesize.
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
Repo: {}\nFiles changed: {}\nInitial request: {}\nMessages (chronological):\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        s.repo,
        files,
        s.ask,
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

/// Reduce a cluster's member-unit summaries into one theme summary. Clustering
/// is by summary embedding, so the members are genuinely on-theme.
fn prompt_for_topic(members: &[String]) -> String {
    let mut items = String::new();
    for m in members {
        items.push_str("- ");
        items.push_str(m);
        items.push('\n');
    }
    format!(
        "<|im_start|>user\nYou are describing a cluster of related engineering work for an index. \
From the one-line summaries of its items below, write ONE self-contained sentence (max 26 words) naming what this area is about and what was done across it. \
Name concrete subjects; do not just list the items. No preamble, no quotes, no lists.\n\n\
Items:\n{items}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
}

/// Name a topic with a short Title-Case heading, GROUNDED in the cluster's
/// distinctive terms and a few representative items (not just the reduced
/// summary). Concrete evidence makes the small model build the name from terms
/// the cluster actually uses, so it passes the faithfulness gate instead of
/// free-associating. (No example *name* is given — the 0.6B parrots those;
/// terms and items are data.)
fn prompt_for_topic_name(summary: &str, terms: &[String], examples: &[String]) -> String {
    let kw = terms.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
    let mut items = String::new();
    for e in examples {
        items.push_str("- ");
        items.push_str(e);
        items.push('\n');
    }
    format!(
        "<|im_start|>user\nName this cluster of engineering work with a short Title Case heading of 2 to 4 words, like a chapter title. Build it from the key terms and items below — prefer the most distinctive, specific nouns. Never use a repository or product name alone. Output only the title: no quotes, no period, no commas.\n\n\
Key terms: {kw}\nItems:\n{items}Description: {summary}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
}

/// Lowercased content words of one summary: ≥3 chars, not generic developer
/// vocabulary, and containing at least one letter (a bare issue number like
/// "955" is cluster-unique by accident, not a topic term). Shared by the term
/// scorer and the idf map so the two tokenizations cannot drift.
fn content_words(s: &str) -> std::collections::HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w) && w.chars().any(|c| c.is_alphabetic()))
        .map(str::to_string)
        .collect()
}

/// Smoothed inverse cluster frequency over every cluster's member summaries:
/// ln((1+N)/(1+cf)) + 1, where cf counts the clusters that mention the term at
/// all. Corpus-background words ("sie" shows up in 60 of 82 live clusters) keep
/// a small positive weight instead of zeroing out, and with a single cluster
/// the weights are uniform — ranking degrades to plain frequency.
fn idf_map(clusters: &[Vec<String>]) -> std::collections::HashMap<String, f64> {
    let mut cf: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for members in clusters {
        let mut seen = std::collections::HashSet::new();
        for m in members {
            seen.extend(content_words(m));
        }
        for w in seen {
            *cf.entry(w).or_default() += 1;
        }
    }
    let n = clusters.len() as f64;
    cf.into_iter().map(|(w, c)| (w, ((1.0 + n) / (1.0 + c as f64)).ln() + 1.0)).collect()
}

/// A cluster's most *distinctive* terms (roadmap I2/I3's c-TF-IDF): per-term
/// document frequency over the member summaries, weighted by the smoothed
/// inverse cluster frequency. Plain frequency made the corpus-wide background
/// vocabulary head almost every cluster's list — and a name gate that accepts
/// "SIE" everywhere stops nothing — so the terms must separate THIS cluster
/// from the others, not describe the whole corpus.
fn cluster_terms(members: &[String], k: usize, idf: &std::collections::HashMap<String, f64>) -> Vec<String> {
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in members {
        for w in content_words(m) {
            *freq.entry(w).or_default() += 1;
        }
    }
    let n = members.len().max(1) as f64;
    let mut v: Vec<(String, f64)> = freq
        .into_iter()
        .map(|(w, c)| {
            let weight = idf.get(&w).copied().unwrap_or(1.0);
            (w, c as f64 / n * weight)
        })
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|(w, _)| w).collect()
}

/// True if the generated name shares at least one distinctive term with the
/// cluster — otherwise it is about something the members are not, and is
/// rejected.
fn name_grounded(name: &str, terms: &[String]) -> bool {
    if name.is_empty() {
        return false;
    }
    let low = name.to_lowercase();
    let words: std::collections::HashSet<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 3).collect();
    terms.iter().any(|t| words.contains(t.as_str()))
}

/// Generic developer vocabulary that must not count as a cluster's term at all —
/// a name should match on a concrete subject, not "update"/"fix"/"feature".
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "into", "are", "has", "have",
    "had", "not", "added", "add", "adds", "fix", "fixes", "fixed", "update", "updates", "updated",
    "updating", "implement", "implemented", "implementing", "support", "new", "using", "use", "used",
    "via", "across", "their", "its", "which", "while", "when", "also", "now", "set", "get", "include",
    "includes", "including", "improve", "improved", "improving", "enhance", "enhanced", "enhancing",
    "project", "work", "feature", "features", "changes", "change", "code", "file", "files", "repo",
    "repository", "dependencies", "dependency", "data", "based", "various", "tools", "system",
    "feat", "chore", "subject", // commit-convention ceremony leaking from PR-title summaries
];

/// Meta-opener clauses the 0.6B stacks in front of summaries, generated as every
/// SUBJECT × VERB combination ("this area addresses …", "the project involves …",
/// "the cluster focuses on …") plus a bare "about". A generator beats a hand-list
/// because the model freely mixes subjects and verbs; built once, sorted
/// longest-first so the most specific prefix strips.
static OPENERS: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
    const SUBJECTS: &[&str] = &[
        "this theme", "the theme", "this cluster", "the cluster", "this area", "the area",
        "this project", "the project", "this work", "the work", "this group", "the group",
        "this collection", "the collection", "this set", "the set", "this topic", "the topic",
        "this effort", "the effort", "this",
    ];
    const VERBS: &[&str] = &[
        "focuses on", "focuses around", "is focused on", "centers on", "centers around",
        "is centered on", "revolves around", "is about", "is", "involves", "covers",
        "includes", "describes", "addresses", "explores", "examines", "deals with",
        "relates to", "pertains to", "concerns", "regards", "encompasses", "captures",
        "represents", "consists of", "comprises", "contains", "groups", "handles",
        "details", "documents", "tracks",
    ];
    let mut v: Vec<String> =
        SUBJECTS.iter().flat_map(|s| VERBS.iter().map(move |verb| format!("{s} {verb} "))).collect();
    v.push("about ".to_string());
    v.sort_by_key(|o| std::cmp::Reverse(o.len()));
    v
});

/// Drop a *complete* meta-opener clause including its verb ("This theme focuses
/// on …", "The theme is …") so the remainder stays grammatical, then
/// re-capitalize. Removing only "This theme " (without the verb) would leave a
/// dangling "is/includes …", so the whole clause must go.
fn strip_opener(s: &str) -> String {
    // Iterate: the model stacks openers ("The area is about developing …"), so
    // strip up to a few in a row. Capped to avoid eating real leading words.
    let mut s = s.to_string();
    for _ in 0..3 {
        let low = s.to_lowercase();
        let Some(op) = OPENERS.iter().find(|op| low.starts_with(op.as_str())) else { break };
        let rest = s[op.len()..].trim_start();
        let mut c = rest.chars();
        s = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
    }
    s
}

/// Strip any reasoning block, surrounding quotes, and extra lines; collapse to
/// one capped line.
fn clean(s: &str) -> String {
    let s = s.rsplit("</think>").next().unwrap_or(s);
    let s = s.trim().trim_matches('"').trim();
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or(s);
    let line = strip_opener(&crate::excerpt(line, 240));
    // Reject degenerate outputs that just echo a prompt field; the caller then
    // falls back to the extractive line.
    let low = line.to_lowercase();
    let echo = ["repo:", "files changed:", "initial request:", "messages"]
        .iter()
        .any(|p| low.starts_with(p));
    if line.len() < 15 || echo {
        return String::new();
    }
    line
}

/// Clean a short topic name. The 0.6B titles its summary well but SHOUTS in
/// all-caps and sometimes snake_cases, so *normalize* (underscores→spaces,
/// Title Case keeping acronyms) rather than reject. Only genuine non-titles —
/// comma-lists, whole sentences (>6 words), anything too long to be a heading
/// (a truncated title is not a title), hyphen-mush compounds, field echoes,
/// the bare generic word — are rejected, falling back to the keyword label.
fn clean_name(s: &str) -> String {
    let s = s.rsplit("</think>").next().unwrap_or(s).trim().trim_matches('"').trim();
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or(s).replace('_', " ");
    let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
    let line = line.trim_end_matches(['.', ':', ',']).trim();
    let low = line.to_lowercase();
    let echo = ["keyphrase", "items", "title", "repo", "update:", "description"].iter().any(|p| low.starts_with(p));
    let words = line.split_whitespace().count();
    let mush = line.split_whitespace().any(|w| w.matches('-').count() >= 3);
    if line.is_empty() || echo || line.contains(',') || !(1..=6).contains(&words) || line.chars().count() > 48 || mush {
        return String::new();
    }
    let t = title_case(line);
    let t = t.trim_end_matches(" Cluster").trim_end_matches("-Cluster").trim();
    // A bare generic word is no title — it carries no topic signal, so route it
    // to the (informative) summary fallback. Meaningful one-word names (NATS,
    // Azure, a model/brand) are NOT here; only words that name nothing.
    if t.is_empty()
        || matches!(t.to_lowercase().as_str(), "cluster" | "chores" | "update" | "fix" | "main" | "misc" | "various" | "general")
        || looks_like_slug(t)
    {
        String::new()
    } else {
        t.to_string()
    }
}

/// A mashed identifier rather than a prose title: a single whitespace-token
/// whose hyphen-separated pieces are ≥2 word-like parts ("Terraform-SIE-Update",
/// "SIE-Server", "MCP-Claude-Project"). The 0.6B emits these branch-name-style
/// under the "short heading" pressure; they read as slugs and (when the same
/// terms recur) collide into duplicate names. Reject to the summary fallback. A
/// spaced title is never a slug, and a version/code piece ("GPT-5.5" → "5.5",
/// "Florence-2" → "2") isn't a word, so codes survive.
fn looks_like_slug(name: &str) -> bool {
    if name.split_whitespace().count() != 1 {
        return false;
    }
    name.split('-').filter(|p| p.chars().any(|c| c.is_ascii_alphabetic())).count() >= 2
}

/// Domain acronyms kept uppercase in titles. The model SHOUTS, so acronyms
/// can't be inferred from case — hence the small curated allowlist.
const ACRO: &[&str] = &[
    "GPU", "CPU", "AWS", "GCP", "TEI", "API", "SIE", "NATS", "OCR", "CLI", "UI", "UX", "JSON",
    "LLM", "TLS", "VDB", "CTA", "SDK", "HTTP", "AI", "ML", "RAG", "SQL", "K8S", "GHCR", "TUI",
    "MTP", "PR", "CI", "CD", "BYO", "VM", "VMS", "CUDA", "JIT", "EKS", "GKE", "AKS", "GCS",
    "MTEB", "NPM", "VPC", "TTL", "DNS", "GPT", "MCP", "GTM", "GPC",
];
/// Canonical mixed-case brands — neither the acronym list (all-caps) nor the
/// code rule can produce "ColBERT", and the SHOUTed input carries no case to
/// preserve, so a curated map is the only source.
const BRAND: &[(&str, &str)] = &[
    ("colbert", "ColBERT"),
    ("deepseek", "DeepSeek"),
    ("github", "GitHub"),
    ("lora", "LoRA"),
    ("pypi", "PyPI"),
    ("qwen3", "Qwen3"),
    ("sglang", "SGLang"),
    ("vectorhub", "VectorHub"),
];
const STOP: &[&str] = &["and", "or", "for", "to", "the", "a", "an", "of", "in", "on", "at", "with", "by", "vs", "from", "into"];

/// Title-case a phrase: restore brand casing (ColBERT), keep known domain
/// acronyms (GPU, SIE, NATS…) and codes (S3, M4, GPT-5.5) uppercase, lowercase
/// function words mid-title, capitalize the rest. Hyphen/slash compounds are
/// cased per component ("sie-internal" → "SIE-Internal") — but only after the
/// whole-token checks, so a dotted code like GPT-5.5 still matches as one unit.
fn title_case(s: &str) -> String {
    s.split_whitespace()
        .enumerate()
        .map(|(i, w)| case_word(w, i == 0))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One whitespace token: whole-token brand/acronym/code first, then per
/// '-'/'/' component, then the stopword/capitalize default. `first` marks the
/// leading token, whose leading component never stop-lowercases.
fn case_word(w: &str, first: bool) -> String {
    if let Some(t) = case_core(w) {
        return t;
    }
    if !w.contains(['-', '/']) {
        return cap(w, first);
    }
    let mut out = String::new();
    let mut comp = String::new();
    for c in w.chars() {
        if c == '-' || c == '/' {
            let f = first && out.is_empty();
            out.push_str(&case_core(&comp).unwrap_or_else(|| cap(&comp, f)));
            out.push(c);
            comp.clear();
        } else {
            comp.push(c);
        }
    }
    let f = first && out.is_empty();
    out.push_str(&case_core(&comp).unwrap_or_else(|| cap(&comp, f)));
    out
}

/// Brand, acronym, or short numeric code (S3, M4, GPT-5.5) — the casings that
/// override the title rules. Brands run first: the code rule would otherwise
/// catch "qwen3" and SHOUT it.
fn case_core(w: &str) -> Option<String> {
    if w.is_empty() {
        return None;
    }
    if let Some((_, canon)) = BRAND.iter().find(|(b, _)| *b == w.to_lowercase()) {
        return Some((*canon).to_string());
    }
    let core: String = w.to_uppercase().chars().filter(|c| c.is_alphanumeric()).collect();
    let code = core.len() <= 6 && core.chars().any(|c| c.is_numeric()) && core.chars().all(|c| c.is_uppercase() || c.is_numeric());
    if ACRO.contains(&core.as_str()) || code {
        return Some(w.to_uppercase());
    }
    None
}

fn cap(w: &str, first: bool) -> String {
    if !first && STOP.contains(&w.to_lowercase().as_str()) {
        return w.to_lowercase();
    }
    let mut ch = w.chars();
    match ch.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &ch.as_str().to_lowercase(),
        None => String::new(),
    }
}

/// 8-byte content hash of arbitrary parts, salted by the prompt version (a
/// prompt change invalidates the cache) and by the summarizer model when it
/// is not the default — different models must not share fleet summaries. The
/// default is pinned to an empty salt, so this costs existing caches nothing;
/// a deliberate default-model change comes with retuned prompts, i.e. a
/// PROMPT_VERSION bump, which regenerates everything anyway.
fn hash_parts(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    h.update(PROMPT_VERSION.as_bytes());
    let spec = std::env::var("SYNTY_LLM").unwrap_or_else(|_| REPO.to_string());
    if spec != REPO {
        h.update(spec.as_bytes());
    }
    for p in parts {
        h.update(b"\0");
        h.update(p.as_bytes());
    }
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn input_hash(s: &SessionInput) -> String {
    hash_parts(&[&s.ask, &s.files.join(","), &s.turns.join("\n")])
}

fn doc_hash(d: &DocInput) -> String {
    hash_parts(&[d.kind, &d.title, &d.text])
}

/// One unit of summarization work: a cache key, its input hash, and the prompt.
/// `short` jobs are topic names (cleaned as a title, not a sentence). `gate`,
/// when set, holds the cluster's distinctive terms: a name sharing none of them
/// is rejected — this is what stops "Colpali" on error-handling PRs or "Update
/// Dependencies" on synty work. `ban` holds the topic's normalized repo slugs
/// and their components: the repo is already a facet on every surface, so a
/// "name" that is just a repo fragment carries nothing. Either rejection falls
/// back to the extractive keyword label, so a topic never ends up titled by a
/// whole sentence.
struct Job {
    key: String,
    hash: String,
    prompt: String,
    label: String,
    short: bool,
    gate: Option<Vec<String>>,
    ban: Vec<String>,
    /// For a name job, the topic summary it titles — the fallback when the
    /// model's own title is rejected (its lead clause beats keyword tokens).
    summary: Option<String>,
}

/// Salt for the topic reduce / name prompts; bump to regenerate them only.
/// t8: reduce inputs ordered by centrality, medoid first (the 0.6B attends to
/// early tokens, so the theme should lead, not the most recent member).
/// s4: distinctive (c-TF-IDF) terms in the prompt and gate, keyword fallback,
/// repo-slug ban — and the name's hash now covers only the summary it titles,
/// so a version bump is the one lever that forces a fleet-wide name refresh
/// after term-algorithm changes.
/// s5: centrality-ordered examples + the embedding-faithfulness gate; isolates
/// these names from machines still generating ungated s4 ones.
const TOPIC_PROMPT_VERSION: &str = "t8";
const TOPIC_NAME_VERSION: &str = "s12";

/// Unit jobs: one per session and per PR/issue.
fn unit_jobs() -> Result<Vec<Job>> {
    let sessions = units::session_inputs()?;
    let docs = units::doc_inputs()?;
    let mut jobs = Vec::with_capacity(sessions.len() + docs.len());
    for s in &sessions {
        jobs.push(Job { key: s.id.clone(), hash: input_hash(s), prompt: prompt_for(s), label: crate::short(&s.id), short: false, gate: None, ban: Vec::new(), summary: None });
    }
    for d in &docs {
        jobs.push(Job { key: d.key.clone(), hash: doc_hash(d), prompt: prompt_for_doc(d), label: d.key.clone(), short: false, gate: None, ban: Vec::new(), summary: None });
    }
    Ok(jobs)
}

/// Representative member lines for the name prompt: the most central members
/// (the input arrives medoid-first) that are real sentences — the degenerate
/// slug echoes ("sie-internal: #955") prime the small model to answer in
/// slugs, so anything under 5 words is skipped, falling back to the raw list
/// when a tiny cluster has nothing better.
fn pick_examples(members: &[String]) -> Vec<String> {
    let wf: Vec<&String> = members.iter().filter(|s| s.split_whitespace().count() >= 5).collect();
    let pick: Vec<&String> = if wf.len() < 3 { members.iter().collect() } else { wf };
    pick.iter().take(3).map(|s| crate::excerpt(s, 90)).collect()
}

/// Topic jobs: a one-line summary reduced from the member-unit summaries, plus a
/// short name that *titles that summary* (a small model shortens a sentence far
/// better than it abstracts a name from a list of items). The summary's hash is
/// over its members; the name's is over the summary text it titles — NOT the
/// terms or examples, which shift whenever any other cluster does (idf is
/// global): a name regenerates exactly when the summary it titles changes, and
/// a TOPIC_NAME_VERSION bump forces the rest.
fn topic_jobs() -> Result<Vec<Job>> {
    // First pass: every cluster's member summaries (capped exactly as the jobs
    // consume them), so terms can be weighted by cross-cluster rarity. Members
    // are ordered by centrality (medoid first, from `cluster`'s rank; stable
    // sort keeps recency among unranked) — the 0.6B attends to early tokens,
    // so the reduce prompt and the examples lead with the theme, not with
    // whatever happens to be most recent.
    let topics = units::topic_units(12)?;
    let memberships: Vec<Vec<String>> = topics
        .iter()
        .map(|t| {
            let mut ordered: Vec<&units::Unit> = t.units.iter().collect();
            // One rerun's summary is enough: collapsed duplicates would pad the
            // reduce prompt with the same line and skew the idf and terms.
            ordered.retain(|u| !u.dup);
            ordered.sort_by_key(|u| u.rank);
            ordered.iter().filter_map(|u| u.summary.clone()).take(40).collect()
        })
        .collect();
    let populated: Vec<Vec<String>> = memberships.iter().filter(|m| !m.is_empty()).cloned().collect();
    let idf = idf_map(&populated);

    let mut jobs = Vec::new();
    for (t, members) in topics.iter().zip(&memberships) {
        if members.is_empty() {
            continue;
        }
        let mut sorted: Vec<&str> = members.iter().map(|s| s.as_str()).collect();
        sorted.sort_unstable();
        let mut sum_parts = vec![TOPIC_PROMPT_VERSION];
        sum_parts.extend(sorted.iter().copied());
        jobs.push(Job {
            key: units::topic_key(&t.cache_key),
            hash: hash_parts(&sum_parts),
            prompt: prompt_for_topic(members),
            label: format!("topic:{}", t.id),
            short: false,
            gate: None,
            ban: Vec::new(),
            summary: None,
        });
        // The name titles the topic summary, so it depends on it. The summary job
        // above runs first in the same pass; for a topic whose summary isn't
        // cached yet the name regenerates next run, once the summary exists. The
        // gate carries the cluster's distinctive terms and `ban` its repo slugs,
        // so an off-theme or repo-echo name falls back — to the summary's lead
        // clause, kept on the job, or the keyword label as a last resort.
        if let Some(sum) = &t.summary {
            let terms = cluster_terms(members, 12, &idf);
            let examples = pick_examples(members);
            jobs.push(Job {
                key: units::topic_name_key(&t.cache_key),
                hash: hash_parts(&[TOPIC_NAME_VERSION, sum.as_str()]),
                prompt: prompt_for_topic_name(sum, &terms, &examples),
                label: format!("name:{}", t.id),
                short: true,
                gate: Some(terms),
                ban: repo_ban(&t.repos),
                summary: Some(sum.clone()),
            });
        }
    }
    Ok(jobs)
}

/// Jobs whose cached hash is missing or stale, narrowed by the debug knobs
/// `SYNTY_LLM_ONLY=<substr,...>` and `SYNTY_LLM_LIMIT=N`, then shuffled with a
/// per-machine seed: viewers working the same fleet-wide pending list start
/// from different ends, so concurrent summarize passes divide the units
/// between them (the write-once store de-duplicates the overlap) instead of
/// generating the same ones in the same order.
fn pending<'a>(jobs: &'a [Job], cache: &units::SummaryCache) -> Vec<&'a Job> {
    let mut todo: Vec<&Job> = jobs.iter().filter(|j| cache.get(&j.key).map(|c| c.hash != j.hash).unwrap_or(true)).collect();
    if let Ok(only) = std::env::var("SYNTY_LLM_ONLY") {
        let want: Vec<String> = only.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        todo.retain(|j| want.iter().any(|w| j.key.contains(w.as_str())));
    }
    if let Ok(n) = std::env::var("SYNTY_LLM_LIMIT") {
        if let Ok(n) = n.parse::<usize>() {
            todo.truncate(n);
        }
    }
    shuffle(&mut todo, crate::index::fnv1a(crate::identity::machine_id().as_bytes()));
    todo
}

/// Deterministic Fisher–Yates with an xorshift stream — vary the seed, vary
/// the order; no RNG dependency.
fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut s = seed | 1;
    for i in (1..v.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.swap(i, (s as usize) % (i + 1));
    }
}

/// Lowercase alphanumeric skeleton of a title or repo slug, so "SIE-Internal",
/// "sie_internal", and "sie-internal" all compare equal.
fn normalize_slug(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

/// The banned names for a topic: each repo slug plus its components ("sie-web"
/// bans "sieweb", "sie", and "web"). A name that IS a repo fragment carries
/// nothing the repos facet doesn't already show — and the components double as
/// the skip-list that keeps repo fragments out of the keyword fallback.
fn repo_ban(repos: &[String]) -> Vec<String> {
    let mut ban = Vec::new();
    for r in repos {
        ban.push(normalize_slug(r));
        ban.extend(r.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|c| c.len() >= 3).map(str::to_string));
    }
    ban.sort_unstable();
    ban.dedup();
    ban
}

/// Extractive fallback title: the cluster's top distinctive terms, title-cased
/// and joined (roadmap I2's keyword-join label) — grounded by construction, so
/// a topic whose LLM name is rejected still gets a name-shaped title instead
/// of surfacing a whole sentence. Banned (repo-slug) terms are skipped lest
/// the fallback reproduce the very name the ban rejected.
fn keyword_label(terms: &[String], ban: &[String]) -> String {
    let kw: Vec<String> = terms
        .iter()
        .filter(|t| !ban.contains(&normalize_slug(t)))
        .take(3)
        .map(|t| title_case(t))
        .collect();
    crate::excerpt(&kw.join(" · "), 48)
}

/// Most-central members scored per topic by the embedding gate — the head of
/// the centrality-ordered list represents the theme; a hub's periphery adds
/// nothing but noise.
const GATE_MEMBERS: usize = 20;
/// Run-relative faithfulness floor: an LLM name scoring below this fraction of
/// the run's median is rejected (mirrors the clustering FLOOR). Run-relative
/// because MaxSim is non-metric — absolute thresholds don't transfer across
/// corpora or models (see quality-roadmap, cross-cutting tradeoffs).
const EMBED_FLOOR: f32 = 0.6;
/// Below this many scored names the median is noise — apply no rejections.
const EMBED_MIN_SCORED: usize = 8;

/// Length-normalized MaxSim of a name against its topic's members: mean over
/// members of (per-name-token best dot, summed) ÷ name tokens. A hallucinated
/// token finds no good match in any member and drags the mean down, while the
/// length norm keeps 2-word and 5-word names comparable.
fn name_score(name: &Array2<f32>, members: &[Array2<f32>]) -> f32 {
    if name.nrows() == 0 || members.is_empty() {
        return 0.0;
    }
    let s: f32 = members.iter().map(|m| crate::topics::maxsim(name, m)).sum();
    s / members.len() as f32 / name.nrows() as f32
}

/// Which of the (index, score) pairs fail the run-relative floor.
fn embed_rejects(scores: &[(usize, f32)]) -> Vec<usize> {
    if scores.len() < EMBED_MIN_SCORED {
        return Vec::new();
    }
    let mut v: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
    v.sort_by(f32::total_cmp);
    let med = v[v.len() / 2];
    scores.iter().filter(|(_, s)| *s < EMBED_FLOOR * med).map(|(i, _)| *i).collect()
}

// ── self-calibrating name gate (SYNTY_NAME_GATE=selfcal) ──────────────────
// The run-relative median floor above is a no-op in practice (a whole run of
// mediocre names floats the median, so almost nothing clears 0.6× it) and the
// real rejecting is done by the cruder unigram `name_grounded`. The
// self-calibrating gate judges each name against ITS OWN cluster's coherence:
// a faithful abstractive title scores at least as well as the cluster's
// least-aligned member, while a hallucinated token falls below all of them.
// Sound because vectors are L2-normalized, so name_score is a bounded
// mean-best-match cosine (see evals/tuning.md).

/// A name whose mean-best-match cosine to its members falls below this is
/// off-theme regardless of corpus — the floor-of-the-floor against a
/// degenerate (all-low-coherence) cluster passing garbage.
const ABS_FLOOR: f32 = 0.5;
/// Reject below this fraction of the cluster's low-coherence tail (COH_PCTILE).
const COH_BETA: f32 = 0.85;
/// The tail percentile the name must clear — the cluster's least-aligned member.
const COH_PCTILE: usize = 10;
/// Below this many members the percentile is noise → use ABS_FLOOR alone.
const MIN_MEMBERS_FOR_COH: usize = 5;

/// One name's faithfulness verdict against its cluster's members.
struct NameVerdict {
    name_score: f32,
    coh_p10: f32,
    members: usize,
    reject: bool,
}

/// Each member's leave-one-out faithfulness to the rest of the cluster — the
/// distribution a faithful name should sit within.
fn member_coherences(members: &[Array2<f32>]) -> Vec<f32> {
    (0..members.len())
        .map(|i| {
            let others: Vec<Array2<f32>> =
                members.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, m)| m.clone()).collect();
            name_score(&members[i], &others)
        })
        .collect()
}

/// The p-th percentile (0..=100, nearest-rank) of a score list. Empty → 0.
fn percentile(scores: &[f32], p: usize) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut v = scores.to_vec();
    v.sort_by(f32::total_cmp);
    let idx = ((p * (v.len() - 1)) + 50) / 100;
    v[idx.min(v.len() - 1)]
}

/// Score a name against its cluster and decide faithfulness self-relative to the
/// cluster's own coherence (reused by the gate and by `synty eval --names`).
fn score_name(name: &Array2<f32>, members: &[Array2<f32>]) -> NameVerdict {
    let ns = name_score(name, members);
    let coh = member_coherences(members);
    let p10 = percentile(&coh, COH_PCTILE);
    // Tiny clusters: the percentile is noise — lean on the absolute floor only.
    let floor = if members.len() < MIN_MEMBERS_FOR_COH { ABS_FLOOR } else { ABS_FLOOR.max(COH_BETA * p10) };
    NameVerdict { name_score: ns, coh_p10: p10, members: members.len(), reject: ns < floor }
}

/// Top-GATE_MEMBERS member embeddings for a topic, loaded from the store
/// (centrality-ordered, dup-excluded by `topic_member_embed_hashes`). Missing
/// vectors (summaries drifted since `cluster` encoded them) are dropped.
fn load_members(store: &crate::store::EmbStore, hashes: &[u64]) -> Result<Vec<Array2<f32>>> {
    let mut mem = Vec::new();
    for h in hashes.iter().take(GATE_MEMBERS) {
        if let Some(e) = store.get(*h)? {
            mem.push(e);
        }
    }
    Ok(mem)
}

/// The self-calibrating gate is the default semantic backstop (each name judged
/// against its own cluster's coherence); the old run-relative median floor — a
/// near no-op — stays reachable as `SYNTY_NAME_GATE=median` for one release.
/// This is a backstop AFTER clean_name + name_grounded, which the name eval
/// showed still earn their keep (name_grounded catches mashed-token garbage the
/// embedding gate can't — a mashed token embeds on-theme; see evals/tuning.md).
fn selfcal_gate() -> bool {
    std::env::var("SYNTY_NAME_GATE").as_deref() != Ok("median")
}

/// I2's embedding-faithfulness gate, over every valid LLM topic name — the
/// unigram gate cannot see a hallucinated half ("Stablebridge Dashboard" on a
/// dashboard cluster). Each name is embedded with the retrieval encoder
/// (content-addressed, so usually a store hit) and MaxSim-scored against the
/// member embeddings `cluster` already produced; run-relative outliers are
/// replaced with the keyword label in the LOCAL cache only. The write-once
/// fleet store keeps the raw generation — every machine applies the same
/// deterministic verdict, so corrections converge without coordination.
/// Returns (scored, rejected).
fn embed_gate_names(tjobs: &[Job], cache: &mut units::SummaryCache, bucket: &str) -> Result<(usize, usize)> {
    let model = crate::model_id();
    let store = crate::store::EmbStore::open(bucket, &model)?;
    let member_hashes = units::topic_member_embed_hashes()?;

    // Candidates: cached LLM-authored names. Fallback names (summary lead
    // clause or keyword label) are extractive — grounded by construction — and
    // skipped.
    let mut cand: Vec<(&Job, String)> = Vec::new();
    for j in tjobs.iter().filter(|j| j.short) {
        let Some(c) = cache.get(&j.key) else { continue };
        if c.summary.is_empty() || c.summary == fallback_name(j) {
            continue;
        }
        cand.push((j, c.summary.clone()));
    }
    if cand.is_empty() {
        return Ok((0, 0));
    }

    let mut embs: Vec<Option<Array2<f32>>> = Vec::with_capacity(cand.len());
    for (_, name) in &cand {
        embs.push(store.get(crate::index::fnv1a(name.as_bytes()))?);
    }
    let miss: Vec<usize> = (0..cand.len()).filter(|&i| embs[i].is_none()).collect();
    if !miss.is_empty() {
        let mut enc = crate::encode::Encoder::load(&model)?;
        let texts: Vec<String> = miss.iter().map(|&i| cand[i].1.clone()).collect();
        for (&i, e) in miss.iter().zip(enc.encode_docs(&texts)?) {
            store.put(crate::index::fnv1a(cand[i].1.as_bytes()), &e)?;
            embs[i] = Some(e);
        }
    }

    let selfcal = selfcal_gate();
    let mut scores: Vec<(usize, f32)> = Vec::new();
    let mut verdicts: Vec<(usize, NameVerdict)> = Vec::new();
    for (i, (j, _)) in cand.iter().enumerate() {
        let stable = j.key.strip_prefix("topicname:").unwrap_or(&j.key);
        let Some(hashes) = member_hashes.get(stable) else { continue };
        let mem = load_members(&store, hashes)?;
        if mem.is_empty() {
            continue; // summaries drifted since `cluster` encoded them — abstain
        }
        if let Some(e) = &embs[i] {
            let v = score_name(e, &mem);
            scores.push((i, v.name_score));
            verdicts.push((i, v));
        }
    }

    // Debug visibility, like the cluster quality report: the lowest-scoring
    // names and (self-cal) each one's per-cluster floor — the tuning data.
    if !scores.is_empty() {
        let mut by_score = scores.clone();
        by_score.sort_by(|a, b| a.1.total_cmp(&b.1));
        let med = by_score[by_score.len() / 2].1;
        let floor = if selfcal { "per-cluster coh".to_string() } else { format!("{:.2}", EMBED_FLOOR * med) };
        eprintln!("  name gate ({}): {} scored, median {med:.2}, floor {floor} — lowest:", if selfcal { "selfcal" } else { "median" }, scores.len());
        for (i, s) in by_score.iter().take(5) {
            let p10 = verdicts.iter().find(|(x, _)| x == i).map(|(_, v)| v.coh_p10).unwrap_or(0.0);
            eprintln!("    [{s:.2} vs coh_p10 {p10:.2}] {}", cand[*i].1);
        }
    }

    let rejects: Vec<usize> = if selfcal {
        verdicts.iter().filter(|(_, v)| v.reject).map(|(i, _)| *i).collect()
    } else {
        embed_rejects(&scores)
    };
    for &i in &rejects {
        let (j, name) = &cand[i];
        let fallback = fallback_name(j);
        let score = scores.iter().find(|(x, _)| *x == i).map(|(_, s)| *s).unwrap_or(0.0);
        eprintln!("  name gate: “{name}” unfaithful to its members ({score:.2}) → “{fallback}”");
        if let Some(c) = cache.get_mut(&j.key) {
            c.summary = fallback;
        }
    }
    Ok((scores.len(), rejects.len()))
}

/// One topic's row in the name-quality eval (`synty eval --names`): the data
/// to eyeball/rate, plus the self-cal verdict. `rating` is human-filled.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(crate) struct NameRow {
    pub key: String,
    pub name: String,
    pub summary: String,
    pub members: Vec<String>,
    pub gate_terms: Vec<String>,
    pub name_score: f32,
    pub coh_p10: f32,
    pub member_n: usize,
    pub scored: bool,
    pub is_fallback: bool,
    pub would_reject: bool,
    #[serde(default)]
    pub rating: Option<String>,
}

/// Score every topic's cached name against its cluster's members — the
/// read-only data behind `synty eval --names`. Mirrors `embed_gate_names`'
/// loading and scoring exactly (shared `score_name`), so the report reflects
/// what the gate would do.
pub(crate) fn eval_names(bucket: &str) -> Result<Vec<NameRow>> {
    let model = crate::model_id();
    let store = crate::store::EmbStore::open(bucket, &model)?;
    let member_hashes = units::topic_member_embed_hashes()?;
    let cache = units::load_summary_cache();
    let jobs = topic_jobs()?;

    // Member summaries (for the eyeball dump), by stable cache key.
    let mut member_sums: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for t in units::topic_units(12)? {
        let sums: Vec<String> =
            t.units.iter().filter(|u| !u.dup).filter_map(|u| u.summary.clone()).take(6).collect();
        member_sums.insert(t.cache_key, sums);
    }

    // Candidate names (cached LLM names + fallbacks), embedded store-first with
    // a single batched encode for any misses (same as the gate).
    let cands: Vec<(&Job, String, bool)> = jobs
        .iter()
        .filter(|j| j.short)
        .filter_map(|j| {
            let c = cache.get(&j.key)?;
            (!c.summary.is_empty()).then(|| (j, c.summary.clone(), c.summary == fallback_name(j)))
        })
        .collect();
    let mut embs: Vec<Option<Array2<f32>>> = Vec::with_capacity(cands.len());
    for (_, name, _) in &cands {
        embs.push(store.get(crate::index::fnv1a(name.as_bytes()))?);
    }
    let miss: Vec<usize> = (0..cands.len()).filter(|&i| embs[i].is_none()).collect();
    if !miss.is_empty() {
        let mut enc = crate::encode::Encoder::load(&model)?;
        let texts: Vec<String> = miss.iter().map(|&i| cands[i].1.clone()).collect();
        for (&i, e) in miss.iter().zip(enc.encode_docs(&texts)?) {
            store.put(crate::index::fnv1a(cands[i].1.as_bytes()), &e)?;
            embs[i] = Some(e);
        }
    }

    let mut rows = Vec::new();
    for (i, (j, name, is_fallback)) in cands.iter().enumerate() {
        let stable = j.key.strip_prefix("topicname:").unwrap_or(&j.key).to_string();
        let mem = member_hashes.get(&stable).map(|h| load_members(&store, h)).transpose()?.unwrap_or_default();
        let (ns, p10, scored, would) = match (&embs[i], mem.is_empty()) {
            (Some(e), false) => {
                let v = score_name(e, &mem);
                (v.name_score, v.coh_p10, true, v.reject)
            }
            _ => (0.0, 0.0, false, false),
        };
        rows.push(NameRow {
            members: member_sums.get(&stable).cloned().unwrap_or_default(),
            key: stable,
            name: name.clone(),
            summary: j.summary.clone().unwrap_or_default(),
            gate_terms: j.gate.clone().unwrap_or_default(),
            name_score: ns,
            coh_p10: p10,
            member_n: mem.len(),
            scored,
            is_fallback: *is_fallback,
            would_reject: would,
            rating: None,
        });
    }
    Ok(rows)
}

/// Would this name be rejected at the given BETA (for the eval's sensitivity
/// sweep)? Mirrors `score_name`'s floor with a variable BETA.
pub(crate) fn rejects_at(name_score: f32, coh_p10: f32, members: usize, beta: f32) -> bool {
    let floor = if members < MIN_MEMBERS_FOR_COH { ABS_FLOOR } else { ABS_FLOOR.max(beta * coh_p10) };
    name_score < floor
}

/// Public for the name eval's slug-count aggregate.
pub(crate) fn is_slug(name: &str) -> bool {
    looks_like_slug(name)
}

/// A short heading distilled from a topic's one-line summary: the lead clause,
/// title-cased. The fallback for a rejected name — the 0.6B writes a coherent
/// summary even when its compressed-to-a-title attempt drifts or echoes a slug,
/// so "Updating Terraform Configurations" beats the keyword soup
/// "Terraform · SIE · Update". Empty when the summary is too thin to title;
/// the caller then drops to the keyword label.
fn summary_heading(summary: &str) -> String {
    let clause = summary.trim().split(|c| matches!(c, ',' | ';' | ':' | '.')).next().unwrap_or("").trim();
    let mut words: Vec<&str> = clause.split_whitespace().take(8).collect();
    // Capping mid-clause can leave a dangling function word ("…Tracking
    // Through", "…Exhaustion By"); trim any trailing preposition/conjunction/
    // copula so the heading ends on a content word. (Content-word truncations
    // like "…a Self-Contained" can't be detected here — that's the cap's edge.)
    const TRAIL: &[&str] = &[
        "and", "or", "the", "a", "an", "to", "of", "for", "with", "in", "on", "at", "by", "via",
        "from", "into", "through", "then", "that", "which", "was", "were", "is", "are", "as", "using",
    ];
    while words.last().map(|w| TRAIL.contains(&w.to_lowercase().as_str())).unwrap_or(false) {
        words.pop();
    }
    if words.len() < 2 {
        return String::new();
    }
    title_case(&words.join(" "))
}

/// The name a topic falls back to when its generated title is rejected: the
/// summary's lead clause if there is a usable summary, else the extractive
/// keyword label. Shared by `finish` and the embedding gate so the rejection
/// path is identical wherever it fires.
fn fallback_name(j: &Job) -> String {
    if let Some(sum) = &j.summary {
        let h = summary_heading(sum);
        if !h.is_empty() {
            return h;
        }
    }
    keyword_label(j.gate.as_deref().unwrap_or_default(), &j.ban)
}

/// Clean a raw generation per the job type and apply the faithfulness gates: a
/// topic name must be well-formed, share a distinctive term with its cluster,
/// and not be a bare repo slug — otherwise it falls back (to the summary's lead
/// clause, else the keyword label).
fn finish(j: &Job, raw: &str) -> String {
    if !j.short {
        return clean(raw);
    }
    let name = clean_name(raw);
    let ok = !name.is_empty()
        && j.gate.as_ref().map(|terms| name_grounded(&name, terms)).unwrap_or(true)
        && !j.ban.contains(&normalize_slug(&name));
    if ok {
        name
    } else {
        fallback_name(j)
    }
}

/// How many model workers to run. `SYNTY_LLM_WORKERS` wins; Metal builds
/// default to 1 (one GPU — parallel workers contend, not compound); CPU
/// defaults to cores/4 (each candle matmul already spreads over ~4 threads),
/// capped so the f32 weights (~2.4 GB per worker) stay reasonable.
fn worker_count(env: Option<usize>, metal: bool, cores: usize, jobs: usize) -> usize {
    let def = if metal { 1 } else { (cores / 4).max(1).min(4) };
    env.unwrap_or(def).clamp(1, 8).min(jobs.max(1))
}

fn llm_workers(jobs: usize) -> usize {
    worker_count(
        std::env::var("SYNTY_LLM_WORKERS").ok().and_then(|v| v.parse().ok()),
        cfg!(feature = "metal"),
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4),
        jobs,
    )
}

/// Resolve one job: the fleet store first (another viewer may have generated
/// it — a GET beats a generation), the model only on a store miss, sharing the
/// result back. Returns (summary, prompt_tok, out_tok, from_store) — or None
/// when generation failed, in which case nothing is cached or shared: a
/// transient model failure must not fleet-persist a fallback as if the model
/// had really been consulted. The next run retries.
fn resolve(j: &Job, llm: &mut Summarizer, store: &crate::store::SummaryStore) -> Option<(String, usize, usize, bool)> {
    if let Ok(Some(s)) = store.get(&j.key, &j.hash) {
        return Some((s, 0, 0, true));
    }
    let (raw, pt, ot) = match llm.generate(&j.prompt) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  {} failed: {e}", j.label);
            return None;
        }
    };
    let summary = finish(j, &raw);
    let _ = store.put(&j.key, &j.hash, &summary);
    Some((summary, pt, ot, false))
}

/// Generate `todo`, updating and periodically persisting the cache. One worker
/// runs in place (reusing `llm` across passes); more fan out over scoped
/// threads, each owning its own model, pulling jobs from a shared counter.
fn run_jobs(
    todo: &[&Job],
    cache: &mut units::SummaryCache,
    llm: &mut Option<Summarizer>,
    store: &crate::store::SummaryStore,
    kind: &str,
) -> Result<(usize, usize)> {
    let workers = llm_workers(todo.len());
    if workers <= 1 {
        if llm.is_none() {
            *llm = Some(Summarizer::load()?);
        }
        let llm = llm.as_mut().expect("summarizer loaded");
        let (mut in_tok, mut out_tok) = (0usize, 0usize);
        for (n, j) in todo.iter().enumerate() {
            if let Some((summary, pt, ot, fetched)) = resolve(j, llm, store) {
                in_tok += pt;
                out_tok += ot;
                let tag = if fetched { " (fleet)" } else { "" };
                eprintln!("  [{kind} {}/{}] {}{tag} — {}", n + 1, todo.len(), j.label, summary);
                cache.insert(j.key.clone(), CachedSummary { hash: j.hash.clone(), summary });
                if (n + 1) % 10 == 0 {
                    units::save_summary_cache(cache)?;
                }
            }
            crate::progress::phase(&format!("summarizing {kind}s"), n + 1, todo.len());
        }
        return Ok((in_tok, out_tok));
    }

    eprintln!("summarize: {workers} model workers");
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let shared = std::sync::Mutex::new(&mut *cache);
    let mut totals = (0usize, 0usize);
    std::thread::scope(|s| -> Result<()> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| -> Result<(usize, usize)> {
                    let mut llm = Summarizer::load()?;
                    let (mut in_tok, mut out_tok) = (0usize, 0usize);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(j) = todo.get(i) else { break };
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        crate::progress::phase(&format!("summarizing {kind}s"), n, todo.len());
                        let Some((summary, pt, ot, fetched)) = resolve(j, &mut llm, store) else { continue };
                        in_tok += pt;
                        out_tok += ot;
                        let tag = if fetched { " (fleet)" } else { "" };
                        eprintln!("  [{kind} {n}/{}] {}{tag} — {}", todo.len(), j.label, summary);
                        let mut c = shared.lock().expect("cache lock");
                        c.insert(j.key.clone(), CachedSummary { hash: j.hash.clone(), summary });
                        if n % 10 == 0 {
                            units::save_summary_cache(&c)?;
                        }
                    }
                    Ok((in_tok, out_tok))
                })
            })
            .collect();
        for h in handles {
            let (i, o) = h.join().expect("summary worker")?;
            totals = (totals.0 + i, totals.1 + o);
        }
        Ok(())
    })?;
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_drops_field_echoes() {
        assert_eq!(clean("repo: synty"), ""); // field echo → extractive fallback
        assert!(clean("Integrated MinerU into the SIE server.").contains("MinerU"));
    }

    #[test]
    fn strip_opener_removes_full_clause_grammatically() {
        assert_eq!(strip_opener("This theme focuses on enhancing OCR adapters."), "Enhancing OCR adapters.");
        assert_eq!(strip_opener("The theme includes updating trends and a binary."), "Updating trends and a binary.");
        assert_eq!(strip_opener("The theme is the implementation of a cache."), "The implementation of a cache.");
        assert_eq!(strip_opener("This cluster focuses on sandbox provisioning."), "Sandbox provisioning.");
        assert_eq!(strip_opener("This area involves Terraform state locking."), "Terraform state locking.");
        assert_eq!(strip_opener("About designing the inference deck."), "Designing the inference deck.");
        assert_eq!(strip_opener("The area is about developing an RLM in a box."), "Developing an RLM in a box.");
        // Combinations the hand-list missed: subject "project", verb "addresses",
        // and the "the cluster" variant.
        assert_eq!(strip_opener("This area addresses image-quality regressions."), "Image-quality regressions.");
        assert_eq!(strip_opener("The project involves a native session tracker."), "A native session tracker.");
        assert_eq!(strip_opener("The cluster focuses on NATS routing and health checks."), "NATS routing and health checks.");
        // Real leading words that merely start like an opener are kept.
        assert_eq!(strip_opener("Integrated MinerU into SIE."), "Integrated MinerU into SIE.");
        assert_eq!(strip_opener("This binary indexes documents with ColBERT."), "This binary indexes documents with ColBERT.");
    }

    // The faithfulness gate rejects a name that shares no distinctive term with
    // its cluster, but keeps one that does. With a single cluster the idf is
    // uniform, so ranking degrades to plain frequency — the smoothing guard.
    #[test]
    fn name_gate_rejects_off_theme() {
        let members = vec![
            "Added InputTooLongError and an overflow policy to the extract SDK".to_string(),
            "gliclass returns all label scores and bounds input length".to_string(),
        ];
        let idf = idf_map(std::slice::from_ref(&members));
        let terms = cluster_terms(&members, 12, &idf);
        assert!(name_grounded("Overflow Policy Handling", &terms)); // shares "overflow"/"policy"
        assert!(!name_grounded("Colpali Visual Retrieval", &terms)); // shares nothing
        assert!(!name_grounded("Update Dependencies", &terms)); // generic words, not cluster terms
    }

    // A term every cluster shares ("sie") must not outrank a cluster's own
    // vocabulary, even when it tops the raw frequency count — four live topics
    // were all named after the corpus-wide product because of this.
    #[test]
    fn cluster_terms_demote_background_words() {
        let mk = |lines: &[&str]| lines.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let a = mk(&["sie voyage reranker benchmark", "sie voyage cohere reranker", "sie cohere benchmark"]);
        let b = mk(&["sie terraform helm provisioning", "sie helm chart", "sie terraform destroy"]);
        let c = mk(&["sie nats queue routing", "sie nats telemetry", "sie queue router"]);
        let idf = idf_map(&[a.clone(), b.clone(), c.clone()]);
        let top = cluster_terms(&a, 3, &idf);
        assert!(!top.contains(&"sie".to_string()), "background word must be demoted: {top:?}");
        assert!(top.contains(&"voyage".to_string()) || top.contains(&"reranker".to_string()), "cluster vocabulary must lead: {top:?}");
        // Pure numbers are not topic terms, however cluster-unique.
        let nums = mk(&["closes #955 in the worker", "follow-up to #955 cleanup", "tracked under #955 still"]);
        let idf2 = idf_map(std::slice::from_ref(&nums));
        assert!(!cluster_terms(&nums, 12, &idf2).contains(&"955".to_string()));
    }

    // A name that fails the gate or cleans to nothing yields the extractive
    // keyword title when there's no usable summary — never an empty name.
    #[test]
    fn finish_falls_back_to_keywords() {
        let j = Job {
            key: "topicname:x".into(),
            hash: String::new(),
            prompt: String::new(),
            label: String::new(),
            short: true,
            gate: Some(vec!["voyage".into(), "reranker".into(), "cohere".into(), "benchmark".into()]),
            ban: vec!["sieinternal".into()],
            summary: None,
        };
        assert_eq!(finish(&j, "Colpali Visual Retrieval"), "Voyage · Reranker · Cohere"); // off-theme
        assert_eq!(finish(&j, "Sharing models, cutting costs."), "Voyage · Reranker · Cohere"); // not a title
        assert_eq!(finish(&j, "Voyage Reranker Benchmarks"), "Voyage Reranker Benchmarks"); // grounded → kept
    }

    // When a summary is on the job, a rejected name falls back to the summary's
    // lead clause (a real phrase) rather than the keyword soup — the soup is
    // the last resort, only when the summary is too thin to title.
    #[test]
    fn finish_prefers_summary_lead_clause_over_keyword_soup() {
        let with_summary = |sum: &str| Job {
            key: "topicname:x".into(),
            hash: String::new(),
            prompt: String::new(),
            label: String::new(),
            short: true,
            gate: Some(vec!["voyage".into(), "reranker".into()]),
            ban: vec![],
            summary: Some(sum.into()),
        };
        // Off-theme name rejected → the summary's first clause, title-cased.
        let j = with_summary("Migrating the gateway to NATS, then wiring telemetry.");
        assert_eq!(finish(&j, "Colpali Visual Retrieval"), "Migrating the Gateway to NATS");
        // A one-word summary is too thin to title → keyword soup is the last resort.
        let thin = with_summary("Cleanup.");
        assert_eq!(finish(&thin, "Colpali Visual Retrieval"), "Voyage · Reranker");
    }

    // The summary heading takes the lead clause, caps the length, and drops a
    // dangling conjunction so it reads as a phrase.
    #[test]
    fn summary_heading_distills_a_lead_clause() {
        assert_eq!(
            summary_heading("Updating Terraform configurations, migrating infrastructure to SIE."),
            "Updating Terraform Configurations"
        );
        // Caps at 8 words and won't end on a dangling "and"/"to".
        assert_eq!(
            summary_heading("Refactoring the adapter registry to support pluggable backends and more"),
            "Refactoring the Adapter Registry to Support Pluggable Backends"
        );
        assert_eq!(summary_heading("Misc."), "", "too thin to title");
    }

    // The repo is already a facet on every surface — a "name" that is just the
    // repo slug OR one of its fragments carries nothing and falls back to the
    // keyword title, with banned terms skipped so the fallback can't reproduce
    // the rejected name (or lead with repo fragments like "internal").
    #[test]
    fn finish_bans_repo_slug_names() {
        let j = Job {
            key: "topicname:x".into(),
            hash: String::new(),
            prompt: String::new(),
            label: String::new(),
            short: true,
            gate: Some(vec!["sie".into(), "pool".into(), "warmup".into()]),
            ban: repo_ban(&["sie-internal".into()]),
            summary: None,
        };
        // Grounded (shares "sie") but still just the repo / a repo fragment.
        assert_eq!(finish(&j, "Sie-Internal"), "Pool · Warmup");
        assert_eq!(finish(&j, "SIE"), "Pool · Warmup");
        assert_eq!(repo_ban(&["sie-web".into()]), vec!["sie", "sieweb", "web"]);
    }

    // The exemplars shown to the model must be real sentences, not the slug
    // echoes that happen to be shortest ("sie-internal: #955" primes the 0.6B
    // to answer in slugs). A tiny cluster with nothing better still gets some.
    #[test]
    fn pick_examples_skips_degenerate_slugs() {
        let members: Vec<String> = vec![
            "sie-internal: #955".into(),
            "sie-router-rust".into(),
            "Added a retry path to the gateway transport for 503 responses".into(),
            "Tuned warm pool startup probes to cut cold-start latency in half".into(),
            "Moved sidecar config into the worker chart and documented the flags".into(),
        ];
        let ex = pick_examples(&members);
        assert_eq!(ex.len(), 3);
        assert!(ex.iter().all(|e| !e.contains("#955") && !e.contains("sie-router-rust")), "{ex:?}");
        let tiny: Vec<String> = vec!["sie-internal: #955".into(), "fix".into()];
        assert_eq!(pick_examples(&tiny).len(), 2); // fallback: better than nothing
    }

    // A name with a hallucinated token scores visibly below a fully-grounded
    // one — the length norm makes the bad half count, instead of letting the
    // good half's matches paper over it.
    #[test]
    fn name_score_penalizes_hallucinated_tokens() {
        use ndarray::arr2;
        let members = vec![arr2(&[[1.0f32, 0.0]]), arr2(&[[0.9f32, 0.1]])];
        let grounded = arr2(&[[1.0f32, 0.0]]);
        let half_hallucinated = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]); // 2nd token matches nothing
        let g = name_score(&grounded, &members);
        let h = name_score(&half_hallucinated, &members);
        assert!(g > 0.9, "grounded name scores high: {g}");
        assert!(h < 0.6 * g, "hallucinated half drags it below the floor: {h} vs {g}");
        assert_eq!(name_score(&grounded, &[]), 0.0);
    }

    // The run-relative floor rejects only clear outliers, and abstains
    // entirely when too few names are scored for the median to mean anything.
    #[test]
    fn embed_rejects_only_clear_outliers() {
        let mut scores: Vec<(usize, f32)> = (0..7).map(|i| (i, 1.0)).collect();
        scores.push((7, 0.2));
        assert_eq!(embed_rejects(&scores), vec![7]); // 8 scored: the outlier goes
        scores[7] = (7, 0.7);
        assert!(embed_rejects(&scores).is_empty()); // above 0.6×median: kept
        let few: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.1)];
        assert!(embed_rejects(&few).is_empty()); // too few to judge
    }

    // The self-calibrating verdict keeps an on-theme name and rejects an
    // off-theme one judged against the cluster's own members (unit vectors →
    // maxsim is cosine). Exact thresholds are tuned in Phase 2; these cases are
    // unambiguous either side of any reasonable floor.
    #[test]
    fn score_name_keeps_on_theme_rejects_off_theme() {
        use ndarray::arr2;
        let members: Vec<Array2<f32>> = (0..6).map(|_| arr2(&[[1.0f32, 0.0]])).collect();
        assert!(!score_name(&arr2(&[[1.0f32, 0.0]]), &members).reject, "on-theme (cos~1) kept");
        assert!(score_name(&arr2(&[[0.0f32, 1.0]]), &members).reject, "orthogonal/hallucinated (cos~0) rejected");
    }

    // Below MIN_MEMBERS_FOR_COH the per-cluster percentile is noise, so the gate
    // leans on the absolute cosine floor alone (sound because vectors are
    // L2-normalized) rather than a 3-sample tail.
    #[test]
    fn score_name_tiny_cluster_uses_absolute_floor() {
        use ndarray::arr2;
        let members: Vec<Array2<f32>> = (0..3).map(|_| arr2(&[[1.0f32, 0.0]])).collect();
        assert!(!score_name(&arr2(&[[1.0f32, 0.0]]), &members).reject); // above ABS_FLOOR
        assert!(score_name(&arr2(&[[0.0f32, 1.0]]), &members).reject); //  below ABS_FLOOR
    }

    #[test]
    fn percentile_nearest_rank() {
        let s = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(percentile(&s, 0), 0.1);
        assert_eq!(percentile(&s, 100), 0.5);
        assert_eq!(percentile(&s, 50), 0.3);
        assert_eq!(percentile(&[], 10), 0.0);
    }

    // Hyphen/slash compounds are cased per component (the acronym list can
    // never match inside "SIE-INTERNAL" as one token), whole-token codes like
    // GPT-5.5 stay intact, and brands get their canonical mixed case.
    #[test]
    fn title_case_components_and_brands() {
        assert_eq!(clean_name("SIE-INTERNAL CLEANUP"), "SIE-Internal Cleanup");
        assert_eq!(clean_name("GPT-5.5 ROLLOUT"), "GPT-5.5 Rollout");
        assert_eq!(clean_name("COLBERT INDEXING"), "ColBERT Indexing");
        assert_eq!(clean_name("QWEN3 NAMING"), "Qwen3 Naming"); // brand beats the code rule
        assert_eq!(clean_name("CTA/GITHUB BUTTON FIXES"), "CTA/GitHub Button Fixes");
        assert_eq!(clean_name("CUDA JIT COMPILATION"), "CUDA JIT Compilation");
        // Not titles: hyphen-mush compounds and anything too long to display
        // untruncated (an ellipsis in a heading is worse than the fallback).
        assert_eq!(clean_name("Mcp-Claud-Eq-Reduction-Generation"), "");
        assert_eq!(clean_name("Web Deployment Notes Synchronization Prerequisites"), "");
    }

    // A mashed identifier (one token, ≥2 hyphen-joined words) is a slug, not a
    // title — the 0.6B's branch-name reflex. Reject it (→ summary fallback) so
    // "Terraform-SIE-Update"/"SIE-Server" stop slipping through and colliding.
    // A spaced title is never a slug; a version/code piece doesn't count.
    #[test]
    fn slug_style_names_are_rejected() {
        assert_eq!(clean_name("TERRAFORM-SIE-UPDATE"), "", "three-part slug");
        assert_eq!(clean_name("SIE-SERVER"), "", "two-part slug");
        assert_eq!(clean_name("MCP-CLAUDE-PROJECT"), "");
        // Spared: spaced titles, and codes whose hyphen-piece isn't a word.
        assert_eq!(clean_name("SIE-INTERNAL CLEANUP"), "SIE-Internal Cleanup", "has a space → not a slug");
        assert_eq!(clean_name("GPT-5.5"), "GPT-5.5", "version code, not two words");
        assert_eq!(clean_name("FLORENCE-2"), "Florence-2", "code piece, not a word");
    }

    // Two machines shuffle the same pending list differently (so concurrent
    // viewers split the work), and one machine's order is stable.
    #[test]
    fn shuffle_is_seeded_and_machine_varied() {
        let base: Vec<usize> = (0..50).collect();
        let (mut a, mut a2, mut b) = (base.clone(), base.clone(), base.clone());
        shuffle(&mut a, 0x1111);
        shuffle(&mut a2, 0x1111);
        shuffle(&mut b, 0x2222);
        assert_eq!(a, a2, "same seed, same order");
        assert_ne!(a, b, "different machines, different order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, base, "a permutation, nothing lost");
    }

    // Worker policy: Metal stays single (one GPU), CPU scales by cores/4, an
    // explicit override wins, and we never spawn more workers than jobs.
    #[test]
    fn worker_count_policy() {
        assert_eq!(worker_count(None, true, 12, 1000), 1); // metal: one GPU
        assert_eq!(worker_count(None, false, 12, 1000), 3); // cpu: cores/4
        assert_eq!(worker_count(None, false, 32, 1000), 4); // capped at 4
        assert_eq!(worker_count(Some(6), true, 12, 1000), 6); // override wins
        assert_eq!(worker_count(Some(99), false, 12, 1000), 8); // hard clamp
        assert_eq!(worker_count(None, false, 32, 2), 2); // never more than jobs
    }

    #[test]
    fn clean_name_normalizes_shouting() {
        // The 0.6B SHOUTS and snake_cases good titles — normalize, don't reject.
        assert_eq!(clean_name("SPARSE OPTIMIZATION"), "Sparse Optimization");
        assert_eq!(clean_name("GPU INFRASTRUCTURE OPTIMIZATION"), "GPU Infrastructure Optimization");
        assert_eq!(clean_name("QUEUE_ROUTING_NATS_HEALTH"), "Queue Routing NATS Health"); // snake → spaces
        assert_eq!(clean_name("CONFIG API IMPLEMENTATION AND EXPANSION"), "Config API Implementation and Expansion");
        assert_eq!(clean_name("VISION IMPROVEMENT CLUSTER"), "Vision Improvement"); // drop generic suffix
        // Genuine non-titles still fall back to the representative-member label.
        assert_eq!(clean_name("Sharing experiments, proposing models, and cutting costs."), "");
        // A bare generic word names nothing → fall back to the summary.
        assert_eq!(clean_name("MAIN"), "");
        assert_eq!(clean_name("Misc"), "");
        // …but a meaningful one-word name (acronym / product) is kept.
        assert_eq!(clean_name("NATS"), "NATS");
        assert_eq!(clean_name("AZURE"), "Azure");
    }
}

/// A dry run for prompt tuning: generate (but do not cache) the first `n` topic
/// summaries — or those matching `SYNTY_LLM_ONLY` — and print them to stdout.
pub fn sample(n: usize) -> Result<()> {
    let all = topic_jobs()?;
    let mut sel: Vec<&Job> = all.iter().collect();
    if let Ok(only) = std::env::var("SYNTY_LLM_ONLY") {
        let want: Vec<String> = only.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        sel.retain(|j| want.iter().any(|w| j.key.contains(w.as_str())));
    }
    sel.truncate(n);
    if sel.is_empty() {
        eprintln!("no topic jobs to sample");
        return Ok(());
    }
    let mut llm = Summarizer::load()?;
    for j in &sel {
        let (raw, _, _) = llm.generate(&j.prompt)?;
        // The full finishing pipeline (clean + gate + fallback), so the dry run
        // shows exactly what a real pass would cache.
        println!("{} — {}", j.label, finish(j, &raw));
    }
    Ok(())
}

/// Refresh summaries in two passes — units (sessions, PRs, issues), then topics
/// reduced from each cluster's representative documents. The model loads once,
/// lazily, only if there is work the fleet hasn't already done: summaries are
/// shared write-once through the bucket, so this machine first materializes
/// what other viewers generated, then resolves each remaining job store-first.
pub fn summarize_all(bucket: &str) -> Result<()> {
    let mut cache = units::load_summary_cache();
    let store = crate::store::SummaryStore::open(bucket)?;
    // Topic entries are only valid for the current clustering: the same rule
    // gates fleet pulls (or orphans would cycle pull → prune → pull) and the
    // prune at the end of the pass.
    let valid: std::collections::HashSet<String> =
        units::topic_units(12)?.iter().map(|t| t.cache_key.clone()).collect();
    let keep = |key: &str| match key.strip_prefix("topic:").or_else(|| key.strip_prefix("topicname:")) {
        Some(stable) => valid.contains(stable),
        None => true,
    };
    match store.sync_cache(&mut cache, &keep) {
        Ok((pulled, pushed)) if pulled + pushed > 0 => {
            eprintln!("summarize: fleet store — pulled {pulled}, shared {pushed}");
            if pulled > 0 {
                units::save_summary_cache(&cache)?;
            }
        }
        Ok(_) => {}
        Err(e) => eprintln!("summarize: fleet sync skipped ({e})"),
    }
    let mut llm: Option<Summarizer> = None;
    let t = std::time::Instant::now();
    let (mut in_tok, mut out_tok) = (0usize, 0usize);

    // Pass 1: units.
    let ujobs = unit_jobs()?;
    let utodo = pending(&ujobs, &cache);
    if !utodo.is_empty() {
        eprintln!("summarizing {} unit(s) with {REPO}…", utodo.len());
        let (i, o) = run_jobs(&utodo, &mut cache, &mut llm, &store, "unit")?;
        (in_tok, out_tok) = (in_tok + i, out_tok + o);
        units::save_summary_cache(&cache)?;
    }

    // Pass 2: topics, reduced from each cluster's representative documents.
    let tjobs = topic_jobs()?;
    let ttodo = pending(&tjobs, &cache);
    if !ttodo.is_empty() {
        eprintln!("summarizing {} topic(s)…", ttodo.len());
        let (i, o) = run_jobs(&ttodo, &mut cache, &mut llm, &store, "topic")?;
        (in_tok, out_tok) = (in_tok + i, out_tok + o);
    }

    // Pass 3: the embedding-faithfulness gate over every LLM name — including
    // ones pulled from the fleet or cached by earlier runs, so a bad name is
    // corrected on every machine no matter where it was generated.
    let (scored, rejected) = match embed_gate_names(&tjobs, &mut cache, bucket) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("summarize: name gate skipped ({e})");
            (0, 0)
        }
    };
    if rejected > 0 {
        eprintln!("summarize: name gate replaced {rejected}/{scored} unfaithful name(s)");
    }
    // Prune orphaned topic entries — stable keys from superseded clusterings,
    // left behind when re-clustering changes a cluster's medoid/membership.
    // Session and doc summaries (keyed by id / gh:repo#n) are untouched.
    cache.retain(|k, _| keep(k));
    units::save_summary_cache(&cache)?;

    let n = utodo.len() + ttodo.len();
    let secs = t.elapsed().as_secs_f64();
    if n == 0 {
        eprintln!("summaries up to date ({} units, {} topics)", ujobs.len(), tjobs.len());
    } else {
        eprintln!("done: {n} items in {:.1}s — {:.1} output tok/s ({in_tok} prompt + {out_tok} output tok)", secs, out_tok as f64 / secs);
    }

    // Standardized coverage/throughput metrics (stderr block + metrics.jsonl line).
    let covered = ujobs.iter().filter(|j| cache.get(&j.key).map(|c| !c.summary.is_empty()).unwrap_or(false)).count();
    let named = cache.iter().filter(|(k, v)| k.starts_with("topicname:") && !v.summary.is_empty()).count();
    let topics = cache.keys().filter(|k| k.starts_with("topic:")).count();
    // Name quality: topics sharing an identical name (a name's one job is to
    // tell topics apart), how often the LLM's title was rejected for a fallback
    // (exact right after a regen pass), and how many of those dropped all the
    // way to the keyword label rather than the summary's lead clause.
    let mut name_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut fallback = 0usize;
    let mut kw_last_resort = 0usize;
    for j in tjobs.iter().filter(|j| j.short) {
        let Some(c) = cache.get(&j.key) else { continue };
        if c.summary.is_empty() {
            continue;
        }
        *name_counts.entry(c.summary.as_str()).or_default() += 1;
        if c.summary == fallback_name(j) {
            fallback += 1;
            if c.summary == keyword_label(j.gate.as_deref().unwrap_or_default(), &j.ban) {
                kw_last_resort += 1;
            }
        }
    }
    let name_dupes: usize = name_counts.values().filter(|&&c| c > 1).sum();
    crate::metrics::Run::new("summarize")
        .set("units", ujobs.len())
        .set("unit_coverage_pct", 100.0 * covered as f64 / ujobs.len().max(1) as f64)
        .set("topics", topics)
        .set("topics_named", named)
        .set("name_dupes", name_dupes)
        .set("names_fallback", fallback)
        .set("names_kw_last_resort", kw_last_resort)
        .set("names_scored", scored)
        .set("name_faithful_pct", if scored > 0 { 100.0 * (scored - rejected) as f64 / scored as f64 } else { 100.0 })
        .set("regenerated", n)
        .set("prompt_tok", in_tok)
        .set("output_tok", out_tok)
        .set("secs", secs)
        .set("out_tok_per_s", if secs > 0.0 { out_tok as f64 / secs } else { 0.0 })
        .emit();
    Ok(())
}

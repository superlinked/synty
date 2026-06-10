// Local abstractive session summaries via Qwen3-0.6B on candle. One concise
// sentence per session, generated offline and cached by an input hash so the
// reader never runs the model at view time. Greedy decode (temperature 0) keeps
// it reproducible. This is the ONLY place a generative model is used — retrieval
// and clustering stay LLM-free, and nothing leaves the machine.

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

/// Name a topic with a short Title-Case heading, GROUNDED in the cluster's salient
/// terms and a few representative items (not just the reduced summary). Concrete
/// evidence makes the small model build the name from terms the cluster actually
/// uses, so it passes the faithfulness gate instead of free-associating. (No
/// example *name* is given — the 0.6B parrots those; terms and items are data.)
fn prompt_for_topic_name(summary: &str, terms: &[String], examples: &[String]) -> String {
    let kw = terms.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
    let mut items = String::new();
    for e in examples {
        items.push_str("- ");
        items.push_str(e);
        items.push('\n');
    }
    format!(
        "<|im_start|>user\nName this cluster of engineering work with a short Title Case heading of 2 to 4 words, like a chapter title. Build it from the key terms and items below — prefer the most specific, recurring nouns. Output only the title: no quotes, no period, no commas.\n\n\
Key terms: {kw}\nItems:\n{items}Description: {summary}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
}

/// Frequent content words across a cluster's member summaries (document frequency,
/// ≥3 chars, minus stopwords) — the cluster's salient terms for the name gate.
fn cluster_terms(members: &[String], k: usize) -> Vec<String> {
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in members {
        let mut seen = std::collections::HashSet::new();
        for w in m.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 3 && !STOPWORDS.contains(w)) {
            if seen.insert(w.to_string()) {
                *freq.entry(w.to_string()).or_default() += 1;
            }
        }
    }
    let mut v: Vec<(String, usize)> = freq.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|(w, _)| w).collect()
}

/// True if the generated name shares at least one salient term with the cluster —
/// otherwise it is about something the members are not, and is rejected.
fn name_grounded(name: &str, terms: &[String]) -> bool {
    if name.is_empty() {
        return false;
    }
    let low = name.to_lowercase();
    let words: std::collections::HashSet<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 3).collect();
    terms.iter().any(|t| words.contains(t.as_str()))
}

/// Generic developer vocabulary that must not count as a cluster's salient term —
/// a name should match on a concrete subject, not "update"/"fix"/"feature".
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "into", "are", "has", "have",
    "had", "not", "added", "add", "adds", "fix", "fixes", "fixed", "update", "updates", "updated",
    "updating", "implement", "implemented", "implementing", "support", "new", "using", "use", "used",
    "via", "across", "their", "its", "which", "while", "when", "also", "now", "set", "get", "include",
    "includes", "including", "improve", "improved", "improving", "enhance", "enhanced", "enhancing",
    "project", "work", "feature", "features", "changes", "change", "code", "file", "files", "repo",
    "repository", "dependencies", "dependency", "data", "based", "various", "tools", "system",
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
/// comma-lists, whole sentences (>6 words), field echoes, the bare generic
/// word — fall back to the representative-member label.
fn clean_name(s: &str) -> String {
    let s = s.rsplit("</think>").next().unwrap_or(s).trim().trim_matches('"').trim();
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or(s).replace('_', " ");
    let line = crate::excerpt(&line, 48);
    let line = line.trim_end_matches(['.', ':', ',']).trim();
    let low = line.to_lowercase();
    let echo = ["keyphrase", "items", "title", "repo", "update:", "description"].iter().any(|p| low.starts_with(p));
    let words = line.split_whitespace().count();
    if line.is_empty() || echo || line.contains(',') || !(1..=6).contains(&words) {
        return String::new();
    }
    let t = title_case(line);
    let t = t.trim_end_matches(" Cluster").trim();
    if t.is_empty() || matches!(t.to_lowercase().as_str(), "cluster" | "chores" | "update" | "fix") {
        String::new()
    } else {
        t.to_string()
    }
}

/// Title-case a phrase: keep known domain acronyms (GPU, SIE, NATS…) and codes
/// (S3, M4, GPT-5.5) uppercase, lowercase function words mid-title, capitalize
/// the rest. The model SHOUTS, so we can't infer acronyms from case — hence the
/// small curated allowlist.
fn title_case(s: &str) -> String {
    const ACRO: &[&str] = &[
        "GPU", "CPU", "AWS", "GCP", "TEI", "API", "SIE", "NATS", "OCR", "CLI", "UI", "UX", "JSON",
        "LLM", "TLS", "VDB", "CTA", "SDK", "HTTP", "AI", "ML", "RAG", "SQL", "K8S", "GHCR", "TUI",
        "MTP", "PR", "CI", "CD", "TLS", "BYO", "VM", "VMS",
    ];
    const STOP: &[&str] = &["and", "or", "for", "to", "the", "a", "an", "of", "in", "on", "at", "with", "by", "vs", "from", "into"];
    s.split_whitespace()
        .enumerate()
        .map(|(i, w)| {
            let core: String = w.to_uppercase().chars().filter(|c| c.is_alphanumeric()).collect();
            let code = core.len() <= 6 && core.chars().any(|c| c.is_numeric()) && core.chars().all(|c| c.is_uppercase() || c.is_numeric());
            if ACRO.contains(&core.as_str()) || code {
                w.to_uppercase()
            } else if i > 0 && STOP.contains(&w.to_lowercase().as_str()) {
                w.to_lowercase()
            } else {
                let mut ch = w.chars();
                match ch.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + &ch.as_str().to_lowercase(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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
    hash_parts(&[&s.ask, &s.files.join(","), &s.turns.join("\n")])
}

fn doc_hash(d: &DocInput) -> String {
    hash_parts(&[d.kind, &d.title, &d.text])
}

/// One unit of summarization work: a cache key, its input hash, and the prompt.
/// `short` jobs are topic names (cleaned as a title, not a sentence). `gate`, when
/// set, holds the cluster's salient terms: a name sharing none of them is rejected
/// (→ empty, so `title()` falls through to the grounded summary) — this is what
/// stops "Colpali" on error-handling PRs or "Update Dependencies" on synty work.
struct Job {
    key: String,
    hash: String,
    prompt: String,
    label: String,
    short: bool,
    gate: Option<Vec<String>>,
}

/// Salt for the topic reduce / name prompts; bump to regenerate them only.
/// t7: combinatorial opener stripping (drops "the cluster focuses …" et al.).
const TOPIC_PROMPT_VERSION: &str = "t7";
const TOPIC_NAME_VERSION: &str = "s3";

/// Unit jobs: one per session and per PR/issue.
fn unit_jobs() -> Result<Vec<Job>> {
    let sessions = units::session_inputs()?;
    let docs = units::doc_inputs()?;
    let mut jobs = Vec::with_capacity(sessions.len() + docs.len());
    for s in &sessions {
        jobs.push(Job { key: s.id.clone(), hash: input_hash(s), prompt: prompt_for(s), label: crate::short(&s.id), short: false, gate: None });
    }
    for d in &docs {
        jobs.push(Job { key: d.key.clone(), hash: doc_hash(d), prompt: prompt_for_doc(d), label: d.key.clone(), short: false, gate: None });
    }
    Ok(jobs)
}

/// Topic jobs: a one-line summary reduced from the member-unit summaries, plus a
/// short name that *titles that summary* (a small model shortens a sentence far
/// better than it abstracts a name from a list of items). The summary's hash is
/// over its members; the name's is over the summary text it titles.
fn topic_jobs() -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for t in units::topic_units(12)? {
        let members: Vec<String> = t.units.iter().filter_map(|u| u.summary.clone()).take(40).collect();
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
            prompt: prompt_for_topic(&members),
            label: format!("topic:{}", t.id),
            short: false,
            gate: None,
        });
        // The name titles the topic summary, so it depends on it. The summary job
        // above runs first in the same pass; for a topic whose summary isn't
        // cached yet the name regenerates next run, once the summary exists. The
        // gate carries the cluster's salient terms so an off-theme name is rejected.
        if let Some(sum) = &t.summary {
            let terms = cluster_terms(&members, 12);
            let mut examples: Vec<&String> = members.iter().collect();
            examples.sort_by_key(|s| s.len()); // shortest = most title-like
            let examples: Vec<String> = examples.iter().take(3).map(|s| crate::excerpt(s, 90)).collect();
            jobs.push(Job {
                key: units::topic_name_key(&t.cache_key),
                hash: hash_parts(&[TOPIC_NAME_VERSION, sum.as_str(), &terms.join(",")]),
                prompt: prompt_for_topic_name(sum, &terms, &examples),
                label: format!("name:{}", t.id),
                short: true,
                gate: Some(terms),
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

/// Clean a raw generation per the job type and apply the faithfulness gate: a
/// topic name sharing no salient term with its cluster is rejected (→ empty →
/// `title()` falls through to the grounded summary).
fn finish(j: &Job, raw: &str) -> String {
    let summary = if j.short { clean_name(raw) } else { clean(raw) };
    match &j.gate {
        Some(terms) if !name_grounded(&summary, terms) => String::new(),
        _ => summary,
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
/// result back. Returns (summary, prompt_tok, out_tok, from_store).
fn resolve(j: &Job, llm: &mut Summarizer, store: &crate::store::SummaryStore) -> (String, usize, usize, bool) {
    if let Ok(Some(s)) = store.get(&j.key, &j.hash) {
        return (s, 0, 0, true);
    }
    let (raw, pt, ot) = match llm.generate(&j.prompt) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  {} failed: {e}", j.label);
            (String::new(), 0, 0)
        }
    };
    let summary = finish(j, &raw);
    let _ = store.put(&j.key, &j.hash, &summary);
    (summary, pt, ot, false)
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
            let (summary, pt, ot, fetched) = resolve(j, llm, store);
            in_tok += pt;
            out_tok += ot;
            let tag = if fetched { " (fleet)" } else { "" };
            eprintln!("  [{kind} {}/{}] {}{tag} — {}", n + 1, todo.len(), j.label, summary);
            crate::progress::phase(&format!("summarizing {kind}s"), n + 1, todo.len());
            cache.insert(j.key.clone(), CachedSummary { hash: j.hash.clone(), summary });
            if (n + 1) % 10 == 0 {
                units::save_summary_cache(cache)?;
            }
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
                        let (summary, pt, ot, fetched) = resolve(j, &mut llm, store);
                        in_tok += pt;
                        out_tok += ot;
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        let tag = if fetched { " (fleet)" } else { "" };
                        eprintln!("  [{kind} {n}/{}] {}{tag} — {}", todo.len(), j.label, summary);
                        crate::progress::phase(&format!("summarizing {kind}s"), n, todo.len());
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

    // The faithfulness gate rejects a name that shares no salient term with its
    // cluster, but keeps one that does.
    #[test]
    fn name_gate_rejects_off_theme() {
        let members = vec![
            "Added InputTooLongError and an overflow policy to the extract SDK".to_string(),
            "gliclass returns all label scores and bounds input length".to_string(),
        ];
        let terms = cluster_terms(&members, 12);
        assert!(name_grounded("Overflow Policy Handling", &terms)); // shares "overflow"/"policy"
        assert!(!name_grounded("Colpali Visual Retrieval", &terms)); // shares nothing
        assert!(!name_grounded("Update Dependencies", &terms)); // generic words, not cluster terms
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
        let out = if j.short { clean_name(&raw) } else { clean(&raw) };
        println!("{} — {out}", j.label);
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
    match store.sync_cache(&mut cache) {
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
    // Prune orphaned topic entries — stable keys from superseded clusterings, left
    // behind when re-clustering changes a cluster's medoid/membership. Session and
    // doc summaries (keyed by id / gh:repo#n) are untouched.
    let valid: std::collections::HashSet<String> = units::topic_units(12)?.iter().map(|t| t.cache_key.clone()).collect();
    cache.retain(|k, _| match k.strip_prefix("topic:").or_else(|| k.strip_prefix("topicname:")) {
        Some(stable) => valid.contains(stable),
        None => true,
    });
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
    crate::metrics::Run::new("summarize")
        .set("units", ujobs.len())
        .set("unit_coverage_pct", 100.0 * covered as f64 / ujobs.len().max(1) as f64)
        .set("topics", topics)
        .set("topics_named", named)
        .set("regenerated", n)
        .set("prompt_tok", in_tok)
        .set("output_tok", out_tok)
        .set("secs", secs)
        .set("out_tok_per_s", if secs > 0.0 { out_tok as f64 / secs } else { 0.0 })
        .emit();
    Ok(())
}

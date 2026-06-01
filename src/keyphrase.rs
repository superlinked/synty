// Extractive cluster labels, no LLM: class-based TF-IDF (c-TF-IDF). Each
// cluster is treated as one "class document"; a term scores high when it is
// frequent *within* the cluster but rare across the others. The top phrases
// read as topic names ("generation isolation", "dependabot alerts", "docs
// search") — far better than the dominant repo/label, which collapsed every
// session topic to "local".
//
// Unigrams and bigrams are scored together; a unigram already covered by a
// chosen bigram is dropped so labels don't repeat ("generation, generation
// isolation" → "generation isolation").

use std::collections::{HashMap, HashSet};

/// English function words + markdown/URL/agent noise. Domain verbs that carry
/// signal in PR titles (fix, add, feat, bug) are intentionally kept.
const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of", "to", "in", "on",
    "at", "by", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "with", "without", "from", "into", "out", "up", "down", "over",
    "under", "we", "you", "i", "he", "she", "they", "them", "his", "her", "their", "our", "your",
    "my", "me", "us", "can", "will", "would", "should", "could", "may", "might", "must", "do",
    "does", "did", "done", "have", "has", "had", "not", "no", "yes", "so", "than", "too", "very",
    "just", "also", "any", "all", "some", "more", "most", "other", "such", "only", "own", "same",
    "what", "which", "who", "whom", "when", "where", "why", "how", "there", "here", "about",
    "again", "once", "now", "use", "used", "using", "via", "per", "each", "both", "few", "make",
    "made", "like", "want", "need", "get", "got", "let", "see", "way", "one", "two", "able",
    "http", "https", "www", "com", "org", "md", "html", "txt", "href", "com", "github",
];

/// Lowercase word tokens of length ≥ 3, stopwords and pure-number tokens
/// dropped. Splits on any non-alphanumeric (keeps `c++`-style as `c`).
fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.len() >= 3 && !cur.bytes().all(|b| b.is_ascii_digit()) {
            let stop: HashSet<&str> = STOP.iter().copied().collect();
            if !stop.contains(cur.as_str()) {
                out.push(cur.clone());
            }
        }
        cur.clear();
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Per-cluster unigram+bigram counts.
fn term_counts(texts: &[&str]) -> (HashMap<String, f64>, f64) {
    let mut counts: HashMap<String, f64> = HashMap::new();
    let mut total: f64 = 0.0;
    for t in texts {
        let toks = tokens(t);
        for w in &toks {
            *counts.entry(w.clone()).or_insert(0.0) += 1.0;
            total += 1.0;
        }
        for w in toks.windows(2) {
            *counts.entry(format!("{} {}", w[0], w[1])).or_insert(0.0) += 1.0;
        }
    }
    (counts, total.max(1.0))
}

/// c-TF-IDF labels: `k` phrases per cluster. `clusters[c]` is the list of
/// document texts in cluster c.
pub fn labels(clusters: &[Vec<&str>], k: usize) -> Vec<Vec<String>> {
    let per: Vec<(HashMap<String, f64>, f64)> = clusters.iter().map(|c| term_counts(c)).collect();

    // Global frequency of each term across all clusters and the average class
    // length, for the idf term: ln(1 + avg_len / f_t).
    let mut global: HashMap<String, f64> = HashMap::new();
    let mut sum_len = 0.0;
    for (counts, len) in &per {
        sum_len += *len;
        for (t, c) in counts {
            *global.entry(t.clone()).or_insert(0.0) += *c;
        }
    }
    let avg_len = (sum_len / per.len().max(1) as f64).max(1.0);

    per.iter()
        .map(|(counts, len)| {
            let mut scored: Vec<(String, f64)> = counts
                .iter()
                .filter(|(t, c)| **c >= 2.0 || t.contains(' ')) // drop hapax unigrams
                .map(|(t, c)| {
                    let tf = c / len;
                    let idf = (1.0 + avg_len / global.get(t).copied().unwrap_or(1.0)).ln();
                    // Bigrams are more nameable; nudge them ahead of unigrams.
                    let bonus = if t.contains(' ') { 1.35 } else { 1.0 };
                    (t.clone(), tf * idf * bonus)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
            pick(scored, k)
        })
        .collect()
}

/// Take the top `k` phrases, skipping a unigram already contained in a chosen
/// bigram (and a bigram subsumed by a chosen one).
fn pick(scored: Vec<(String, f64)>, k: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut covered: HashSet<String> = HashSet::new();
    for (t, _) in scored {
        if out.len() >= k {
            break;
        }
        let words: Vec<&str> = t.split(' ').collect();
        // Skip a unigram that a chosen bigram already contains.
        if words.len() == 1 && covered.contains(&t) {
            continue;
        }
        // Skip a near-duplicate bigram sharing a word with a chosen phrase.
        if words.len() == 2 && words.iter().any(|w| out.iter().any(|o| o == w)) {
            continue;
        }
        out.push(t.clone());
        for w in words {
            covered.insert(w.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_lowercases_and_drops_stop_and_numbers() {
        let t = tokens("Fix the Generation Isolation 123 in gateway");
        assert!(t.contains(&"fix".to_string()));
        assert!(t.contains(&"generation".to_string()));
        assert!(t.contains(&"isolation".to_string()));
        assert!(t.contains(&"gateway".to_string()));
        assert!(!t.contains(&"the".to_string())); // stopword
        assert!(!t.contains(&"123".to_string())); // number
    }

    // A distinctive phrase in one cluster outscores a word common to all.
    #[test]
    fn ctfidf_surfaces_distinctive_phrase() {
        let clusters = vec![
            vec![
                "generation isolation guardrails on the gateway",
                "strengthen generation isolation in the gateway queue",
            ],
            vec!["docling ocr adapter for documents", "mineru ocr document parsing"],
            vec!["readme docker quickstart deployment", "fix dead links in readme"],
        ];
        let ls = labels(&clusters, 3);
        assert!(
            ls[0].iter().any(|l| l.contains("generation") || l.contains("isolation")),
            "cluster 0 labels: {:?}",
            ls[0]
        );
        assert!(ls[1].iter().any(|l| l.contains("ocr") || l.contains("document")), "{:?}", ls[1]);
        // "gateway" appears in cluster 0 only → should be available; "the" never.
        assert!(ls.iter().flatten().all(|l| !l.split(' ').any(|w| w == "the")));
    }

    #[test]
    fn pick_drops_unigram_covered_by_bigram() {
        let scored = vec![
            ("generation isolation".to_string(), 3.0),
            ("generation".to_string(), 2.5),
            ("queue".to_string(), 2.0),
        ];
        let got = pick(scored, 3);
        assert_eq!(got[0], "generation isolation");
        assert!(!got.contains(&"generation".to_string()), "got: {got:?}");
        assert!(got.contains(&"queue".to_string()));
    }
}

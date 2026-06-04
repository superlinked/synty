// Standardized run metrics. Any operation that produces health or quality
// numbers builds a `Run`, records named fields as it works, then `emit`s them:
// a one-line block to stderr (the info log) and one JSON object appended to
// metrics.jsonl. So the same numbers are tracked the same way every run and can
// be inspected across iterations — `tail metrics.jsonl`, or diff the silhouette
// between two clusterings — instead of being recomputed ad hoc.

use std::io::Write;

const LOG: &str = "metrics.jsonl";

/// One operation's metrics, accumulated then emitted once.
pub struct Run {
    op: &'static str,
    fields: Vec<(String, serde_json::Value)>,
}

impl Run {
    pub fn new(op: &'static str) -> Self {
        Self { op, fields: Vec::new() }
    }

    /// Record a field (number or text). Chainable.
    pub fn set(&mut self, key: &str, v: impl Into<serde_json::Value>) -> &mut Self {
        self.fields.push((key.to_string(), v.into()));
        self
    }

    /// Print a one-line block to stderr and append one JSON object to the log.
    pub fn emit(&self) {
        let block: Vec<String> = self.fields.iter().map(|(k, v)| format!("{k}={}", show(v))).collect();
        eprintln!("[metrics {}] {}", self.op, block.join(" · "));
        if let Err(e) = self.persist() {
            eprintln!("  (metrics log skipped: {e})");
        }
    }

    fn persist(&self) -> anyhow::Result<()> {
        let mut obj = serde_json::Map::new();
        obj.insert("op".into(), serde_json::json!(self.op));
        obj.insert("ts".into(), serde_json::json!(chrono::Utc::now().to_rfc3339()));
        for (k, v) in &self.fields {
            obj.insert(k.clone(), v.clone());
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(LOG)?;
        writeln!(f, "{}", serde_json::Value::Object(obj))?;
        Ok(())
    }
}

/// Compact display: trim float noise to 3 dp, leave ints and text as-is.
fn show(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Number(n) => match n.as_f64() {
            Some(f) if f.fract() != 0.0 => format!("{f:.3}"),
            _ => n.to_string(),
        },
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A Run renders its fields as a readable `key=value` block, floats trimmed.
    #[test]
    fn block_is_readable() {
        let mut r = Run::new("cluster");
        r.set("units", 1082_usize).set("silhouette", 0.0791_f64).set("label", "ok");
        let block: Vec<String> = r.fields.iter().map(|(k, v)| format!("{k}={}", show(v))).collect();
        assert_eq!(block, ["units=1082", "silhouette=0.079", "label=ok"]);
    }
}

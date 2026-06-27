// Standardized run metrics. Any operation that produces health or quality
// numbers builds a `Run`, records named fields as it works, then `emit`s them as
// one `[metrics <op>]` block to stderr — so the same numbers are produced the
// same way every run instead of being recomputed ad hoc. Persisting is the
// caller's business: redirect stderr (`2>> runs.log`) if you want a history.

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

    /// Print a one-line `key=value` block to stderr.
    pub fn emit(&self) {
        let block: Vec<String> = self.fields.iter().map(|(k, v)| format!("{k}={}", show(v))).collect();
        eprintln!("[metrics {}] {}", self.op, block.join(" · "));
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

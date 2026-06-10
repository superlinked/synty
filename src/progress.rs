// Machine-readable build progress. When SYNTY_PROGRESS is set (the TUI sets it
// on the freshen child it spawns), pipeline stages emit one `@phase name d/t`
// line per step to stderr; the TUI tails the child's log file and renders the
// last phase in its status footer. A tiny line protocol beats parsing prose —
// progress survives rewording, and `\r`-style tickers never reach it.

/// Emit a phase marker (only when the parent asked for them).
pub fn phase(name: &str, done: usize, total: usize) {
    if std::env::var_os("SYNTY_PROGRESS").is_some() {
        eprintln!("@phase {name} {done}/{total}");
    }
}

/// Parse a phase marker line: `@phase encode 120/470` → ("encode", 120, 470).
pub fn parse(line: &str) -> Option<(String, usize, usize)> {
    let rest = line.strip_prefix("@phase ")?;
    let (name, counts) = rest.rsplit_once(' ')?;
    let (done, total) = counts.split_once('/')?;
    Some((name.to_string(), done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// Human form for the TUI footer: "⟳ encode 120/470" (or "⟳ encode" when the
/// step has no meaningful count).
pub fn describe(name: &str, done: usize, total: usize) -> String {
    if total <= 1 {
        format!("⟳ {name}")
    } else {
        format!("⟳ {name} {done}/{total}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_lines_roundtrip() {
        assert_eq!(parse("@phase encode 120/470"), Some(("encode".into(), 120, 470)));
        assert_eq!(parse("@phase summarize units 3/40"), Some(("summarize units".into(), 3, 40)));
        assert_eq!(parse("encoded 12/40"), None, "prose is not protocol");
        assert_eq!(parse("@phase broken"), None);
        assert_eq!(describe("encode", 120, 470), "⟳ encode 120/470");
        assert_eq!(describe("cluster", 0, 1), "⟳ cluster");
    }
}

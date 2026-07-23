// Shared, model-independent vocabulary filters. Retrieval query derivation and
// local topic naming use the same developer-ceremony stopwords so feature
// selection cannot change what `synty related` searches for.

pub(crate) const DEVELOPER_STOPWORDS: &[&str] = &[
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "into", "are", "has",
    "have", "had", "not", "added", "add", "adds", "fix", "fixes", "fixed", "update", "updates",
    "updated", "updating", "implement", "implemented", "implementing", "support", "new", "using",
    "use", "used", "via", "across", "their", "its", "which", "while", "when", "also", "now",
    "set", "get", "include", "includes", "including", "improve", "improved", "improving",
    "enhance", "enhanced", "enhancing", "project", "work", "feature", "features", "changes",
    "change", "code", "file", "files", "repo", "repository", "dependencies", "dependency",
    "data", "based", "various", "tools", "system", "feat", "chore", "subject",
];

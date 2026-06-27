// Machine-independent GitHub ingestion. Pulls a trailing window of PRs and
// issues per repo straight from the GitHub GraphQL API with a token, so it runs
// anywhere (CI, a server) without a developer machine or the `gh` CLI. Output
// is the per-repo JSON `ingest` reads: corpus/github/{prs,issues}-<repo>.json,
// plus a manifest (hashes + scraped_at) that shares the corpus through the
// bucket and floors the next scrape.
//
// Scrapes are INCREMENTAL: ordered by UPDATED_AT and floored at the last
// scrape time, so a steady-state run fetches only items that changed since —
// including state flips (OPEN→MERGED) the old created-window approach only
// refreshed by re-downloading everything.
//
// Token resolution: $GITHUB_TOKEN, else `gh auth token` for local convenience.

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;

const ENDPOINT: &str = "https://api.github.com/graphql";
const PAGE: i64 = 50;
/// Overlap subtracted from the last scrape time when flooring an incremental
/// fetch — clock skew and in-flight updates must not slip between scrapes.
const FLOOR_SLACK_MIN: i64 = 60;

pub fn run(owner: &str, repos: Option<String>, since_days: u64, out: &str) -> Result<()> {
    let token = resolve_token()?;
    let cutoff = days_ago_rfc3339(since_days);
    let floor = scrape_floor(&crate::sync::load_github_manifest(out).scraped_at, &cutoff);
    if floor != cutoff {
        eprintln!("github: incremental — fetching items updated since {floor}");
    }
    // Explicit --repos wins; otherwise pull the org's most-recently-active set.
    let repos: Vec<String> = match repos {
        Some(s) => s.split(',').map(|r| r.trim().to_string()).filter(|r| !r.is_empty()).collect(),
        None => active_repos(owner, crate::config::load().backfill_repos.unwrap_or(crate::config::DEFAULT_REPOS))?,
    };
    // Remember the resolved set so session-repo folding can use the real names.
    let mut cfg = crate::config::load();
    cfg.repos = repos.clone();
    let _ = crate::config::save(&cfg);
    std::fs::create_dir_all(out)?;
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        .build();

    // Pin this account's login (best-effort) so local sessions attribute to the
    // same identity as the PRs we're about to pull — reusing the token in hand.
    let login = post(&agent, &token, "query { viewer { login } }", json!({}))
        .ok()
        .and_then(|d| d["viewer"]["login"].as_str().map(str::to_string));
    if let Some(login) = login {
        crate::identity::cache_github_login(&login);
    }

    let scraped_at = days_ago_rfc3339(0);
    let mut manifest = crate::sync::GithubManifest { scraped_at, ..Default::default() };
    let mut tot_pr = 0;
    let mut tot_is = 0;
    for repo in &repos {
        // A repo we've never scraped gets the full window even mid-increment.
        let mut write = |kind: Kind, fname: String, count: &mut usize| -> Result<()> {
            let path = Path::new(out).join(&fname);
            let existing: Vec<Value> = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            let f = if existing.is_empty() { &cutoff } else { &floor };
            let fetched = fetch(&agent, &token, owner, repo, kind, f)?;
            let merged = merge_items(existing, fetched, &cutoff);
            *count += merged.len();
            let body = serde_json::to_string(&merged)?;
            manifest.files.insert(fname, format!("{:016x}", crate::index::fnv1a(body.as_bytes())));
            std::fs::write(path, body)?;
            Ok(())
        };
        write(Kind::Pr, format!("prs-{repo}.json"), &mut tot_pr)?;
        write(Kind::Issue, format!("issues-{repo}.json"), &mut tot_is)?;
    }
    // Drop files for repos no longer in the active set, then publish the
    // manifest — ingest reads the directory, so orphans would linger forever.
    for entry in std::fs::read_dir(out)?.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if (name.starts_with("prs-") || name.starts_with("issues-")) && !manifest.files.contains_key(&name) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    crate::write_atomic(
        &Path::new(out).join(crate::sync::GH_MANIFEST).to_string_lossy(),
        &serde_json::to_vec(&manifest)?,
    )?;
    eprintln!("github: {tot_pr} PRs + {tot_is} issues across {} repos (window since {cutoff}) → {out}", repos.len());

    // Cache the org's members so the fleet roster scopes its untracked list to
    // the team, not every external PR author. Best-effort: a user account or a
    // token without org-read just leaves it unset (roster uses all authors).
    match org_members(owner) {
        Ok(m) if !m.is_empty() => {
            crate::fleet::save_org_members(owner, &m);
            eprintln!("github: cached {} org members for {owner}", m.len());
        }
        Ok(_) => {}
        Err(e) => eprintln!("github: org members unavailable ({e}) — roster will use all active authors"),
    }
    Ok(())
}

/// Floor for an incremental fetch: the last scrape time minus slack, never
/// older than the retention window (a long-idle machine just re-backfills).
fn scrape_floor(scraped_at: &str, window_cutoff: &str) -> String {
    use chrono::{DateTime, SecondsFormat, Utc};
    let Some(t) = DateTime::parse_from_rfc3339(scraped_at).ok() else { return window_cutoff.into() };
    let floored = (t.with_timezone(&Utc) - chrono::Duration::minutes(FLOOR_SLACK_MIN))
        .to_rfc3339_opts(SecondsFormat::Secs, true);
    if floored.as_str() > window_cutoff { floored } else { window_cutoff.into() }
}

/// Merge an incremental fetch into a repo's existing items: replace by number,
/// insert new, drop anything created before the retention window, newest
/// (by creation) first — the same shape a full scrape produces.
fn merge_items(existing: Vec<Value>, fetched: Vec<Value>, created_cutoff: &str) -> Vec<Value> {
    let mut by_number: std::collections::BTreeMap<i64, Value> = existing
        .into_iter()
        .filter_map(|v| v["number"].as_i64().map(|n| (n, v)))
        .collect();
    for v in fetched {
        if let Some(n) = v["number"].as_i64() {
            by_number.insert(n, v);
        }
    }
    let mut out: Vec<Value> = by_number
        .into_values()
        .filter(|v| v["createdAt"].as_str().unwrap_or("") >= created_cutoff)
        .collect();
    out.sort_by(|a, b| b["createdAt"].as_str().unwrap_or("").cmp(a["createdAt"].as_str().unwrap_or("")));
    out
}

/// Whether the local GitHub corpus is older than `minutes` (or was never
/// scraped) — the build's cue to refresh it if it has a token. RFC3339 UTC
/// strings compare lexicographically; an empty scraped_at is always stale.
pub fn stale(dir: &str, minutes: i64) -> bool {
    crate::sync::load_github_manifest(dir).scraped_at < days_ago_rfc3339_minutes(minutes)
}

fn days_ago_rfc3339_minutes(minutes: i64) -> String {
    use chrono::{SecondsFormat, Utc};
    (Utc::now() - chrono::Duration::minutes(minutes)).to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[derive(Clone, Copy)]
enum Kind {
    Pr,
    Issue,
}

/// Paginate a repo's PRs or issues most-recently-UPDATED first, keeping nodes
/// updated at or after `floor` and stopping once they fall older — so an
/// incremental scrape touches only what changed (new items AND state flips on
/// old ones). Each node is reshaped into the JSON `ingest` expects (number,
/// title, body, author.login, url, labels, state, createdAt — plus, for PRs,
/// additions/deletions/mergedAt for the LOC stats).
fn fetch(agent: &ureq::Agent, token: &str, owner: &str, repo: &str, kind: Kind, floor: &str) -> Result<Vec<Value>> {
    let conn = match kind {
        Kind::Pr => "pullRequests",
        Kind::Issue => "issues",
    };
    // PR-only fields — requesting them on Issue nodes is a GraphQL error.
    let extra = match kind {
        Kind::Pr => " additions deletions mergedAt",
        Kind::Issue => "",
    };
    let query = format!(
        r#"query($owner:String!,$name:String!,$cursor:String){{
  repository(owner:$owner,name:$name){{
    {conn}(first:{PAGE},after:$cursor,orderBy:{{field:UPDATED_AT,direction:DESC}}){{
      pageInfo{{hasNextPage endCursor}}
      nodes{{number title body state url createdAt updatedAt author{{login}} labels(first:20){{nodes{{name}}}}{extra}}}
    }}
  }}
}}"#
    );

    let mut out = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let vars = json!({"owner": owner, "name": repo, "cursor": cursor});
        let data = post(agent, token, &query, vars)?;
        let conn_obj = &data["repository"][conn];
        for n in conn_obj["nodes"].as_array().cloned().unwrap_or_default() {
            let updated = n["updatedAt"].as_str().unwrap_or("");
            if updated < floor {
                return Ok(out); // newest-updated first: everything below is older
            }
            out.push(reshape(&n));
        }
        if conn_obj["pageInfo"]["hasNextPage"].as_bool() != Some(true) {
            break;
        }
        cursor = conn_obj["pageInfo"]["endCursor"].as_str().map(String::from);
        if cursor.is_none() {
            break;
        }
    }
    Ok(out)
}

/// GraphQL node → the flat shape `ingest::github_docs` reads.
fn reshape(n: &Value) -> Value {
    let labels: Vec<Value> = n["labels"]["nodes"]
        .as_array()
        .map(|a| a.iter().filter_map(|l| l["name"].as_str()).map(|name| json!({"name": name})).collect())
        .unwrap_or_default();
    json!({
        "number": n["number"],
        "title": n["title"],
        "body": n["body"],
        "state": n["state"],
        "url": n["url"],
        "createdAt": n["createdAt"],
        "author": {"login": n["author"]["login"].as_str().unwrap_or("")},
        "labels": labels,
        // PR-only; null on issues. The LOC stats read these from the raw
        // corpus directly — ingest's doc shape is unaffected.
        "additions": n["additions"],
        "deletions": n["deletions"],
        "mergedAt": n["mergedAt"],
    })
}

fn post(agent: &ureq::Agent, token: &str, query: &str, vars: Value) -> Result<Value> {
    let resp = agent
        .post(ENDPOINT)
        .set("Authorization", &format!("Bearer {token}"))
        .set("User-Agent", "synty")
        .set("Accept", "application/vnd.github+json")
        .send_json(json!({"query": query, "variables": vars}));
    let body: Value = match resp {
        Ok(r) => r.into_json().map_err(|e| anyhow!("github: bad JSON: {e}"))?,
        Err(ureq::Error::Status(code, r)) => {
            let snippet: String = r.into_string().unwrap_or_default().chars().take(200).collect();
            bail!("github: HTTP {code}: {snippet}");
        }
        Err(e) => bail!("github: request failed: {e}"),
    };
    if let Some(errs) = body["errors"].as_array() {
        if !errs.is_empty() {
            let msgs: Vec<&str> = errs.iter().filter_map(|e| e["message"].as_str()).collect();
            bail!("github: graphql errors: {}", msgs.join("; "));
        }
    }
    body.get("data").cloned().ok_or_else(|| anyhow!("github: empty data"))
}

fn resolve_token() -> Result<String> {
    if let Ok(t) = std::env::var("GITHUB_TOKEN") {
        if !t.trim().is_empty() {
            return Ok(t.trim().to_string());
        }
    }
    // Local convenience: borrow the gh CLI's token if it is logged in.
    if let Ok(o) = std::process::Command::new("gh").args(["auth", "token"]).output() {
        let t = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if o.status.success() && !t.is_empty() {
            return Ok(t);
        }
    }
    bail!("no GitHub token: set GITHUB_TOKEN (a PAT with repo scope) or run `gh auth login`")
}

fn quick_agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout_read(Duration::from_secs(30)).build()
}

/// The authenticated login plus every org the token can see — the accounts
/// `synty init` pins from (personal account first, then orgs).
pub fn accounts() -> Result<Vec<String>> {
    let token = resolve_token()?;
    let data = post(&quick_agent(), &token, "query { viewer { login organizations(first:100) { nodes { login } } } }", json!({}))?;
    let mut out = Vec::new();
    if let Some(login) = data["viewer"]["login"].as_str() {
        out.push(login.to_string());
    }
    if let Some(nodes) = data["viewer"]["organizations"]["nodes"].as_array() {
        out.extend(nodes.iter().filter_map(|o| o["login"].as_str()).map(str::to_string));
    }
    Ok(out)
}

/// A release asset: its filename and the API URL that serves its bytes.
pub struct ReleaseAsset {
    pub name: String,
    pub api_url: String,
}

/// A published release: the tag (e.g. `v0.2.0`) and its uploaded assets. Used by
/// `synty upgrade` — the binary distributes through GitHub Releases, fetched
/// with the same token synty already uses for PRs/issues.
pub struct Release {
    pub tag: String,
    pub assets: Vec<ReleaseAsset>,
}

impl Release {
    pub fn asset(&self, name: &str) -> Option<&ReleaseAsset> {
        self.assets.iter().find(|a| a.name == name)
    }
    pub fn asset_names(&self) -> String {
        self.assets.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", ")
    }
}

/// The repo's latest published release (REST). Ok(None) when the repo has no
/// releases yet (404). Needs a token (private repos); callers that nag treat a
/// token error as "don't know" and stay quiet.
pub fn latest_release(repo: &str) -> Result<Option<Release>> {
    let token = resolve_token()?;
    let url = format!("https://api.github.com/repos/{repo}/releases/latest");
    let body: Value = match quick_agent()
        .get(&url)
        .set("Authorization", &format!("Bearer {token}"))
        .set("User-Agent", "synty")
        .set("Accept", "application/vnd.github+json")
        .call()
    {
        Ok(r) => r.into_json().map_err(|e| anyhow!("github: bad JSON: {e}"))?,
        Err(ureq::Error::Status(404, _)) => return Ok(None),
        Err(ureq::Error::Status(code, r)) => {
            let snippet: String = r.into_string().unwrap_or_default().chars().take(200).collect();
            bail!("github: HTTP {code}: {snippet}");
        }
        Err(e) => bail!("github: request failed: {e}"),
    };
    let tag = body["tag_name"].as_str().unwrap_or_default().to_string();
    let assets = body["assets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    Some(ReleaseAsset { name: x["name"].as_str()?.to_string(), api_url: x["url"].as_str()?.to_string() })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Some(Release { tag, assets }))
}

/// Download a release asset's bytes from its API `url`. GitHub answers with a
/// 302 to a signed object-store URL that *rejects* an Authorization header, so
/// we disable auto-follow and re-request the redirect target without auth.
pub fn download_asset(api_url: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let token = resolve_token()?;
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(180))
        .build();
    let resp = agent
        .get(api_url)
        .set("Authorization", &format!("Bearer {token}"))
        .set("User-Agent", "synty")
        .set("Accept", "application/octet-stream")
        .call();
    let mut reader = match resp {
        // Direct bytes (some setups don't redirect).
        Ok(r) if r.status() == 200 => r.into_reader(),
        // The expected case: follow the signed redirect WITHOUT the auth header.
        Ok(r) if (300..400).contains(&r.status()) => {
            let loc = r.header("location").ok_or_else(|| anyhow!("github: asset redirect without Location"))?.to_string();
            quick_agent()
                .get(&loc)
                .set("User-Agent", "synty")
                .call()
                .map_err(|e| anyhow!("github: signed asset fetch failed: {e}"))?
                .into_reader()
        }
        Ok(r) => bail!("github: unexpected asset status {}", r.status()),
        Err(ureq::Error::Status(code, r)) => {
            let snippet: String = r.into_string().unwrap_or_default().chars().take(200).collect();
            bail!("github: HTTP {code}: {snippet}");
        }
        Err(e) => bail!("github: asset download failed: {e}"),
    };
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map_err(|e| anyhow!("github: read asset: {e}"))?;
    Ok(buf)
}

/// Every member of `org` the token can see — the roster's "core users", so
/// fleet coverage flags org members who go untracked, not every external
/// contributor who happened to open a PR. Errors when `org` is a user account
/// or the token lacks org-read visibility; the caller treats that as "unknown"
/// and the roster falls back to all active authors.
pub fn org_members(org: &str) -> Result<Vec<String>> {
    let token = resolve_token()?;
    let agent = quick_agent();
    let q = r#"query($org:String!,$cursor:String){
  organization(login:$org){
    membersWithRole(first:100,after:$cursor){
      pageInfo{hasNextPage endCursor}
      nodes{login}
    }
  }
}"#;
    let mut out = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let data = post(&agent, &token, q, json!({"org": org, "cursor": cursor}))?;
        let conn = &data["organization"]["membersWithRole"];
        for n in conn["nodes"].as_array().cloned().unwrap_or_default() {
            if let Some(l) = n["login"].as_str() {
                out.push(l.to_string());
            }
        }
        if conn["pageInfo"]["hasNextPage"].as_bool() != Some(true) {
            break;
        }
        cursor = conn["pageInfo"]["endCursor"].as_str().map(String::from);
        if cursor.is_none() {
            break;
        }
    }
    Ok(out)
}

/// The `k` most-recently-pushed, non-archived repos owned by `owner` (an org or
/// a user — tries org first, falls back to user so a personal account works).
pub fn active_repos(owner: &str, k: usize) -> Result<Vec<String>> {
    repos_under("organization", "", owner, k).or_else(|_| repos_under("user", ", ownerAffiliations:[OWNER]", owner, k))
}

fn repos_under(root: &str, affil: &str, owner: &str, k: usize) -> Result<Vec<String>> {
    let token = resolve_token()?;
    let q = format!(
        "query($owner:String!,$k:Int!){{ {root}(login:$owner){{ repositories(first:$k, isArchived:false{affil}, orderBy:{{field:PUSHED_AT,direction:DESC}}){{ nodes{{ name }} }} }} }}"
    );
    let data = post(&quick_agent(), &token, &q, json!({"owner": owner, "k": k as i64}))?;
    let nodes = data[root]["repositories"]["nodes"].as_array().ok_or_else(|| anyhow!("{owner}: no repositories"))?;
    Ok(nodes.iter().filter_map(|n| n["name"].as_str().map(str::to_string)).collect())
}

/// RFC3339 cutoff `since_days` before now, computed from the system clock.
fn days_ago_rfc3339(days: u64) -> String {
    use chrono::{SecondsFormat, Utc};
    (Utc::now() - chrono::Duration::days(days as i64)).to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(number: i64, created: &str, state: &str) -> Value {
        json!({"number": number, "title": format!("t{number}"), "createdAt": created, "state": state})
    }

    // An incremental fetch merges into the existing corpus: a state flip
    // replaces the old item, new items insert, items older than the retention
    // window drop, and the result stays newest-created-first like a full scrape.
    #[test]
    fn incremental_merge_updates_inserts_and_expires() {
        let existing = vec![
            item(1, "2026-05-01T00:00:00Z", "OPEN"),
            item(2, "2026-03-01T00:00:00Z", "OPEN"), // now outside the window
        ];
        let fetched = vec![
            item(1, "2026-05-01T00:00:00Z", "MERGED"), // state flip
            item(3, "2026-06-01T00:00:00Z", "OPEN"),   // new
        ];
        let merged = merge_items(existing, fetched, "2026-04-01T00:00:00Z");
        let numbers: Vec<i64> = merged.iter().map(|v| v["number"].as_i64().unwrap()).collect();
        assert_eq!(numbers, vec![3, 1], "newest first; #2 expired");
        assert_eq!(merged[1]["state"], "MERGED", "the fetch's state wins");
    }

    // The fetch floor: last scrape minus slack, never older than the window —
    // and a never-scraped corpus backfills the whole window.
    #[test]
    fn floor_is_last_scrape_minus_slack_bounded_by_window() {
        let window = "2026-03-01T00:00:00Z";
        assert_eq!(scrape_floor("", window), window, "no manifest → full window");
        assert_eq!(scrape_floor("2026-06-10T12:00:00Z", window), "2026-06-10T11:00:00Z");
        assert_eq!(scrape_floor("2026-01-01T00:00:00Z", window), window, "stale scrape → window");
    }

    // Staleness drives the build's refresh decision: never-scraped and old
    // manifests are stale, a fresh one is not.
    #[test]
    fn staleness_reads_the_manifest() {
        let dir = std::env::temp_dir().join(format!("synty-ghstale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        assert!(stale(d, 60), "no manifest → stale");
        let m = crate::sync::GithubManifest { scraped_at: days_ago_rfc3339_minutes(0), ..Default::default() };
        std::fs::write(dir.join(crate::sync::GH_MANIFEST), serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(!stale(d, 60), "just scraped → fresh");
        let m = crate::sync::GithubManifest { scraped_at: days_ago_rfc3339_minutes(120), ..Default::default() };
        std::fs::write(dir.join(crate::sync::GH_MANIFEST), serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(stale(d, 60), "two hours old → stale");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

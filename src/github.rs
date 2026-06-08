// Machine-independent GitHub ingestion. Pulls a trailing window of PRs and
// issues per repo straight from the GitHub GraphQL API with a token, so it runs
// anywhere (CI, a server) without a developer machine or the `gh` CLI. Output
// is the per-repo JSON `ingest` reads: corpus/github/{prs,issues}-<repo>.json.
//
// Token resolution: $GITHUB_TOKEN, else `gh auth token` for local convenience.

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;

const ENDPOINT: &str = "https://api.github.com/graphql";
const PAGE: i64 = 50;

pub fn run(owner: &str, repos: Option<String>, since_days: u64, out: &str) -> Result<()> {
    let token = resolve_token()?;
    let cutoff = days_ago_rfc3339(since_days);
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

    let mut tot_pr = 0;
    let mut tot_is = 0;
    for repo in &repos {
        let prs = fetch(&agent, &token, owner, repo, Kind::Pr, &cutoff)?;
        let issues = fetch(&agent, &token, owner, repo, Kind::Issue, &cutoff)?;
        std::fs::write(Path::new(out).join(format!("prs-{repo}.json")), serde_json::to_string(&prs)?)?;
        std::fs::write(Path::new(out).join(format!("issues-{repo}.json")), serde_json::to_string(&issues)?)?;
        tot_pr += prs.len();
        tot_is += issues.len();
        eprintln!("{:<22} prs={:<4} issues={}", repo, prs.len(), issues.len());
    }
    eprintln!("github: {tot_pr} PRs + {tot_is} issues from {} repos (since {cutoff}) → {out}", repos.len());
    Ok(())
}

#[derive(Clone, Copy)]
enum Kind {
    Pr,
    Issue,
}

/// Paginate a repo's PRs or issues (newest first), keeping nodes created at or
/// after `cutoff` and stopping once they fall older. Each node is reshaped into
/// the JSON `ingest` expects (number, title, body, author.login, url, labels,
/// state, createdAt).
fn fetch(agent: &ureq::Agent, token: &str, owner: &str, repo: &str, kind: Kind, cutoff: &str) -> Result<Vec<Value>> {
    let conn = match kind {
        Kind::Pr => "pullRequests",
        Kind::Issue => "issues",
    };
    let query = format!(
        r#"query($owner:String!,$name:String!,$cursor:String){{
  repository(owner:$owner,name:$name){{
    {conn}(first:{PAGE},after:$cursor,orderBy:{{field:CREATED_AT,direction:DESC}}){{
      pageInfo{{hasNextPage endCursor}}
      nodes{{number title body state url createdAt author{{login}} labels(first:20){{nodes{{name}}}}}}
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
            let created = n["createdAt"].as_str().unwrap_or("");
            if created < cutoff {
                return Ok(out); // newest-first: everything below is older
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
/// `synty setup` offers to backfill from (personal account first, then orgs).
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

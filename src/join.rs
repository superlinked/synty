// `synty join [bucket]` — one-command onboarding. Idempotent and
// non-interactive: point a machine at the team bucket (or run local with no
// arg), pin its GitHub identity so sessions attribute to the same login as the
// person's PRs, turn on the login-time tracker, and run the first build — so a
// single command goes from nothing to "tracking + a viewer". Re-running with a
// bucket is the local→bucket switch (the build's event sync does the rest).
//
// This replaces the old interactive `setup`: synty isn't released yet, so there
// is one onboarding path, not two.

use crate::{config, github, identity, track, up};
use anyhow::Result;

pub fn run(bucket: Option<String>, machine: &str, no_build: bool) -> Result<()> {
    let mut cfg = config::load();

    // 1. Bucket: setting it is the local→bucket switch; absent → stay local
    //    (resolve_bucket falls back to the local .synty store).
    if let Some(b) = &bucket {
        cfg.bucket = Some(b.clone());
    }

    // 2. GitHub identity — best-effort, no prompts. With a token, pin the login
    //    (so sessions merge with the person's PRs) and default the backfill org
    //    for a solo user; a bucket joiner gets the corpus from the bucket, so
    //    this is optional. Without a token, sessions attribute to the git
    //    identity (see identity::actor).
    match github::accounts() {
        Ok(accounts) if !accounts.is_empty() => {
            identity::cache_github_login(&accounts[0]);
            if cfg.org.is_none() {
                cfg.org = Some(accounts[0].clone());
            }
            eprintln!("join: signed in as {} — sessions attribute to this login", accounts[0]);
        }
        _ => eprintln!(
            "join: no GitHub token — sessions attribute to your git identity; set GITHUB_TOKEN (or `gh auth login`) to link your PRs"
        ),
    }
    config::save(&cfg)?;

    // 3. Track at login.
    match track::autostart_set(true) {
        Ok(()) => eprintln!("join: login-time tracker enabled"),
        Err(e) => eprintln!("join: autostart unavailable ({e}) — run `synty up` to track in the foreground"),
    }

    // 4. First build: track → ingest → index → summarize → cluster, so the
    //    viewer has something to show immediately.
    if !no_build {
        up::build(&config::resolve_bucket(bucket), machine, 1.0, false)?;
    }

    // 5. Tell the user where they stand on the local→bucket ramp.
    match &cfg.bucket {
        Some(b) => eprintln!("\njoin: activated — tracking → {b}. Open the viewer: synty tui"),
        None => eprintln!(
            "\njoin: tracking locally. When you're ready to join your team, run `synty join <bucket>`. Open the viewer: synty tui"
        ),
    }
    Ok(())
}

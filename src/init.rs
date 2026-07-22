// `synty init [bucket]` — one-command onboarding. Idempotent and
// non-interactive: initialize synty on this machine — point it at the team
// bucket (or run local with no arg), pin its GitHub identity so sessions
// attribute to the same login as the person's PRs, turn on the login-time
// tracker, and run the first build — so a single command goes from nothing to
// "tracking + a viewer". Re-running with a bucket is the local→bucket switch
// (the build's event sync does the rest).
//
// This is the single onboarding path: one command, not the old interactive
// `setup` plus something else.

use crate::{bucket, config, github, identity, track, up};
use anyhow::{Context, Result};

pub struct Opts {
    pub bucket: Option<String>,
    pub aws_profile: Option<String>,
    pub capture_since: Option<String>,
    pub upload_interval: Option<u64>,
    pub capture_repos: Vec<String>,
    pub upload_redaction: Option<String>,
    pub mcp_redaction: Option<String>,
    pub campaign: Option<String>,
    pub role: Option<String>,
    pub machine: String,
    pub no_build: bool,
    pub no_autostart: bool,
}

pub fn run(opts: Opts) -> Result<()> {
    let Opts {
        bucket,
        aws_profile,
        capture_since,
        upload_interval,
        capture_repos,
        upload_redaction,
        mcp_redaction,
        campaign,
        role,
        machine,
        no_build,
        no_autostart,
    } = opts;
    let mut cfg = config::load();

    // 1. Bucket: setting it is the local→bucket switch; absent → stay local
    //    (resolve_bucket falls back to the local .synty store).
    if let Some(b) = &bucket {
        cfg.bucket = Some(b.clone());
    }
    if let Some(profile) = aws_profile {
        cfg.aws_profile = Some(profile);
    }
    if let Some(raw) = capture_since {
        cfg.capture_since = Some(config::normalize_capture_since(&raw)?);
    }
    if let Some(secs) = upload_interval {
        cfg.upload_interval_secs = Some(secs.max(1));
    }
    if !capture_repos.is_empty() {
        cfg.capture_repos = capture_repos;
    }
    if let Some(profile) = upload_redaction {
        cfg.upload_redaction = Some(profile.parse::<crate::redact::Profile>()?.as_str().into());
    }
    if let Some(profile) = mcp_redaction {
        cfg.mcp_redaction = Some(profile.parse::<crate::redact::Profile>()?.as_str().into());
    }
    if let Some(campaign) = campaign.filter(|value| !value.is_empty()) {
        cfg.campaign_id = Some(campaign);
    }
    if let Some(role) = role.filter(|value| !value.is_empty()) {
        cfg.campaign_role = Some(role);
    }

    // 2. GitHub identity — best-effort, no prompts. With a token, pin the login
    //    (so sessions merge with the person's PRs) and default the backfill org
    //    for a solo user; a bucket joiner gets the corpus from the bucket, so
    //    this is optional. Without a token, sessions attribute to the git
    //    identity (see identity::actor).
    match github::accounts() {
        Ok(accounts) if !accounts.is_empty() => {
            cfg.github_login = Some(accounts[0].clone());
            if cfg.org.is_none() {
                cfg.org = Some(accounts[0].clone());
            }
            eprintln!("init: signed in as {} — sessions attribute to this login", accounts[0]);
        }
        _ => eprintln!(
            "init: no GitHub token — sessions attribute to your git identity; set GITHUB_TOKEN (or `gh auth login`) to link your PRs"
        ),
    }
    config::save(&cfg)?;

    // Fail before installing a watcher when this binary lacks the cloud
    // backend, the durable provider cannot mint credentials, or the role
    // cannot list/read/write the bucket. The config stays saved so the error is fixable
    // in place by re-running init.
    if let Some(b) = &cfg.bucket {
        verify_bucket_access(b, &machine).with_context(|| format!("verify team bucket {b}"))?;
        eprintln!("init: verified read/write access to {b}");
    }

    // 3. First build: its one-shot tracker drains and uploads before the
    //    long-lived watcher exists, so two processes never race the cursor or
    //    upload ledgers during initialization.
    if !no_build {
        up::build(&config::resolve_bucket(bucket), &machine, 1.0, false)?;
    }

    // 4. Track at login after the initial one-shot has completed. Containers
    //    supervise `track --watch` themselves and opt out explicitly.
    if no_autostart {
        eprintln!("init: autostart skipped — run `synty track --watch` under your supervisor");
    } else {
        track::autostart_set(true).context(
            "enable login-time tracker (configuration was saved, but initialization is not complete)",
        )?;
        eprintln!("init: login-time tracker enabled");
    }

    // 5. Tell the user where they stand on the local→bucket ramp.
    match &cfg.bucket {
        Some(b) => eprintln!("\ninit: activated — tracking → {b}. Open the viewer: synty tui"),
        None => eprintln!(
            "\ninit: tracking locally. When you're ready to join your team, run `synty init <bucket>`. Open the viewer: synty tui"
        ),
    }
    Ok(())
}

/// Exercise the operations a fleet member needs before declaring activation:
/// recursive event listing, conditional immutable writes, and reads. The
/// machine-scoped marker is useful bucket metadata and never contains session
/// content or overwrites another member's object.
fn verify_bucket_access(bucket_uri: &str, machine: &str) -> Result<()> {
    let store = bucket::open(bucket_uri)?;
    store.list("events")?;
    let machine = identity::resolve_machine(machine);
    let key = format!("members/{machine}/activation.json");
    let body = serde_json::to_vec(&serde_json::json!({
        "machine": machine,
        "actor": identity::actor(),
        "synty_version": env!("CARGO_PKG_VERSION"),
    }))?;
    store.put_if_absent(&key, &body)?;
    store
        .get(&key)?
        .with_context(|| format!("activation marker {key} was not readable"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_activation_markers_are_machine_scoped() {
        let root = std::env::temp_dir().join(format!("synty-init-access-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        verify_bucket_access(root.to_str().unwrap(), "dev-a").unwrap();
        verify_bucket_access(root.to_str().unwrap(), "dev-b").unwrap();
        assert!(root.join("members/dev-a/activation.json").is_file());
        assert!(root.join("members/dev-b/activation.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }
}

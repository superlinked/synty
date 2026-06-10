// `synty setup` — first-run onboarding. Verifies GitHub access, lets you pick
// the org/account to back-fill, previews its most-active repos, offers login-time
// autostart, and writes .synty/config.json so the tracker, backfill, and TUI
// share the choices. Re-runnable any time to change them.

use crate::{config, github, track};
use anyhow::Result;
use std::io::{self, Write};

pub fn run() -> Result<()> {
    println!("synty setup\n");
    let mut cfg = config::load();

    // 1. GitHub: verify the token and discover the accounts it can see.
    let accounts = match github::accounts() {
        Ok(a) if !a.is_empty() => a,
        Ok(_) => {
            println!("Your GitHub token sees no accounts. Skipping GitHub for now.");
            Vec::new()
        }
        Err(e) => {
            println!("GitHub not connected: {e}");
            println!("Set GITHUB_TOKEN (a PAT with repo scope) or run `gh auth login`, then re-run `synty setup`.\n");
            Vec::new()
        }
    };

    if !accounts.is_empty() {
        cfg.github_login = Some(accounts[0].clone());
        println!("Signed in as {}.\n", accounts[0]);

        // 2. Which org/account to back-fill from.
        println!("Back-fill PRs & issues from which account?");
        for (i, a) in accounts.iter().enumerate() {
            let tag = if i == 0 { "  (you)" } else { "" };
            println!("  {}. {}{}", i + 1, a, tag);
        }
        let idx = ask("Choose", 1, accounts.len()) - 1;
        cfg.org = Some(accounts[idx].clone());

        // 3. How many of its most-active repos.
        let k = ask("How many of its most-active repos to track?", config::DEFAULT_REPOS, 200);
        cfg.backfill_repos = Some(k);

        // Preview so the choice is concrete.
        match github::active_repos(&accounts[idx], k) {
            Ok(repos) if !repos.is_empty() => {
                println!("\nWill track these {} repos in {}:", repos.len(), accounts[idx]);
                println!("  {}\n", repos.join(", "));
            }
            Ok(_) => println!("\n(no active repos found in {})\n", accounts[idx]),
            Err(e) => println!("\n(couldn't preview repos: {e})\n"),
        }
    }

    // 4. Fleet bucket (optional): the shared backplane every machine's tracker
    // pushes to and every viewer builds from. Blank = solo (local only).
    let cur = cfg.bucket.clone().unwrap_or_default();
    let hint = if cur.is_empty() { "blank = solo".to_string() } else { format!("current: {cur}") };
    let b = prompt(&format!("Shared fleet bucket (s3://… / gs://… / path) [{hint}]: "));
    let b = b.trim();
    if !b.is_empty() {
        cfg.bucket = Some(b.to_string());
    }

    // 5. Autostart.
    if yes_no("Start synty's tracker at login (keeps your memory fresh)?", true) {
        match track::autostart_set(true) {
            Ok(()) => println!("autostart enabled — synty tracks in the background from now on."),
            Err(e) => println!("autostart skipped: {e}"),
        }
    } else {
        println!("autostart left off — toggle it any time with `a` on the TUI's Status tab.");
    }

    config::save(&cfg)?;
    println!("\nSaved .synty/config.json. Run `synty up` to build, or `synty tui` to browse.");
    Ok(())
}

/// Prompt for a number in `[lo, hi]`, returning `default` on blank/invalid input.
fn ask(label: &str, default: usize, hi: usize) -> usize {
    let raw = prompt(&format!("{label} [default {default}]: "));
    raw.trim().parse::<usize>().ok().filter(|n| (1..=hi).contains(n)).unwrap_or(default)
}

fn yes_no(label: &str, default_yes: bool) -> bool {
    let hint = if default_yes { "[Y/n]" } else { "[y/N]" };
    match prompt(&format!("{label} {hint}: ")).trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default_yes,
    }
}

fn prompt(msg: &str) -> String {
    print!("{msg}");
    let _ = io::stdout().flush();
    let mut s = String::new();
    let _ = io::stdin().read_line(&mut s);
    s
}

# Contributing to synty

synty is one self-contained Rust binary. No build server, no services; you
develop and test it entirely offline. This file covers the day-to-day. The full
working rules live in [`AGENTS.md`](AGENTS.md), and the architecture lives in
[`docs/design.md`](docs/design.md).

## Build, test, run

```sh
cargo test                              # scenario suite: pure, no model/corpus/network
cargo build --release                   # the shipped build: plain CPU, portable
cargo build --release --features metal  # Apple Silicon GPU encode (~5.7x faster), for dev
```

`cargo test` stays pure on purpose, so anyone can run the whole suite in seconds.
Quality checks that need the model, a corpus, or the network live under
[`evals/`](evals/) and the on-demand `scripts/fleet-smoke.sh`, never in the unit
suite. `accelerate` (macOS CPU BLAS) and `mkl` (Linux) are the other opt-in
backends; none of the three is ever the default.

Run the pipeline end to end with `synty build`, or step through it: `ingest →
index → summarize → cluster → summarize`. `docs/design.md` explains why each stage
exists.

## Tests

Every behavioral change gets a scenario-style test written from what a user
expects, not from the implementation; the existing `#[cfg(test)]` blocks are the
pattern. When a change is meant to move a quality number, read the metric it
emits (a `[metrics <op>]` block on stderr) instead of eyeballing the output.

## Commits and PRs

Write the subject in the imperative, scoped to the area (`M1: Louvain topics
...`). The body explains why, and prefers numbers (docs/s, cluster counts, repo
and PR ids, dates) over adjectives. When you change a command, flag, default, or
metric, grep `README.md` / `docs/design.md` / `AGENTS.md` and reconcile them in the
same commit; stale docs drift one sentence at a time.

## Releasing

Binaries ship as GitHub Release assets, built by CI. Cut a release by bumping
the `Cargo.toml` version and pushing a matching tag; `.github/workflows/release.yml`
runs the test suite, builds each platform, and attaches `synty-<os>-<arch>`
(plus a `.sha256`) to the release:

```sh
# bump the Cargo.toml version, commit, then:
git tag v0.2.0 && git push origin v0.2.0
```

The matrix builds `macos-14` (`--features metal,s3,gcs`, Apple Silicon) and
`ubuntu-latest` (`--features s3,gcs,mcp-http`, Linux x64); the `s3`/`gcs` features read
the team's data bucket and are independent of where the binary ships. Add rows
for more platforms (Intel Mac `macos-13`, `ubuntu-24.04-arm`) as needed. Users
then update with `synty upgrade`, which reads the same release.

The same tag publishes an immutable `linux/amd64` container to
`851725219920.dkr.ecr.eu-central-1.amazonaws.com/synty:<version>`.
Deploy `deploy/aws/ecr-publisher.yaml` once in account `851725219920`, then set
the GitHub Actions repository variable `AWS_ECR_PUBLISH_ROLE_ARN` to the stack's
`PublisherRoleArn` output. The role trusts only version tags from
`superlinked/synty` and can push only the retained `synty` repository. The Helm
chart's empty image tag resolves to `Chart.appVersion`, so bump it with the
Cargo version before tagging.

```sh
aws cloudformation deploy --region eu-central-1 \
  --stack-name synty-ecr-publisher \
  --template-file deploy/aws/ecr-publisher.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Writing style

Prose in this repo (README, design docs, this file) avoids the usual AI tells:
no em dashes, no filler vocabulary, no "not X, but Y." Be specific, take a
position, vary sentence length, use semicolons. Read it aloud before committing.
It should sound like a person, not a press release.

## Rendering the demo GIFs

The README's GIFs come from `.tape` scripts under `docs/`, rendered with
[vhs](https://github.com/charmbracelet/vhs) (`brew install vhs`):

```sh
vhs docs/install.tape   # → docs/install.gif
vhs docs/tui.tape       # → docs/tui.gif
vhs docs/cli.tape       # → docs/cli.gif
```

The tapes are plain text. Re-render when the UI changes, and tune the `Sleep`s
to your machine; the first wait covers the model load plus the background
freshen. Each tape runs the real binary, so it captures whatever corpus it
points at. `tui` and `cli` are read-only; `install` is host-safe (a throwaway
`$SYNTY_HOME`, no login agent, no build).

## License

By contributing, you agree that your contributions are licensed under the
[Apache-2.0](LICENSE) license that covers this project.

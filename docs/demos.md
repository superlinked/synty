# Demos

Animated walkthroughs, generated from the `.tape` scripts in this folder with
[vhs](https://github.com/charmbracelet/vhs) (`brew install vhs`):

```sh
vhs docs/install.tape   # → docs/install.gif
vhs docs/tui.tape       # → docs/tui.gif
vhs docs/cli.tape       # → docs/cli.gif
```

The tapes are plain text — re-render whenever the UI changes, and tune the
`Sleep`s to your machine (the first wait covers the model load + the background
freshen). Each tape runs the **real binary**, so it captures whatever corpus
it's pointed at — render on a demo/shareable bucket, not your internal one.
`tui`/`cli` are read-only; `install` is host-safe (throwaway `$SYNTY_HOME`, no
login agent, no build).

## Onboarding — local trial → activated
![onboarding: local trial, then join the team](install.gif)

## Browse — the TUI
![the TUI: topics, drill-down, search, stats, status](tui.gif)

## Agent surface — Markdown from the CLI
![synty related / search / status printing Markdown](cli.gif)

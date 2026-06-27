# Render:  vhs docs/cli.tape   →   docs/cli.gif
# The agent-facing surface: Markdown to stdout. `related` derives a query from
# the repo's recent commits + changed files (no query needed), `search` is the
# explicit form, `status` shows health + activation. Run it from a repo and on a
# corpus you're happy to show (the GIF captures the real output).
Output docs/cli.gif
Set Theme "Catppuccin Mocha"
Set FontSize 15
Set Width 1240
Set Height 820
Set Padding 16

Type "synty related"  Enter
Sleep 8s

Type "synty search 'rate limiting middleware' --filter kind=pull_request"  Enter
Sleep 8s

Type "synty status"  Enter
Sleep 8s

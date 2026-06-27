# Render:  vhs docs/tui.tape   →   docs/tui.gif
# Browse synty's work memory: Topics → drill into one → Search → Stats → Status.
# Run it where you're happy to SHOW the data (a demo bucket, not your internal
# corpus — the GIF captures whatever `synty tui` displays). Tune the Sleeps to
# your machine: the first one waits out the model load + the background freshen.
Output docs/tui.gif
Set Theme "Catppuccin Mocha"
Set FontSize 15
Set Width 1240
Set Height 760
Set Padding 16

Hide
Type "synty tui"  Enter
Sleep 6s
Show

Sleep 2s              # Topics — the emergent themes (the default tab)
Down  Sleep 700ms
Down  Sleep 700ms
Enter Sleep 2500ms    # drill into a topic → its sessions / PRs / issues
Escape Sleep 1200ms   # back out to the topic list

Type "3"  Sleep 1200ms            # Search tab
Type "rate limiting"  Sleep 700ms
Enter  Sleep 2800ms               # ranked cards across the fleet

Type "4"  Sleep 2600ms            # Stats — what agents consume vs what shipped
Type "5"  Sleep 3000ms            # Status — activation + the fleet roster
Type "q"

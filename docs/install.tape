# Render:  vhs docs/install.tape   →   docs/install.gif
# The onboarding STORY: one paste → tracking → activated, shown via the
# local→bucket ramp. Host-safe by design — it runs in a throwaway $SYNTY_HOME
# and never enables your real login tracker or rebuilds anything. (The real
# install one-liner DOES register a login agent + run a first build; render that
# on a VM if you want the live version.)
Output docs/install.gif
Set Theme "Catppuccin Mocha"
Set FontSize 15
Set Width 1240
Set Height 560
Set Padding 16

Type "export SYNTY_HOME=$(mktemp -d) && mkdir -p $SYNTY_HOME/.synty"  Enter
Sleep 700ms
Type "# one paste from nothing to tracking + a viewer:"  Enter
Type "#   curl -fsSL <internal-url>/install.sh | sh -s -- gs://acme-synty"  Enter
Sleep 2s
Type "# start local (no bucket) to see what's in there:"  Enter
Type "synty status | head -3"  Enter
Sleep 2800ms
Type "# ready to join the team — point at the bucket, and you're activated:"  Enter
Type "printf '{\"bucket\":\"gs://acme-synty\"}' > $SYNTY_HOME/.synty/config.json"  Enter
Type "synty status | head -3"  Enter
Sleep 3s

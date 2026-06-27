# Render:  vhs docs/install.tape   →   docs/install.gif
# Onboarding: start local, then join the team. Rendered in a throwaway
# $HOME/$SYNTY_HOME so it never touches your real config or tracker. (The real
# install one-liner —  curl … | sh -s -- <bucket>  — just runs `synty init`.)
Output docs/install.gif
Set Theme "Catppuccin Mocha"
Set FontSize 15
Set Width 1240
Set Height 560
Set Padding 16

Hide
Type `H=$(mktemp -d); export HOME=$H SYNTY_HOME=$H; cd $H`  Enter
Type "clear"  Enter
Show

Type "synty init --no-build"  Enter
Sleep 3s
Type "synty status | head -3"  Enter
Sleep 2500ms
Type "synty init gs://acme-synty --no-build"  Enter
Sleep 3s
Type "synty status | head -3"  Enter
Sleep 3s

Output .github/assets/demo/demo-light.mp4
Output .github/assets/demo/demo-light.gif

Set Theme { "name": "Whimsy Light", "black": "#24292f", "red": "#d0355a", "green": "#1a7c3e", "yellow": "#9a6700", "blue": "#2160c4", "magenta": "#8250df", "cyan": "#1b7c83", "white": "#ffffff", "brightBlack": "#57606a", "brightRed": "#ef6487", "brightGreen": "#1a7c3e", "brightYellow": "#bf8700", "brightBlue": "#0969da", "brightMagenta": "#aa7ff0", "brightCyan": "#3192aa", "brightWhite": "#8b949e", "background": "#f6f8fa", "foreground": "#24292f", "selection": "#add6ff", "cursor": "#0969da" }

Set Shell "bash"
Set FontSize 28
Set Width 1620
Set Height 950

Hide
Type "export VHS_NO_SANDBOX='true'" Enter
Type "export PS1='$ '" Enter
Type "clear" Enter

Show
Type@150ms "things-eeg2 --help" Sleep 1s Enter
Sleep 5s

Show
Type@150ms "things-eeg2 --install-completion" Sleep 1s Enter
Sleep 2s

Show
Type@150ms "source ~/.bashrc" Sleep 1s Enter
Sleep 2s

Show
Type "clear"
Enter

Show
Type "# Now you can press 'Tab' for autocompletion"
Sleep 2
Enter

Show
Type "# The following assumes that you have downloaded the raw_data to ~/things_eeg2"
Enter
Sleep 3

Show
Type@150ms "things-eeg2 preprocess --"
Sleep 500ms
Tab 2
Sleep 2s

Show
Type@150ms "su"
Sleep 500ms
Tab 2
Type "1"
Sleep 1s
Type@150ms " --prof"
Tab 2
Enter
Sleep 2s
Ctrl+C
Ctrl+C
Sleep 3s
Type@120ms "# Preprocessing interrupted for demo" Enter
Sleep 2s
Output .github/assets/demo/demo-dark.mp4
Output .github/assets/demo/demo-dark.gif

Set Theme {"name":"Whimsy","black":"#535178","red":"#ef6487","green":"#5eca89","yellow":"#fdd877","blue":"#65aef7","magenta":"#aa7ff0","cyan":"#43c1be","white":"#ffffff","brightBlack":"#535178","brightRed":"#ef6487","brightGreen":"#5eca89","brightYellow":"#fdd877","brightBlue":"#65aef7","brightMagenta":"#aa7ff0","brightCyan":"#43c1be","brightWhite":"#ffffff","background":"#0d1117","foreground":"#b3b0d6","selection":"#3d3c58","cursor":"#b3b0d6"}

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
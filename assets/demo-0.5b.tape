# Renders the 0.5B base-vs-LoRA demo to MP4 + GIF. Rebuild from the repo root:
#   vhs assets/demo-0.5b.tape
# The demo loads Qwen2.5-0.5B (MLX) and generates every answer LIVE, greedy, at
# record time — nothing is canned. Needs the converted adapter (setup_adapter.py
# --model qwen-0.5b) and `uv pip install raif-format`. stderr (HF fetch bars) is
# silenced so only the demo's own clean output is recorded.
Output "assets/demo-0.5b.mp4"
Output "assets/demo-0.5b.gif"

Set Shell "bash"
Set FontSize 22
Set Width 1360
Set Height 820
Set Padding 28
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful

Hide
Type "export PS1='$ '"
Enter
Type "uv run python -c 'pass'"
Enter
Type "clear"
Enter
Show

Type "uv run python examples/demo.py 2>/dev/null"
Sleep 700ms
Enter
# Ends during the demo's ~8s outro hold — on the call-to-action, not the
# blinking shell prompt. Bump if a slower machine cuts the last slide.
Sleep 52s

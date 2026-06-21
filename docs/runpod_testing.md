# RunPod GPU testing runbook (vLLM e2e)

vLLM doesn't run on Apple Silicon, so the plugin end-to-end
(`cuda/cloud/serve_smoke_v019.sh`, all three OpenAI paths) runs on a CUDA GPU.
This captures the gotchas so we don't rediscover them each time. See also
`vllm_tool_calling.md` (the tool-path fix) and `vllm_e2e_results.md` (results).

## 1. Provision ONE pod — with a teardown backstop

- One pod at a time. Run `runpodctl pod list` first; if it isn't `[]`, stop.
- GPU: **A40** (48 GB) or any 24 GB+ Ampere/Ada card. A 3B + LoRA needs little VRAM.
- Image: a **RunPod PyTorch** image — it ships `openssh` so you can SSH in. The official
  `vllm/vllm-openai` image has no sshd and its entrypoint is the server, so it's awkward
  for interactive runs.
- Always set `--terminate-after` as a backstop in case cleanup is missed.

```bash
TERM_AT=$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)        # macOS; Linux: date -u -d '+3 hours' +%Y-%m-%dT%H:%M:%SZ
runpodctl pod create --name raif-vllm --gpu-id "NVIDIA A40" --gpu-count 1 \
  --image runpod/pytorch:<tag> --container-disk-in-gb 40 \
  --volume-in-gb 30 --volume-mount-path /workspace \
  --ssh --ports 22/tcp --cloud-type SECURE --terminate-after "$TERM_AT"
```

## 2. SSH — the proxy is flaky under load

- Endpoint: `runpodctl ssh info <POD_ID>` → a direct `root@<ip> -p <port>` you can
  rsync/scp over (key `~/.runpod/ssh/runpodctl-ssh-key`).
- Use **plain one-shot SSH with retries**, NOT `ControlMaster` multiplexing. Under a
  high host load average (seen ~18 while vLLM loads), the multiplexed master gets
  throttled and dies with `exit 255`; one-shot connections are more robust.
- `255` is almost always transient throttling, not a bad command — retry with a short
  backoff.
- For ephemeral hosts: `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`, and
  `-n` so backgrounded local commands don't hang on stdin.
- rsync with `--no-owner --no-group` (the volume denies chown). The shell here is zsh,
  which does **not** word-split unquoted variables — inline ssh `-i/-o/-p` options or
  use `${=VAR}`, or they get passed as a single argument.

## 3. Run long jobs in tmux; log to /workspace

- `/root` is the container's **ephemeral** layer; `/workspace` is the **persistent volume**.
  Put scripts and logs on `/workspace` so a container restart doesn't wipe them.
- Launch pattern that survives the flaky proxy: create an **empty** session, then send the
  command — embedding a long command directly in `tmux new-session "<cmd>"` sometimes drops
  the connection.
  ```bash
  tmux new-session -d -s smoke
  tmux send-keys -t smoke 'cd /workspace/raif && WORKROOT=/workspace/raif \
    bash raif-lora/cuda/cloud/serve_smoke_v019.sh > /workspace/smoke.log 2>&1' Enter
  ```
- Tail `run.log` over the multiplexed connection and watch for a **terminal marker**
  (`OVERALL`, `EXITCODE=`, `FATAL`) — not just the success line, or a crash looks like "still running".

## 4. Version pins (stock image vs the GPU driver)

The single-plugin flow targets **vLLM 0.19** (the last CUDA-12 vLLM, which carries the
reasoning-parser decode + `render_chat` inject hooks the plugin needs).
`serve_smoke_v019.sh` already encodes these pins; know them so failures are recognisable:

| Pin | Why |
|---|---|
| `vllm==0.19.0` | last CUDA-12 vLLM; its torch runs on an A40's CUDA-12.x driver. Newer vLLM ships cu130 torch → `NVIDIA driver too old, found version 12090`. |
| `fastapi==0.115.6` | pins starlette `<0.42`; vLLM's prometheus instrumentator breaks on starlette 1.x → `'_IncludedRouter' object has no attribute 'path'` (server never becomes healthy). Expect loud pip "dependency conflict" lines about sse-starlette / prometheus-fastapi-instrumentator wanting newer starlette — they are benign; the pin is deliberate. |
| use `python3.12` | the image's `python3` is an empty 3.10; torch/vllm live under 3.12. |

## 5. Run the e2e

`serve_smoke_v019.sh` expects the two working trees already present as siblings under
`$WORKROOT` (it does **not** clone — the plugin lives in `raif-lora/packages/vllm` and the
`raif` format lib with `schema_bridge`/`stream` in `raif-standard/packages/py`). rsync the
needed subtrees, then run it in tmux:

```bash
# from the laptop — rsync only what the smoke needs (skip .venv/.git/data)
RSH='ssh -i ~/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no -p <port>'
rsync -az --no-owner --no-group -e "$RSH" raif-standard/packages/py  root@<ip>:/workspace/raif/raif-standard/packages/
rsync -az --no-owner --no-group -e "$RSH" raif-lora/packages/vllm    root@<ip>:/workspace/raif/raif-lora/packages/
rsync -az --no-owner --no-group -e "$RSH" raif-lora/cuda/cloud/      root@<ip>:/workspace/raif/raif-lora/cuda/cloud/
rsync -az --no-owner --no-group -e "$RSH" raif-lora/examples/smoke_plugin.py root@<ip>:/workspace/raif/raif-lora/examples/

# on the pod
WORKROOT=/workspace/raif bash raif-lora/cuda/cloud/serve_smoke_v019.sh
```

- It installs `vllm==0.19.0` + `fastapi==0.115.6`, editable-installs both `raif` packages,
  serves base + LoRA with `VLLM_PLUGINS=raif --reasoning-parser raif --tool-call-parser raif`
  and the `--chat-template`, then runs `examples/smoke_plugin.py` across all five paths.
- `--enforce-eager` + `--max-model-len 8192` cut startup time.
- The model caches in `/workspace/.hf-cache`, so re-runs skip the download (~6 GB).

## 6. ALWAYS tear down and verify

```bash
runpodctl pod delete <POD_ID>
runpodctl pod list          # must print []
```

Then confirm `runpodctl me` shows spend/hr `0`. Don't leave a GPU idle while you think.

# RunPod GPU testing runbook (vLLM e2e)

vLLM doesn't run on Apple Silicon, so the tool-call end-to-end (`cuda/cloud/serve_test.sh`)
runs on a CUDA GPU. This captures the gotchas so we don't rediscover them each time.
See also `vllm_tool_calling.md` (the fix) and `vllm_e2e_results.md` (results).

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

- Endpoint: `runpodctl ssh info <POD_ID>`.
- Use **SSH connection multiplexing** (`ControlMaster=auto`, a `ControlPath` socket,
  `ControlPersist=10m`). Repeated fresh connections get throttled — symptom is `exit 255`
  with no output, especially while vLLM is loading (host load average spikes ~9).
- `255` is almost always transient throttling, not a bad command. Retry, or reuse the master.
- For ephemeral hosts: `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`, and
  `-n` so backgrounded local commands don't hang on stdin.

## 3. Run long jobs in tmux; log to /workspace

- `/root` is the container's **ephemeral** layer; `/workspace` is the **persistent volume**.
  Put scripts and logs on `/workspace` so a container restart doesn't wipe them.
- Launch pattern that survives the flaky proxy: create an **empty** session, then send the
  command — embedding a long command directly in `tmux new-session "<cmd>"` sometimes drops
  the connection.
  ```bash
  tmux new-session -d -s job
  tmux send-keys -t job 'bash /workspace/serve_test.sh > /workspace/run.log 2>&1' Enter
  ```
- Tail `run.log` over the multiplexed connection and watch for a **terminal marker**
  (`OVERALL`, `EXITCODE=`, `FATAL`) — not just the success line, or a crash looks like "still running".

## 4. Version pins (stock image vs the GPU driver)

The stock RunPod `pytorch:...-cu1290` image is bleeding-edge and breaks vLLM 0.11 three ways.
`serve_test.sh` already encodes these pins; know them so failures are recognisable:

| Pin | Why |
|---|---|
| `vllm==0.11.0` | torch 2.8/cu128 runs on an A40's CUDA-12.9 driver. Latest vLLM (0.23) ships cu130 torch → `NVIDIA driver too old, found version 12090`. |
| `transformers>=4.56,<5` | 5.x dropped `all_special_tokens_extended` → `TokenizersBackend has no attribute ...` on tokenizer init. |
| `fastapi==0.115.6` | pins starlette `<1.0`; vLLM 0.11's metrics break on starlette 1.x → `'_IncludedRouter' object has no attribute 'path'` (server never becomes healthy). |
| use `python3.12` | the image's `python3` is an empty 3.10; torch/vllm live under 3.12. |

## 5. Run the e2e

`serve_test.sh` clones both sibling repos, installs the pinned deps, serves base + LoRA with
the `raif` tool parser and `--chat-template`, then runs the smoke + shim tests. Until the
branches are merged to `main`, pass branch overrides:

```bash
STD_BRANCH=<branch> LORA_BRANCH=<branch> WORKROOT=/workspace/raif bash /workspace/serve_test.sh
```

- `--enforce-eager` + `--max-model-len 8192` cut startup time.
- The model caches in `/workspace/.hf-cache`, so re-runs skip the download (~6 GB).

## 6. ALWAYS tear down and verify

```bash
runpodctl pod delete <POD_ID>
runpodctl pod list          # must print []
```

Then confirm `runpodctl me` shows spend/hr `0`. Don't leave a GPU idle while you think.

# RAIF LoRA — cloud run (RunPod / SSH GPU box)

Run the `cuda/` ladder on a rented GPU when the local 5070 Ti box is
unavailable. RunPod-first; adaptable to any SSH-reachable Ubuntu+CUDA box by
overriding `WORKROOT` and the file-transfer steps. Written for a **24 GB
Ada/Ampere card (RTX 4090 / L4 / A10)** — ~$0.3–0.7/hr on-demand.

> This is the **first execution of the CUDA port on a GPU**. Expect to debug
> unsloth/trl/bitsandbytes version drift — see *Troubleshooting* below. The
> smoke run is the validation gate: it must roughly reproduce the MLX numbers
> (**69 % valid / 23 % holdout fidelity**) before any longer run is trusted.

## 0. One-time account setup

Add your SSH public key to RunPod **before** launching, so every pod accepts
your key (official templates inject `$PUBLIC_KEY` into `authorized_keys` and run
sshd automatically):

```sh
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub   # or paste the .pub in console → Settings → SSH Keys
```

To move files off the pod later, install `runpodctl` on your **laptop** too:

```sh
brew install runpod/runpodctl/runpodctl        # macOS; configure once if prompted: runpodctl config --apiKey <key>
```

## 1. Provision the pod

- **GPU:** 24 GB is the documented **minimum** (full-precision LoRA,
  `load_in_4bit=False`) — RTX 4090 / L4 / A10. 16 GB is **experimental**: requires
  `--micro-batch 2` and may need a precision fallback (see Troubleshooting).
- **Template:** official **RunPod PyTorch** template (it runs sshd + injects your
  key out of the box). Pick a **recent one (torch ≥ 2.5, CUDA ≥ 12.1)** so unsloth's
  torch floor is already satisfied and the bootstrap reuses the image torch instead
  of re-pulling a multi-GB wheel. Set the pod's **CUDA Version filter ≥ 12.1**.
- **Volume disk:** ≥ 40 GB. It mounts at **`/workspace`** and survives stop/restart
  but is **deleted on _terminate_** — the container disk is small and wiped on every
  stop, so all caches + adapters must live under `/workspace` (the bootstrap does this).
  For results you want to keep across pods, attach a **Network Volume** (also mounts
  at `/workspace`, persists independently).
- **SSH:** the **Connect** tab shows two options. *Basic SSH* (`ssh <POD_ID>@ssh.runpod.io`)
  is enough to run the bootstrap. Only *full SSH* (rent a **public IP** + exposed TCP
  port 22, shown as `ssh root@<ip> -p <port>`) supports **scp/rsync**; without it, use
  `runpodctl` for file transfer (§4). The bootstrap itself is just an outbound `curl`,
  so it works on either.

## 2. One-shot bootstrap (on the pod)

SSH in (either mode), then:

```sh
curl -fsSL https://raw.githubusercontent.com/skrrt-sh/raif-lora/main/cuda/cloud/bootstrap.sh | bash
```

That single command (idempotent — safe to re-run): checks the GPU, clones
`raif-lora` + `raif-standard` as siblings under `/workspace/raif`, installs the
userland stack (`cuda/cloud/requirements-cloud.txt`) **into the image's torch
interpreter** — no venv, so pip reuses the preinstalled torch instead of
re-pulling it — installs **bun** + the prototype decoder, regenerates the
gitignored data (`make_data.sh full`, seed 0 → the same 1235/65/500 split),
runs the **eval-meter oracle gate**, then trains + evals the **smoke** stage
and prints the numbers to compare against MLX.

Knobs:

```sh
curl -fsSL …/bootstrap.sh | RUN_STAGE=0 bash        # set up only, don't train
curl -fsSL …/bootstrap.sh | STAGE=warm bash         # bootstrap straight into warm
curl -fsSL …/bootstrap.sh | OPTIM=adamw_torch bash  # if bitsandbytes errors
```

## 3. Climb the ladder

After smoke validates, from `/workspace/raif/raif-lora`:

```sh
# no venv — the stack is installed into the image's python (`python` already has it)
export PATH="$HOME/.bun/bin:$PATH" HF_HOME=/workspace/raif/.hf-cache

# stage 2 — warm. Gate: valid fidelity ≥ 75% AND holdout > smoke's 23%
python cuda/train_unsloth.py --stage warm && python cuda/eval_cuda.py --adapter ./adapters-cuda/warm --n 13

# stage 4 — full (~2–4 hr). Acceptance gate: parse ≥98%, fid ≥95%, token Δ ≤ −8%, no holdout regression
python cuda/train_unsloth.py --stage full && python cuda/eval_cuda.py --adapter ./adapters-cuda/full --n 13
```

Gates and pinned results: `../../ITERATION_PLAN.md`. Parity table and the
deliberate CUDA divergences: `../README.md`.

> Long runs: launch under `tmux`/`nohup` so an SSH drop doesn't kill training.
> `python cuda/train_unsloth.py --stage full 2>&1 | tee logs/cuda-full.log`.

## 4. Pull the adapter back (BEFORE teardown)

Adapters are gitignored and `/workspace` is wiped on **terminate** — copy the
result off the box first, or it's gone.

**Primary — `runpodctl` (works on any pod, no public IP):** it sends one file,
so tar the adapter dir first.

```sh
# on the POD:
cd /workspace/raif/raif-lora
tar czf /workspace/full.tgz -C adapters-cuda full
runpodctl send /workspace/full.tgz          # prints a one-time code

# on your LAPTOP:
runpodctl receive <code>                    # e.g. 8338-galileo-collect-fidel
tar xzf full.tgz -C ./adapters/             # → ./adapters/full
```

**Alternative — scp (only if you rented full SSH / a public IP):**

```sh
scp -P <port> -r root@<ip>:/workspace/raif/raif-lora/adapters-cuda/full ./adapters/cuda-full
scp -P <port> root@<ip>:/workspace/raif/raif-lora/logs/'cuda-*' ./logs/
```

(Or push the adapter to a GitHub Release / HF Hub from the pod.) **Then stop the
pod** to pause billing while keeping `/workspace`, or **terminate** it to stop
all charges — terminate erases the volume, so only do it after the copy above.

## Troubleshooting (the expected first-run drift)

| symptom | fix |
|---|---|
| `bitsandbytes` / `adamw_8bit` errors | re-run trainer with `--optim adamw_torch` (LoRA optimizer state is tiny — no real cost) |
| CUDA OOM on the `full` run (2048 seq) | `python cuda/train_unsloth.py --stage full --micro-batch 2` (grad-accum auto-rises to keep examples-seen identical) |
| `bun: command not found` in eval | `export PATH="$HOME/.bun/bin:$PATH"` (bootstrap sets it; new shells need it too) |
| eval can't find the decoder | the two repos must be **siblings**; confirm `/workspace/raif/raif-standard/prototype` exists and `bun install` ran there |
| `unsloth`/`trl` `SFTConfig` / `train_on_responses_only` API mismatch | versions drifted past the `requirements.txt` floors — check current args in context7 docs and adjust `cuda/train_unsloth.py` |
| HF rate-limit / gated model | base is `unsloth/Llama-3.2-3B-Instruct` (ungated, no token needed); if rate-limited, set `HF_TOKEN` |
| torch `is_available()` False but capability prints | driver predates the wheels — update the pod's NVIDIA driver / pick a newer template |
| container disk fills up mid-install | caches must be on `/workspace` — the bootstrap sets `HF_HOME`/`PIP_CACHE_DIR` there; in a fresh shell re-export them before pip/training |
| `scp`/`rsync` "connection refused" | you're on basic proxy SSH (no public IP) — use `runpodctl send`/`receive` (§4) or re-rent with a public IP |
| pip wants to **upgrade torch** during step 3a | image torch is older than unsloth's floor — pick a newer PyTorch template (torch ≥ 2.5), or `pip install` the unsloth version matching the image torch |
| `pip` dependency-resolution conflict on `unsloth`/`trl`/`transformers` | the pins in `requirements-cloud.txt` drifted vs the installed unsloth — reconcile against `unsloth`'s declared ranges (`pip show unsloth`) |

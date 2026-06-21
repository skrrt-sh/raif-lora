# raif-vllm

One vLLM plugin for transparent RAIF token savings. Install it and existing
OpenAI clients get RAIF on `tools` and `response_format` with no proxy and no
client changes.

```sh
pip install raif-vllm
VLLM_PLUGINS=raif vllm serve <model> --enable-lora --lora-modules raif=<lora> \
  --reasoning-parser raif
```

See `raif-lora/docs/` for the full serving guide.

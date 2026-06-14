/**
 * Talk to the local RAIF LoRA through the Vercel AI SDK, then decode the output.
 *
 * The model runs on your machine via `examples/serve.sh` (mlx-lm's
 * OpenAI-compatible server). The AI SDK doesn't care that it's local — it's just
 * another OpenAI-compatible endpoint. The one RAIF-specific step is at the
 * boundary: the model emits RAIF, so we run `decode()` from `raif-format` to get
 * a plain JS value back.
 *
 *   examples/serve.sh                 # terminal 1: start the server
 *   cd examples/ai-sdk && npm install && npm run demo   # terminal 2
 *
 * Why the custom `fetch`: mlx-lm 0.31.x doesn't apply the CLI --adapter-path per
 * request, so we inject the adapter into the request body as `adapters`. Drop
 * this once the upstream bug is fixed and the server alone will serve RAIF.
 */
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText } from "ai";
import { decode } from "raif-format";

const BASE_URL = process.env.RAIF_BASE_URL ?? "http://127.0.0.1:8899/v1";
// Match whatever serve.sh printed ("Client must send … adapters": "<path>").
const ADAPTER = process.env.RAIF_ADAPTER ?? "./adapters/llama-3b-mlx";

// Inject the MLX adapter path into every request body (see header note).
const injectAdapter: typeof fetch = async (url, options) => {
	if (options?.body && typeof options.body === "string") {
		const body = JSON.parse(options.body);
		body.adapters = ADAPTER;
		options = { ...options, body: JSON.stringify(body) };
	}
	return fetch(url, options);
};

const mlx = createOpenAICompatible({
	name: "mlx-raif",
	baseURL: BASE_URL,
	fetch: injectAdapter,
});

async function toRaif(payload: unknown): Promise<string> {
	const { text } = await generateText({
		model: mlx("default_model"),
		temperature: 0,
		maxOutputTokens: 1024,
		prompt: `Rewrite this JSON payload as RAIF:\n${JSON.stringify(payload)}`,
	});
	return text.trim();
}

async function main() {
	const payload = {
		user: "ada",
		tasks: ["write", "test", "ship"],
		done: false,
		count: 3,
	};

	console.log("→ input JSON:", JSON.stringify(payload));
	const raif = await toRaif(payload);
	console.log(`\n── RAIF (model output) ──\n ${raif}`);

	const result = decode(raif);
	console.log("\n── decode() → JSON ──");
	if (result.ok) {
		console.log(JSON.stringify(result.value, null, 2));
		const repairs = result.repairs ?? [];
		if (repairs.length)
			console.log(`(recovered via ${repairs.length} repair(s))`);
	} else {
		console.error("decode failed:", result.error);
		process.exit(1);
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});

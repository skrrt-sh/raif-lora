"""Oracle tests for eval_smoke's scoring meter.

The warm run's old 0% fidelity turned out to be a data artifact, not a model
failure — so before trusting any number this harness prints, prove the meter
itself: a perfect model must score 100/100, a value-corrupting model must
fail fidelity but not parse, and a grammar-breaking model must fail parse.

Run from raif-lora root:  uv run python src/test_eval_smoke.py
Uses the real bun decoder subprocess and real data/valid.jsonl (integration
style on purpose — the meter includes the decode path).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import eval_smoke


class FakeTok:
    """Stands in for the HF tokenizer; eval_group only calls apply_chat_template."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return messages[0]["content"]


def make_generate(outputs: list[str]):
    """A fake mlx_lm.generate that returns scripted outputs in call order."""
    it = iter(outputs)

    def generate(model, tok, prompt=None, max_tokens=0, verbose=False):
        return next(it)

    return generate


def stratified_examples(n: int = 6) -> list[dict]:
    examples = eval_smoke.load_examples(Path("./data/valid.jsonl"))
    return eval_smoke.sample_examples(examples, n, seed=0)


class MeterOracle(unittest.TestCase):
    def test_value_corruption_fails_fidelity_but_parses(self):
        examples = stratified_examples(4)
        outputs = [
            ex["messages"][1]["content"] + "\nzzz_oracle=injected by meter test"
            for ex in examples
        ]
        stats = eval_smoke.eval_group(
            "oracle-corrupt", examples, None, FakeTok(), make_generate(outputs)
        )
        self.assertEqual(stats["parse"], stats["n"])
        self.assertEqual(stats["fidelity"], 0)

    def test_grammar_garbage_fails_parse(self):
        examples = stratified_examples(3)
        outputs = ["I cannot help with that request" for _ in examples]
        stats = eval_smoke.eval_group(
            "oracle-garbage", examples, None, FakeTok(), make_generate(outputs)
        )
        self.assertEqual(stats["parse"], 0)
        self.assertEqual(stats["fidelity"], 0)

    def test_refusal_with_colon_parses_via_repair_and_is_counted(self):
        # Caveat made explicit: TIER 1 separator coercion (':' → '=') turns a
        # prose refusal containing a colon into a one-leaf object. Such outputs
        # count as parse ✓ — so the meter must surface how many parses needed
        # repairs, or the 98% parse gate is softer than it looks.
        examples = stratified_examples(3)
        outputs = [
            "I'm sorry, here is your object: it has three fields."
            for _ in examples
        ]
        stats = eval_smoke.eval_group(
            "oracle-refusal", examples, None, FakeTok(), make_generate(outputs)
        )
        self.assertEqual(stats["parse"], stats["n"])
        self.assertEqual(stats["fidelity"], 0)
        self.assertEqual(stats["repaired"], stats["n"])

    def test_perfect_echo_scores_full_parse_and_fidelity(self):
        examples = stratified_examples(6)
        self.assertGreater(len(examples), 0, "valid.jsonl is empty")
        outputs = [ex["messages"][1]["content"] for ex in examples]
        stats = eval_smoke.eval_group(
            "oracle-perfect", examples, None, FakeTok(), make_generate(outputs)
        )
        self.assertEqual(stats["skipped"], 0)
        self.assertEqual(stats["n"], len(examples))
        self.assertEqual(stats["parse"], stats["n"])
        self.assertEqual(stats["fidelity"], stats["n"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

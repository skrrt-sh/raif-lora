# RAIF grammars

- `raif.gbnf` — llama.cpp GBNF grammar for the RAIF wire format
  (grammar-constrained decoding). Ground truth for what it must accept is
  `raif-standard/prototype/src/raif.ts` `encode()`; everything
  context-sensitive (multiline nonce equality, table arity/prefix agreement)
  is delegated to the decoder's repair pass.
- `grammar_lint.ts` — lint test with a built-in GBNF interpreter (Earley
  acceptor). Asserts every corpus encoding is accepted and a negative set
  (`key=[` with no closer, empty key, `>>>` runs in bare values, …) is
  rejected.

Run the lint:

```sh
bun grammars/grammar_lint.ts
```

(Any cwd works; paths resolve relative to the file.)

Validate the grammar with llama.cpp itself:

```sh
llama-gbnf-validator grammars/raif.gbnf <raif-doc>
```

#!/usr/bin/env bun
// RAIF GBNF grammar lint.
//
// Run:  bun grammars/grammar_lint.ts                       (from raif-lora/)
//  or:  bun /path/to/raif-lora/grammars/grammar_lint.ts    (any cwd)
//
// Asserts that grammars/raif.gbnf accepts every `encode(corpusCase)` wire
// string from the raif-standard prototype (the encoder is ground truth),
// plus hand-written positives, and rejects a negative set.
//
// Contains a small GBNF interpreter sufficient for raif.gbnf:
//   - literals with escapes, char classes (negation, ranges, escapes),
//     alternation, grouping, repetition (* + ?), rule references;
//   - acceptance via an Earley chart parser with per-set deduplication:
//     ambiguity stays polynomial, never exponential.
// Not a full GBNF implementation (no {m,n} repetition, no 8-hex escapes).

import { corpus } from "../../raif-standard/packages/js/bench/corpus.ts";
import { encode } from "../../raif-standard/packages/js/src/raif.ts";

// ─── GBNF parsing ───────────────────────────────────────────────────────────

type CharSet = { neg: boolean; ranges: Array<[number, number]> };
type Sym = { kind: "ref"; name: string } | { kind: "char"; set: CharSet };
interface Prod {
  lhs: string;
  rhs: Sym[];
}

function parseGbnf(text: string): Prod[] {
  const prods: Prod[] = [];
  for (const { name, body } of splitRuleBlocks(stripComments(text))) {
    new BodyParser(body, name, prods).parseTopLevel();
  }
  return prods;
}

// Strip `#` comments, respecting string literals and char classes.
function stripComments(text: string): string {
  let out = "";
  let i = 0;
  let inStr = false;
  let inCls = false;
  while (i < text.length) {
    const c = text[i]!;
    if (inStr || inCls) {
      out += c;
      if (c === "\\") {
        out += text[i + 1] ?? "";
        i += 2;
        continue;
      }
      if (inStr && c === '"') inStr = false;
      else if (inCls && c === "]") inCls = false;
      i++;
      continue;
    }
    if (c === "#") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "[") inCls = true;
    out += c;
    i++;
  }
  return out;
}

function splitRuleBlocks(text: string): Array<{ name: string; body: string }> {
  const re = /(?:^|\n)[ \t]*([a-zA-Z][a-zA-Z0-9-]*)[ \t]*::=/g;
  const hits: Array<{ name: string; bodyStart: number; matchStart: number }> = [];
  for (let m = re.exec(text); m !== null; m = re.exec(text)) {
    hits.push({ name: m[1]!, bodyStart: m.index + m[0].length, matchStart: m.index });
  }
  return hits.map((h, i) => ({
    name: h.name,
    body: text.slice(h.bodyStart, i + 1 < hits.length ? hits[i + 1]!.matchStart : text.length),
  }));
}

class BodyParser {
  private pos = 0;
  private synth = 0;

  constructor(
    private src: string,
    private rule: string,
    private prods: Prod[],
  ) {}

  parseTopLevel(): void {
    const alts = this.parseAlternates();
    this.skipWs();
    if (this.pos < this.src.length) {
      throw new Error(
        `rule ${this.rule}: trailing junk: ${JSON.stringify(this.src.slice(this.pos, this.pos + 20))}`,
      );
    }
    for (const rhs of alts) this.prods.push({ lhs: this.rule, rhs });
  }

  private skipWs(): void {
    while (this.pos < this.src.length && /[ \t\r\n]/.test(this.src[this.pos]!)) this.pos++;
  }

  private parseAlternates(): Sym[][] {
    const alts: Sym[][] = [this.parseSequence()];
    this.skipWs();
    while (this.src[this.pos] === "|") {
      this.pos++;
      alts.push(this.parseSequence());
      this.skipWs();
    }
    return alts;
  }

  private parseSequence(): Sym[] {
    const syms: Sym[] = [];
    for (;;) {
      this.skipWs();
      const c = this.src[this.pos];
      if (c === undefined || c === "|" || c === ")") break;
      let item: Sym[];
      if (c === '"') item = this.parseLiteral();
      else if (c === "[") item = [this.parseClass()];
      else if (c === "(") item = [this.parseGroup()];
      else if (/[a-zA-Z]/.test(c)) item = [this.parseRef()];
      else
        throw new Error(`rule ${this.rule}: unexpected ${JSON.stringify(c)} at offset ${this.pos}`);
      syms.push(...this.applyPostfix(item));
    }
    return syms;
  }

  // Desugar * + ? into synthetic right-recursive rules.
  private applyPostfix(item: Sym[]): Sym[] {
    const c = this.src[this.pos];
    if (c !== "*" && c !== "+" && c !== "?") return item;
    this.pos++;
    const name = `${this.rule}'${this.synth++}`;
    const ref: Sym = { kind: "ref", name };
    if (c === "?") {
      this.prods.push({ lhs: name, rhs: item }, { lhs: name, rhs: [] });
    } else if (c === "*") {
      this.prods.push({ lhs: name, rhs: [...item, ref] }, { lhs: name, rhs: [] });
    } else {
      this.prods.push({ lhs: name, rhs: [...item, ref] }, { lhs: name, rhs: item });
    }
    return [ref];
  }

  private parseLiteral(): Sym[] {
    this.pos++; // opening "
    const syms: Sym[] = [];
    while (this.src[this.pos] !== '"') {
      if (this.pos >= this.src.length) throw new Error(`rule ${this.rule}: unterminated literal`);
      const code = this.readChar();
      syms.push({ kind: "char", set: { neg: false, ranges: [[code, code]] } });
    }
    this.pos++; // closing "
    return syms;
  }

  private parseClass(): Sym {
    this.pos++; // [
    let neg = false;
    if (this.src[this.pos] === "^") {
      neg = true;
      this.pos++;
    }
    const ranges: Array<[number, number]> = [];
    while (this.src[this.pos] !== "]") {
      if (this.pos >= this.src.length)
        throw new Error(`rule ${this.rule}: unterminated char class`);
      const lo = this.readChar();
      let hi = lo;
      if (this.src[this.pos] === "-" && this.src[this.pos + 1] !== "]") {
        this.pos++;
        hi = this.readChar();
      }
      ranges.push([lo, hi]);
    }
    this.pos++; // ]
    return { kind: "char", set: { neg, ranges } };
  }

  private parseGroup(): Sym {
    this.pos++; // (
    const alts = this.parseAlternates();
    this.skipWs();
    if (this.src[this.pos] !== ")") throw new Error(`rule ${this.rule}: expected ')'`);
    this.pos++;
    const name = `${this.rule}'${this.synth++}`;
    for (const rhs of alts) this.prods.push({ lhs: name, rhs });
    return { kind: "ref", name };
  }

  private parseRef(): Sym {
    const m = /^[a-zA-Z][a-zA-Z0-9-]*/.exec(this.src.slice(this.pos))!;
    this.pos += m[0].length;
    return { kind: "ref", name: m[0] };
  }

  private readChar(): number {
    const c = this.src[this.pos]!;
    if (c !== "\\") {
      this.pos++;
      return c.charCodeAt(0);
    }
    const e = this.src[this.pos + 1]!;
    this.pos += 2;
    switch (e) {
      case "n":
        return 10;
      case "r":
        return 13;
      case "t":
        return 9;
      case "\\":
        return 92;
      case '"':
        return 34;
      case "[":
        return 91;
      case "]":
        return 93;
      case "^":
        return 94;
      case "-":
        return 45;
      case "x": {
        const h = this.src.slice(this.pos, this.pos + 2);
        this.pos += 2;
        return Number.parseInt(h, 16);
      }
      case "u": {
        const h = this.src.slice(this.pos, this.pos + 4);
        this.pos += 4;
        return Number.parseInt(h, 16);
      }
      default:
        throw new Error(`rule ${this.rule}: unsupported escape \\${e}`);
    }
  }
}

// ─── Earley acceptor ─────────────────────────────────────────────────────────
// Polynomial in input length regardless of grammar ambiguity (items are
// deduplicated per chart set). Handles ε-productions from ? / * desugaring
// via the standard completed-in-set fix.

function compile(prods: Prod[], root = "root"): (input: string) => boolean {
  const byLhs = new Map<string, number[]>();
  prods.forEach((p, i) => {
    const list = byLhs.get(p.lhs);
    if (list) list.push(i);
    else byLhs.set(p.lhs, [i]);
  });
  for (const p of prods) {
    for (const s of p.rhs) {
      if (s.kind === "ref" && !byLhs.has(s.name)) {
        throw new Error(`undefined rule referenced: ${s.name} (from ${p.lhs})`);
      }
    }
  }
  if (!byLhs.has(root)) throw new Error(`no '${root}' rule`);

  const matches = (set: CharSet, code: number): boolean => {
    let hit = false;
    for (const [lo, hi] of set.ranges) {
      if (code >= lo && code <= hi) {
        hit = true;
        break;
      }
    }
    return set.neg ? !hit : hit;
  };

  interface Item {
    prod: number;
    dot: number;
    origin: number;
  }

  return function accepts(input: string): boolean {
    const n = input.length;
    const sets: Array<Set<string>> = Array.from({ length: n + 1 }, () => new Set());
    const queues: Item[][] = Array.from({ length: n + 1 }, () => []);
    const wants: Array<Map<string, Item[]>> = Array.from({ length: n + 1 }, () => new Map());
    const done: Array<Set<string>> = Array.from({ length: n + 1 }, () => new Set());

    const add = (i: number, prod: number, dot: number, origin: number): void => {
      const k = `${prod},${dot},${origin}`;
      if (sets[i]!.has(k)) return;
      sets[i]!.add(k);
      queues[i]!.push({ prod, dot, origin });
    };

    for (const p of byLhs.get(root)!) add(0, p, 0, 0);

    let accepted = false;
    for (let i = 0; i <= n; i++) {
      const q = queues[i]!;
      while (q.length > 0) {
        const it = q.pop()!;
        const { lhs, rhs } = prods[it.prod]!;
        if (it.dot < rhs.length) {
          const sym = rhs[it.dot]!;
          if (sym.kind === "ref") {
            let w = wants[i]!.get(sym.name);
            if (!w) {
              w = [];
              wants[i]!.set(sym.name, w);
            }
            w.push(it);
            for (const p of byLhs.get(sym.name)!) add(i, p, 0, i);
            if (done[i]!.has(sym.name)) add(i, it.prod, it.dot + 1, it.origin);
          } else if (i < n && matches(sym.set, input.charCodeAt(i))) {
            add(i + 1, it.prod, it.dot + 1, it.origin);
          }
        } else {
          // complete
          if (it.origin === i) done[i]!.add(lhs);
          const parents = wants[it.origin]!.get(lhs);
          if (parents) {
            for (const parent of parents) add(i, parent.prod, parent.dot + 1, parent.origin);
          }
          if (i === n && lhs === root && it.origin === 0) accepted = true;
        }
      }
    }
    return accepted;
  };
}

// ─── Test cases ──────────────────────────────────────────────────────────────

const grammarText = await Bun.file(new URL("./raif.gbnf", import.meta.url)).text();
const accepts = compile(parseGbnf(grammarText));

let failures = 0;
let total = 0;

function check(label: string, input: string, expected: boolean): void {
  total++;
  const got = accepts(input);
  const ok = got === expected;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  [${expected ? "accept" : "reject"}]  ${label}`);
  if (!ok) console.log(`        input: ${JSON.stringify(input).slice(0, 200)}`);
}

console.log("── corpus encodings (encoder is ground truth) ──");
for (const entry of corpus) check(entry.name, encode(entry.json), true);

console.log("── corpus encodings, generation profile (ADR-0019) ──");
for (const entry of corpus) {
  check(
    `${entry.name} (generation+markers)`,
    encode(entry.json, { profile: "generation", markers: true }),
    true,
  );
}
check("glued marker is not framing", "s=</raif>", true); // a value, not a marker

console.log("── hand positives ──");
check("empty document (encode of {})", encode({}), true);
check("nonce multiline via encode (random nonce)", encode({ body: "Hello,\n\n>>>\nrest" }), true);
const handPositives: Array<[string, string]> = [
  ["empty string document", ""],
  ["bare value with embedded < (no wrap)", "from=John <j@x.com"],
  ["outermost-slice wrapped value", "field=<<<a>>>b>>>"],
  ["wrapped key segment mid-path", "user.<<<a.b>>>=1"],
  ["wrapped key with index", "<<<k>>>[0]=1"],
  ["bare multiline block", "body=<<<\nHello,\n\nworld\n>>>"],
  [
    "nonce multiline w/ literal >>> content line",
    "body=<<<7f2a\nHello,\n\n>>>\nthat line was a literal closer\n>>>7f2a",
  ],
  ["inline object with wrapped key", "mixed[0]={<<<user.email>>>=x@y.z,role=admin}"],
  [
    "table header + rows incl. wrapped cell",
    "items::id,name\nitems[0]=1,<<<a,b>>>\nitems[1]=2,bar",
  ],
  [
    "array literal w/ inline obj, wrapped ] row, number",
    "events=[\n{type=click,target=a.cta}\n<<<]>>>\n42\n]",
  ],
  ["typed separators", "id:s=42\nflag:b=true\ncount:n=0\nnote:t=hi"],
  ["wrapped empty string value", "s=<<<>>>"],
  ["wrapped cell with trailing > merge", "row={a=<<<x,>>>>}"],
];
for (const [label, input] of handPositives) check(label, input, true);

console.log("── negatives ──");
const negatives: Array<[string, string]> = [
  ["unterminated array literal at EOF", "a=[\nfoo"],
  ["empty key", "=novalue"],
  ["lone [ as finished leaf value", "k=["],
  ["array opener then leaf, no ] closer", "a=[\nfoo=bar"],
  ["unterminated <<< wrap", "a=<<<oops"],
  ["bare value with >>> run (encoder always wraps)", "a=>>>x"],
];
for (const [label, input] of negatives) check(label, input, false);

console.log(
  `\n${total - failures}/${total} checks passed${failures ? ` — ${failures} FAILED` : ""}`,
);
process.exit(failures ? 1 : 0);

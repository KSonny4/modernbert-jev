# modernbert-jev design notes

## Goal

Typed decision models (choice, score, noul) with a TypeSafe-compatible API
that run on 1 CPU and 1 GB of RAM. Private, local, no per-token fees.

## Architecture

```
state + questions  ->  hypothesis builder  ->  NLI cross-encoder (ONNX int8)
   ->  per-head temperature  ->  typed answers + probabilities + confidence
```

- **Backbone.** A small NLI cross-encoder (DeBERTa-v3-small, 142 M params,
  Apache-2.0) exported to ONNX and dynamically quantized to int8 (~165 MB).
  Zero-shot decisions reuse the entailment head, the same family as the
  public `nli` and `verdict` decision models.
- **Choice/score.** One hypothesis per option/level, softmax over per-option
  entailment logits with a fitted temperature. Score is the expected level.
  Choice subtracts each hypothesis's null-premise ("N/A") bias first
  (contextual calibration à la Zhao et al. 2021, weight 0.5): it doubles
  18-way routing accuracy and leaves few-option sets bitwise unchanged.
- **Known limit: abstract options.** Content descriptions ("Bugs or
  integration problems") work; meta labels ("entailment / neutral /
  contradiction") score near chance on small backbones. Label-word
  hypotheses fix meta-options on the base backbone but wreck concrete
  ones, so descriptions stay the default. Fine-tuning should resolve
  the fork with data instead of heuristics.
- **Noul.** The instructions are stripped to the core question (up to the
  first `?`) plus any yes/no criteria; the answer renormalizes entailment
  against contradiction. Tried and dropped: heuristic question-to-statement
  rewriting, which corrupts multi-word subjects for no measured gain.
- **Long states.** Inputs past ~1.5 kB are split into overlapping word
  windows (300 words, capped at 6, stride widened to span the doc), scored
  in one batch, and max-pooled per option (best-evidence window for noul).
  Short inputs keep a single fast pass. A cap of 12 scored identically on
  hard-public but took 2x longer, so 6 is the default (`MJEV_CHUNK_CAP`
  overrides). Worst case ~17 s for a 15 kB 5-option item on one M4 core,
  which sets the p95; the Speed axis takes the p50/p95 mean, so this stays
  above the sub-50 gate. Future: retrieve top windows with a bi-encoder
  before cross-encoding.
- **Confidence.** Exact TypeSafe formulas (`src/modernbert_jev/confidence.py`), taken
  from their published docs.

## Why not the big-model tricks

Evaluated and rejected for this goal (see session notes):

- **MoE / expert streaming, flash offloading (LLM in a Flash style),
  GGUF GPU offload (HyperQwen, Strata).** These solve VRAM capacity on
  machines that still have fast compute. Our binding constraint is 1 CPU
  of compute plus the JevBench Speed axis: a 1 B-active-parameter forward
  pass takes tens of seconds on one core and trips the sub-50 Speed gate,
  which multiplies the whole composite score down. A small dense encoder
  fully resident in RAM answers in tens of milliseconds.
- **Rust rewrite of the server.** Inference dominates latency (>95%); the
  hot path already runs in Rust/C++ (HF `tokenizers`, ONNX Runtime).
  A Rust server would save RAM, not milliseconds. Revisit only if the
  1 GB budget demands it; current peak is ~730 MB with room to tune the
  ORT arena down.

## Memory budget (target 1.5–2 GB, 2026-10-08 relaxation)

Was 1 GB strict; owner now accepts up to 2 GB. This admits
ModernBERT-large int8 (~450 MB weights + ~550 MB runtime ≈ 1 GB)
with headroom. Base models stay the small-device option.

| Part | Size |
| --- | --- |
| int8 ONNX weights | ~165 MB |
| Tokenizer | ~9 MB |
| ORT arena + Python + server | ~550 MB measured |
| **Peak RSS (1 thread)** | **~730 MB** |

Tuning knobs if this regresses: ORT `arena_extend_strategy`,
`session.use_env_allocators`, smaller `CHUNK_CAP`.

## Quantization is per-platform (measured 2026-10-07)

Dynamic int8 is NOT portable across CPU architectures for these graphs.
Same file, same ORT 1.30, same inputs:

| weights | ARM64 ORT | x86_64 ORT |
| --- | --- | --- |
| signed QInt8 | correct, 97 ms/q | argmax-flipping garbage |
| unsigned QUInt8 | correct, 244 ms/q | correct, ~17 ms/pair |

Verified against torch fp32 ground truth (`[3.74, -0.96, -2.02]`
vs x86-QInt8 `[0.67, 1.25, 0.65]`). Not an optimizer fusion (all
`GraphOptimizationLevel`s identical) and not missing shape inference
(`quant_pre_process` changed nothing). Rule: **ship QInt8 for ARM64,
QUInt8 for x86_64** (`cli.PLATFORM_ONNX` selects). `train.py` emits
QUInt8 (`model.onnx`) since training runs on x86; re-quantize to QInt8
for ARM deployments. Every export must print `NLI-SANITY: PASS`.

## Calibration

One temperature per head in `calibration.json`, fitted by minimizing NLL
on labelled decisions (temperature scaling). ECE/Brier are tracked on the
same split. Never fit on the sealed JevBench items; they are held out.
Shipped `modernbert-jev` values: choice 2.0 (ECE 0.097), score 1.0 (ECE 0.217),
noul 4.0 (ECE 0.136).

## JevBench protocol

No custom adapter is needed: the stock `typesafe` adapter speaks our
`POST /v1/systemone` directly.

```bash
modernbert-jev serve --model models/modernbert-jev --port 11436 &
jevbench --adapter typesafe --endpoint http://127.0.0.1:11436 \
  --model modernbert-jev --key-env "" --results out.jsonl
```

Local runs cover the 231 public items only. The sealed 308 items need the
official harness; report public numbers as unofficial.

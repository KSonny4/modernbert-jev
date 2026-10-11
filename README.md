# modernbert-jev — Jev-class decisions on 1 CPU and 2 GB of RAM

Ask typed questions (choice, score, yes/no) about any text or JSON and get
calibrated answers in milliseconds — private, local, no per-token fees.
Speaks the TypeSafe `POST /v1/systemone` shapes, so existing clients work
by changing only the base URL.

Status: working preview, submission pending. Numbers below are unofficial
local measurements on public items; sealed-set board placement needs the
official JevBench harness.

## What we created

A complete tiny-Jev stack: Python library + HTTP server + CLI + training
loop, built around one idea — **a small NLI cross-encoder is already a
decision model**. Given a state (premise) and an option (hypothesis),
entailment probability *is* the option's probability. No generation, no
JSON parsing, no GPU.

- `choice`: max-pooled entailment per option → softmax → probabilities
- `score`: entailment per level → expected value over the legend
- `noul`: entailment vs contradiction renormalized → P(yes)

Long states are split into overlapping 300-word windows (max-pooled);
an 18-way routing trick (null-premise PMI subtraction) fixes
generic-option bias; one temperature per head calibrates confidence.
Everything runs in one ONNX Runtime thread: ~100 ms per short decision,
~330 MB peak RAM for the model.

Training is KL distillation: the student mimics a teacher's soft decision
distributions (50k–655k rows of `SargeDev/jev-distill-corpus-v3`) plus an
MNLI auxiliary that keeps the head honest NLI. Full fine-tune of 150M
params on a single RX 7900 XT via WSL (~0.18 s/step at batch 4).

## Models

| Model | What | JevBench public /231 | RAM |
| --- | --- | --- | --- |
| `modernbert-jev-zero` | `tasksource/ModernBERT-base-nli`, zero-shot | 0.528 | ~330 MB |
| `modernbert-jev-v2` (**submit candidate**) | above + 50k-row distill | **0.571** (ARM) / 0.554 (x86) | ~360 MB |
| `modernbert-jev-v3` | above + full 655k corpus | 0.541 — overfit, see below | ~360 MB |

v2 detail (ARM/QInt8 file): easy 0.979, original 0.708, hard 0.306.
Per type: choice 0.78, noul 0.50, score 0.75 on original. Temps refit:
choice 0.9 (ECE 0.110), score 0.9 (ECE 0.144), noul 2.5 (ECE 0.081).
Bespoke: MASSIVE 18-way 0.643, BoolQ 0.630.

**Quantization is per-platform** (measured, verified against torch fp32):
signed QInt8 is correct on ARM64 but argmax-flipping garbage on x86_64 ORT;
unsigned QUInt8 is correct on both (2.5× slower on ARM, ~17 ms/pair on
x86). Ship QInt8 for ARM, QUInt8 for x86. Every export must print
`NLI-SANITY: PASS`.

**The v3 lesson:** the fully-converged run (loss 0.007) lost 6 easy items
vs the 50k run. More teacher-mimicry ≠ better decisions — the short run's
early stop was implicit regularization. Hard tier (~0.31) wants
adversarial data (ANLI), not more of the same; that's v4.

## Where this sits

Live JevBench board (v1.4.2.2, 91 ranked): top systems score 60+ at
$0.02/1k decisions. Our public accuracy 0.554 ties rank-64 Verdict
(exact: 128/231); neighboring encoder rows score 10–20, which is our
projected band — above the bottom ~15 rows (rerankers, GLiNER-small,
Certo). By score-per-dollar we project ~#2 on the board (~10,000, behind
only Laya), at ~$0.001–0.002/1k decisions on commodity CPU. The niche the
Cost axis can't see: this runs where nothing else does — 1 CPU, no GPU,
no server, no account.

## Requirements

- 1 CPU, up to 2 GB of RAM (base models fit 1 GB), ~200 MB disk per model
- Python 3.10+ ([uv](https://docs.astral.sh/uv/) recommended)

## Quickstart

```bash
git clone <this-repo> && cd modernbert-jev
uv venv && uv pip install -e .
modernbert-jev pull modernbert-jev
modernbert-jev serve --model models/modernbert-jev --port 11436
```

Ask it something (second shell):

```bash
curl http://127.0.0.1:11436/v1/systemone -d '{
  "model": "modernbert-jev",
  "state": "Where is my package? I ordered it last week.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which intent does the message express?",
      "criteria": {
        "track_order": "Wants to know where an order is",
        "cancel_order": "Wants to cancel an order"
      }
    },
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"}
  }
}'
```

Or point the TypeSafe SDK at it:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:11436
export TYPESAFE_API_KEY=local  # any value works
```

One-shot answers without the server:

```bash
modernbert-jev run --model models/modernbert-jev --request examples/support-ticket.json
```

## API

Three question types, same shapes as TypeSafe:

| Type | Returns |
| --- | --- |
| `choice` | `choice`, `probabilities` per option, `confidence` |
| `score` | `score` (expected level), `legend`, `probabilities`, `confidence` |
| `noul` | `noul` (probability of yes) |

`GET /v1/models` lists installed models. See [docs/DESIGN.md](docs/DESIGN.md)
for architecture, memory budget, and calibration notes.

## Benchmark (unofficial, local, public items only)

v2 (QInt8/ARM), 1 thread, fitted temperatures:

| Split | Accuracy |
| --- | --- |
| easy (48) | 0.979 |
| original (72) | 0.708 |
| hard-public (111) | 0.306 |
| **overall** | **0.571** |

x86 (QUInt8) file: 0.938 / 0.681 / 0.306 → 0.554. Median ~100–250 ms per
decision on one CPU (short items; worst case ~40 s for a 15 kB 5-option
long-policy item, which sets the p95 — windowing retrieval is future work).

For scale, small public models on the official sealed-inclusive board sit
between ~6 (GLiNER-small class) and ~30 (Laya 421 M). Our public accuracy
already clears the bottom rows (rerankers ~0.34, Certo 0.32), with
top-tier Speed and Cost still to be scored; sealed items are harder and
untested here, so treat this as a floor estimate, not a board claim.

Run the official protocol yourself (needs no custom adapter):

```bash
modernbert-jev serve --model models/modernbert-jev --port 11436 &
jevbench --adapter typesafe --endpoint http://127.0.0.1:11436 \
  --model modernbert-jev --key-env "" --results out.jsonl
```

Or submit this repo + weights at `benchmarkheaven.com/submit` (free queue).

## Train

```bash
modernbert-jev-train --backbone tasksource/ModernBERT-base-nli \
  --out /root/mjev-v2 --data distill --max-rows 50000 \
  --batch-rows 4 --max-length 192 --epochs 1
```

Needs a GPU box (we use a Windows/WSL RX 7900 XT box; runbook:
[docs/WINDOWS.md](docs/WINDOWS.md), ROCm setup: [docs/TRAINING.md](docs/TRAINING.md)).
Long runs launch via Windows Scheduled Task (the OS owns the process; SSH
only triggers and polls — plain `&` backgrounding dies with the session).

## Develop

```bash
MJEV_TEST_MODEL=/tmp/mjev-models/modernbert-jev-zero .venv/bin/python -m unittest discover -s tests
```

Layout: `src/modernbert_jev/` (`api.py`, `confidence.py`, `runtime.py`, `server.py`,
`cli.py`, `train.py`), `tests/`, `docs/`, `examples/`.

## jevman (Pac-Man) results

| Player | Threads | Games | Mean score (± se) | Answer latency | Model size | Training |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost ranker (depth 5, 265 trees) | 1 | 100 | 3701 ± 99 | 8 ms | 0.94 MB | 7 s on 1 CPU thread |
| XGBoost ranker (depth 6, 300 trees, old defaults) | 1 | 100 | 3137 ± 85 | 2.3 ms | 1.85 MB | 5 s on 1 CPU thread |
| jev oracle6k | 12 | 100 | 3427 ± 84 | 200 ms | 147 MB | about 8 h on Apple MPS |

- Hardware: 14-vCPU Windows box, WSL Ubuntu-24.04. The bench runs one game for each call with `--parallel 1`.
- The depth-5 run shared the box with a 12-thread job. That job explains the 8 ms latency.
- Latency sets the score. At 1 thread, XGBoost scores about 3 times jev. The depth-5 XGBoost at 1 thread leads jev at 12 threads by 274 points (about 2.1 se).
- XGBoost is not jev-based. It reads hand-written option text from the bench. It does not transfer to new tasks.

## Licenses

Runtime code is Apache-2.0. Weights derive from
`tasksource/ModernBERT-base-nli` (Apache-2.0) and `SargeDev/jev-distill-corpus-v3`
(Apache-2.0). JevBench items used for evaluation only, never training.

"""Train a ModernBERT backbone and export it to quantized ONNX.

Two data modes:

- ``mnli``: plain 3-way NLI fine-tuning (keeps neutral semantics sharp).
- ``distill``: KL distillation from teacher decision distributions
  (SargeDev/jev-distill-corpus-v3): per option, the model's entailment
  logits go through log-softmax and match the teacher's target
  distribution — exactly the computation the runtime serves, plus a
  small MNLI auxiliary loss so the NLI head keeps its meaning.

A manual single-device loop (CUDA/ROCm, MPS, or CPU); full fine-tuning,
no LoRA needed at 184 M params. Temperatures still need fitting
afterwards on held-out decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from modernbert_jev.api import parse_request, state_text
from modernbert_jev.runtime import (
    MAX_STATE_CHARS,
    NULL_PREMISE,
    PMI_ALPHA,
    SINGLE_PASS_CHARS,
    DecisionModel,
    label_indices,
)

GLUE_NAMES = ("entailment", "neutral", "contradiction")


def aux_label_map(id2label: object) -> list[int]:
    """MNLI GLUE label (0=E 1=N 2=C) -> backbone head index, by name."""
    names = {str(v).lower(): int(k) for k, v in dict(id2label or {}).items()}
    if all(n in names for n in GLUE_NAMES):
        return [names[n] for n in GLUE_NAMES]
    return [1, 2, 0]  # cross-encoder fallback: 0=C 1=E 2=N


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(backbone: str, labels: int = 3):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=labels
    )
    return tokenizer, model


def export_onnx(model, tokenizer, out_path: str, opset: int = 17) -> None:
    model.eval()
    dummy = tokenizer(
        "premise text for export", "hypothesis text for export", return_tensors="pt"
    )
    input_names = list(dummy.keys())
    dynamic = {name: {0: "batch", 1: "seq"} for name in input_names}
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(dummy[name] for name in input_names),
            out_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={**dynamic, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def quantize_dynamic(onnx_path: str, out_path: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # QUInt8, not QInt8: signed dynamic weights mis-execute on x86 ORT
    # (argmax-flipping garbage, correct on ARM), unsigned matches fp32.
    quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QUInt8)


def write_model_dir(
    target: str, tokenizer, onnx_path: str, backbone: str, temperatures=None
) -> None:
    import shutil

    from transformers import AutoConfig

    os.makedirs(target, exist_ok=True)
    shutil.copy(onnx_path, os.path.join(target, "model.onnx"))
    tokenizer.backend_tokenizer.save(os.path.join(target, "tokenizer.json"))
    AutoConfig.from_pretrained(backbone).save_pretrained(target)
    with open(os.path.join(target, "calibration.json"), "w") as fh:
        json.dump(temperatures or {"choice": 1.0, "score": 1.0, "noul": 1.0}, fh)
    with open(os.path.join(target, "MODEL_CARD.txt"), "w") as fh:
        fh.write(f"backbone: {backbone}\nquant: dynamic-quint8\n")


def distill_pairs(row: dict) -> list[tuple[str, str]]:
    """(premise, hypothesis) pairs for one teacher row."""
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, sort_keys=True, ensure_ascii=False)
    question = row["question"].strip()
    return [(state, f"{question} {label}") for label in row["options"]]


def train_distill(
    backbone: str,
    out: str,
    max_rows: int = 150_000,
    epochs: int = 1,
    max_steps: int = -1,
    batch_rows: int = 8,
    lr: float = 2e-5,
    aux_weight: float = 0.15,
    max_length: int = 256,
    seed: int = 7,
    start_from: str | None = None,
) -> str:
    from datasets import load_dataset

    device = pick_device()
    print(f"device: {device}")
    random.seed(seed)
    torch.manual_seed(seed)
    if start_from:
        print(f"resuming weights from: {start_from}")
    tokenizer, model = build_model(start_from or backbone)
    model.to(device)
    model.train()
    contra_idx, ent_idx = label_indices(model.config.id2label)
    to_backbone = aux_label_map(model.config.id2label)
    print(f"label-order: entail={ent_idx} contra={contra_idx} aux-map={to_backbone}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    rows: list[dict] = []
    for r in load_dataset(
        "SargeDev/jev-distill-corpus-v3", split="train", streaming=True
    ):
        rows.append(r)
        if len(rows) >= max_rows:
            break
    mnli = load_dataset("nyu-mll/glue", "mnli", split="train").shuffle(seed=seed)
    mnli_iter = iter(mnli)

    import time as _time

    step, epoch_rows, t_start = 0, 0, _time.perf_counter()
    for epoch in range(epochs):
        random.shuffle(rows)
        for start in range(0, len(rows), batch_rows):
            chunk = rows[start : start + batch_rows]
            pairs, bounds, targets = [], [0], []
            for r in chunk:
                row_pairs = distill_pairs(r)
                pairs.extend(row_pairs)
                bounds.append(bounds[-1] + len(row_pairs))
                targets.append(torch.tensor(r["target"], dtype=torch.float))
            enc = tokenizer(
                [p for p, _ in pairs], [h for _, h in pairs],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            entail = model(**enc).logits[:, ent_idx]
            loss = torch.zeros((), device=device)
            for i, tgt in enumerate(targets):
                lo, hi = bounds[i], bounds[i + 1]
                logp = F.log_softmax(entail[lo:hi], dim=0)
                loss = loss + F.kl_div(logp, tgt.to(device), reduction="sum")
            loss = loss / len(targets)
            if aux_weight > 0:
                try:
                    mrow = [next(mnli_iter) for _ in range(batch_rows)]
                except StopIteration:
                    mnli_iter = iter(mnli.shuffle(seed=seed + epoch))
                    mrow = [next(mnli_iter) for _ in range(batch_rows)]
                menc = tokenizer(
                    [m["premise"] for m in mrow], [m["hypothesis"] for m in mrow],
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                )
                menc = {k: v.to(device) for k, v in menc.items()}
                mlogits = model(**menc).logits
                mlabel = torch.tensor([to_backbone[m["label"]] for m in mrow], device=device)
                loss = loss + aux_weight * F.cross_entropy(mlogits, mlabel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            epoch_rows += len(chunk)
            if step % 100 == 0 or step <= 5:
                el = _time.perf_counter() - t_start
                print(f"epoch={epoch} step={step} rows={epoch_rows} loss={loss.item():.4f} {el / max(step, 1):.2f}s/step", flush=True)
            if step % 2000 == 0:
                model.save_pretrained(f"{out}-ckpt-step{step}")
                print(f"checkpoint: {out}-ckpt-step{step}", flush=True)
            if 0 < max_steps <= step:
                break
        if 0 < max_steps <= step:
            break
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def jevman_example(row: dict, premise_src: str = "state") -> dict:
    """One strong-player decision -> runtime premise, hypotheses and target index.

    premise_src: "state" (state_text, as the runtime) or "instructions" (question text).
    """
    state = row["state"]
    if isinstance(state, dict):
        state = dict(state)
        state.pop("maze", None)  # same as examples/jevman_endpoint.py
    req = parse_request({
        "state": state,
        "model": "train",
        "questions": {"q": {**row["question"], "type": "choice"}},
    })
    question = req.questions["q"]
    labels = list(question.criteria)
    target = labels.index(row["choice"])
    best = row.get("best") or [row["choice"]]
    premise = state_text(question.instructions if premise_src == "instructions" else req.state)[:MAX_STATE_CHARS]
    windows = [premise] if len(premise) <= SINGLE_PASS_CHARS else DecisionModel._windows(premise)
    return {
        "premises": windows,
        "hyps": [DecisionModel._choice_hypothesis(question, label) for label in labels],
        "desc": [question.criteria[label] for label in labels],
        "target": target,
        "targets": sorted({labels.index(k) for k in best} | {target}),
    }


def split_held_out(rows: list[dict], seed: int = 7, frac: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Hold out whole games (submission, game): about ``frac`` of them, fixed seed."""
    groups = sorted({r["group"] for r in rows}, key=repr)
    random.Random(seed).shuffle(groups)
    n_held = max(1, round(frac * len(groups))) if len(groups) > 1 else 0
    held = set(groups[:n_held])
    return ([r for r in rows if r["group"] not in held],
            [r for r in rows if r["group"] in held])


def _risky(desc: str) -> bool:
    return "DANGER" in desc or "TRAP" in desc


def decision_scores(model, tokenizer, batch, device, max_length, ent_idx, pmi_alpha=PMI_ALPHA) -> list[torch.Tensor]:
    """Runtime choice logits per decision, as serve: max over windows of entail(window, hyp), minus pmi_alpha * entail(N/A, hyp).
    pmi_alpha == 0 skips the N/A pairs."""
    pairs = [(w, hyp) for ex in batch for w in ex["premises"] for hyp in ex["hyps"]]
    n = len(pairs)
    if pmi_alpha:
        pairs += [(NULL_PREMISE, hyp) for ex in batch for hyp in ex["hyps"]]
    enc = tokenizer(
        [p for p, _ in pairs], [h for _, h in pairs],
        padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    entail = model(**{k: v.to(device) for k, v in enc.items()}).logits[:, ent_idx]
    main, null = entail[:n], entail[n:]
    out, pos, null_pos = [], 0, 0
    for ex in batch:
        w, k = len(ex["premises"]), len(ex["hyps"])
        block = main[pos : pos + w * k].reshape(w, k)
        pos += w * k
        scores = block.max(dim=0).values
        if pmi_alpha:
            scores = scores - pmi_alpha * null[null_pos : null_pos + k]
        out.append(scores)
        null_pos += k
    return out


@torch.no_grad()
def jevman_eval(model, tokenizer, examples, device, max_length, ent_idx, batch_rows, pmi_alpha=PMI_ALPHA) -> dict:
    model.eval()
    agree = risky = 0
    for lo in range(0, len(examples), batch_rows):
        batch = examples[lo : lo + batch_rows]
        for ex, scores in zip(batch, decision_scores(model, tokenizer, batch, device, max_length, ent_idx, pmi_alpha)):
            pick = int(scores.argmax())
            agree += pick in ex["targets"]
            risky += _risky(ex["desc"][pick])
    model.train()
    n = max(len(examples), 1)
    return {"agree": agree / n, "risky": risky / n}


def jevman_loss(scores: torch.Tensor, targets: list[int]) -> torch.Tensor:
    """-log(sum of softmax probs over targets). One target gives plain cross-entropy."""
    return -(torch.logsumexp(scores[targets], 0) - torch.logsumexp(scores, 0))


def save_state(path: str, model, opt, step: int, epoch: int, lo: int) -> None:
    """Atomic: a kill during torch.save leaves the previous state.pt intact."""
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "epoch": epoch, "lo": lo}, tmp)
    os.replace(tmp, path)


def train_jevman(start, jsonl, epochs, max_steps, batch_rows, lr, max_length=None, premise_src="state", seed=7,
                 pmi_alpha=PMI_ALPHA, ckpt_dir=None, ckpt_every=500):
    """Listwise fine-tune: softmax CE over the runtime choice logits, one step per batch of decisions.

    ckpt_dir: save state.pt every ckpt_every steps and at the end; resume from it when present.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    rows = [{**r, "group": (r["submission"], r["game"])} for r in load_jsonl(jsonl)]
    train_rows, held_rows = split_held_out(rows, seed)
    train_ex = [jevman_example(r, premise_src) for r in train_rows]
    held_ex = [jevman_example(r, premise_src) for r in held_rows]
    device = pick_device()
    print(f"device: {device} train-decisions={len(train_ex)} held-out-decisions={len(held_ex)}", flush=True)
    tokenizer, model = build_model(start)
    model.to(device)
    model.gradient_checkpointing_enable()
    _, ent_idx = label_indices(model.config.id2label)
    if max_length is None:  # serve truncates at the config limit
        max_length = int(model.config.max_position_embeddings)
    strong_risky = sum(_risky(ex["desc"][ex["target"]]) for ex in held_ex) / max(len(held_ex), 1)

    def report(tag: str) -> None:
        if not held_ex:
            print(f"{tag}: no held-out games", flush=True)
            return
        m = jevman_eval(model, tokenizer, held_ex, device, max_length, ent_idx, batch_rows, pmi_alpha)
        print(f"{tag}: held-out top-1 agreement={m['agree']:.3f} "
              f"DANGER/TRAP share of picks={m['risky']:.3f} "
              f"(strong player {strong_risky:.3f}) n={len(held_ex)}", flush=True)

    # ponytail: no MNLI aux loss. train_distill's aux is inline and pulls GLUE; reuse needs a refactor.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    state_path = None
    step, ep0, first0 = 0, 0, 0  # first0: batch start to run first in epoch ep0
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        state_path = os.path.join(ckpt_dir, "state.pt")
    if state_path and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, ep0 = state["step"], state["epoch"]
        first0 = state["lo"] + batch_rows  # the saved batch is done
        print(f"resumed at step={step}", flush=True)
    else:
        report("before")
    model.train()
    last_saved, epoch, lo = step, ep0, first0
    for epoch in range(epochs):
        random.shuffle(train_ex)  # once per epoch, skipped epochs too: same data order
        if epoch < ep0:
            continue
        for lo in range(first0 if epoch == ep0 else 0, len(train_ex), batch_rows):
            if 0 < max_steps <= step:
                break
            batch = train_ex[lo : lo + batch_rows]
            scores = decision_scores(model, tokenizer, batch, device, max_length, ent_idx, pmi_alpha)
            loss = sum(jevman_loss(s, ex["targets"]) for s, ex in zip(scores, batch)) / len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0 or step <= 2:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
            if state_path and step % ckpt_every == 0:
                save_state(state_path, model, opt, step, epoch, lo)
                last_saved = step
        if 0 < max_steps <= step:
            break
    if state_path and step != last_saved:
        save_state(state_path, model, opt, step, epoch, lo)
    report("after")
    model.to("cpu")  # export_onnx traces with CPU inputs
    return tokenizer, model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="modernbert-jev-train", description="Train a ModernBERT backbone")
    parser.add_argument("--backbone", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--out", required=True, help="Output model directory")
    parser.add_argument("--data", choices=["mnli", "distill", "jevman"], default="distill")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=150_000)
    parser.add_argument("--batch-rows", type=int, default=None,
                        help="Rows per step (default: 1 for --data jevman, 8 otherwise)")
    parser.add_argument("--start-from", "--resume-from", dest="resume_from", default=None,
                        help="torch checkpoint dir or HF id; ONNX model dirs cannot be fine-tuned")
    parser.add_argument("--jsonl", default=None,
                        help="Strong-player decisions, one JSON object per line (--data jevman)")
    parser.add_argument("--premise", choices=["state", "instructions"], default="state",
                        help="Premise text for --data jevman (must match serve text)")
    parser.add_argument("--pmi-alpha", type=float, default=PMI_ALPHA,
                        help="PMI weight for --data jevman (must match serve; 0 for --no-pmi)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Token limit (default: 256 for distill; model max_position_embeddings for jevman)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="Resumable state.pt dir (--data jevman): resumes when it exists")
    parser.add_argument("--ckpt-every", type=int, default=500, help="Steps between state.pt saves")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training; export + quantize the backbone as-is")
    args = parser.parse_args(argv)
    args.batch_rows = args.batch_rows or (1 if args.data == "jevman" else 8)

    backbone = args.backbone
    if args.export_only:
        tokenizer, model = build_model(args.backbone)
    elif args.data == "distill":
        ckpt = args.out + "-ckpt"
        train_distill(
            args.backbone, ckpt, args.max_rows, args.epochs, args.max_steps,
            args.batch_rows, args.lr, args.aux_weight, args.max_length or 256,
            start_from=args.resume_from,
        )
        tokenizer, model = build_model(ckpt)
    elif args.data == "jevman":
        if not args.jsonl:
            raise SystemExit("--data jevman needs --jsonl")
        backbone = args.resume_from or args.backbone  # config source for write_model_dir
        tokenizer, model = train_jevman(
            backbone, args.jsonl, args.epochs, args.max_steps,
            args.batch_rows, args.lr, args.max_length, premise_src=args.premise,
            pmi_alpha=args.pmi_alpha, ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
        )
    else:
        raise SystemExit("mnli-only mode was removed; use --data distill (it includes MNLI aux)")
    tmp = args.out + "-fp32.onnx"
    tmp_q = args.out + "-int8.onnx"
    export_onnx(model, tokenizer, tmp)
    quantize_dynamic(tmp, tmp_q)
    write_model_dir(args.out, tokenizer, tmp_q, backbone)
    os.remove(tmp)
    os.remove(tmp_q)
    print(f"wrote {args.out}")
    print(nli_sanity(args.out), flush=True)
    return 0


def nli_sanity(model_dir: str) -> str:
    """Fail-loud check that the exported head kept NLI semantics."""
    from modernbert_jev.runtime import DecisionModel

    model = DecisionModel(model_dir, threads=1)
    logits, _ = model._logits(
        [
            ("The cat sat on the mat.", "The cat is on the mat."),
            ("The cat sat on the mat.", "The dog flew to the moon."),
        ]
    )
    ent_ok = int(logits[0].argmax()) == model.entail_idx
    con_ok = int(logits[1].argmax()) == model.contra_idx
    verdict = "PASS" if ent_ok and con_ok else "FAIL"
    ent_s = "PASS" if ent_ok else "WRONG"
    con_s = "PASS" if con_ok else "WRONG"
    return f"NLI-SANITY: {verdict} entail-pair={ent_s} contra-pair={con_s}"


if __name__ == "__main__":
    raise SystemExit(main())

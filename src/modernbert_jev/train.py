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

from modernbert_jev.runtime import label_indices

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="modernbert-jev-train", description="Train a ModernBERT backbone")
    parser.add_argument("--backbone", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--out", required=True, help="Output model directory")
    parser.add_argument("--data", choices=["mnli", "distill"], default="distill")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=150_000)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to resume training from (weights only)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training; export + quantize the backbone as-is")
    args = parser.parse_args(argv)

    if args.export_only:
        tokenizer, model = build_model(args.backbone)
    elif args.data == "distill":
        ckpt = args.out + "-ckpt"
        train_distill(
            args.backbone, ckpt, args.max_rows, args.epochs, args.max_steps,
            args.batch_rows, args.lr, args.aux_weight, args.max_length,
            start_from=args.resume_from,
        )
        tokenizer, model = build_model(ckpt)
    else:
        raise SystemExit("mnli-only mode was removed; use --data distill (it includes MNLI aux)")
    tmp = args.out + "-fp32.onnx"
    tmp_q = args.out + "-int8.onnx"
    export_onnx(model, tokenizer, tmp)
    quantize_dynamic(tmp, tmp_q)
    write_model_dir(args.out, tokenizer, tmp_q, args.backbone)
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

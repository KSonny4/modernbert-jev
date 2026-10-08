"""Command line: ``modernbert-jev pull``, ``serve`` and ``run``."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys

from modernbert_jev.api import ApiError, parse_request
from modernbert_jev.runtime import DecisionModel
from modernbert_jev.server import DEFAULT_PORT, serve

BACKBONES = {
    # Name always means "current best pullable"; fine-tuned releases take
    # over this slot once published with per-platform onnx/ files.
    "modernbert-jev": "tasksource/ModernBERT-base-nli",
}

# Shipped temperatures, fitted by NLL on labelled public decisions.
SHIPPED_CALIBRATION = {
    "modernbert-jev": {"choice": 0.6, "score": 0.6, "noul": 2.0},
}

# Quantized ONNX picked per CPU; plain fp32 fallback otherwise.
PLATFORM_ONNX = {
    "arm64": "onnx/model_qint8_arm64.onnx",
    "aarch64": "onnx/model_qint8_arm64.onnx",
    "x86_64": "onnx/model_quint8_avx2.onnx",
    "AMD64": "onnx/model_quint8_avx2.onnx",
}

# Unsigned weights for x86 (signed mis-executes there), signed for ARM.
PLATFORM_QUANT = {
    "arm64": "qint8",
    "aarch64": "qint8",
    "x86_64": "quint8",
    "AMD64": "quint8",
}


def default_models_dir() -> str:
    return os.environ.get("MJEV_MODELS_DIR", os.path.join(os.getcwd(), "models"))


def cmd_pull(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    name = args.name
    if name not in BACKBONES:
        print(f"unknown model {name!r}; known: {sorted(BACKBONES)}", file=sys.stderr)
        return 2
    repo = BACKBONES[name]
    machine = platform.machine()
    onnx_path = PLATFORM_ONNX.get(machine, "onnx/model.onnx")
    target = os.path.join(default_models_dir(), name)
    os.makedirs(target, exist_ok=True)
    try:
        snap = snapshot_download(
            repo,
            allow_patterns=["tokenizer.json", "config.json", onnx_path, "README.md"],
        )
        shutil.copy(os.path.join(snap, onnx_path), os.path.join(target, "model.onnx"))
        for fname in ("tokenizer.json", "config.json"):
            shutil.copy(os.path.join(snap, fname), os.path.join(target, fname))
    except Exception:
        # No shipped onnx/ for this backbone: export locally instead.
        try:
            from modernbert_jev.train import build_model, export_onnx
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print("pull needs the train extras for export: pip install -e '.[train]'",
                  file=sys.stderr)
            return 2
        quant = PLATFORM_QUANT.get(machine, "quint8")
        print(f"no {onnx_path} upstream; exporting {quant} locally (slow, once)")
        tokenizer, model = build_model(repo)
        tmp = os.path.join(target, "fp32.onnx")
        export_onnx(model, tokenizer, tmp)
        quantize_dynamic(
            tmp, os.path.join(target, "model.onnx"),
            weight_type=QuantType.QUInt8 if quant == "quint8" else QuantType.QInt8,
        )
        os.remove(tmp)
        tokenizer.backend_tokenizer.save(os.path.join(target, "tokenizer.json"))
        from transformers import AutoConfig
        AutoConfig.from_pretrained(repo).save_pretrained(target)
    cal_path = os.path.join(target, "calibration.json")
    if not os.path.exists(cal_path):
        with open(cal_path, "w") as fh:
            json.dump(SHIPPED_CALIBRATION.get(name, {"choice": 1.0, "score": 1.0, "noul": 1.0}), fh)
    with open(os.path.join(target, "MODEL_CARD.txt"), "w") as fh:
        fh.write(f"name: {name}\nbackbone: {repo}\nonnx: {onnx_path}\n")
    print(f"pulled {name} into {target}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    model_dir = args.model or os.path.join(default_models_dir(), "modernbert-jev")
    if not os.path.isfile(os.path.join(model_dir, "model.onnx")):
        print(f"no model at {model_dir}; run `modernbert-jev pull modernbert-jev` first", file=sys.stderr)
        return 2
    serve(model_dir, host=args.host, port=args.port, threads=args.threads)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    model_dir = args.model or os.path.join(default_models_dir(), "modernbert-jev")
    try:
        model = DecisionModel(model_dir, threads=args.threads)
    except Exception as exc:
        print(f"cannot load model at {model_dir}: {exc}", file=sys.stderr)
        return 2
    try:
        with open(args.request) as fh:
            request = parse_request(json.load(fh))
    except (OSError, ValueError, ApiError) as exc:
        print(f"bad request file: {exc}", file=sys.stderr)
        return 2
    response = model.ask(request)
    answers = response.to_dict()["answers"]
    width = max([len("question")] + [len(k) for k in answers])
    print(f"{'question':<{width}}  answer  details")
    for name, answer in answers.items():
        if answer["type"] == "choice":
            detail = ", ".join(f"{k}={v:.2f}" for k, v in answer["probabilities"].items())
            print(f"{name:<{width}}  {answer['choice']}  conf={answer['confidence']:.2f} {detail}")
        elif answer["type"] == "score":
            print(f"{name:<{width}}  {answer['score']:.2f}  conf={answer['confidence']:.2f}")
        else:
            print(f"{name:<{width}}  {answer['noul']:.2f}")
    print(f"input_tokens={response.input_tokens}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modernbert-jev", description="Tiny typed decision models")
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="Download and install a model")
    pull.add_argument("name", nargs="?", default="modernbert-jev")
    pull.set_defaults(func=cmd_pull)

    serve_p = sub.add_parser("serve", help="Serve a model over HTTP")
    serve_p.add_argument("--model", default=None)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_p.add_argument("--threads", type=int, default=1)
    serve_p.set_defaults(func=cmd_serve)

    run = sub.add_parser("run", help="Answer one request file")
    run.add_argument("--model", default=None)
    run.add_argument("--request", required=True, help="JSON file with a systemone body")
    run.add_argument("--threads", type=int, default=1)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

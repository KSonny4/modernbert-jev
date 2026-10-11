"""XGBoost Pac-Man player for the jevman bench. Trained on the same oracle file as the jev model.

Train:
    python examples/jevman_xgb.py train ORACLE.jsonl MODEL.json

Serve (same HTTP contract as examples/jevman_endpoint.py, one thread):
    python examples/jevman_xgb.py serve MODEL.json PORT

Features come from the request only: the state and the option texts the bench sends.
The endpoint drops state["maze"], so the grid is not available to the model.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import xgboost as xgb

SEED = 7
HELD_FRAC = 0.1
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
FEATURES = [
    "pellet_steps", "corridor_pellets", "danger_steps", "ghost_approaching", "junction_steps",
    "ghost_margin", "power_steps", "frightened_steps", "trap", "danger", "trap_margin",
    "is_reverse", "chase", "frightened_secs", "lives", "pellets_left",
]


def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else math.nan


def option_features(state, label, desc):
    """One option -> FEATURES. Numbers come from the option text; missing values are NaN."""
    pac = state.get("pacman") or {}
    trap = re.search(r"can reach the next junction in (\d+) steps?, you need (\d+)", desc)
    if "coming toward you" in desc:
        approaching = 1.0
    elif "moving away" in desc:
        approaching = 0.0
    else:
        approaching = math.nan
    return [
        _num(r"nearest pellet (\d+) steps? away", desc),
        _num(r"(\d+) pellets? in the next corridor", desc),
        _num(r"nearest dangerous ghost (\d+) steps? away", desc),
        approaching,
        _num(r"you reach the next junction in (\d+) steps?", desc),
        _num(r"(\d+) steps? before any ghost", desc),
        _num(r"power pellet (\d+) steps? away", desc),
        _num(r"frightened ghost (\d+) steps? away", desc),
        float("TRAP:" in desc),
        float("DANGER:" in desc),
        float(trap.group(2)) - float(trap.group(1)) if trap else math.nan,
        float(OPPOSITE.get(pac.get("dir")) == label),
        float(state.get("mode") == "chase"),
        float(state.get("frightened_seconds_left", 0)),
        float(state.get("lives", 0)),
        float(state.get("pellets_left", 0)),
    ]


def decision_features(state, question):
    """One choice question -> (labels, matrix with one row per option)."""
    labels = list(question["criteria"])
    X = np.array([option_features(state, lab, question["criteria"][lab]) for lab in labels], dtype=float)
    return labels, X


def load_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_held_out(rows, seed=SEED, frac=HELD_FRAC):
    """Copy of src/modernbert_jev/train.py split_held_out (that module imports torch).
    tests/test_jevman_xgb.py checks it against the original when torch is installed."""
    import random

    groups = sorted({r["group"] for r in rows}, key=repr)
    random.Random(seed).shuffle(groups)
    n_held = max(1, round(frac * len(groups))) if len(groups) > 1 else 0
    held = set(groups[:n_held])
    return ([r for r in rows if r["group"] not in held],
            [r for r in rows if r["group"] in held])


def fit(rows, n_estimators=265, max_depth=5, learning_rate=0.1):
    """Ranker: rank:pairwise, one group per decision, label 1 for each option in best."""
    Xs, ys, sizes = [], [], []
    for r in rows:
        labels, X = decision_features(r["state"], r["question"])
        best = set(r.get("best") or [r["choice"]]) | {r["choice"]}
        Xs.append(X)
        ys.extend(float(lab in best) for lab in labels)
        sizes.append(len(labels))
    dtrain = xgb.DMatrix(np.vstack(Xs), label=np.array(ys), feature_names=FEATURES, nthread=1)
    dtrain.set_group(sizes)
    params = {"objective": "rank:pairwise", "eta": learning_rate, "max_depth": max_depth,
              "nthread": 1, "seed": SEED}
    return xgb.train(params, dtrain, num_boost_round=n_estimators)


def scores(booster, state, question):
    labels, X = decision_features(state, question)
    raw = booster.predict(xgb.DMatrix(X, feature_names=FEATURES, nthread=1))
    return labels, raw


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def agreement(booster, rows):
    hits = 0
    for r in rows:
        labels, raw = scores(booster, r["state"], r["question"])
        pick = labels[int(np.argmax(raw))]
        hits += pick in set(r.get("best") or [r["choice"]])
    return hits / max(len(rows), 1)


def respond(booster, model_name, raw):
    """Bench contract: body -> (status, SystemOneResponse dict or error dict)."""
    if not isinstance(raw, dict):
        return 400, {"error": {"message": "body must be a JSON object"}}
    if "state" not in raw:
        return 400, {"error": {"message": "request needs 'state'"}}
    questions = raw.get("questions")
    if not isinstance(questions, dict) or not questions:
        return 400, {"error": {"message": "request needs a non-empty 'questions' map"}}
    answers = {}
    for name, q in questions.items():
        if not isinstance(q, dict) or q.get("type") != "choice":
            return 400, {"error": {"message": f"question {name!r}: only type choice is supported"}}
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            return 400, {"error": {"message": f"question {name!r} needs non-empty 'instructions'"}}
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            return 400, {"error": {"message": f"choice question {name!r} needs a non-empty 'criteria' map"}}
        if not all(isinstance(k, str) and k and isinstance(v, str) for k, v in criteria.items()):
            return 400, {"error": {"message": f"choice question {name!r} needs string options"}}
        try:
            labels, raw_scores = scores(booster, raw["state"], q)
        except Exception as exc:
            return 500, {"error": {"message": f"inference failed: {exc}"}}
        probs = softmax(raw_scores)
        top = int(np.argmax(probs))
        answers[name] = {
            "type": "choice",
            "choice": labels[top],
            "confidence": float(probs[top]),
            "probabilities": {lab: float(p) for lab, p in zip(labels, probs)},
        }
    return 200, {"model": model_name, "answers": answers,
                 "usage": {"input_tokens": 0, "output_tokens": 0}}


def make_server(model_path, port):
    booster = xgb.Booster()
    booster.load_model(model_path)
    booster.set_param({"nthread": 1})
    name = os.path.splitext(os.path.basename(model_path))[0]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"null"
                status, out = respond(booster, name, json.loads(body))
            except ValueError as exc:
                status, out = 400, {"error": {"message": str(exc)}}
            data = json.dumps(out).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def train(data_path, out_path, **fit_opts):
    rows = [{**r, "group": (r["submission"], r["game"])} for r in load_rows(data_path)]
    train_rows, held_rows = split_held_out(rows)
    t0 = time.perf_counter()
    booster = fit(train_rows, **fit_opts)
    secs = time.perf_counter() - t0
    booster.save_model(out_path)
    booster.set_param({"nthread": 1})
    print(f"train decisions={len(train_rows)} held-out decisions={len(held_rows)}")
    print(f"held-out top-1 agreement={agreement(booster, held_rows):.3f} n={len(held_rows)}")
    print(f"train seconds={secs:.1f} model bytes={os.path.getsize(out_path)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("oracle")
    p_train.add_argument("model")
    p_train.add_argument("--trees", type=int, help="boosting rounds (default: fit() n_estimators)")
    p_train.add_argument("--depth", type=int, help="max tree depth (default: fit() max_depth)")
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("model")
    p_serve.add_argument("port", type=int)
    args = parser.parse_args(argv)
    if args.cmd == "train":
        opts = {"n_estimators": args.trees, "max_depth": args.depth}
        train(args.oracle, args.model, **{k: v for k, v in opts.items() if v is not None})
    else:
        server = make_server(args.model, args.port)
        print(f"jevman xgb endpoint: {args.model} on http://127.0.0.1:{args.port} (1 thread)", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())

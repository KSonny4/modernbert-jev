"""XGBoost jevman player: trains on a few oracle rows and answers one bench request over HTTP."""

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request

import pytest

pytest.importorskip("xgboost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import jevman_xgb as J  # noqa: E402

DATA = os.environ.get("JEVMAN_ORACLE", "")
requires_oracle = pytest.mark.skipif(not os.path.exists(DATA), reason="set JEVMAN_ORACLE to jevman-oracle-v2.jsonl")


@requires_oracle
def test_xgb_serve_returns_legal_option():
    rows = J.load_rows(DATA)[:300]
    booster = J.fit(rows, n_estimators=20)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.json")
        booster.save_model(path)
        server = J.make_server(path, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            row = rows[0]
            body = json.dumps({"state": row["state"], "questions": {"move": row["question"]}}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.server_address[1]}/x", data=body,
                headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req).read())
        finally:
            server.shutdown()
            server.server_close()
    answer = out["answers"]["move"]
    assert answer["type"] == "choice"
    assert answer["choice"] in row["question"]["criteria"]
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6


def test_delay_ms_sleeps_before_answer():
    state = {"pacman": {"dir": "up"}, "lives": 3, "pellets_left": 10}
    question = {"type": "choice", "instructions": "Pick a move.",
                "criteria": {"up": "nearest pellet 2 steps away", "down": "nearest dangerous ghost 1 steps away"}}
    rows = [{"state": state, "question": question, "choice": "up", "best": ["up"]}] * 4
    booster = J.fit(rows, n_estimators=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.json")
        booster.save_model(path)
        server = J.make_server(path, 0, delay_ms=200)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            body = json.dumps({"state": state, "questions": {"move": question}}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.server_address[1]}/x", data=body,
                headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            urllib.request.urlopen(req).read()
            elapsed = time.perf_counter() - t0
        finally:
            server.shutdown()
            server.server_close()
    assert elapsed >= 0.2


@requires_oracle
def test_split_matches_train_module():
    torch_train = pytest.importorskip("modernbert_jev.train")
    rows = [{**r, "group": (r["submission"], r["game"])} for r in J.load_rows(DATA)[:2000]]
    assert J.split_held_out(rows) == torch_train.split_held_out(rows)

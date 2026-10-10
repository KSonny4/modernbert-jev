"""XGBoost jevman player: trains on a few oracle rows and answers one bench request over HTTP."""

import json
import os
import sys
import tempfile
import threading
import urllib.request

import pytest

pytest.importorskip("xgboost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import jevman_xgb as J  # noqa: E402

DATA = os.environ.get("JEVMAN_ORACLE", "")
if not os.path.exists(DATA):
    pytest.skip("set JEVMAN_ORACLE to jevman-oracle-v2.jsonl", allow_module_level=True)


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


def test_split_matches_train_module():
    torch_train = pytest.importorskip("modernbert_jev.train")
    rows = [{**r, "group": (r["submission"], r["game"])} for r in J.load_rows(DATA)[:2000]]
    assert J.split_held_out(rows) == torch_train.split_held_out(rows)

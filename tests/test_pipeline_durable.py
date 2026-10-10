"""Kill test: SIGKILL the pipeline during bench-8, rerun, train and ship run once."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "examples" / "jevman_pipeline.py"

FAKE_TRAIN = """
import pathlib, sys, time
model_dir, counter = pathlib.Path(sys.argv[1]), sys.argv[2]
open(counter, "a").write("train\\n")
time.sleep(1)
model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "model.onnx").write_bytes(b"fake onnx" * 100)
"""

# Mimics examples/jevman_bench_durable.py: flock, resume from jsonl, DONE line.
FAKE_RUNNER = """
import fcntl, json, os, random, sys, time
out = sys.argv[2]
games = int(sys.argv[sys.argv.index("--games") + 1])
lock = open(out + ".lock", "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("LOCKED")
    sys.exit(0)
have = sum(1 for line in open(out) if line.endswith("\\n")) if os.path.exists(out) else 0
for g in range(have, games):
    time.sleep(1)
    with open(out, "a") as f:
        f.write(json.dumps({"game": g + 1, "score": random.randint(1000, 3000)}) + "\\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"  game {g + 1}: score {g}", flush=True)
print(f"DONE games={games} mean=2000 sd=1 se=1 latency_ms=1 fallback_pct=0", flush=True)
"""


def _config(tmp: Path) -> Path:
    (tmp / "fake_train.py").write_text(FAKE_TRAIN)
    (tmp / "fake_runner.py").write_text(FAKE_RUNNER)
    (tmp / "box").mkdir()
    cfg = {
        "run_id": "kill-test",
        "state_dir": str(tmp / "state"),
        "train_cmd": [sys.executable, str(tmp / "fake_train.py"), str(tmp / "model"), str(tmp / "counter")],
        "model_dir": str(tmp / "model"),
        "box_cmd": [],
        "box_dir": str(tmp / "box"),
        "runner_cmd": [sys.executable, str(tmp / "fake_runner.py"), str(tmp / "model")],
        "lock_poll_s": 0.5,
        "benches": [
            {"label": "8", "games": 8, "threads": 1},
            {"label": "100", "games": 3, "threads": 1, "min_prev_mean": 2900},
        ],
    }
    path = tmp / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_kill_and_rerun(tmp_path):
    cfg = _config(tmp_path)
    log1 = tmp_path / "run1.log"
    with open(log1, "w") as out:
        proc = subprocess.Popen([sys.executable, str(PIPELINE), str(cfg)], stdout=out, stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    while "  game 2" not in log1.read_text():
        assert proc.poll() is None, "pipeline exited before bench-8 started"
        assert time.time() < deadline, "bench-8 never started"
        time.sleep(0.2)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()

    rerun = subprocess.run([sys.executable, str(PIPELINE), str(cfg)], capture_output=True, text=True, timeout=300)
    log2 = rerun.stdout + rerun.stderr
    assert rerun.returncode == 0, log2

    assert (tmp_path / "counter").read_text().splitlines() == ["train"]
    ships = (log1.read_text() + log2).count("step ship start")
    assert ships == 1
    lines = (tmp_path / "box" / "bench-8.jsonl").read_text().splitlines()
    assert len(lines) == 8
    assert "DONE games=8" in log2
    assert "bench-100: skipped" in log2
    assert "PIPELINE DONE run_id=kill-test" in log2

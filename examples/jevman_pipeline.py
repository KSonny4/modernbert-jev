#!/usr/bin/env python3
"""Durable jevman pipeline: train, ship, bench-<label>. One DBOS workflow per run.

Usage: python examples/jevman_pipeline.py CONFIG.json

Config keys:
  run_id, state_dir, train_cmd (argv), model_dir, box_dir, runner_cmd (argv up to
  MODEL_DIR on the box), box_cmd (argv prefix, [] = run here), benches
  [{label, games, threads, min_prev_mean?}], lock_poll_s (default 60).
The step appends OUT_JSONL, --games and --threads to runner_cmd.

State: state_dir/dbos.sqlite (DBOS system DB, SQLite). Rerun the same command after
a crash. Finished steps are not run again. The step that was running is retried.
"""
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from dbos import DBOS, SetWorkflowID

DONE_RE = re.compile(r"^DONE\s+(.*)$")


def _log(msg):
    print(msg, flush=True)


def _stream(argv):
    """Run argv, echo its output to our log. Return (rc, lines)."""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        _log(line)
    return p.wait(), lines


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1_remote(cfg, path):
    """sha1 of path on the box, or None when missing."""
    if not cfg["box_cmd"]:
        return _sha1_file(path) if os.path.exists(path) else None
    r = subprocess.run(cfg["box_cmd"] + ["sha1sum", path], capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None


def _num(v):
    try:
        return float(v)
    except ValueError:
        return v


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=30)
def train_step(cfg):
    model = Path(cfg["model_dir"]) / "model.onnx"
    if model.exists():
        _log("train: model.onnx exists, skip")
        return "skipped"
    _log("step train start")
    rc, _ = _stream(cfg["train_cmd"])
    if rc or not model.exists():
        raise RuntimeError(f"train failed, rc={rc}")
    return "trained"


@DBOS.step(retries_allowed=True, max_attempts=5, interval_seconds=30)
def ship_step(cfg):
    _log("step ship start")
    model_dir = Path(cfg["model_dir"])
    tgz = Path(cfg["state_dir"]) / "model.tgz"
    if not tgz.exists():
        # Fixed gzip mtime and sorted entries, so a retry builds the same bytes.
        tmp = tgz.with_name("model.tgz.tmp")
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as t:
                for p in sorted(model_dir.rglob("*")):
                    t.add(p, arcname=str(Path(model_dir.name) / p.relative_to(model_dir)), recursive=False)
        os.replace(tmp, tgz)
    sha = _sha1_file(tgz)
    remote = f"{cfg['box_dir']}/model.tgz"
    if _sha1_remote(cfg, remote) == sha:
        _log("ship: box copy is current, skip upload")
    else:
        _log("ship: upload")
        with open(tgz, "rb") as src:
            r = subprocess.run(cfg["box_cmd"] + ["tee", remote], stdin=src, stdout=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"upload failed, rc={r.returncode}")
        if _sha1_remote(cfg, remote) != sha:
            raise RuntimeError("box sha1 does not match")
    rc, _ = _stream(cfg["box_cmd"] + ["tar", "-xzf", remote, "-C", cfg["box_dir"]])
    if rc:
        raise RuntimeError(f"remote untar failed, rc={rc}")
    return sha


@DBOS.step(retries_allowed=True, max_attempts=5, interval_seconds=60)
def bench_step(cfg, bench):
    label = bench["label"]
    name = f"bench-{label}.jsonl"
    remote = f"{cfg['box_dir']}/{name}"
    argv = cfg["box_cmd"] + cfg["runner_cmd"] + [
        remote, "--games", str(bench["games"]), "--threads", str(bench["threads"])]
    _log(f"step bench-{label} start")
    while True:
        rc, lines = _stream(argv)
        # ponytail: fixed poll, no backoff. Lock holder is a long bench, so 60 s is fine.
        if "LOCKED" in lines and not any(DONE_RE.match(x) for x in lines):
            _log(f"bench-{label}: another runner holds the lock, retry in {cfg.get('lock_poll_s', 60)} s")
            time.sleep(cfg.get("lock_poll_s", 60))
            continue
        break
    done = [m.group(1) for m in map(DONE_RE.match, lines) if m]
    if rc or not done:
        raise RuntimeError(f"bench-{label} failed, rc={rc}")
    dest = Path(cfg["state_dir"]) / name
    with open(dest.with_name(name + ".tmp"), "w") as f:
        subprocess.run(cfg["box_cmd"] + ["cat", remote], stdout=f, check=True)
    os.replace(dest.with_name(name + ".tmp"), dest)
    result = {k: _num(v) for k, v in re.findall(r"(\w+)=(\S+)", done[-1])}
    return {"label": label, **result}


@DBOS.workflow()
def pipeline(cfg):
    train_step(cfg)
    ship_step(cfg)
    results, prev_mean = {}, None
    for bench in cfg["benches"]:
        floor = bench.get("min_prev_mean")
        if floor is not None and prev_mean is not None and prev_mean < floor:
            _log(f"bench-{bench['label']}: skipped, previous mean {prev_mean} < {floor}")
            results[bench["label"]] = "skipped"
            continue
        results[bench["label"]] = bench_step(cfg, bench)
        prev_mean = results[bench["label"]].get("mean", prev_mean)
    return results


def main(argv):
    with open(argv[1]) as f:
        cfg = json.load(f)
    state = Path(cfg["state_dir"]).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    cfg["state_dir"] = str(state)
    cfg["model_dir"] = str(Path(cfg["model_dir"]).expanduser())
    DBOS(config={"name": "jevman_pipeline", "system_database_url": f"sqlite:///{state}/dbos.sqlite"})
    DBOS.launch()
    run_id = cfg["run_id"]
    status = DBOS.get_workflow_status(run_id)
    if status is not None and status.status == "ERROR":
        _log("workflow ended in ERROR earlier, resume it")
        DBOS.resume_workflow(run_id)
    with SetWorkflowID(run_id):
        handle = DBOS.start_workflow(pipeline, cfg)
    results = handle.get_result()
    for label, r in results.items():
        _log(f"RESULT bench-{label}: {r}")
    _log(f"PIPELINE DONE run_id={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

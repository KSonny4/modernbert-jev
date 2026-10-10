#!/usr/bin/env python3
"""Resumable jevman bench runner: one game per bench call, one JSON line per game.

A killed run resumes at the next missing game. Standard library only.

Run:
    python examples/jevman_bench_durable.py MODEL_DIR OUT_JSONL --games N --threads T
    python examples/jevman_bench_durable.py MODEL_DIR OUT_JSONL --summary
"""

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAME_RE = re.compile(
    r"^\s*game \d+: survived ([\d.]+)s, score (\d+), pellets (\d+), deaths (\d+), "
    r"level \d+, ghosts (\d+), calls (\d+), fallbacks (\d+)/(\d+), \$[\d.]+",
    re.M,
)
MEAN_RE = re.compile(r"^MEAN .* latency (\d+) ms", re.M)
DEATHS_RE = re.compile(r"^deaths by situation: (\{.*\})\s*$", re.M)


def parse_game(text):
    g = GAME_RE.search(text)
    m = MEAN_RE.search(text)
    if not g or not m:
        return None
    d = DEATHS_RE.search(text)
    try:
        deaths_by = json.loads(d[1]) if d else {}
    except ValueError:
        return None
    return {
        "score": int(g[2]),
        "survived_s": float(g[1]),
        "pellets": int(g[3]),
        "deaths": int(g[4]),
        "ghosts": int(g[5]),
        "calls": int(g[6]),
        "fallbacks": int(g[7]),
        "decisions": int(g[8]),
        "latency_ms": int(m[1]),
        "deaths_by": deaths_by,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw": g[0].strip(),
    }


def load(path, repair):
    """Return complete lines as dicts. A trailing partial line is dropped (and cut off if repair)."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    end = data.rfind(b"\n") + 1
    if repair and end != len(data):
        with open(path, "r+b") as f:
            f.truncate(end)
    return [json.loads(line) for line in data[:end].splitlines()]


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stats(records):
    n = len(records)
    scores = [r["score"] for r in records]
    mean = sum(scores) / n if n else 0.0
    sd = statistics.stdev(scores) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    latency = sum(r["latency_ms"] for r in records) / n if n else 0.0
    decisions = sum(r["decisions"] for r in records)
    fallbacks = sum(r["fallbacks"] for r in records)
    pct = 100 * fallbacks / decisions if decisions else 0.0
    return (
        f"DONE games={n} mean={mean:.1f} sd={sd:.1f} se={se:.1f} "
        f"latency_ms={latency:.0f} fallback_pct={pct:.1f}"
    )


class Endpoint:
    """One endpoint process. Its pid is kept in a file so a later run can stop a stale one."""

    def __init__(self, args):
        self.args = args
        self.pidfile = args.out_jsonl + ".endpoint.pid"
        self.proc = None
        if args.endpoint_cmd:
            self.cmd = shlex.split(args.endpoint_cmd)
        else:
            self.cmd = [
                sys.executable, "examples/jevman_endpoint.py", args.model_dir, str(args.port),
                "--threads", str(args.threads), "--premise", "instructions", "--no-pmi",
            ]
        self.env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))

    def kill_stale(self):
        # ponytail: pid reuse after a reboot could hit an unrelated process; add a cmdline check if it happens.
        try:
            pid = int(open(self.pidfile).read())
        except (OSError, ValueError):
            return
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):
                os.kill(pid, 0)
                time.sleep(0.2)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.remove(self.pidfile)

    def ensure(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        if self.proc is not None:
            print(f"endpoint exited ({self.proc.returncode}), restarting", file=sys.stderr)
        self.proc = subprocess.Popen(self.cmd, cwd=REPO, env=self.env)
        with open(self.pidfile, "w") as f:
            f.write(str(self.proc.pid))
        self.wait_port()

    def wait_port(self):
        end = time.time() + 90
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("endpoint exited during start")
            try:
                socket.create_connection(("127.0.0.1", self.args.port), timeout=1).close()
                return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"endpoint not up on port {self.args.port} after 90 s")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)


def play_once(args):
    cmd = shlex.split(args.bench_cmd) + [
        "--endpoint", f"http://127.0.0.1:{args.port}/x", "--games", "1", "--parallel", "1",
    ]
    env = dict(os.environ)
    if args.path_prepend:
        env["PATH"] = args.path_prepend + os.pathsep + env["PATH"]
    try:
        p = subprocess.run(
            cmd, cwd=args.bench_dir, env=env, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    return parse_game(p.stdout + p.stderr)


def run(args, records):
    ep = Endpoint(args)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        ep.kill_stale()
        while len(records) < args.games:
            rec = None
            for attempt in range(1, 4):
                ep.ensure()
                rec = play_once(args)
                if rec:
                    break
                print(f"game {len(records) + 1}: attempt {attempt} failed", file=sys.stderr)
            if rec is None:
                return 2
            append(args.out_jsonl, rec)
            records.append(rec)
    finally:
        ep.stop()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir")
    ap.add_argument("out_jsonl")
    ap.add_argument("--games", type=int)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--port", type=int, default=11437)
    ap.add_argument("--bench-dir", default="/root/mjev-bench/jevman-benchmark")
    ap.add_argument("--path-prepend", default="")
    ap.add_argument("--bench-cmd", default="npm run bench --")
    ap.add_argument("--endpoint-cmd", help="full endpoint command; overrides the default")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    if args.summary:
        print(stats(load(args.out_jsonl, repair=False)))
        return 0
    if args.games is None:
        ap.error("--games is required")

    lock = open(args.out_jsonl + ".lock", "a")  # held until exit; the kernel frees it on SIGKILL
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("LOCKED")
        return 0

    records = load(args.out_jsonl, repair=True)
    if len(records) < args.games:
        rc = run(args, records)
        if rc:
            return rc
    print(stats(records[: args.games]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

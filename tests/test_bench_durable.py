"""Kill test for examples/jevman_bench_durable.py. Run: python3 tests/test_bench_durable.py

Uses a fake bench command and a fake endpoint (python -m http.server).
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "examples", "jevman_bench_durable.py")

FAKE_BENCH = """
import random, time
time.sleep(1)
s = random.randint(500, 2000)
print(f"  game 1: survived 1.0s, score {s}, pellets 90, deaths 3, level 1, ghosts 0, calls 32, fallbacks 4/29, $0.0000")
print('deaths by situation: {"waiting at junction":3}')
print(f"MEAN pacman=http://127.0.0.1/x ghosts=greedy: score {s}, survived 1.0s, pellets 110, pellets/life 37, ghosts 0, fruit eaten 8/8, fallbacks 6.7%, latency 848 ms, cost $0.0000/game")
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    return data[: data.rfind(b"\n") + 1].splitlines()


def wait_for(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.5)
    raise AssertionError("timed out")


class KillTest(unittest.TestCase):
    def test_sigkill_then_resume(self):
        tmp = tempfile.mkdtemp(prefix="bench-durable-")
        bench = os.path.join(tmp, "fake_bench.py")
        with open(bench, "w") as f:
            f.write(FAKE_BENCH)
        out = os.path.join(tmp, "out.jsonl")
        port = free_port()

        def start(log):
            cmd = [
                sys.executable, RUNNER, tmp, out, "--games", "10", "--threads", "1",
                "--port", str(port), "--bench-dir", tmp,
                "--bench-cmd", f"{sys.executable} {bench}",
                "--endpoint-cmd", f"{sys.executable} -m http.server {port} --bind 127.0.0.1",
            ]
            return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

        def endpoint_pid(runner_pid):
            r = subprocess.run(["pgrep", "-P", str(runner_pid), "-f", "http.server"],
                               capture_output=True, text=True)
            return int(r.stdout.split()[0])

        try:
            # Run 1: SIGKILL the runner after 4 lines exist.
            log1 = open(os.path.join(tmp, "run1.log"), "w")
            p1 = start(log1)
            wait_for(lambda: len(lines(out)) >= 4, 120)
            head = lines(out)[:4]
            p1.send_signal(signal.SIGKILL)
            p1.wait()
            with open(out, "ab") as f:  # simulate a write cut off by the kill
                f.write(b'{"score": 12')

            # Run 2: resume. Kill the endpoint mid-run; the runner must restart it.
            log2 = os.path.join(tmp, "run2.log")
            log2_f = open(log2, "w")
            p2 = start(log2_f)
            wait_for(lambda: len(lines(out)) >= 6, 120)
            os.kill(endpoint_pid(p2.pid), signal.SIGKILL)
            # A second runner on the same file must report LOCKED.
            locked = subprocess.run(
                [sys.executable, RUNNER, tmp, out, "--games", "10", "--port", str(port),
                 "--endpoint-cmd", "true"],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(locked.stdout.strip(), "LOCKED")
            self.assertEqual(p2.wait(timeout=300), 0)

            all_lines = lines(out)
            self.assertEqual(len(all_lines), 10)
            self.assertEqual(all_lines[:4], head)
            for line in all_lines:
                json.loads(line)
            with open(log2) as f:
                self.assertIn("restarting", f.read())
        finally:
            subprocess.run(["pkill", "-f", f"http.server {port}"])


if __name__ == "__main__":
    unittest.main()

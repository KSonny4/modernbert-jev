"""Kill test: --data jevman resumes from state.pt after SIGKILL.

Needs torch and transformers. Set JEVMAN_TEST_DATA to a dir holding jevman-oracle-v2.jsonl and
out-oracle6k-ckpt/ (tokenizer.json, tokenizer_config.json). The test skips
when the data is missing.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
DATA = Path(os.environ.get("JEVMAN_TEST_DATA", "/nonexistent"))


def make_tiny_model(model_dir: Path) -> None:
    from transformers import ModernBertConfig, ModernBertForSequenceClassification

    labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    cfg = ModernBertConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=512, num_labels=3, id2label=labels,
        label2id={v: k for k, v in labels.items()},
    )
    ModernBertForSequenceClassification(cfg).save_pretrained(model_dir)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(DATA / "out-oracle6k-ckpt" / name, model_dir / name)


@unittest.skipUnless((DATA / "jevman-oracle-v2.jsonl").exists(), "set JEVMAN_TEST_DATA")
class ResumeKillTest(unittest.TestCase):
    def test_sigkill_then_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_tiny_model(tmp / "tiny")
            with open(DATA / "jevman-oracle-v2.jsonl") as src, open(tmp / "data.jsonl", "w") as dst:
                dst.writelines(src.readlines()[:300])
            # CPU only: a tiny model on MPS runs about 4x slower than on CPU here.
            # os._exit skips the teardown, where onnxruntime aborts on this Mac (exit -6).
            cpu_main = (
                "import os, sys, torch\n"
                "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n"
                "torch.backends.mps.is_available = lambda: False\n"
                "from modernbert_jev.train import main\n"
                "code = main(sys.argv[1:])\n"
                "sys.stdout.flush()\n"
                "os._exit(code)\n"
            )
            cmd = [
                sys.executable, "-u", "-c", cpu_main,
                "--data", "jevman", "--jsonl", str(tmp / "data.jsonl"),
                "--start-from", str(tmp / "tiny"), "--out", str(tmp / "out"),
                "--epochs", "1", "--max-steps", "100", "--max-length", "128",
                "--ckpt-dir", str(tmp / "ckpt"), "--ckpt-every", "20",
            ]
            env = {**os.environ, "PYTHONPATH": str(SRC)}

            # First run: SIGKILL right after the step=50 log line.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            first_log = []
            for line in proc.stdout:
                first_log.append(line)
                if re.search(r"step=50\b", line):
                    proc.send_signal(signal.SIGKILL)
                    break
            proc.stdout.close()
            proc.wait()
            self.assertEqual(proc.returncode, -signal.SIGKILL, "".join(first_log[-5:]))
            self.assertTrue((tmp / "ckpt" / "state.pt").exists())

            # Rerun the same command: it must resume from step 40 and finish.
            done = subprocess.run(cmd, capture_output=True, text=True, env=env)
            log = done.stdout + done.stderr
            self.assertEqual(done.returncode, 0, log[-2000:])
            self.assertIn("resumed at step=40", log)
            self.assertNotIn("before:", log)
            self.assertRegex(log, r"step=100\b")
            self.assertIn("after:", log)


if __name__ == "__main__":
    unittest.main()

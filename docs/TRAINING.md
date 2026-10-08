# Training v2 on the 7800 XT box

Target: fine-tune DeBERTa-v3-base (184 M) on decision data with KL
distillation, export to int8 ONNX, stay in the 1 CPU / 1 GB serving
budget. Serving code needs no changes: a trained directory drops into
`modernbert-jev serve` / `modernbert-jev run` as-is.

## 1. ROCm on WSL2 (one time, ~30 min)

No dual-boot needed: ROCm 6.x ships a WSL runtime and the RX 7800 XT
(gfx1101) works through `/dev/dxg`. In WSL Ubuntu 24.04:

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update && sudo apt-get install -y wget gnupg2
wget https://repo.radeon.com/amdgpu-install/6.4.4/ubuntu/noble/amdgpu-install_6.4.60404-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.4.60404-1_all.deb
sudo amdgpu-install -y --usecase=rocm --no-dkms  # no DKMS: WSL uses the Windows driver
rocminfo | grep -i "marketing\|gfx"  # want gfx1101 / RX 7800 XT
```

Then the project env (venv lives in the checkout, per repo convention):

```bash
git clone <modernbert-jev-repo> ~/modernbert-jev && cd ~/modernbert-jev
python3 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
.venv/bin/pip install -e ".[train]"
# WSL gotcha: swap the wheel-bundled HSA runtime for the WSL one,
# or torch.cuda stays False:
TORCH_LIB=$(.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
cp /opt/rocm/lib/libhsa-runtime64.so.1 $TORCH_LIB/libhsa-runtime64.so
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"  # want True
```

`torch.cuda` is the API name even on ROCm; `modernbert-jev-train` picks CUDA
automatically (`pick_device`: cuda > mps > cpu).

## 2. Data (do not improvise splits)

`SargeDev/jev-distill-corpus-v3` (Apache-2.0, 741 k rows, teacher: Jev 1.13):

| Split | Rows | Use |
| --- | --- | --- |
| train | 655,806 | training pool |
| validation | 14,111 | early stopping only |
| calibration | 13,766 | temperature fitting only |
| test / ood / test_set_30k | 14k / 13k / 30k | EVALUATION ONLY, never train |

Rules: never train on test/ood/test_set_30k. JevBench public 231 is a
regression check, not a tuning set anymore (templates were already
selected on it). MNLI trains only as the auxiliary loss (`--aux-weight`).

## 3. Runs (escalate, don't jump)

```bash
# Smoke: 3 steps, proves the loop + export on your box (~10 min):
python -m modernbert_jev.train --out models/modernbert-jev-v2-smoke --max-rows 32 \
  --batch-rows 4 --max-steps 3

# Short: 50 k rows, check the loss curve + a 30 k-sample eval:
python -m modernbert_jev.train --out models/modernbert-jev-v2-short --max-rows 50000 \
  --batch-rows 32 --epochs 1

# Full: 150 k rows overnight, then all 656 k only if the curve earns it:
python -m modernbert_jev.train --out models/modernbert-jev-v2 --max-rows 150000 \
  --batch-rows 32 --epochs 2
```

Rough sizing for a 7800 XT (fp32, batch-rows 32): expect single-digit
seconds per step; extrapolate from the first 100 steps (`loss` prints
every 100). 150 k rows ≈ 4.7 k steps ≈ one night. Full 656 k ≈ days.
16 GB VRAM fits batch-rows 32 comfortably; lower it on OOM, never raise
the learning rate to compensate (default 2e-5).

## 4. After training

1. Fit temperatures on the corpus `calibration` split (adapt
   `/tmp/mjev_calibrate.py` or ask; do NOT reuse v1's 2.0/1.0/4.0 blindly).
2. Evaluate, in this order: `test_set_30k` sample (primary), `ood`
   (generalization), JevBench public 231 (regression — must stay ≥ 0.528),
   BoolQ + massive (external sanity).
3. Ship: the directory already contains `model.onnx`,
   `tokenizer.json`, `config.json`, `calibration.json`. Peak-RAM check:
   must stay under 1 GB at `--threads 1`.
4. Board submission needs a public GitHub/HF link + contact email at
   benchmarkheaven.com/submit (free queue). Publish weights first.

## What success looks like

v1 (zero-shot small): 0.528 public, hard-tier near chance, meta-options
broken. v2 wins if: hard-public clearly above chance, meta-options work,
and the 30 k held-out beats v1's public number. If the loss falls but
the 30 k doesn't move, the run failed — say so and change the data, not
the metric.

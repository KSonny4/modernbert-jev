# Running tasks on the Windows / WSL training box

The GPU box is a Windows desktop (RX 7900 XT) with WSL2 Ubuntu 24.04.
All compute runs inside WSL, driven over SSH from the Mac. This file
records the patterns that work; each rule below cost a failed launch.

## Connection

- SSH alias `windows` -> `Ksonn@<lan-ip>`, key auth, `BatchMode=yes`.
- The box uses DHCP: the IP has moved before (`.91` -> `.100`).
  If SSH times out, sweep the LAN and match the host key, then update
  `~/.ssh/config`.
- Keep the box awake: `powercfg -change -standby-timeout-ac 0` and
  disable hibernate. A sleeping box kills runs and reboots WSL.

## One `wsl` call per SSH invocation

The Windows SSH server runs commands via `cmd`, which mangles chains:

```sh
# WRONG: dies with "The system cannot find the path specified"
ssh windows "wsl -d Ubuntu-24.04 ls /root; wsl -d Ubuntu-24.04 nvidia-smi"

# RIGHT: one wsl call, compound logic inside a script
ssh windows "wsl -d Ubuntu-24.04 tee /tmp/job.sh" < job.sh
ssh windows "wsl -d Ubuntu-24.04 bash /tmp/job.sh"
```

Pipes and `tail`/`grep` inside the SSH string also break; keep them
inside the script file.

## Ship files over stdin, never scp to /tmp

Windows has no `/tmp`, so `scp file windows:/tmp/file` fails.
`tee` over stdin is binary-safe and works for tarballs too:

```sh
ssh windows "wsl -d Ubuntu-24.04 tee /tmp/ship.tar.gz" < ship.tar.gz
```

## Persist under /root, never /tmp

`/tmp` inside WSL does not survive the reboot that follows a sleep.
Checkpoints, logs, data caches, and model outputs all go under `/root`:

- code: `/root/modernbert-jev`, venv: `/root/modernbert-jev/.venv`
- run log: `/root/modernbert-jev-train-<name>.log`
- model out: `/root/modernbert-jev-v2-<name>`
- HF cache: `/root/hf-cache` (export `HF_HOME` to match)
- checkpoints every 2000 steps + `--resume-from` support in `train.py`

## Launch pattern

Plain `&` backgrounding is NOT enough: when the launching SSH session
closes, the `wsl.exe` teardown kills the whole process tree. Two runs
died this way (silent, no traceback, frozen log). Always detach with
`setsid -f`, and trap the exit code so a real crash is distinguishable
from a session teardown:

```sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf-cache
cat > /root/modernbert-jev-run-<name>.sh <<'INNER'
#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf-cache
/root/modernbert-jev/.venv/bin/python -m modernbert_jev.train --backbone <hf-id> \
  --out /root/modernbert-jev-v2-<name> --data distill \
  --max-rows 50000 --batch-rows 4 --max-length 192 --epochs 1 \
  > /root/modernbert-jev-train-<name>.log 2>&1
echo "EXIT-CODE=$? FINISHED-AT=$(date -u +%FT%TZ)" >> /root/modernbert-jev-train-<name>.log
INNER
chmod +x /root/modernbert-jev-run-<name>.sh
setsid -f /root/modernbert-jev-run-<name>.sh < /dev/null > /dev/null 2>&1
sleep 150
tail -n 5 /root/modernbert-jev-train-<name>.log
pgrep -af "modernbert_jev.train" > /dev/null && echo TRAIN-RUNNING || echo TRAIN-DEAD
```

## Scheduled tasks, not background jobs

When the run must survive the SSH session, register a Windows Scheduled
Task (the polymarket pattern: the OS owns the process, SSH only triggers
and polls):

```cmd
schtasks /Create /TN mjev_train_<name> /SC ONCE /ST 23:59 /TR "wsl -d Ubuntu-24.04 bash /root/modernbert-jev-run-<name>.sh" /F
schtasks /Run /TN mjev_train_<name>
schtasks /Query /TN mjev_train_<name> /FO LIST   & rem Status: Running?
```

**Delete the task when done** (`schtasks /Delete /TN <name> /F`).
The `/SC ONCE /ST 23:59` trigger is real: a spent 50k task re-fired at
23:59 and contended the GPU with the overnight run until it was killed.
Two `modernbert_jev.train` processes at ~250% CPU each is the signature.

Notes:

- `--data` is a mode flag (`distill` | `mnli`), not a path. The corpus
  streams from the HF cache; there is no local parquet to check for.
- `wsl: Failed to start the systemd user session` on stderr is noise;
  the command still runs.
- Verify with `tail` on the persistent log and `ls` on the out dir.
  `pgrep` exit 1 means nothing is running.

## Environment setup

ROCm install, venv, and PyTorch versions are in `docs/TRAINING.md`.
When in doubt, `rocminfo | grep gfx` and
`python -c "import torch; print(torch.cuda.is_available())"` inside WSL
confirm the GPU path is alive before launching anything long.

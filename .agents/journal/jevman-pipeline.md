# Jevman pipeline journal

## 2026-10-10: DBOS workflow (issue KSonny4/modernbert-jev#1)

- Added `examples/jevman_pipeline.py`: one DBOS workflow per run (train, ship, bench-<label>).
- DBOS 3.2.0 runs on SQLite (`sqlite:///<state_dir>/dbos.sqlite`). No Postgres.
- Added `examples/jevman_pipeline.plist.example` (launchd KeepAlive, restart on crash only).
- Kill test `tests/test_pipeline_durable.py`: SIGKILL during bench-8, rerun. Train ran once, ship ran once, pipeline finished.
- 2026-10-10: `tests/test_pipeline_durable.py` now skips (not fails) when `dbos` is missing (`pytest.importorskip`).

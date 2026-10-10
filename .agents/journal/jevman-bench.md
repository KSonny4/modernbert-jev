# Jevman bench journal

## 2026-10-10 - Durable bench runner (PR: jevman-bench-durable)

- Added `examples/jevman_bench_durable.py`. One game per bench call. One JSON line per game, fsynced.
- A killed run resumes at the next missing game. A partial last line is cut off.
- A file lock stops a second runner on the same output file.
- The endpoint pid is kept in a file. A new run stops a stale endpoint and starts a fresh one.
- Test: `python3 tests/test_bench_durable.py`. It sends SIGKILL to the runner and to the endpoint.

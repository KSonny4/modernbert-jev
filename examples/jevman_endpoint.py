"""Jevman bench endpoint: serve one model on any path, for the jevman bench.

Run:
    PYTHONPATH=src python examples/jevman_endpoint.py <model_dir> <port> [flags]

Flags:
    --threads N            ONNX intra-op threads (default 1)
    --dump FILE            append each raw request body to FILE, one JSON per line
    --premise state|instructions
                           state (default): premise is the request state.
                           instructions: premise is the first question's instructions.
    --no-pmi               disable null-premise (PMI) subtraction for choice questions

Accepts POST on any path. Drops state["maze"] (too long to score), fills
"model" when missing, and returns the SystemOneResponse JSON.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from modernbert_jev import runtime
from modernbert_jev.api import ApiError, parse_request
from modernbert_jev.runtime import DecisionModel

args = sys.argv[1:]
threads = 1
if "--threads" in args:
    i = args.index("--threads")
    threads = int(args[i + 1])
    del args[i : i + 2]
dump_path = None
if "--dump" in args:
    i = args.index("--dump")
    dump_path = args[i + 1]
    del args[i : i + 2]
premise = "state"
if "--premise" in args:
    i = args.index("--premise")
    premise = args[i + 1]
    del args[i : i + 2]
    if premise not in ("state", "instructions"):
        sys.exit("--premise must be state or instructions")
if "--no-pmi" in args:
    args.remove("--no-pmi")
    runtime.PMI_ALPHA = 0.0
model_dir, port = args[0], int(args[1])
model = DecisionModel(model_dir, threads=threads)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"null")
            if not isinstance(raw, dict):
                raise ApiError("body must be a JSON object")
            if dump_path:
                try:
                    with open(dump_path, "a") as fh:
                        fh.write(json.dumps(raw) + "\n")
                except OSError as exc:
                    print(f"dump failed: {exc}", file=sys.stderr, flush=True)
            if isinstance(raw.get("state"), dict):
                raw["state"].pop("maze", None)
            # runtime cuts the premise to MAX_STATE_CHARS in ask(); train.py --premise instructions builds the same text.
            if premise == "instructions":
                questions = raw.get("questions")
                values = questions.values() if isinstance(questions, dict) else []
                first = next((q for q in values if isinstance(q, dict) and q.get("type") == "choice"), None)
                if first is not None and isinstance(first.get("instructions"), str):
                    raw["state"] = first["instructions"]
            raw.setdefault("model", model.name)
            req = parse_request(raw)
            try:
                status, body = 200, model.ask(req).to_dict()
            except Exception as exc:
                status, body = 500, {"error": {"message": f"inference failed: {exc}"}}
        except (ValueError, ApiError) as exc:
            status, body = 400, {"error": {"message": str(exc)}}
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


print(f"jevman endpoint: {model.name} on http://127.0.0.1:{port} ({threads} thread(s))", flush=True)
HTTPServer(("127.0.0.1", port), Handler).serve_forever()

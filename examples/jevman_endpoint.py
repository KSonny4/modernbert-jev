"""Jevman bench endpoint: serve one model on any path, for the jevman bench.

Run:
    PYTHONPATH=src python examples/jevman_endpoint.py <model_dir> <port> [--threads N]

Accepts POST on any path. Drops state["maze"] (too long to score), fills
"model" when missing, and returns the SystemOneResponse JSON.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from modernbert_jev.api import ApiError, parse_request
from modernbert_jev.runtime import DecisionModel

args = sys.argv[1:]
threads = 1
if "--threads" in args:
    i = args.index("--threads")
    threads = int(args[i + 1])
    del args[i : i + 2]
model_dir, port = args[0], int(args[1])
model = DecisionModel(model_dir, threads=threads)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"null")
            if not isinstance(raw, dict):
                raise ApiError("body must be a JSON object")
            if isinstance(raw.get("state"), dict):
                raw["state"].pop("maze", None)
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

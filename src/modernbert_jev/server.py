"""Local HTTP server with the TypeSafe request/response shapes.

Serves ``POST /v1/systemone`` and ``GET /v1/models`` from the standard
library only, so the runtime stays small enough for a 1 CPU / 1 GB box.
Point any TypeSafe-compatible client at it by setting the base URL.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from modernbert_jev.api import ApiError, parse_request
from modernbert_jev.runtime import DecisionModel

DEFAULT_PORT = 11436


def find_models(models_dir: str) -> list[str]:
    """Names of installed models: subdirs holding a ``model.onnx`` file."""
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        name
        for name in os.listdir(models_dir)
        if os.path.isfile(os.path.join(models_dir, name, "model.onnx"))
    )


def dispatch(
    model: DecisionModel, models_dir: str, method: str, path: str, body: bytes
) -> tuple[int, dict]:
    """Route one request; pure logic shared by the handler and the tests."""
    if method == "GET" and path == "/v1/models":
        return 200, {"models": [{"name": name} for name in find_models(models_dir)]}
    if method == "POST" and path == "/v1/systemone":
        try:
            raw = json.loads(body or b"null")
        except json.JSONDecodeError as exc:
            return 400, {"error": {"message": f"invalid JSON: {exc}"}}
        try:
            request = parse_request(raw)
        except ApiError as exc:
            return 400, {"error": {"message": str(exc)}}
        try:
            response = model.ask(request)
        except Exception as exc:  # never leak a traceback over HTTP
            return 500, {"error": {"message": f"inference failed: {exc}"}}
        return 200, response.to_dict()
    return 404, {"error": {"message": f"unknown path {path}"}}


class Handler(BaseHTTPRequestHandler):
    model: DecisionModel
    models_dir: str
    server_version = "modernbert-jev/0.1.0"

    def log_message(self, *args: object) -> None:  # quiet by default
        pass

    def _send(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self, method: str) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        status, payload = dispatch(
            self.model, self.models_dir, method, path, self.rfile.read(length)
        )
        self._send(status, payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        self._handle("POST")


def serve(model_dir: str, host: str, port: int, threads: int) -> None:
    """Load one model and serve it until interrupted."""
    model = DecisionModel(model_dir, threads=threads)
    Handler.model = model
    Handler.models_dir = os.path.dirname(os.path.abspath(model_dir))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"modernbert-jev serving {model.name} on http://{host}:{port} ({threads} thread(s))")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, cast


class FakeInstalledProvider:
    """Small OpenAI-compatible provider used by installed-artifact subprocesses."""

    def __init__(self, *, block_streaming: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self._authorization_matches: list[bool] = []
        self._block_streaming = block_streaming
        self.streaming_started = threading.Event()
        self.release_streaming = threading.Event()
        self._request_seen = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._active_requests = 0
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = cast(dict[str, Any], json.loads(self.rfile.read(length)))
                authorization_matches = (
                    self.headers.get("Authorization")
                    == "Bearer synthetic-provider-credential"
                )
                with parent._lock:
                    parent.requests.append(payload)
                    parent._authorization_matches.append(authorization_matches)
                    parent._active_requests += 1
                    parent._idle.clear()
                    parent._request_seen.set()
                try:
                    chunks = [
                        {
                            "role": "assistant",
                            "content": "Hello from installed fake provider",
                        },
                        {"role": "assistant", "content": ""},
                    ]
                    if not payload.get("stream"):
                        body = json.dumps({
                            "id": "installed-fake-id",
                            "object": "chat.completion",
                            "model": "installed-fake-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": chunks[0]["content"],
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for index, delta in enumerate(chunks):
                        chunk = {
                            "id": "installed-fake-id",
                            "object": "chat.completion.chunk",
                            "created": index,
                            "model": "installed-fake-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": "stop" if index == 1 else None,
                                }
                            ],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if parent._block_streaming:
                            parent.streaming_started.set()
                            parent.release_streaming.wait()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                finally:
                    with parent._lock:
                        parent._active_requests -= 1
                        if parent._active_requests == 0:
                            parent._idle.set()

        return Handler

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def server_port(self) -> int:
        return self._server.server_port

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self.requests)

    @property
    def authorization_matches(self) -> list[bool]:
        with self._lock:
            return list(self._authorization_matches)

    def wait_for_request(self, count: int, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while self.request_count < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._request_seen.wait(remaining):
                raise AssertionError(f"provider did not receive request {count}")

    def wait_for_idle(self, timeout: float = 30) -> None:
        if not self._idle.wait(timeout):
            raise AssertionError("provider still has an active request")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.release_streaming.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)

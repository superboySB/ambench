# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Small, versioned HTTP protocol for ACT and Diffusion Policy inference.

Arrays use base64 encoded contiguous bytes so image layout and floating point
values survive a round trip without pickle or a policy framework dependency.
"""

from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib import error, request

import numpy as np

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 32 * 1024 * 1024
ALLOWED_DTYPES = {"uint8", "float32", "float64", "int32", "int64", "bool"}


def encode_value(value: Any) -> Any:
    """Convert nested NumPy and Torch values into JSON compatible values."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.name not in ALLOWED_DTYPES:
            raise TypeError(f"Unsupported RPC array dtype: {value.dtype}.")
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": base64.b64encode(contiguous.tobytes()).decode("ascii"),
            "dtype": contiguous.dtype.name,
            "shape": list(contiguous.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported RPC value type: {type(value).__name__}.")


def decode_value(value: Any) -> Any:
    """Restore arrays while rejecting object dtypes and malformed payloads."""
    if isinstance(value, dict):
        if "__ndarray__" in value:
            if set(value) != {"__ndarray__", "dtype", "shape"}:
                raise ValueError("Invalid RPC array fields.")
            dtype_name = value["dtype"]
            shape = value["shape"]
            if dtype_name not in ALLOWED_DTYPES or not isinstance(shape, list):
                raise ValueError("Invalid RPC array dtype or shape.")
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise ValueError("Invalid RPC array dimensions.")
            raw = base64.b64decode(value["__ndarray__"], validate=True)
            dtype = np.dtype(dtype_name)
            expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(raw) != expected_size:
                raise ValueError("RPC array byte count does not match its shape.")
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        return {key: decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


def dumps(value: Any) -> bytes:
    """Serialize a protocol message."""
    return json.dumps(encode_value(value), separators=(",", ":"), allow_nan=False).encode("utf-8")


def loads(payload: bytes) -> Any:
    """Deserialize a protocol message."""
    return decode_value(json.loads(payload.decode("utf-8")))


class RemotePolicyClient:
    """Synchronous client; simulator stepping waits for each policy response."""

    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def call(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else dumps({"version": PROTOCOL_VERSION, **payload})
        method = "GET" if data is None else "POST"
        req = request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                result = loads(response.read())
        except error.HTTPError as exc:
            try:
                message = loads(exc.read()).get("error", str(exc))
            except (ValueError, AttributeError):
                message = str(exc)
            raise RuntimeError(f"Remote policy {path} failed: {message}") from exc
        if not isinstance(result, dict) or result.get("version") != PROTOCOL_VERSION:
            raise ValueError("Remote policy returned an incompatible protocol version.")
        return result

    def info(self) -> dict[str, Any]:
        return self.call("info")

    def reset(self, **payload: Any) -> dict[str, Any]:
        return self.call("reset", payload)

    def infer(self, **payload: Any) -> dict[str, Any]:
        return self.call("infer", payload)


def make_handler(backend: Any) -> type[BaseHTTPRequestHandler]:
    """Bind one model backend to the protocol's HTTP routes."""

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status: int, body: dict[str, Any]) -> None:
            payload = dumps({"version": PROTOCOL_VERSION, **body})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._respond(200, {"status": "ready", "policy": backend.policy_name})
            elif self.path == "/info":
                self._respond(200, backend.info())
            else:
                self._respond(404, {"error": "Unknown route."})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in ("/reset", "/infer"):
                self._respond(404, {"error": "Unknown route."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid or oversized RPC request.")
                body = loads(self.rfile.read(length))
                if not isinstance(body, dict) or body.pop("version", None) != PROTOCOL_VERSION:
                    raise ValueError("Incompatible RPC protocol version.")
                result = backend.reset(**body) if self.path == "/reset" else backend.infer(**body)
                self._respond(200, result)
            except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                self._respond(400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(backend: Any, host: str, port: int) -> None:
    """Serve one policy instance; requests are processed serially."""
    with HTTPServer((host, port), make_handler(backend)) as server:
        print(f"Serving {backend.policy_name} on http://{host}:{port}", flush=True)
        server.serve_forever()

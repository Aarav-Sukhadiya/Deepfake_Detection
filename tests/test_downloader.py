"""Tests for the weight downloader: hashing, atomic download, verification."""
from __future__ import annotations

import hashlib
import http.server
import socket
import threading
from pathlib import Path

import pytest

from dfdetect.models.downloader import (
    ChecksumError,
    compute_sha256,
    download_file,
    ensure_weights_file,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(tmp_path: Path):
    """Serve ``tmp_path`` over HTTP on a free port for the duration of a test."""
    port = _free_port()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp_path), **kw)

        def log_message(self, *_):  # silence noise
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", tmp_path
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_compute_sha256(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello world")
    assert compute_sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def test_download_file_verifies_hash(http_server, tmp_path: Path):
    base, served = http_server
    payload = b"deepfake weights payload" * 1024
    (served / "weights.pt").write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    dest = tmp_path / "cache" / "weights.pt"
    out = download_file(f"{base}/weights.pt", dest, expected_sha256=expected)
    assert out == dest
    assert dest.read_bytes() == payload


def test_download_file_rejects_bad_hash(http_server, tmp_path: Path):
    base, served = http_server
    (served / "weights.pt").write_bytes(b"tampered")

    dest = tmp_path / "cache" / "weights.pt"
    with pytest.raises(ChecksumError):
        download_file(f"{base}/weights.pt", dest, expected_sha256="00" * 32)
    assert not dest.exists()
    # The partial file should be cleaned up too.
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_ensure_weights_reuses_cached_file(http_server, tmp_path: Path):
    base, served = http_server
    payload = b"cached payload"
    (served / "weights.pt").write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    cache = tmp_path / "cache"
    first = ensure_weights_file(
        model_name="efficientnet_b4",
        weights_dir=cache,
        configured_url=f"{base}/weights.pt",
        configured_sha256=expected,
    )
    assert first is not None and first.exists()
    mtime = first.stat().st_mtime

    # Second call should NOT re-download (file present, hash matches).
    second = ensure_weights_file(
        model_name="efficientnet_b4",
        weights_dir=cache,
        configured_url=f"{base}/weights.pt",
        configured_sha256=expected,
    )
    assert second == first
    assert second.stat().st_mtime == mtime


def test_ensure_weights_returns_none_when_no_source(tmp_path: Path):
    assert (
        ensure_weights_file(
            model_name="efficientnet_b4",
            weights_dir=tmp_path,
        )
        is None
    )

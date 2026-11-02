"""Serve Python functions, log densities and samplers as FlatPPL black boxes.

A FlatPPL model loads a black box as a module of ``external(anything)``
members. A FlatPPL engine then sends the box primitive queries over HTTP, with
data as Arrow IPC (see ``docs/protocol.md``).

>>> import numpy as np
>>> import flatppl_bridge as fb
>>> box = fb.Box()
>>> box.function("square", lambda x: x**2, vjp=lambda x, ct: 2 * x * ct)
>>> with fb.Server({"demo": box}) as server:
...     url = server.url
"""

from __future__ import annotations

import os
import signal
import threading
from typing import Mapping

from . import _native
from ._box import Box, InvalidInput

__all__ = ["Box", "InvalidInput", "Server", "serve", "PROTOCOL_VERSION"]

PROTOCOL_VERSION: int = _native.PROTOCOL_VERSION


class Server:
    """A running box server. It serves in background threads until :meth:`close`.

    :param boxes: Python boxes by name.
    :param host: The address to listen on.
    :param port: The port. ``0`` picks a free one; read it from :attr:`url`.
    :param libraries: Plugin libraries by box name, as ``(path, config)``,
        for example the BridgeStan plugin.
    :param token: Require ``Authorization: Bearer <token>``.
    :param store: Keep uploaded objects in this directory instead of memory.
    :param store_bytes: The in-memory object store size.
    :param cache_bytes: The reply cache size. ``0`` turns memoization off.
    :param body_limit: The largest request body.

    Serve over HTTPS outside a trusted network, for example behind a reverse
    proxy or with the ``flatppl-bridge-server`` binary.
    """

    def __init__(
        self,
        boxes: Mapping[str, Box] | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        libraries: Mapping[str, tuple[str | os.PathLike, str]] | None = None,
        token: str | None = None,
        store: str | os.PathLike | None = None,
        store_bytes: int = 1 << 30,
        cache_bytes: int = 1 << 28,
        body_limit: int = 1 << 30,
    ) -> None:
        native_boxes = [(name, box._dispatch, box.describe()) for name, box in (boxes or {}).items()]
        native_libraries = [(name, os.fspath(path), config) for name, (path, config) in (libraries or {}).items()]
        if not native_boxes and not native_libraries:
            raise ValueError("give at least one box or library")
        self._native = _native.Server(
            native_boxes,
            native_libraries,
            host,
            port,
            token,
            None if store is None else os.fspath(store),
            store_bytes,
            cache_bytes,
            body_limit,
        )

    @property
    def url(self) -> str:
        """The base URL, for example ``http://127.0.0.1:8737``."""
        return self._native.url

    def close(self) -> None:
        """Stop serving and wait for running queries to finish."""
        self._native.close()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def serve(boxes: Mapping[str, Box], host: str = "127.0.0.1", port: int = 8737, **options) -> None:
    """Serve ``boxes`` until interrupted. Takes the options of :class:`Server`."""
    server = Server(boxes, host=host, port=port, **options)
    print(f"serving {server.url}", flush=True)
    stop = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)
        server.close()

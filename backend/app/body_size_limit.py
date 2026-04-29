from __future__ import annotations

from collections.abc import Awaitable, Callable


class BodySizeLimitMiddleware:
    def __init__(self, app, *, max_body_bytes: int, paths: set[str]) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.paths = paths

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] != "http.request":
                return message
            body = message.get("body", b"")
            received_bytes += len(body)
            if received_bytes > self.max_body_bytes:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"detail":"Request body is too large."}',
                        "more_body": False,
                    }
                )
                return {"type": "http.disconnect"}
            return message

        await self.app(scope, limited_receive, send)

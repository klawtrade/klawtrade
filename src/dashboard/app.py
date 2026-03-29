"""FastAPI web dashboard for the Sentinel trading system."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class Dashboard:
    """Manages the FastAPI app and real-time WebSocket connections."""

    def __init__(self) -> None:
        self.app = FastAPI(title="Sentinel Dashboard", docs_url=None, redoc_url=None)
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=True,
        )
        self._connections: list[WebSocket] = []

        # References set by the orchestrator after construction
        self._get_state: Optional[Any] = None
        self._get_trades: Optional[Any] = None
        self._get_signals: Optional[Any] = None
        self._kill_switch: Optional[Any] = None
        self._resume_switch: Optional[Any] = None

        self._register_routes()

    # ------------------------------------------------------------------
    # Orchestrator wiring
    # ------------------------------------------------------------------

    def set_callbacks(
        self,
        get_state: Any,
        get_trades: Any,
        get_signals: Any,
        kill_switch: Any,
        resume_switch: Any,
    ) -> None:
        """Wire up callbacks from the main orchestrator.

        Args:
            get_state: Callable returning a dict of the current portfolio state.
            get_trades: Async callable returning a list of recent trade dicts.
            get_signals: Async callable returning a list of recent signal dicts.
            kill_switch: Callable to trigger the circuit breaker.
            resume_switch: Callable to clear the circuit breaker.
        """
        self._get_state = get_state
        self._get_trades = get_trades
        self._get_signals = get_signals
        self._kill_switch = kill_switch
        self._resume_switch = resume_switch

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            template = self._jinja.get_template("index.html")
            state = self._get_state() if self._get_state else {}
            html = template.render(state=state)
            return HTMLResponse(content=html)

        @app.get("/api/state")
        async def api_state() -> dict:
            if self._get_state:
                return self._get_state()
            return {}

        @app.get("/api/trades")
        async def api_trades() -> list[dict]:
            if self._get_trades:
                return await self._get_trades()
            return []

        @app.get("/api/signals")
        async def api_signals() -> list[dict]:
            if self._get_signals:
                return await self._get_signals()
            return []

        @app.post("/api/kill")
        async def api_kill() -> dict:
            if self._kill_switch:
                self._kill_switch()
                return {"status": "halted", "message": "Circuit breaker triggered"}
            return {"status": "error", "message": "Kill switch not configured"}

        @app.post("/api/resume")
        async def api_resume() -> dict:
            if self._resume_switch:
                self._resume_switch()
                return {"status": "resumed", "message": "Circuit breaker cleared"}
            return {"status": "error", "message": "Resume switch not configured"}

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            self._connections.append(ws)
            logger.info("WebSocket client connected", extra={"total": len(self._connections)})
            try:
                while True:
                    # Keep connection alive; client can send pings
                    await ws.receive_text()
            except WebSocketDisconnect:
                self._connections.remove(ws)
                logger.info("WebSocket client disconnected", extra={"total": len(self._connections)})

    # ------------------------------------------------------------------
    # Real-time push
    # ------------------------------------------------------------------

    async def broadcast_state(self, state: dict) -> None:
        """Push the current state to all connected WebSocket clients."""
        if not self._connections:
            return
        payload = json.dumps(state, default=str)
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._connections.remove(ws)

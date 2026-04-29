from __future__ import annotations

from typing import Dict

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response


def register_base_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

"""Translate ApiException subclasses into the unified error envelope.

ApiException extends plain Exception (not HTTPException), so FastAPI won't
convert it on its own — this handler does, emitting:
    {"error": {"code", "message", "reason"?, "fields"?}}
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import ApiException


def add_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiException)
    async def _handle_api_exception(_: Request, exc: ApiException) -> JSONResponse:
        error: dict[str, object] = {"code": exc.code, "message": exc.message}
        if exc.reason is not None:
            error["reason"] = exc.reason
        if exc.fields is not None:
            error["fields"] = exc.fields
        return JSONResponse(status_code=exc.status, content={"error": error})

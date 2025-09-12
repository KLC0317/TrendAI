"""
Custom exceptions and error handlers
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse


def setup_exception_handlers(app: FastAPI):
    """Setup application exception handlers"""
    
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found"}
        )

    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

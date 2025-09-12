"""
TrendAI Backend API - Legacy Entry Point
This file maintains backward compatibility while using the new modular structure.
"""
from app.main import app

# Re-export app for backward compatibility
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

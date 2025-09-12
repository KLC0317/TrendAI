"""
TrendAI Backend API - Main Application Entry Point
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.config import settings
from app.core.middleware import setup_middleware
from app.core.exceptions import setup_exception_handlers
from app.routers import health, trends, predictions, analysis
from app.services.model_loader import get_model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting TrendAI Backend API...")
    
    # Initialize model on startup
    model_loader = get_model_loader()
    model_loader.initialize_model()
    
    yield
    
    # Shutdown
    print("Shutting down TrendAI Backend API...")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(trends.router, prefix="/api", tags=["trends"])
app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

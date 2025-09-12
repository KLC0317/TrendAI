"""
Dependency injection for FastAPI
"""
from app.services.trend_service import TrendService
from app.services.prediction_service import PredictionService
from app.services.model_loader import ModelLoaderService


def get_trend_service() -> TrendService:
    """Get trend service instance"""
    return TrendService()


def get_prediction_service() -> PredictionService:
    """Get prediction service instance"""
    return PredictionService()


def get_model_loader() -> ModelLoaderService:
    """Get model loader service instance"""
    return ModelLoaderService()

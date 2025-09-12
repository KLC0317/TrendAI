"""
Application configuration
"""
from typing import List


class Settings:
    """Application settings"""
    
    # API Info
    API_TITLE: str = "TrendAI Backend API"
    API_DESCRIPTION: str = "Backend API for TrendAI trend analysis data with trained model integration"
    API_VERSION: str = "1.0.0"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://localhost:5173", 
        "*"
    ]
    
    # Cache Settings
    CACHE_DURATION: int = 300  # 5 minutes
    
    # Model Settings
    MODEL_BASE_PATH: str = './saved_trend_models'
    MODEL_NAME: str = 'enhanced_trend_ai_with_export_20250906_161418'
    
    # Data file paths
    COMMENT_FILES: List[str] = [
        './data/comments1.csv',
        './data/comments2.csv',
        './data/comments3.csv',
        './data/comments4.csv',
        './data/comments5.csv'
    ]
    VIDEO_FILE: str = './data/videos.csv'


settings = Settings()

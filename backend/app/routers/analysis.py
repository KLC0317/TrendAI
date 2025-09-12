"""
Analysis endpoints
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.services.trend_service import TrendService
import os
import json

router = APIRouter()
trend_service = TrendService()


@router.get("/analysis/info")
async def get_analysis_info():
    """Get information about the underlying analysis"""
    try:
        metadata_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', 
            'notebook', 
            'enhanced_trend_ai_with_export_20250906_161418', 
            'model_metadata.json'
        )
        
        analysis_info = {
            "model_type": "EnhancedTrendAI",
            "total_comments_analyzed": 3651864,
            "total_videos_analyzed": 92759,
            "topics_identified": 25,
            "analysis_date": "2025-09-05T15:11:22",
            "top_trending_topics": [
                {"rank": 1, "topic": "india, beautiful, pakistan", "growth": 123.5},
                {"rank": 2, "topic": "russian, russia, makeup", "growth": 67.8},
                {"rank": 3, "topic": "people, why, dont", "growth": 65.7},
                {"rank": 4, "topic": "que, linda, para", "growth": 60.8},
                {"rank": 5, "topic": "name, girl, cute", "growth": 37.8}
            ]
        }
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                analysis_info.update(metadata)
        
        return analysis_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analysis info: {str(e)}")


@router.get("/model/status")
async def get_model_status():
    """Get information about loaded model status"""
    try:
        from app.services.model_loader import ModelLoaderService
        model_loader = ModelLoaderService()
        
        loaded_model = model_loader.get_loaded_model()
        date_predictor = model_loader.get_date_predictor()
        
        status = {
            "model_loaded": loaded_model is not None,
            "date_predictor_ready": date_predictor is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if loaded_model:
            # Check what data is available in the loaded model
            available_data = []
            if hasattr(loaded_model, 'comments_df') and loaded_model.comments_df is not None:
                available_data.append("comments_df")
            if hasattr(loaded_model, 'videos_df') and loaded_model.videos_df is not None:
                available_data.append("videos_df")
            if hasattr(loaded_model, 'combined_trend_data') and loaded_model.combined_trend_data is not None:
                available_data.append("combined_trend_data")
            
            status.update({
                "available_data": available_data,
                "model_attributes": [attr for attr in dir(loaded_model) if not attr.startswith('_')]
            })
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model status: {str(e)}")

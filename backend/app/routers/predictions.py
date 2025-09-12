"""
Prediction endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime
from app.models.schemas import DatePredictionRequest, DatePredictionResponse, TopicPrediction
from app.services.prediction_service import PredictionService
from app.utils.date_utils import parse_date

router = APIRouter()
prediction_service = PredictionService()


@router.get("/predict")
async def predict_trends_get(target_date: str, days_ahead: Optional[int] = None):
    """GET endpoint for predictions using trained model"""
    try:
        # Parse and validate the target date
        parsed_date = parse_date(target_date)
        calculated_days_ahead = (parsed_date - datetime.now()).days
        
        # Validate date range
        if abs(calculated_days_ahead) > 365:
            raise HTTPException(
                status_code=400, 
                detail="Target date must be within 365 days of current date"
            )
        
        # Use prediction service
        result = prediction_service.predict_for_date(target_date)
        
        # Convert to the expected format
        predictions = {}
        for topic_name, pred_data in result.get('predictions', {}).items():
            predictions[topic_name] = TopicPrediction(
                cluster_id=pred_data['cluster_id'],
                predicted_score=pred_data['predicted_score'],
                growth_rate=pred_data['growth_rate'],
                trend_slope=pred_data['trend_slope'],
                current_score=pred_data['current_score'],
                topic_words=pred_data['topic_words'],
                topic_tags=pred_data['topic_tags'],
                confidence=pred_data['confidence']
            )
        
        return DatePredictionResponse(
            target_date=target_date,
            days_ahead=calculated_days_ahead,
            predictions=predictions,
            total_topics=len(predictions),
            generated_at=datetime.now().isoformat(),
            method=result.get('method', 'trained_model'),
            warning=result.get('warning')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting trends: {str(e)}")


@router.post("/predict-date", response_model=DatePredictionResponse)
async def predict_trends_for_date(request: DatePredictionRequest):
    """POST endpoint for predictions"""
    try:
        # Parse and validate the target date
        target_date = parse_date(request.target_date)
        days_ahead = (target_date - datetime.now()).days
        
        # Validate date range
        if abs(days_ahead) > 365:
            raise HTTPException(
                status_code=400, 
                detail="Target date must be within 365 days of current date"
            )
        
        # Use prediction service
        result = prediction_service.predict_for_date(request.target_date)
        
        # Convert to the expected format
        predictions = {}
        for topic_name, pred_data in result.get('predictions', {}).items():
            predictions[topic_name] = TopicPrediction(
                cluster_id=pred_data['cluster_id'],
                predicted_score=pred_data['predicted_score'],
                growth_rate=pred_data['growth_rate'],
                trend_slope=pred_data['trend_slope'],
                current_score=pred_data['current_score'],
                topic_words=pred_data['topic_words'],
                topic_tags=pred_data['topic_tags'],
                confidence=pred_data['confidence']
            )
        
        response = DatePredictionResponse(
            target_date=request.target_date,
            days_ahead=days_ahead,
            predictions=predictions,
            method=result.get('method', 'trained_model'),
            total_topics=len(predictions),
            generated_at=datetime.now().isoformat(),
            warning=result.get('warning')
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting trends: {str(e)}")

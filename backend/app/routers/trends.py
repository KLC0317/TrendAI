"""
Trend analysis endpoints
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import TrendResponse
from app.services.trend_service import TrendService

router = APIRouter()
trend_service = TrendService()


@router.get("/trends", response_model=TrendResponse)
async def get_trend_data(days: int = 20):
    """Get trend data for plotting"""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        data = trend_service.get_cached_data(days)
        return TrendResponse(**data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating trend data: {str(e)}")


@router.get("/trends/summary")
async def get_trend_summary():
    """Get summary statistics of trends"""
    try:
        data = trend_service.get_cached_data()
        return {
            "summary": data["summary_stats"],
            "metadata": data["metadata"],
            "categories": ["Emerging", "Established", "Decaying"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trend summary: {str(e)}")


@router.get("/trends/raw")
async def get_raw_trend_data(days: int = 20):
    """Get raw trend data in simple format with actual topic names"""
    try:
        dates, topic_data, trending_topics = trend_service.generate_realistic_trend_data(days)
        
        response = {"dates": dates}
        
        for topic in trending_topics:
            topic_name = topic['topic_word']
            if topic_name in topic_data:
                response[topic['topic_word']] = topic_data[topic_name]
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting raw trend data: {str(e)}")


@router.get("/growth-ranking")
async def get_growth_ranking():
    """Get top 10 growth ranking data for leaderboard"""
    try:
        return trend_service.get_growth_ranking()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting growth ranking: {str(e)}")

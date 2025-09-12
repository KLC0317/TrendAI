"""
Pydantic models for TrendAI API
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Union


class TrendDataPoint(BaseModel):
    date: str
    value: float


class TrendSeries(BaseModel):
    name: str
    color: str
    data: List[float]
    type: str = "line"


class ChartData(BaseModel):
    x_axis: str
    y_axis: str
    dates: List[str]
    series: List[TrendSeries]


class TrendResponse(BaseModel):
    metadata: Dict
    chart_data: ChartData
    summary_stats: Dict


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class TopicPrediction(BaseModel):
    cluster_id: int
    predicted_score: float
    growth_rate: float
    trend_slope: float
    current_score: float
    topic_words: List[str]
    topic_tags: List[str]
    confidence: str


class DatePredictionRequest(BaseModel):
    target_date: str


class DatePredictionResponse(BaseModel):
    target_date: str
    days_ahead: int
    predictions: Dict[str, TopicPrediction]
    total_topics: int
    generated_at: str
    method: str = "trend_extrapolation"
    warning: Optional[str] = None

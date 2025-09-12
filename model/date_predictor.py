"""Date-specific prediction utilities."""

from typing import Union, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np


class DateSpecificPredictor:
    """
    Fixed version that works with properly loaded models
    """
    
    def __init__(self, loaded_model):
        self.predictor = loaded_model
        
    def predict_growth_rate_for_date(self, target_date: Union[str, datetime], 
                                   topic_keywords: List[str] = None) -> Dict:
        """
        Predict growth rate for a specific date using loaded model data
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        days_ahead = (target_date - pd.Timestamp.now()).days
        
        if days_ahead <= 0:
            raise ValueError("Target date must be in the future")
        
        print(f"Predicting for {target_date.strftime('%Y-%m-%d')} ({days_ahead} days ahead)")
        
        # Check if we have trend data
        if not hasattr(self.predictor, 'combined_trend_data') or self.predictor.combined_trend_data is None:
            return self._fallback_prediction(target_date, topic_keywords)
        
        # Use trend data to make predictions
        return self._trend_based_prediction(target_date, days_ahead, topic_keywords)
    
    def _trend_based_prediction(self, target_date, days_ahead, topic_keywords=None):
        """
        Make predictions based on historical trend data
        """
        trend_data = self.predictor.combined_trend_data
        
        results = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'days_ahead': days_ahead,
            'predictions': {},
            'method': 'trend_extrapolation'
        }
        
        # Get unique clusters
        clusters = trend_data['cluster'].unique()
        
        for cluster_id in clusters:
            cluster_data = trend_data[trend_data['cluster'] == cluster_id].copy()
            
            if len(cluster_data) < 3:  # Need minimum data points
                continue
            
            # Sort by date
            cluster_data = cluster_data.sort_values('date')
            
            # Calculate trend
            recent_scores = cluster_data['combined_trending_score'].tail(5).values
            if len(recent_scores) >= 2:
                # Simple linear projection
                trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                current_score = recent_scores[-1]
                predicted_score = current_score + (trend_slope * (days_ahead / 7))  # Weekly projection
                
                # Calculate growth rate
                growth_rate = trend_slope / max(current_score, 0.001)  # Avoid division by zero
                
                # Get topic information
                topic_words = self.predictor.cluster_topics.get(cluster_id, [])
                topic_tags = self.predictor.cluster_tags.get(cluster_id, [])
                
                # Filter by keywords if provided
                if topic_keywords:
                    topic_text = ' '.join(topic_words + topic_tags).lower()
                    if not any(keyword.lower() in topic_text for keyword in topic_keywords):
                        continue
                
                topic_name = ', '.join(topic_words[:3])
                
                results['predictions'][topic_name] = {
                    'cluster_id': cluster_id,
                    'predicted_score': float(predicted_score),
                    'growth_rate': float(growth_rate),
                    'trend_slope': float(trend_slope),
                    'current_score': float(current_score),
                    'topic_words': topic_words[:5],
                    'topic_tags': topic_tags[:5],
                    'confidence': 'medium'
                }
        
        return results
    
    def _fallback_prediction(self, target_date, topic_keywords=None):
        """
        Fallback prediction using cluster performance data
        """
        results = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'predictions': {},
            'method': 'cluster_analysis_fallback',
            'warning': 'Limited data - using cluster-based estimation'
        }
        
        if not hasattr(self.predictor, 'cluster_topics'):
            results['error'] = 'No cluster data available'
            return results
        
        # Use cluster popularity and tag performance as proxy
        for cluster_id, topic_words in self.predictor.cluster_topics.items():
            if topic_keywords:
                topic_text = ' '.join(topic_words).lower()
                if not any(keyword.lower() in topic_text for keyword in topic_keywords):
                    continue
            
            # Estimate growth based on cluster size and tag popularity
            cluster_size = 0
            if hasattr(self.predictor, 'comments_df') and self.predictor.comments_df is not None:
                cluster_size = len(self.predictor.comments_df[self.predictor.comments_df['cluster'] == cluster_id])
            
            # Simple heuristic: larger clusters with popular tags have higher growth potential
            base_growth = min(cluster_size / 1000, 0.1)  # Cap at 10%
            
            topic_name = ', '.join(topic_words[:3])
            results['predictions'][topic_name] = {
                'cluster_id': cluster_id,
                'estimated_growth_rate': float(base_growth),
                'topic_words': topic_words[:5],
                'cluster_size': cluster_size,
                'confidence': 'low'
            }
        
        return results

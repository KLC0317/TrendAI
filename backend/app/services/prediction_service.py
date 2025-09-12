"""
Prediction service
"""
import random
from datetime import datetime
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from app.services.model_loader import ModelLoaderService


class PredictionService:
    """Service for handling prediction operations"""
    
    def __init__(self):
        self.model_loader = ModelLoaderService()
    
    def predict_for_date(self, target_date: Union[str, datetime]) -> Dict:
        """Predict trends for a specific date"""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        days_ahead = (target_date - pd.Timestamp.now()).days
        
        if days_ahead <= 0:
            raise ValueError("Target date must be in the future")
        
        print(f"Predicting for {target_date.strftime('%Y-%m-%d')} ({days_ahead} days ahead)")
        
        # Try to use trained model first
        loaded_model = self.model_loader.get_loaded_model()
        date_predictor = self.model_loader.get_date_predictor()
        
        if date_predictor and loaded_model:
            try:
                return self._trained_model_prediction(target_date, days_ahead, loaded_model)
            except Exception as e:
                print(f"Error using trained model: {str(e)}")
                # Fall back to legacy method
        
        # Fallback to legacy prediction method
        return self._legacy_prediction(target_date, days_ahead)
    
    def _trained_model_prediction(self, target_date, days_ahead, loaded_model):
        """Make predictions using trained model"""
        # Check if we have trend data
        if not hasattr(loaded_model, 'combined_trend_data') or loaded_model.combined_trend_data is None:
            return self._fallback_prediction(target_date, days_ahead)
        
        # Use trend data to make predictions
        return self._trend_based_prediction(target_date, days_ahead, loaded_model)
    
    def _trend_based_prediction(self, target_date, days_ahead, loaded_model):
        """Make predictions based on historical trend data"""
        trend_data = loaded_model.combined_trend_data
        
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
                topic_words = getattr(loaded_model, 'cluster_topics', {}).get(cluster_id, [])
                topic_tags = getattr(loaded_model, 'cluster_tags', {}).get(cluster_id, [])
                
                topic_name = ', '.join(topic_words[:3]) if topic_words else f'cluster_{cluster_id}'
                
                results['predictions'][topic_name] = {
                    'cluster_id': int(cluster_id),
                    'predicted_score': float(predicted_score),
                    'growth_rate': float(growth_rate * 100),  # Convert to percentage
                    'trend_slope': float(trend_slope),
                    'current_score': float(current_score),
                    'topic_words': topic_words[:5] if topic_words else [],
                    'topic_tags': topic_tags[:5] if topic_tags else [],
                    'confidence': 'medium'
                }
        
        return results
    
    def _legacy_prediction(self, target_date, days_ahead):
        """Legacy prediction method using topic clusters"""
        topic_clusters = self._get_topic_clusters_legacy()
        results = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'days_ahead': days_ahead,
            'predictions': {},
            'method': 'legacy_extrapolation',
            'warning': 'Using legacy prediction method'
        }
        
        for cluster_id, cluster_info in topic_clusters.items():
            prediction_result = self._predict_topic_score_legacy(
                cluster_id, 
                days_ahead, 
                cluster_info['current_score']
            )
            
            topic_key = ", ".join(cluster_info['topic_words'])
            
            results['predictions'][topic_key] = {
                'cluster_id': cluster_id,
                'predicted_score': prediction_result['predicted_score'],
                'growth_rate': prediction_result['growth_rate'],
                'trend_slope': prediction_result['trend_slope'],
                'current_score': cluster_info['current_score'],
                'topic_words': cluster_info['topic_words'],
                'topic_tags': cluster_info['topic_tags'],
                'confidence': prediction_result['confidence']
            }
        
        return results
    
    def _fallback_prediction(self, target_date, days_ahead):
        """Fallback prediction using cluster performance data"""
        results = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'days_ahead': days_ahead,
            'predictions': {},
            'method': 'cluster_analysis_fallback',
            'warning': 'Limited data - using cluster-based estimation'
        }
        
        # Use sample data if no model data available
        sample_predictions = {
            'makeup, beauty, beautiful': {
                'cluster_id': 16,
                'predicted_score': 0.3409462411575563,
                'growth_rate': 26.70161583455357,
                'trend_slope': 0.017650500000000003,
                'current_score': 0.06610274115755627,
                'topic_words': ['makeup', 'beauty', 'beautiful', 'without', 'tutorial'],
                'topic_tags': ['makeup', 'makeup tutorial', 'shorts', 'beauty', 'lipstick'],
                'confidence': 'high'
            },
            'beautiful, pretty, gorgeous': {
                'cluster_id': 21,
                'predicted_score': 0.12816357889297203,
                'growth_rate': 9.970431404669185,
                'trend_slope': 0.0050061776527331224,
                'current_score': 0.05021024115755627,
                'topic_words': ['beautiful', 'pretty', 'gorgeous', 'cute', 'wow'],
                'topic_tags': ['beauty', 'fashion', 'style'],
                'confidence': 'medium'
            },
            'hair, hairstyle, wig': {
                'cluster_id': 4,
                'predicted_score': -0.047014411231051864,
                'growth_rate': -301.92741157556244,
                'trend_slope': -0.0030192741157556244,
                'current_score': 0.0,
                'topic_words': ['hair', 'hairstyle', 'wig', 'beautiful', 'bald'],
                'topic_tags': ['hair', 'hair transformation', 'hairstyle', 'haircut', 'shorts'],
                'confidence': 'medium'
            }
        }
        
        results['predictions'] = sample_predictions
        return results
    
    def _get_topic_clusters_legacy(self):
        """Legacy method - Get topic clusters with their associated words and tags"""
        topic_clusters = {
            0: {
                'topic_words': ['makeup', 'beauty', 'cosmetics', 'lipstick', 'foundation'],
                'topic_tags': ['makeup', 'beauty tutorial', 'cosmetics', 'skincare', 'shorts'],
                'current_score': 0.75
            },
            1: {
                'topic_words': ['skincare', 'routine', 'moisturizer', 'serum', 'cleanser'],
                'topic_tags': ['skincare', 'morning routine', 'night routine', 'glowing skin', 'shorts'],
                'current_score': 0.68
            },
            2: {
                'topic_words': ['hairstyle', 'hair', 'tutorial', 'curls', 'straight'],
                'topic_tags': ['hair', 'hair transformation', 'hairstyle', 'haircut', 'shorts'],
                'current_score': 0.82
            },
            3: {
                'topic_words': ['nails', 'manicure', 'polish', 'design', 'art'],
                'topic_tags': ['nails', 'nail art', 'manicure', 'nail design', 'shorts'],
                'current_score': 0.45
            },
            4: {
                'topic_words': ['hair', 'hairstyle', 'wig', 'beautiful', 'bald'],
                'topic_tags': ['hair', 'hair transformation', 'hairstyle', 'haircut', 'shorts'],
                'current_score': 0.0
            },
            10: {
                'topic_words': ['face', 'try', 'why', 'people', 'dont'],
                'topic_tags': ['face', 'beauty', 'skincare', 'tutorial', 'shorts'],
                'current_score': 0.15
            }
        }
        return topic_clusters
    
    def _predict_topic_score_legacy(self, cluster_id: int, days_ahead: int, current_score: float) -> Dict:
        """Legacy prediction method for topic scores"""
        cluster_trends = {
            0: 0.002,    # Makeup - steady growth
            1: 0.001,    # Skincare - slow growth
            2: -0.001,   # Hair - slight decline
            3: 0.003,    # Nails - growing trend
            4: -0.003,   # Hair/wig - declining
            10: -0.002   # Face/try - declining
        }
        
        trend_slope = cluster_trends.get(cluster_id, 0.0)
        noise_factor = random.uniform(0.8, 1.2)
        trend_slope *= noise_factor
        
        weeks_ahead = days_ahead / 7.0
        predicted_score = current_score + (trend_slope * weeks_ahead)
        
        if current_score > 0:
            growth_rate = ((predicted_score - current_score) / current_score) * 100
        else:
            growth_rate = trend_slope * 100 if trend_slope != 0 else 0
        
        if abs(days_ahead) <= 30:
            confidence = "high"
        elif abs(days_ahead) <= 90:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            'predicted_score': predicted_score,
            'growth_rate': growth_rate,
            'trend_slope': trend_slope,
            'confidence': confidence
        }

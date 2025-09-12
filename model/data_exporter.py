import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import calendar
from enhanced_trend_ai import EnhancedTrendAI
class TrendDataExporter:
    """
    Class to export comprehensive trend analysis data to JSON format
    """
    
    def __init__(self, predictor: EnhancedTrendAI):
        self.predictor = predictor
        self.export_timestamp = datetime.now()
        
    def generate_monthly_leaderboards(self, start_date: str = None, months_ahead: int = 12) -> Dict:
        """
        Generate monthly trend leaderboards for forecasted trends
        
        Args:
            start_date (str): Start date in 'YYYY-MM' format (default: current month)
            months_ahead (int): Number of months to forecast
            
        Returns:
            Dict: Monthly leaderboards with top trending topics, tags, and videos
        """
        if start_date is None:
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(start_date + "-01", "%Y-%m-%d")
        
        monthly_leaderboards = {}
        
        # Generate forecasts for extended periods (up to 12 months)
        extended_horizons = [30 * i for i in range(1, months_ahead + 1)]  # 30, 60, 90... days
        multi_horizon_results = self.predictor.generate_multi_horizon_forecasts(extended_horizons)
        
        for month_offset in range(months_ahead):
            target_date = current_date + timedelta(days=30 * month_offset)
            month_key = target_date.strftime("%B_%Y")  # e.g., "July_2024"
            
            horizon_days = 30 * (month_offset + 1)
            
            if horizon_days in multi_horizon_results:
                horizon_data = multi_horizon_results[horizon_days]
                cluster_metrics = horizon_data['metrics']['cluster_metrics']
                
                # Get top trending topics for this month
                trending_topics = []
                for cluster_id, metrics in cluster_metrics.items():
                    if metrics['growth_rate'] > 0:  # Only positive growth
                        trending_topics.append({
                            'cluster_id': cluster_id,
                            'topic_words': metrics.get('topic_words', [])[:3],
                            'growth_rate': metrics['growth_rate'],
                            'prediction_mean': metrics['prediction_mean'],
                            'trend_strength': metrics['trend_strength']
                        })
                
                # Sort by growth rate and get top 10
                trending_topics.sort(key=lambda x: x['growth_rate'], reverse=True)
                top_10_topics = trending_topics[:10]
                
                # Rank the topics
                for rank, topic in enumerate(top_10_topics, 1):
                    topic['ranking'] = rank
                
                # Get top tags for this period
                top_tags = self._get_top_tags_for_period(cluster_metrics, limit=10)
                
                # Get top videos for this period
                top_videos = self._get_top_videos_for_period(cluster_metrics, limit=2)
                
                monthly_leaderboards[month_key] = {
                    'month': target_date.strftime("%B"),
                    'year': target_date.year,
                    'forecast_date': target_date.strftime("%Y-%m-%d"),
                    'days_ahead': horizon_days,
                    'top_trending_topics': [
                        {
                            'ranking': topic['ranking'],
                            'predicted_trend': ', '.join(topic['topic_words']),
                            'growth_rate': round(topic['growth_rate'], 4),
                            'prediction_confidence': round(topic['prediction_mean'], 4),
                            'trend_strength': round(topic['trend_strength'], 4)
                        }
                        for topic in top_10_topics
                    ],
                    'top_tags': top_tags,
                    'top_videos': top_videos,
                    'total_forecasted_clusters': len(cluster_metrics),
                    'positive_growth_clusters': len([m for m in cluster_metrics.values() if m['growth_rate'] > 0])
                }
        
        return monthly_leaderboards
    
    def generate_forecast_graph_data(self, days_range: int = 100, top_topics: int = 10) -> Dict:
        """
        Generate 100-day forecast data for graphing
        
        Args:
            days_range (int): Number of days to forecast (default 100)
            top_topics (int): Number of top topics to track
            
        Returns:
            Dict: Daily forecast data for top trending topics
        """
        # Generate forecasts for each day from 1 to 100
        daily_horizons = list(range(10, days_range + 1, 10))  # 10, 20, 30... 100
        multi_horizon_results = self.predictor.generate_multi_horizon_forecasts(daily_horizons)
        
        # Identify top performing topics across all horizons
        topic_performance = {}
        for horizon, results in multi_horizon_results.items():
            for cluster_id, metrics in results['metrics']['cluster_metrics'].items():
                if cluster_id not in topic_performance:
                    topic_performance[cluster_id] = {
                        'total_growth': 0,
                        'topic_words': metrics.get('topic_words', [])[:3],
                        'appearances': 0
                    }
                topic_performance[cluster_id]['total_growth'] += metrics['growth_rate']
                topic_performance[cluster_id]['appearances'] += 1
        
        # Calculate average growth and get top topics
        for cluster_id in topic_performance:
            avg_growth = topic_performance[cluster_id]['total_growth'] / topic_performance[cluster_id]['appearances']
            topic_performance[cluster_id]['avg_growth'] = avg_growth
        
        top_topic_clusters = sorted(
            topic_performance.items(), 
            key=lambda x: x[1]['avg_growth'], 
            reverse=True
        )[:top_topics]
        
        # Generate daily data points
        forecast_data = {
            'metadata': {
                'generation_date': self.export_timestamp.isoformat(),
                'forecast_range_days': days_range,
                'top_topics_count': len(top_topic_clusters),
                'data_points_per_topic': len(daily_horizons)
            },
            'topics': {},
            'aggregated_daily_data': []
        }
        
        # Generate data for each top topic
        for cluster_id, topic_data in top_topic_clusters:
            topic_name = ', '.join(topic_data['topic_words'])
            
            daily_growth_rates = []
            for horizon in daily_horizons:
                if (horizon in multi_horizon_results and 
                    cluster_id in multi_horizon_results[horizon]['metrics']['cluster_metrics']):
                    growth_rate = multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['growth_rate']
                else:
                    growth_rate = 0
                
                daily_growth_rates.append({
                    'day': horizon,
                    'growth_rate': round(growth_rate, 6),
                    'date': (datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d")
                })
            
            forecast_data['topics'][topic_name] = {
                'cluster_id': cluster_id,
                'topic_keywords': topic_data['topic_words'],
                'average_growth_rate': round(topic_data['avg_growth'], 6),
                'daily_forecasts': daily_growth_rates
            }
        
        # Generate aggregated daily data for overall market trends
        for horizon in daily_horizons:
            if horizon in multi_horizon_results:
                cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
                
                total_growth = sum(m['growth_rate'] for m in cluster_metrics.values())
                avg_growth = total_growth / len(cluster_metrics) if cluster_metrics else 0
                positive_growth_count = sum(1 for m in cluster_metrics.values() if m['growth_rate'] > 0)
                
                forecast_data['aggregated_daily_data'].append({
                    'day': horizon,
                    'date': (datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d"),
                    'average_market_growth': round(avg_growth, 6),
                    'total_growth_sum': round(total_growth, 6),
                    'positive_trends_count': positive_growth_count,
                    'total_clusters_analyzed': len(cluster_metrics)
                })
        
        return forecast_data
    
    def extract_raw_model_data(self) -> Dict:
        """
        Extract comprehensive raw data from model execution for LLM analysis
        
        Returns:
            Dict: Complete raw data from model analysis
        """
        raw_data = {
            'model_metadata': {
                'model_type': 'EnhancedTrendAI',
                'analysis_timestamp': self.export_timestamp.isoformat(),
                'model_version': getattr(self.predictor, 'version', '1.0.0'),
                'weight_factors': {
                    'video_weight': self.predictor.video_weight,
                    'comment_weight': self.predictor.comment_weight,
                    'tag_weight': self.predictor.tag_weight
                }
            },
            'data_summary': {
                'total_comments_analyzed': len(self.predictor.comments_df) if self.predictor.comments_df is not None else 0,
                'total_videos_analyzed': len(self.predictor.videos_df) if self.predictor.videos_df is not None else 0,
                'total_clusters_identified': len(set(self.predictor.clusters)) if self.predictor.clusters is not None else 0,
                'analysis_period': {
                    'start_date': str(self.predictor.comments_df['publishedAt'].min().date()) if self.predictor.comments_df is not None and not self.predictor.comments_df.empty else None,
                    'end_date': str(self.predictor.comments_df['publishedAt'].max().date()) if self.predictor.comments_df is not None and not self.predictor.comments_df.empty else None
                }
            },
            'cluster_analysis': {},
            'sentiment_analysis': {},
            'generational_analysis': {},
            'tag_analysis': {},
            'video_performance_metrics': {},
            'trending_patterns': {}
        }
        
        # Extract cluster data
        if hasattr(self.predictor, 'cluster_topics') and self.predictor.cluster_topics:
            for cluster_id, topic_words in self.predictor.cluster_topics.items():
                cluster_data = {
                    'cluster_id': cluster_id,
                    'topic_words': topic_words,
                    'topic_tags': self.predictor.cluster_tags.get(cluster_id, []),
                    'cluster_size': 0
                }
                
                if self.predictor.comments_df is not None:
                    cluster_comments = self.predictor.comments_df[self.predictor.comments_df['cluster'] == cluster_id]
                    cluster_data.update({
                        'cluster_size': len(cluster_comments),
                        'avg_sentiment': cluster_comments['compound'].mean() if 'compound' in cluster_comments.columns else 0,
                        'sentiment_distribution': {
                            'positive': len(cluster_comments[cluster_comments['compound'] > 0.1]) if 'compound' in cluster_comments.columns else 0,
                            'neutral': len(cluster_comments[(cluster_comments['compound'] >= -0.1) & (cluster_comments['compound'] <= 0.1)]) if 'compound' in cluster_comments.columns else 0,
                            'negative': len(cluster_comments[cluster_comments['compound'] < -0.1]) if 'compound' in cluster_comments.columns else 0
                        }
                    })
                
                raw_data['cluster_analysis'][str(cluster_id)] = cluster_data
        
        # Extract sentiment analysis
        if self.predictor.comments_df is not None and 'compound' in self.predictor.comments_df.columns:
            sentiment_stats = {
                'overall_sentiment_mean': self.predictor.comments_df['compound'].mean(),
                'overall_sentiment_std': self.predictor.comments_df['compound'].std(),
                'sentiment_distribution': {
                    'very_positive': len(self.predictor.comments_df[self.predictor.comments_df['compound'] > 0.5]),
                    'positive': len(self.predictor.comments_df[(self.predictor.comments_df['compound'] > 0.1) & (self.predictor.comments_df['compound'] <= 0.5)]),
                    'neutral': len(self.predictor.comments_df[(self.predictor.comments_df['compound'] >= -0.1) & (self.predictor.comments_df['compound'] <= 0.1)]),
                    'negative': len(self.predictor.comments_df[(self.predictor.comments_df['compound'] >= -0.5) & (self.predictor.comments_df['compound'] < -0.1)]),
                    'very_negative': len(self.predictor.comments_df[self.predictor.comments_df['compound'] < -0.5])
                }
            }
            raw_data['sentiment_analysis'] = sentiment_stats
        
        # Extract generational analysis
        if hasattr(self.predictor, 'generational_clusters') and self.predictor.generational_clusters:
            generational_data = {}
            generation_distribution = {}
            
            for cluster_id, gen_data in self.predictor.generational_clusters.items():
                generational_data[str(cluster_id)] = {
                    'dominant_generation': gen_data['dominant_generation'],
                    'confidence_score': gen_data['dominant_generation_score'],
                    'generation_distribution': gen_data['generation_distribution'],
                    'avg_generational_scores': gen_data['avg_generational_scores']
                }
                
                # Aggregate generation distribution
                dominant_gen = gen_data['dominant_generation']
                if dominant_gen not in generation_distribution:
                    generation_distribution[dominant_gen] = 0
                generation_distribution[dominant_gen] += 1
            
            raw_data['generational_analysis'] = {
                'cluster_level_analysis': generational_data,
                'overall_generation_distribution': generation_distribution
            }
        
        # Extract tag analysis
        if hasattr(self.predictor, 'popular_tags') and self.predictor.popular_tags is not None:
            tag_data = {}
            for tag, data in self.predictor.popular_tags.head(50).iterrows():  # Top 50 tags
                tag_data[tag] = {
                    'frequency': int(data['frequency']),
                    'total_views': int(data['total_views']),
                    'total_likes': int(data['total_likes']),
                    'popularity_score': float(data['tag_popularity_score']),
                    'avg_views_per_video': float(data['avg_views_per_video']),
                    'avg_likes_per_video': float(data['avg_likes_per_video'])
                }
            
            raw_data['tag_analysis'] = {
                'top_tags': tag_data,
                'total_unique_tags': len(self.predictor.popular_tags)
            }
        
        # Extract video performance metrics
        if self.predictor.videos_df is not None:
            video_stats = {
                'total_views': int(self.predictor.videos_df['viewCount'].sum()),
                'total_likes': int(self.predictor.videos_df['likeCount'].sum()),
                'total_comments': int(self.predictor.videos_df['commentCount'].sum()),
                'avg_engagement_rate': float(self.predictor.videos_df['engagement_rate'].mean()),
                'avg_trending_score': float(self.predictor.videos_df['trending_score'].mean()),
                'top_performing_videos': []
            }
            
            # Get top 10 performing videos
            top_videos = self.predictor.videos_df.nlargest(10, 'trending_score')
            for _, video in top_videos.iterrows():
                video_stats['top_performing_videos'].append({
                    'title': video['title'],
                    'views': int(video['viewCount']),
                    'likes': int(video['likeCount']),
                    'comments': int(video['commentCount']),
                    'trending_score': float(video['trending_score']),
                    'engagement_rate': float(video['engagement_rate']),
                    'published_date': str(video['publishedAt'].date()) if pd.notna(video['publishedAt']) else None
                })
            
            raw_data['video_performance_metrics'] = video_stats
        
        # Extract trending patterns
        if hasattr(self.predictor, 'combined_trend_data') and not self.predictor.combined_trend_data.empty:
            trending_patterns = {
                'temporal_trends': [],
                'cluster_performance_over_time': {}
            }
            
            # Aggregate temporal trends
            temporal_agg = self.predictor.combined_trend_data.groupby('date').agg({
                'combined_trending_score': 'mean',
                'total_views': 'sum',
                'video_likes': 'sum',
                'comment_count': 'sum'
            }).reset_index()
            
            for _, row in temporal_agg.iterrows():
                trending_patterns['temporal_trends'].append({
                    'date': str(row['date'].date()),
                    'avg_trending_score': float(row['combined_trending_score']),
                    'total_daily_views': int(row['total_views']),
                    'total_daily_likes': int(row['video_likes']),
                    'total_daily_comments': int(row['comment_count'])
                })
            
            raw_data['trending_patterns'] = trending_patterns
        
        return raw_data
    
    def _get_top_tags_for_period(self, cluster_metrics: Dict, limit: int = 10) -> List[Dict]:
        """Extract top tags for a specific forecast period"""
        tag_performance = {}
        
        for cluster_id, metrics in cluster_metrics.items():
            if hasattr(self.predictor, 'cluster_tags') and cluster_id in self.predictor.cluster_tags:
                cluster_tags = self.predictor.cluster_tags[cluster_id]
                growth_rate = metrics['growth_rate']
                
                for tag in cluster_tags:
                    if tag not in tag_performance:
                        tag_performance[tag] = {'total_growth': 0, 'cluster_count': 0}
                    tag_performance[tag]['total_growth'] += growth_rate
                    tag_performance[tag]['cluster_count'] += 1
        
        # Calculate average growth for each tag
        for tag in tag_performance:
            avg_growth = tag_performance[tag]['total_growth'] / tag_performance[tag]['cluster_count']
            tag_performance[tag]['avg_growth'] = avg_growth
        
        # Sort and get top tags
        top_tags = sorted(tag_performance.items(), key=lambda x: x[1]['avg_growth'], reverse=True)[:limit]
        
        return [
            {
                'ranking': i + 1,
                'tag': tag,
                'growth_rate': round(data['avg_growth'], 4),
                'cluster_count': data['cluster_count']
            }
            for i, (tag, data) in enumerate(top_tags)
        ]
    
    def _get_top_videos_for_period(self, cluster_metrics: Dict, limit: int = 2) -> List[Dict]:
        """Extract top videos for a specific forecast period"""
        if self.predictor.videos_df is None:
            return []
        
        # Get clusters with highest growth rates
        top_growth_clusters = sorted(
            cluster_metrics.items(), 
            key=lambda x: x[1]['growth_rate'], 
            reverse=True
        )[:5]  # Look at top 5 growing clusters
        
        top_videos = []
        for cluster_id, metrics in top_growth_clusters:
            if 'cluster' in self.predictor.videos_df.columns:
                cluster_videos = self.predictor.videos_df[
                    self.predictor.videos_df['cluster'] == cluster_id
                ].nlargest(1, 'trending_score')  # Get top video from this cluster
                
                for _, video in cluster_videos.iterrows():
                    top_videos.append({
                        'title': video['title'],
                        'views': int(video['viewCount']),
                        'likes': int(video['likeCount']),
                        'trending_score': float(video['trending_score']),
                        'predicted_growth_rate': round(metrics['growth_rate'], 4),
                        'cluster_topics': ', '.join(metrics.get('topic_words', [])[:3])
                    })
                    
                    if len(top_videos) >= limit:
                        break
            
            if len(top_videos) >= limit:
                break
        
        # Add ranking
        for i, video in enumerate(top_videos[:limit], 1):
            video['ranking'] = i
        
        return top_videos[:limit]
    
    def export_complete_analysis(self, output_file: str = None, months_ahead: int = 12) -> Dict:
        """
        Export complete trend analysis to JSON
        
        Args:
            output_file (str): Output JSON file path
            months_ahead (int): Number of months to forecast for leaderboards
            
        Returns:
            Dict: Complete analysis data
        """
        print("Generating comprehensive trend analysis export...")
        
        complete_data = {
            'export_info': {
                'generation_timestamp': self.export_timestamp.isoformat(),
                'export_version': '1.0.0',
                'data_types_included': [
                    'monthly_leaderboards',
                    'forecast_graph_data', 
                    'raw_model_data'
                ]
            }
        }
        
        try:
            # Generate monthly leaderboards
            print("Generating monthly trend leaderboards...")
            complete_data['monthly_leaderboards'] = self.generate_monthly_leaderboards(
                months_ahead=months_ahead
            )
            
            # Generate forecast graph data
            print("Generating 100-day forecast graph data...")
            complete_data['forecast_graph_data'] = self.generate_forecast_graph_data()
            
            # Extract raw model data
            print("Extracting comprehensive raw model data...")
            complete_data['raw_model_data'] = self.extract_raw_model_data()
            
            # Add summary statistics
            complete_data['summary_statistics'] = {
                'total_months_forecasted': len(complete_data['monthly_leaderboards']),
                'total_topics_tracked': len(complete_data['forecast_graph_data']['topics']),
                'total_clusters_analyzed': complete_data['raw_model_data']['data_summary']['total_clusters_identified'],
                'forecast_data_points': len(complete_data['forecast_graph_data']['aggregated_daily_data']),
                'top_growth_rate': max([
                    topic_data['average_growth_rate'] 
                    for topic_data in complete_data['forecast_graph_data']['topics'].values()
                ], default=0)
            }
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(complete_data, f, indent=2, ensure_ascii=False, default=str)
                print(f"Complete analysis exported to: {output_file}")
            
            print("Export completed successfully!")
            return complete_data
            
        except Exception as e:
            print(f"Error during export: {e}")
            raise

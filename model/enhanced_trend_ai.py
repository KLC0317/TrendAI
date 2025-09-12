"""Enhanced TrendAI with ensemble forecasting capabilities."""

from trend_ai import TrendAI
from ensemble_forecaster import EnsembleForecaster
from generational_analyzer import EnhancedGenerationalLanguageAnalyzer
from config import *
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class EnhancedTrendAI(TrendAI):
    """
    Enhanced YouTube Trend Predictor with Ensemble Forecasting capabilities.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.ensemble_forecaster = EnsembleForecaster()
        
    def train_ensemble_forecasting_models(self, forecast_days: int = 30) -> None:
        """
        Train the ensemble forecasting models.
        
        Args:
            forecast_days (int): Number of days to forecast
        """
        print("Training ensemble forecasting models...")
        
        if self.combined_trend_data is None or self.combined_trend_data.empty:
            print("No trend data available. Run prepare_time_series_data first.")
            return
        
        # Train ensemble models
        self.ensemble_forecaster.train_ensemble_models(
            self.combined_trend_data,
            target_col='combined_trending_score'
        )
        
        # Calculate ensemble weights using recent data as validation
        validation_cutoff = self.combined_trend_data['date'].max() - pd.Timedelta(days=14)
        validation_data = self.combined_trend_data[
            self.combined_trend_data['date'] >= validation_cutoff
        ]
        
        if not validation_data.empty:
            self.ensemble_forecaster.calculate_ensemble_weights(
                validation_data, 'combined_trending_score'
            )
    
    def generate_ensemble_forecasts(self, forecast_days: int = 30) -> Dict:
        """
        Generate ensemble forecasts for future trends.
        
        Args:
            forecast_days (int): Number of days to forecast
            
        Returns:
            Dict: Forecast results with confidence intervals
        """
        print(f"Generating ensemble forecasts for {forecast_days} days...")
        
        if self.combined_trend_data is None or self.combined_trend_data.empty:
            print("No trend data available for forecasting")
            return {}
        
        # Generate ensemble predictions
        ensemble_predictions = self.ensemble_forecaster.ensemble_forecast(
            self.combined_trend_data,
            forecast_steps=forecast_days
        )
        
        # Create forecast results with metadata
        last_date = pd.to_datetime(self.combined_trend_data['date'].max())
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast_results = {}
        for cluster_id, predictions in ensemble_predictions.items():
            topic_words = self.cluster_topics.get(cluster_id, [])[:5]
            topic_tags = self.cluster_tags.get(cluster_id, [])[:3]
            
            forecast_results[cluster_id] = {
                'topic_words': topic_words,
                'topic_tags': topic_tags,
                'dates': future_dates.tolist(),
                'predictions': predictions.tolist(),
                'prediction_mean': float(np.mean(predictions)),
                'prediction_trend': 'increasing' if predictions[-1] > predictions[0] else 'decreasing',
                'confidence_level': 'medium',  # Could implement actual confidence intervals
                'forecast_category': self._categorize_forecast(predictions)
            }
        
        return forecast_results
    
    def _categorize_forecast(self, predictions: np.ndarray) -> str:
        """
        Categorize forecast based on prediction patterns.
        
        Args:
            predictions (np.ndarray): Forecast predictions
            
        Returns:
            str: Forecast category
        """
        if len(predictions) < 2:
            return 'stable'
        
        trend_slope = (predictions[-1] - predictions[0]) / len(predictions)
        max_val = np.max(predictions)
        min_val = np.min(predictions)
        volatility = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0
        
        if trend_slope > 0.05 and max_val > predictions[0] * 1.2:
            return 'rapid_growth'
        elif trend_slope > 0.01:
            return 'steady_growth'
        elif trend_slope < -0.05:
            return 'declining'
        elif volatility > 0.3:
            return 'volatile'
        else:
            return 'stable'
    def generate_multi_horizon_forecasts(self, forecast_horizons: List[int] = [20, 30, 40, 60, 80]) -> Dict:
        """
        Generate forecasts for multiple time horizons.
        
        Args:
            forecast_horizons (List[int]): List of forecast horizons in days
            
        Returns:
            Dict: Forecasts organized by horizon and cluster
        """
        print(f"Generating multi-horizon forecasts for {len(forecast_horizons)} horizons...")
        
        multi_horizon_results = {}
        
        for horizon in forecast_horizons:
            print(f"Generating {horizon}-day forecasts...")
            
            # Generate ensemble forecasts for this horizon
            forecast_results = self.generate_ensemble_forecasts(forecast_days=horizon)
            
            # Calculate additional metrics for this horizon
            horizon_metrics = self._calculate_horizon_metrics(forecast_results, horizon)
            
            multi_horizon_results[horizon] = {
                'forecasts': forecast_results,
                'metrics': horizon_metrics,
                'horizon_days': horizon
            }
        
        return multi_horizon_results

    def identify_generational_trending_topics(self, window_days: int = 30) -> Dict:
        """
        Identify trending topics by generation.
        
        Args:
            window_days (int): Number of recent days to analyze
            
        Returns:
            Dict: Trending topics organized by generation
        """
        if not hasattr(self, 'generational_clusters') or self.generational_clusters is None:
            self.analyze_generational_trends_by_cluster()
        
        recent_date = self.comments_df['publishedAt'].max() - timedelta(days=window_days)
        recent_comments = self.comments_df[self.comments_df['publishedAt'] >= recent_date]
        
        generational_trends = {
            'gen_z': [],
            'millennial': [],
            'gen_x': [],
            'boomer': [],
            'neutral': []
        }
        
        for cluster_id, analysis in self.generational_clusters.items():
            dominant_gen = analysis['dominant_generation']
            
            # Calculate trend metrics for this cluster
            cluster_recent_comments = recent_comments[recent_comments['cluster'] == cluster_id]
            
            if len(cluster_recent_comments) > 0:
                # Calculate engagement and growth metrics
                avg_sentiment = cluster_recent_comments['compound'].mean() if 'compound' in cluster_recent_comments.columns else 0
                comment_volume = len(cluster_recent_comments)
                avg_likes = cluster_recent_comments['likeCount'].mean()
                
                # Get video performance for this generation
                video_perf = analysis['video_performance_by_generation'].get(dominant_gen, {})
                
                trend_data = {
                    'cluster_id': cluster_id,
                    'topic_words': analysis['topic_words'][:5],
                    'topic_tags': analysis['topic_tags'][:3],
                    'dominant_generation': dominant_gen,
                    'generation_confidence': analysis['dominant_generation_score'],
                    'recent_comment_volume': comment_volume,
                    'avg_sentiment': avg_sentiment,
                    'avg_comment_likes': avg_likes,
                    'generation_distribution': analysis['generation_distribution'],
                    'video_performance': video_perf
                }
                
                generational_trends[dominant_gen].append(trend_data)
        
        # Sort each generation's trends by relevance
        for generation in generational_trends:
            generational_trends[generation].sort(
                key=lambda x: (x['recent_comment_volume'] * (1 + x['avg_sentiment'])), 
                reverse=True
            )
        
        self.generational_trends = generational_trends
        return generational_trends

    def _calculate_horizon_metrics(self, forecast_results: Dict, horizon: int) -> Dict:
        """
        Calculate trend-related metrics for a specific forecast horizon.
        
        Args:
            forecast_results (Dict): Forecast results for clusters
            horizon (int): Forecast horizon in days
            
        Returns:
            Dict: Calculated metrics
        """
        metrics = {
            'total_clusters': len(forecast_results),
            'avg_prediction_mean': 0.0,
            'total_predicted_growth': 0.0,
            'volatility_score': 0.0,
            'trend_strength': 0.0,
            'category_distribution': {},
            'cluster_metrics': {}
        }
        
        if not forecast_results:
            return metrics
        
        # Calculate aggregate metrics
        prediction_means = []
        growth_rates = []
        volatilities = []
        categories = []
        
        for cluster_id, results in forecast_results.items():
            predictions = np.array(results['predictions'])
            
            # Basic metrics
            pred_mean = np.mean(predictions)
            prediction_means.append(pred_mean)
            
            # Growth rate (start to end)
            if len(predictions) > 1:
                growth_rate = (predictions[-1] - predictions[0]) / predictions[0] if predictions[0] > 0 else 0
                growth_rates.append(growth_rate)
            
            # Volatility
            volatility = np.std(predictions) / pred_mean if pred_mean > 0 else 0
            volatilities.append(volatility)
            
            # Category
            category = results['forecast_category']
            categories.append(category)
            
            # Trend strength (correlation with linear trend)
            x = np.arange(len(predictions))
            if len(predictions) > 2:
                correlation = np.corrcoef(x, predictions)[0, 1] if np.std(predictions) > 0 else 0
                trend_strength = abs(correlation)
            else:
                trend_strength = 0
            
            # Store individual cluster metrics
            metrics['cluster_metrics'][cluster_id] = {
                'prediction_mean': pred_mean,
                'growth_rate': growth_rate if len(predictions) > 1 else 0,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'max_value': np.max(predictions),
                'min_value': np.min(predictions),
                'final_value': predictions[-1] if len(predictions) > 0 else 0,
                'topic_words': results.get('topic_words', [])
            }
        
        # Aggregate metrics
        metrics['avg_prediction_mean'] = np.mean(prediction_means) if prediction_means else 0
        metrics['total_predicted_growth'] = np.sum(growth_rates) if growth_rates else 0
        metrics['volatility_score'] = np.mean(volatilities) if volatilities else 0
        metrics['trend_strength'] = np.mean([m['trend_strength'] for m in metrics['cluster_metrics'].values()])
        
        # Category distribution
        category_counts = pd.Series(categories).value_counts().to_dict()
        metrics['category_distribution'] = category_counts
        
        return metrics

    
    def visualize_ensemble_forecasts(self, forecast_results: Dict, top_n: int = 5) -> None:
        """
        Visualize ensemble forecast results.
        
        Args:
            forecast_results (Dict): Forecast results from generate_ensemble_forecasts
            top_n (int): Number of top clusters to visualize
        """
        print("Creating ensemble forecast visualizations...")
        
        if not forecast_results:
            print("No forecast results to visualize")
            return
        
        # Sort clusters by prediction mean
        sorted_clusters = sorted(
            forecast_results.items(),
            key=lambda x: x[1]['prediction_mean'],
            reverse=True
        )[:top_n]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Forecasting Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Individual Forecast Lines
        ax1 = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_clusters)))
        
        for i, (cluster_id, results) in enumerate(sorted_clusters):
            topic_label = ', '.join(results['topic_words'][:2])
            dates = pd.to_datetime(results['dates'])
            predictions = results['predictions']
            
            ax1.plot(dates, predictions, color=colors[i], linewidth=2, 
                    marker='o', label=f"{topic_label} (C{cluster_id})")
        
        ax1.set_title('Individual Cluster Forecasts', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Predicted Trending Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Forecast Categories
        ax2 = axes[0, 1]
        categories = [results['forecast_category'] for _, results in sorted_clusters]
        category_counts = pd.Series(categories).value_counts()
        
        ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        ax2.set_title('Forecast Categories Distribution', fontweight='bold')
        
        # Plot 3: Prediction Means Comparison
        ax3 = axes[1, 0]
        cluster_names = [', '.join(results['topic_words'][:2]) for _, results in sorted_clusters]
        prediction_means = [results['prediction_mean'] for _, results in sorted_clusters]
        
        bars = ax3.bar(range(len(cluster_names)), prediction_means, color='skyblue')
        ax3.set_xticks(range(len(cluster_names)))
        ax3.set_xticklabels(cluster_names, rotation=45, ha='right')
        ax3.set_title('Average Predicted Trending Scores', fontweight='bold')
        ax3.set_ylabel('Average Trending Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, prediction_means):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Trend Directions
        ax4 = axes[1, 1]
        trend_directions = [results['prediction_trend'] for _, results in sorted_clusters]
        trend_counts = pd.Series(trend_directions).value_counts()
        
        colors_trend = ['green' if trend == 'increasing' else 'red' for trend in trend_counts.index]
        ax4.bar(trend_counts.index, trend_counts.values, color=colors_trend)
        ax4.set_title('Forecast Trend Directions', fontweight='bold')
        ax4.set_ylabel('Number of Clusters')
        
        plt.tight_layout()
        plt.show()

    def visualize_multi_horizon_analysis(self, multi_horizon_results: Dict, top_clusters: int = 8) -> None:
        """
        Create comprehensive visualizations for multi-horizon forecasts.
        
        Args:
            multi_horizon_results (Dict): Results from generate_multi_horizon_forecasts
            top_clusters (int): Number of top clusters to highlight
        """
        if not multi_horizon_results:
            print("No multi-horizon results to visualize")
            return
        
        horizons = sorted(multi_horizon_results.keys())
        
        # Create multiple visualization sets
        self._plot_horizon_trend_metrics(multi_horizon_results, horizons)
        self._plot_cluster_performance_across_horizons(multi_horizon_results, horizons, top_clusters)
        self._plot_forecast_uncertainty_analysis(multi_horizon_results, horizons)
        self._plot_category_evolution_across_horizons(multi_horizon_results, horizons)

        self.plot_cluster_growth_by_horizon(multi_horizon_results, horizons)
        self.plot_cluster_comparison_across_horizons(multi_horizon_results, top_n=top_clusters)
    
    def _plot_horizon_trend_metrics(self, multi_horizon_results: Dict, horizons: List[int]) -> None:
        """Plot trend metrics across different forecast horizons."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trend Metrics Across Forecast Horizons', fontsize=16, fontweight='bold')
        
        # Extract metrics for each horizon
        metrics_data = {
            'horizons': horizons,
            'avg_prediction_mean': [multi_horizon_results[h]['metrics']['avg_prediction_mean'] for h in horizons],
            'total_predicted_growth': [multi_horizon_results[h]['metrics']['total_predicted_growth'] for h in horizons],
            'volatility_score': [multi_horizon_results[h]['metrics']['volatility_score'] for h in horizons],
            'trend_strength': [multi_horizon_results[h]['metrics']['trend_strength'] for h in horizons],
            'total_clusters': [multi_horizon_results[h]['metrics']['total_clusters'] for h in horizons]
        }
        
        # Plot 1: Average Prediction Mean
        axes[0, 0].plot(horizons, metrics_data['avg_prediction_mean'], marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Average Prediction Mean vs Horizon', fontweight='bold')
        axes[0, 0].set_xlabel('Forecast Horizon (days)')
        axes[0, 0].set_ylabel('Average Prediction Mean')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total Predicted Growth
        axes[0, 1].bar(horizons, metrics_data['total_predicted_growth'], color='green', alpha=0.7)
        axes[0, 1].set_title('Total Predicted Growth vs Horizon', fontweight='bold')
        axes[0, 1].set_xlabel('Forecast Horizon (days)')
        axes[0, 1].set_ylabel('Total Predicted Growth')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility Score
        axes[0, 2].plot(horizons, metrics_data['volatility_score'], marker='s', color='orange', linewidth=2, markersize=8)
        axes[0, 2].set_title('Forecast Volatility vs Horizon', fontweight='bold')
        axes[0, 2].set_xlabel('Forecast Horizon (days)')
        axes[0, 2].set_ylabel('Volatility Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Trend Strength
        axes[1, 0].plot(horizons, metrics_data['trend_strength'], marker='^', color='red', linewidth=2, markersize=8)
        axes[1, 0].set_title('Trend Strength vs Horizon', fontweight='bold')
        axes[1, 0].set_xlabel('Forecast Horizon (days)')
        axes[1, 0].set_ylabel('Trend Strength')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Number of Clusters Forecasted
        axes[1, 1].bar(horizons, metrics_data['total_clusters'], color='purple', alpha=0.7)
        axes[1, 1].set_title('Clusters Forecasted vs Horizon', fontweight='bold')
        axes[1, 1].set_xlabel('Forecast Horizon (days)')
        axes[1, 1].set_ylabel('Number of Clusters')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Horizon Comparison Radar
        axes[1, 2].remove()  # Remove this axis for a custom radar chart
        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
        
        # Normalize metrics for radar chart
        normalized_metrics = []
        metric_names = ['Avg Prediction', 'Growth Rate', 'Volatility', 'Trend Strength']
        
        for i, horizon in enumerate([horizons[0], horizons[-1]]):  # Compare first and last horizon
            norm_pred = metrics_data['avg_prediction_mean'][horizons.index(horizon)] / max(metrics_data['avg_prediction_mean']) if max(metrics_data['avg_prediction_mean']) > 0 else 0
            norm_growth = abs(metrics_data['total_predicted_growth'][horizons.index(horizon)]) / max([abs(x) for x in metrics_data['total_predicted_growth']]) if max([abs(x) for x in metrics_data['total_predicted_growth']]) > 0 else 0
            norm_vol = metrics_data['volatility_score'][horizons.index(horizon)] / max(metrics_data['volatility_score']) if max(metrics_data['volatility_score']) > 0 else 0
            norm_trend = metrics_data['trend_strength'][horizons.index(horizon)]
            
            normalized_metrics.append([norm_pred, norm_growth, norm_vol, norm_trend])
        
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['blue', 'red']
        labels = [f'{horizons[0]} days', f'{horizons[-1]} days']
        
        for i, (metrics, color, label) in enumerate(zip(normalized_metrics, colors, labels)):
            metrics += metrics[:1]  # Complete the circle
            ax_radar.plot(angles, metrics, 'o-', linewidth=2, label=label, color=color)
            ax_radar.fill(angles, metrics, alpha=0.25, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metric_names)
        ax_radar.set_title('Horizon Comparison (Normalized)', fontweight='bold', pad=20)
        ax_radar.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_cluster_performance_across_horizons(self, multi_horizon_results: Dict, horizons: List[int], top_clusters: int) -> None:
        """Plot performance of top clusters across different horizons."""
        
        # Identify top clusters based on average performance across all horizons
        cluster_avg_performance = {}
        all_clusters = set()
        
        for horizon in horizons:
            for cluster_id, metrics in multi_horizon_results[horizon]['metrics']['cluster_metrics'].items():
                all_clusters.add(cluster_id)
                if cluster_id not in cluster_avg_performance:
                    cluster_avg_performance[cluster_id] = []
                cluster_avg_performance[cluster_id].append(metrics['prediction_mean'])
        
        # Calculate average performance and get top clusters
        for cluster_id in cluster_avg_performance:
            cluster_avg_performance[cluster_id] = np.mean(cluster_avg_performance[cluster_id])
        
        top_cluster_ids = sorted(cluster_avg_performance.items(), key=lambda x: x[1], reverse=True)[:top_clusters]
        top_cluster_ids = [cid for cid, _ in top_cluster_ids]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Top {top_clusters} Cluster Performance Across Horizons', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_cluster_ids)))
        
        # Plot 1: Prediction Mean Evolution
        ax1 = axes[0, 0]
        for i, cluster_id in enumerate(top_cluster_ids):
            means = []
            for horizon in horizons:
                if cluster_id in multi_horizon_results[horizon]['metrics']['cluster_metrics']:
                    means.append(multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['prediction_mean'])
                else:
                    means.append(0)
            
            # Get topic words for legend
            topic_words = []
            for horizon in horizons:
                if cluster_id in multi_horizon_results[horizon]['metrics']['cluster_metrics']:
                    topic_words = multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['topic_words'][:2]
                    break
            
            label = f"C{cluster_id}: {', '.join(topic_words)}" if topic_words else f"Cluster {cluster_id}"
            ax1.plot(horizons, means, marker='o', color=colors[i], linewidth=2, label=label)
        
        ax1.set_title('Prediction Mean vs Horizon', fontweight='bold')
        ax1.set_xlabel('Forecast Horizon (days)')
        ax1.set_ylabel('Prediction Mean')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Growth Rate Evolution
        ax2 = axes[0, 1]
        for i, cluster_id in enumerate(top_cluster_ids):
            growth_rates = []
            for horizon in horizons:
                if cluster_id in multi_horizon_results[horizon]['metrics']['cluster_metrics']:
                    growth_rates.append(multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['growth_rate'])
                else:
                    growth_rates.append(0)
            
            ax2.plot(horizons, growth_rates, marker='s', color=colors[i], linewidth=2)
        
        ax2.set_title('Growth Rate vs Horizon', fontweight='bold')
        ax2.set_xlabel('Forecast Horizon (days)')
        ax2.set_ylabel('Growth Rate')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Volatility Evolution
        ax3 = axes[1, 0]
        for i, cluster_id in enumerate(top_cluster_ids):
            volatilities = []
            for horizon in horizons:
                if cluster_id in multi_horizon_results[horizon]['metrics']['cluster_metrics']:
                    volatilities.append(multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['volatility'])
                else:
                    volatilities.append(0)
            
            ax3.plot(horizons, volatilities, marker='^', color=colors[i], linewidth=2)
        
        ax3.set_title('Volatility vs Horizon', fontweight='bold')
        ax3.set_xlabel('Forecast Horizon (days)')
        ax3.set_ylabel('Volatility Score')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trend Strength Evolution
        ax4 = axes[1, 1]
        for i, cluster_id in enumerate(top_cluster_ids):
            trend_strengths = []
            for horizon in horizons:
                if cluster_id in multi_horizon_results[horizon]['metrics']['cluster_metrics']:
                    trend_strengths.append(multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['trend_strength'])
                else:
                    trend_strengths.append(0)
            
            ax4.plot(horizons, trend_strengths, marker='d', color=colors[i], linewidth=2)
        
        ax4.set_title('Trend Strength vs Horizon', fontweight='bold')
        ax4.set_xlabel('Forecast Horizon (days)')
        ax4.set_ylabel('Trend Strength')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_forecast_uncertainty_analysis(self, multi_horizon_results: Dict, horizons: List[int]) -> None:
        """Analyze and plot forecast uncertainty across horizons."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Forecast Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # Collect uncertainty metrics
        uncertainty_data = {
            'horizon': [],
            'prediction_std': [],
            'growth_rate_std': [],
            'max_prediction_range': [],
            'coefficient_of_variation': []
        }
        
        for horizon in horizons:
            cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
            
            if not cluster_metrics:
                continue
            
            prediction_means = [m['prediction_mean'] for m in cluster_metrics.values()]
            growth_rates = [m['growth_rate'] for m in cluster_metrics.values()]
            max_values = [m['max_value'] for m in cluster_metrics.values()]
            min_values = [m['min_value'] for m in cluster_metrics.values()]
            
            uncertainty_data['horizon'].append(horizon)
            uncertainty_data['prediction_std'].append(np.std(prediction_means))
            uncertainty_data['growth_rate_std'].append(np.std(growth_rates))
            uncertainty_data['max_prediction_range'].append(np.mean([max_val - min_val for max_val, min_val in zip(max_values, min_values)]))
            
            # Coefficient of variation
            mean_pred = np.mean(prediction_means)
            cv = np.std(prediction_means) / mean_pred if mean_pred > 0 else 0
            uncertainty_data['coefficient_of_variation'].append(cv)
        
        # Plot uncertainty metrics
        axes[0, 0].plot(uncertainty_data['horizon'], uncertainty_data['prediction_std'], 
                        marker='o', color='red', linewidth=2, markersize=8)
        axes[0, 0].set_title('Prediction Standard Deviation', fontweight='bold')
        axes[0, 0].set_xlabel('Forecast Horizon (days)')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(uncertainty_data['horizon'], uncertainty_data['growth_rate_std'], 
                        marker='s', color='orange', linewidth=2, markersize=8)
        axes[0, 1].set_title('Growth Rate Standard Deviation', fontweight='bold')
        axes[0, 1].set_xlabel('Forecast Horizon (days)')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(uncertainty_data['horizon'], uncertainty_data['max_prediction_range'], 
                        marker='^', color='green', linewidth=2, markersize=8)
        axes[1, 0].set_title('Average Prediction Range', fontweight='bold')
        axes[1, 0].set_xlabel('Forecast Horizon (days)')
        axes[1, 0].set_ylabel('Average Range (Max - Min)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(uncertainty_data['horizon'], uncertainty_data['coefficient_of_variation'], 
                        marker='d', color='purple', linewidth=2, markersize=8)
        axes[1, 1].set_title('Coefficient of Variation', fontweight='bold')
        axes[1, 1].set_xlabel('Forecast Horizon (days)')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_category_evolution_across_horizons(self, multi_horizon_results: Dict, horizons: List[int]) -> None:
        """Plot how forecast categories evolve across different horizons."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Forecast Category Evolution Across Horizons', fontsize=16, fontweight='bold')
        
        # Collect category data
        all_categories = set()
        category_data = {}
        
        for horizon in horizons:
            category_dist = multi_horizon_results[horizon]['metrics']['category_distribution']
            category_data[horizon] = category_dist
            all_categories.update(category_dist.keys())
        
        all_categories = sorted(list(all_categories))
        
        # Plot 1: Stacked Bar Chart
        ax1 = axes[0]
        bottom_values = np.zeros(len(horizons))
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_categories)))
        
        for i, category in enumerate(all_categories):
            values = []
            for horizon in horizons:
                values.append(category_data[horizon].get(category, 0))
            
            ax1.bar(horizons, values, bottom=bottom_values, label=category, color=colors[i])
            bottom_values += values
        
        ax1.set_title('Category Distribution Across Horizons', fontweight='bold')
        ax1.set_xlabel('Forecast Horizon (days)')
        ax1.set_ylabel('Number of Clusters')
        ax1.legend()
        
        # Plot 2: Line Chart showing category trends
        ax2 = axes[1]
        for i, category in enumerate(all_categories):
            values = []
            for horizon in horizons:
                values.append(category_data[horizon].get(category, 0))
            
            ax2.plot(horizons, values, marker='o', linewidth=2, label=category, color=colors[i])
        
        ax2.set_title('Category Trends Across Horizons', fontweight='bold')
        ax2.set_xlabel('Forecast Horizon (days)')
        ax2.set_ylabel('Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _plot_cluster_trend_vs_growth_scatter(self, multi_horizon_results: Dict, horizons: List[int]) -> None:
        """Plot cluster trend strength vs growth rate scatter plots for each horizon."""
        
        # Determine subplot layout based on number of horizons
        n_horizons = len(horizons)
        if n_horizons <= 2:
            rows, cols = 1, n_horizons
            figsize = (8 * n_horizons, 6)
        elif n_horizons <= 4:
            rows, cols = 2, 2
            figsize = (16, 12)
        elif n_horizons <= 6:
            rows, cols = 2, 3
            figsize = (18, 12)
        else:
            rows, cols = 3, 3
            figsize = (18, 18)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle('Cluster Trend Strength vs Growth Rate by Forecast Horizon', fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if n_horizons == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        # Color map for different prediction strength categories
        def get_color_by_prediction(prediction_mean, all_predictions):
            if not all_predictions:
                return 'gray'
            percentile_75 = np.percentile(all_predictions, 75)
            percentile_25 = np.percentile(all_predictions, 25)
            
            if prediction_mean >= percentile_75:
                return 'red'  # High prediction
            elif prediction_mean <= percentile_25:
                return 'blue'  # Low prediction
            else:
                return 'green'  # Medium prediction
        
        for idx, horizon in enumerate(horizons):
            ax = axes[idx]
            cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
            
            if not cluster_metrics:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{horizon}-Day Horizon', fontweight='bold')
                continue
            
            # Extract data for this horizon
            trend_strengths = []
            growth_rates = []
            prediction_means = []
            cluster_labels = []
            topic_words_list = []
            
            for cluster_id, metrics in cluster_metrics.items():
                trend_strengths.append(metrics['trend_strength'])
                growth_rates.append(metrics['growth_rate'])
                prediction_means.append(metrics['prediction_mean'])
                cluster_labels.append(f"C{cluster_id}")
                topic_words_list.append(metrics.get('topic_words', [])[:2])
            
            # Create scatter plot with color coding
            colors = [get_color_by_prediction(pred, prediction_means) for pred in prediction_means]
            sizes = [50 + abs(pred) * 10 for pred in prediction_means]  # Size based on prediction magnitude
            
            scatter = ax.scatter(trend_strengths, growth_rates, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add cluster labels
            for i, (x, y, label, topics) in enumerate(zip(trend_strengths, growth_rates, cluster_labels, topic_words_list)):
                # Only label points that are not too crowded
                if len(trend_strengths) <= 20:  # Only add labels if not too many clusters
                    topic_str = ', '.join(topics) if topics else ''
                    full_label = f"{label}: {topic_str}" if topic_str else label
                    ax.annotate(full_label, (x, y), xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Customize the plot
            ax.set_xlabel('Trend Strength')
            ax.set_ylabel('Growth Rate')
            ax.set_title(f'{horizon}-Day Horizon', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(0.02, 0.98, 'Declining\nTrend', transform=ax.transAxes, fontsize=8, alpha=0.6, 
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.3))
            ax.text(0.98, 0.98, 'Growing\nTrend', transform=ax.transAxes, fontsize=8, alpha=0.6, 
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.3))
            ax.text(0.02, 0.02, 'Declining\nWeak Trend', transform=ax.transAxes, fontsize=8, alpha=0.6, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.3))
            ax.text(0.98, 0.02, 'Growing\nWeak Trend', transform=ax.transAxes, fontsize=8, alpha=0.6, 
                   horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.3))
        
        # Hide extra subplots if any
        for idx in range(n_horizons, len(axes)):
            axes[idx].set_visible(False)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='High Prediction (Top 25%)'),
            Patch(facecolor='green', alpha=0.7, label='Medium Prediction (Middle 50%)'),
            Patch(facecolor='blue', alpha=0.7, label='Low Prediction (Bottom 25%)')
        ]
        
        if n_horizons > 1:
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
        else:
            axes[0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def plot_cluster_growth_by_horizon(self, multi_horizon_results: Dict, horizons: List[int] = None, 
                                 max_clusters_per_plot: int = 15) -> None:
        """
        Plot cluster growth rates across different horizons with clusters on x-axis.
        
        Args:
            multi_horizon_results (Dict): Results from generate_multi_horizon_forecasts
            horizons (List[int]): Specific horizons to plot (if None, plots all)
            max_clusters_per_plot (int): Maximum clusters per plot to avoid overcrowding
        """
        if not multi_horizon_results:
            print("No multi-horizon results to visualize")
            return
        
        available_horizons = sorted(multi_horizon_results.keys())
        if horizons is None:
            horizons = available_horizons
        else:
            horizons = [h for h in horizons if h in available_horizons]
        
        if not horizons:
            print("No valid horizons found")
            return
        
        # Collect all unique clusters across horizons
        all_clusters = set()
        for horizon in horizons:
            cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
            all_clusters.update(cluster_metrics.keys())
        
        all_clusters = sorted(list(all_clusters))
        
        # If too many clusters, split into multiple plots
        if len(all_clusters) > max_clusters_per_plot:
            n_plots = (len(all_clusters) + max_clusters_per_plot - 1) // max_clusters_per_plot
            cluster_chunks = [all_clusters[i*max_clusters_per_plot:(i+1)*max_clusters_per_plot] 
                             for i in range(n_plots)]
        else:
            cluster_chunks = [all_clusters]
        
        for chunk_idx, cluster_chunk in enumerate(cluster_chunks):
            # Determine subplot layout
            n_horizons = len(horizons)
            if n_horizons == 1:
                fig, ax = plt.subplots(1, 1, figsize=(16, 8))
                axes = [ax]
            elif n_horizons <= 2:
                fig, axes = plt.subplots(1, n_horizons, figsize=(16, 8))
                if n_horizons == 1:
                    axes = [axes]
            elif n_horizons <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
            elif n_horizons <= 6:
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(3, 3, figsize=(20, 16))
                axes = axes.flatten()
            
            plot_title = f'Cluster Growth Rates by Horizon'
            if len(cluster_chunks) > 1:
                plot_title += f' (Part {chunk_idx + 1}/{len(cluster_chunks)})'
            fig.suptitle(plot_title, fontsize=16, fontweight='bold')
            
            # Plot each horizon
            for idx, horizon in enumerate(horizons):
                ax = axes[idx]
                cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
                
                # Prepare data for this horizon
                cluster_labels = []
                growth_rates = []
                colors = []
                topic_labels = []
                
                for cluster_id in cluster_chunk:
                    if cluster_id in cluster_metrics:
                        metrics = cluster_metrics[cluster_id]
                        growth_rate = metrics['growth_rate']
                        topic_words = metrics.get('topic_words', [])[:2]  # Take first 2 topic words
                        
                        # Create cluster label with topic
                        topic_str = ', '.join(topic_words) if topic_words else 'Unknown'
                        cluster_label = f"C{cluster_id}"
                        topic_labels.append(f"{cluster_label}\n{topic_str}")
                        
                        cluster_labels.append(cluster_label)
                        growth_rates.append(growth_rate)
                        
                        # Color based on growth rate
                        if growth_rate > 0.05:
                            colors.append('green')
                        elif growth_rate > 0:
                            colors.append('lightgreen')
                        elif growth_rate > -0.05:
                            colors.append('orange')
                        else:
                            colors.append('red')
                    else:
                        # Cluster not available for this horizon
                        cluster_label = f"C{cluster_id}"
                        topic_labels.append(f"{cluster_label}\nNo Data")
                        cluster_labels.append(cluster_label)
                        growth_rates.append(0)
                        colors.append('gray')
                
                # Create bar plot
                x_positions = range(len(cluster_labels))
                bars = ax.bar(x_positions, growth_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Customize plot
                ax.set_xlabel('Clusters', fontweight='bold')
                ax.set_ylabel('Growth Rate', fontweight='bold')
                ax.set_title(f'{horizon}-Day Forecast Horizon', fontweight='bold')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                
                # Add value labels on bars
                for bar, value in zip(bars, growth_rates):
                    height = bar.get_height()
                    label_y = height + 0.001 if height >= 0 else height - 0.005
                    ax.text(bar.get_x() + bar.get_width()/2, label_y, f'{value:.3f}', 
                           ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, fontweight='bold')
                
                # Add growth rate categories legend (only on first subplot)
                if idx == 0:
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='green', alpha=0.7, label='High Growth (>5%)'),
                        Patch(facecolor='lightgreen', alpha=0.7, label='Moderate Growth (0-5%)'),
                        Patch(facecolor='orange', alpha=0.7, label='Slight Decline (0 to -5%)'),
                        Patch(facecolor='red', alpha=0.7, label='Strong Decline (<-5%)'),
                        Patch(facecolor='gray', alpha=0.7, label='No Data')
                    ]
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8)
            
            # Hide extra subplots
            for idx in range(len(horizons), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
    
    
    def plot_cluster_comparison_across_horizons(self, multi_horizon_results: Dict, 
                                              selected_clusters: List[int] = None,
                                              top_n: int = 10) -> None:
        """
        Plot growth rate comparison for selected clusters across all horizons.
        
        Args:
            multi_horizon_results (Dict): Results from multi-horizon analysis
            selected_clusters (List[int]): Specific clusters to plot (if None, uses top performers)
            top_n (int): Number of top clusters to show if selected_clusters is None
        """
        if not multi_horizon_results:
            print("No multi-horizon results available")
            return
        
        horizons = sorted(multi_horizon_results.keys())
        
        # Determine clusters to plot
        if selected_clusters is None:
            # Find top performing clusters based on average growth rate
            cluster_avg_growth = {}
            all_clusters = set()
            
            for horizon in horizons:
                cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
                for cluster_id, metrics in cluster_metrics.items():
                    all_clusters.add(cluster_id)
                    if cluster_id not in cluster_avg_growth:
                        cluster_avg_growth[cluster_id] = []
                    cluster_avg_growth[cluster_id].append(metrics['growth_rate'])
            
            # Calculate average growth rates
            for cluster_id in cluster_avg_growth:
                cluster_avg_growth[cluster_id] = np.mean(cluster_avg_growth[cluster_id])
            
            # Get top clusters
            top_clusters = sorted(cluster_avg_growth.items(), key=lambda x: x[1], reverse=True)[:top_n]
            selected_clusters = [cluster_id for cluster_id, _ in top_clusters]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Colors for different clusters
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_clusters)))
        
        # Plot each cluster's growth rate across horizons
        for i, cluster_id in enumerate(selected_clusters):
            growth_rates = []
            topic_words = []
            
            for horizon in horizons:
                cluster_metrics = multi_horizon_results[horizon]['metrics']['cluster_metrics']
                if cluster_id in cluster_metrics:
                    growth_rates.append(cluster_metrics[cluster_id]['growth_rate'])
                    if not topic_words:  # Get topic words from first available horizon
                        topic_words = cluster_metrics[cluster_id].get('topic_words', [])[:2]
                else:
                    growth_rates.append(0)  # No data available
            
            # Create label with topic words
            topic_str = ', '.join(topic_words) if topic_words else 'Unknown'
            label = f"C{cluster_id}: {topic_str}"
            
            # Plot line
            ax.plot(horizons, growth_rates, marker='o', linewidth=2, markersize=8, 
                    color=colors[i], label=label)
        
        # Customize plot
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Growth Rate', fontweight='bold', fontsize=12)
        ax.set_title('Cluster Growth Rate Comparison Across Forecast Horizons', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()


    
    def run_enhanced_pipeline_with_forecasting(self, comment_files: List[str], video_file: str,
                                             n_clusters: int = None, forecast_days: int = 30) -> Dict:
        """
        Run the complete enhanced pipeline with ensemble forecasting.
        
        Args:
            comment_files (List[str]): List of comment CSV files
            video_file (str): Video CSV file path
            n_clusters (int): Number of clusters (if None, will optimize)
            forecast_days (int): Number of days to forecast
            
        Returns:
            Dict: Complete analysis results including ensemble forecasts
        """
        print("Starting enhanced pipeline with ensemble forecasting...")
        
        # Run base pipeline
        base_results = super().run_enhanced_pipeline(comment_files, video_file, n_clusters, forecast_days)
        
        # Train ensemble forecasting models
        self.train_ensemble_forecasting_models(forecast_days)
        
        # Generate ensemble forecasts
        forecast_results = self.generate_ensemble_forecasts(forecast_days)
        
        # Visualize forecasts
        if forecast_results:
            self.visualize_ensemble_forecasts(forecast_results)
        
        # Combine results
        enhanced_results = base_results.copy()
        enhanced_results['ensemble_forecasts'] = forecast_results
        enhanced_results['forecasting_summary'] = {
            'forecast_horizon_days': forecast_days,
            'clusters_forecasted': len(forecast_results),
            'models_used': ['LSTM', 'ARIMA', 'Prophet'],
            'ensemble_method': 'weighted_average',
            'forecast_categories': {
                category: len([r for r in forecast_results.values() if r['forecast_category'] == category])
                for category in ['rapid_growth', 'steady_growth', 'stable', 'declining', 'volatile']
            }
        }
        
        return enhanced_results

    def run_multi_horizon_analysis(self, comment_files: List[str], video_file: str,
                                  n_clusters: int = None, 
                                  forecast_horizons: List[int] = [20, 30, 40, 60, 80]) -> Dict:
        """
        Run complete analysis with multiple forecast horizons.
        
        Args:
            comment_files: List of comment CSV files
            video_file: Video CSV file path
            n_clusters: Number of clusters (if None, will optimize)
            forecast_horizons: List of forecast horizons in days
            
        Returns:
            Dict: Complete analysis results with multi-horizon forecasts
        """
        print("Starting multi-horizon trend analysis...")
        
        # Run base pipeline first
        base_results = self.run_enhanced_pipeline_with_forecasting(
            comment_files, video_file, n_clusters, max(forecast_horizons)
        )
        
        # Generate multi-horizon forecasts
        multi_horizon_results = self.generate_multi_horizon_forecasts(forecast_horizons)
        
        # Create comprehensive visualizations
        if multi_horizon_results:
        # Generate comprehensive generational forecasting visualizations
            print("Generating generational forecasting analysis...")
            self.visualize_all_generations_forecast(forecast_horizons)
            
            # Generate specific generation plots
            for generation in ['gen_z', 'millennial']:
                print(f"Generating {generation} specific growth analysis...")
                self.plot_generational_growth_by_clusters(generation, forecast_horizons[:3])
            # Combine all results
            complete_results = base_results.copy()
            complete_results['multi_horizon_analysis'] = multi_horizon_results
            complete_results['forecast_horizons'] = forecast_horizons
            
        return complete_results
    

class EnhancedGenerationalTrendAI(EnhancedTrendAI):
    """
    Enhanced TrendAI with improved generational analysis.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__(model_name)
        # Replace the analyzer with enhanced version
        self.generational_analyzer = EnhancedGenerationalLanguageAnalyzer()
    
    def analyze_generational_patterns(self) -> None:
        """Enhanced generational analysis with debugging."""
        print("Analyzing generational language patterns with enhanced detection...")
        
        if self.comments_df is None:
            print("No comment data available")
            return
        
        # Sample some comments for debugging
        sample_comments = self.comments_df['cleaned_text'].head(100).tolist()
        print("Testing generational detection on sample comments...")
        
        generation_counts = {'gen_z': 0, 'millennial': 0, 'gen_x': 0, 'boomer': 0, 'neutral': 0}
        
        for comment in sample_comments:
            if pd.notna(comment) and len(str(comment).strip()) > 5:
                classification = self.generational_analyzer.classify_generation(str(comment))
                generation_counts[classification] += 1
        
        print("Sample classification results:")
        for gen, count in generation_counts.items():
            print(f"  {gen}: {count}/100 ({count}%)")
        
        # Analyze all comments
        print("Analyzing all comments...")
        self.comments_df['generational_scores'] = self.comments_df['cleaned_text'].apply(
            lambda x: self.generational_analyzer.analyze_generational_language(str(x)) if pd.notna(x) else {}
        )
        
        # Extract individual generation scores
        for generation in ['gen_z', 'millennial', 'gen_x', 'boomer']:
            self.comments_df[f'{generation}_score'] = self.comments_df['generational_scores'].apply(
                lambda x: x.get(generation, 0) if isinstance(x, dict) else 0
            )
        
        # Classify predominant generation with more liberal approach
        self.comments_df['dominant_generation'] = self.comments_df['cleaned_text'].apply(
            lambda x: self.generational_analyzer.classify_generation(str(x)) if pd.notna(x) else 'neutral'
        )
        
        # Print final distribution
        final_distribution = self.comments_df['dominant_generation'].value_counts()
        print("Final generational distribution:")
        total_comments = len(self.comments_df)
        for gen, count in final_distribution.items():
            percentage = (count / total_comments) * 100
            print(f"  {gen}: {count:,} comments ({percentage:.1f}%)")
        
        # Analyze videos if available
        if self.videos_df is not None:
            print("Analyzing videos for generational patterns...")
            combined_video_text = (
                self.videos_df.get('cleaned_title', '').fillna('') + ' ' + 
                self.videos_df.get('cleaned_description', '').fillna('')
            )
            
            self.videos_df['generational_scores'] = combined_video_text.apply(
                lambda x: self.generational_analyzer.analyze_generational_language(str(x)) if pd.notna(x) and str(x).strip() else {}
            )
            
            # Extract individual generation scores for videos
            for generation in ['gen_z', 'millennial', 'gen_x', 'boomer']:
                self.videos_df[f'{generation}_score'] = self.videos_df['generational_scores'].apply(
                    lambda x: x.get(generation, 0) if isinstance(x, dict) else 0
                )
            
            self.videos_df['dominant_generation'] = combined_video_text.apply(
                lambda x: self.generational_analyzer.classify_generation(str(x)) if pd.notna(x) and str(x).strip() else 'neutral'
            )
            
            # Print video distribution
            video_distribution = self.videos_df['dominant_generation'].value_counts()
            print("Video generational distribution:")
            total_videos = len(self.videos_df)
            for gen, count in video_distribution.items():
                percentage = (count / total_videos) * 100
                print(f"  {gen}: {count} videos ({percentage:.1f}%)")
    
    def analyze_generational_trends_by_cluster(self) -> None:
        """Enhanced cluster analysis with debugging and validation."""
        print("Analyzing generational trends by cluster with enhanced detection...")
        
        if self.comments_df is None or 'cluster' not in self.comments_df.columns:
            print("Comment data or clustering results not available")
            return
        
        # Ensure generational analysis is completed
        if 'dominant_generation' not in self.comments_df.columns:
            self.analyze_generational_patterns()
        
        clusters = self.comments_df['cluster'].unique()
        self.generational_clusters = {}
        
        print(f"Analyzing {len(clusters)} clusters...")
        
        for cluster_id in clusters:
            cluster_comments = self.comments_df[self.comments_df['cluster'] == cluster_id].copy()
            
            if len(cluster_comments) < 5:  # Reduced minimum threshold
                continue
            
            try:
                # Enhanced generational distribution analysis
                generation_distribution = cluster_comments['dominant_generation'].value_counts(normalize=True)
                
                if len(generation_distribution) == 0:
                    continue
                
                # Get the most dominant generation
                dominant_generation = generation_distribution.index[0]
                dominant_generation_score = generation_distribution.iloc[0]
                
                # Calculate average generational scores for this cluster
                avg_generational_scores = {}
                for generation in ['gen_z', 'millennial', 'gen_x', 'boomer']:
                    if f'{generation}_score' in cluster_comments.columns:
                        avg_generational_scores[generation] = cluster_comments[f'{generation}_score'].mean()
                    else:
                        avg_generational_scores[generation] = 0.0
                
                # Get topic information
                topic_words = self.cluster_topics.get(cluster_id, [])[:10]
                topic_tags = self.cluster_tags.get(cluster_id, [])[:5]
                
                # Enhanced video performance analysis by generation
                video_performance_by_generation = {}
                if hasattr(self, 'videos_df') and self.videos_df is not None and 'cluster' in self.videos_df.columns:
                    cluster_videos = self.videos_df[self.videos_df['cluster'] == cluster_id]
                    
                    if not cluster_videos.empty and 'dominant_generation' in cluster_videos.columns:
                        for generation in ['gen_z', 'millennial', 'gen_x', 'boomer', 'neutral']:
                            gen_videos = cluster_videos[cluster_videos['dominant_generation'] == generation]
                            if len(gen_videos) > 0:
                                video_performance_by_generation[generation] = {
                                    'count': len(gen_videos),
                                    'avg_views': gen_videos.get('viewCount', pd.Series([0])).mean(),
                                    'avg_likes': gen_videos.get('likeCount', pd.Series([0])).mean(),
                                    'avg_engagement': gen_videos.get('engagement_rate', pd.Series([0])).mean()
                                }
                
                self.generational_clusters[cluster_id] = {
                    'dominant_generation': dominant_generation,
                    'dominant_generation_score': float(dominant_generation_score),
                    'generation_distribution': generation_distribution.to_dict(),
                    'avg_generational_scores': avg_generational_scores,
                    'topic_words': topic_words,
                    'topic_tags': topic_tags,
                    'video_performance_by_generation': video_performance_by_generation,
                    'cluster_size': len(cluster_comments)
                }
                
                # Debug information for first few clusters
                if len(self.generational_clusters) <= 3:
                    print(f"Cluster {cluster_id} analysis:")
                    print(f"  Size: {len(cluster_comments)} comments")
                    print(f"  Dominant generation: {dominant_generation} ({dominant_generation_score:.2%})")
                    print(f"  Topics: {', '.join(topic_words[:3])}")
                    print(f"  Generation distribution: {dict(generation_distribution.round(3))}")
                    print()
                
            except Exception as e:
                print(f"Error analyzing cluster {cluster_id}: {e}")
                continue
        
        print(f"✅ Generational analysis completed for {len(self.generational_clusters)} clusters")
        
        # Print summary by generation
        generation_cluster_counts = {}
        for cluster_data in self.generational_clusters.values():
            gen = cluster_data['dominant_generation']
            generation_cluster_counts[gen] = generation_cluster_counts.get(gen, 0) + 1
        
        print("Clusters by dominant generation:")
        for gen, count in sorted(generation_cluster_counts.items()):
            print(f"  {gen}: {count} clusters")

"""Visualization utilities for trend analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')


class TrendVisualizer:
    """Handles all visualization tasks for trend analysis."""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_enhanced_trends(self, top_n_clusters: int = 5) -> None:
        """
        Create comprehensive trend visualizations using matplotlib and seaborn including video metrics and tags.
        
        Args:
            top_n_clusters (int): Number of top clusters to visualize
        """
        print("Creating enhanced trend visualizations with tag integration...")
        
        if self.predictor.combined_trend_data.empty:
            print("No trend data available for visualization")
            return
        
        # Get top clusters by combined trending score
        cluster_performance = self.predictor.combined_trend_data.groupby('cluster')['combined_trending_score'].sum().sort_values(ascending=False)
        top_clusters = cluster_performance.head(top_n_clusters).index.tolist()
        
        # Create comprehensive dashboard with subplots
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        fig.suptitle('Enhanced YouTube Beauty Trends Analysis Dashboard (Video+Tag Weighted)', 
                     fontsize=20, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_clusters)))
        
        # Plot 1: Video Views Over Time
        ax1 = axes[0, 0]
        for i, cluster_id in enumerate(top_clusters):
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_words = ', '.join(self.predictor.cluster_topics.get(cluster_id, [])[:3])
            topic_tags = ', '.join(self.predictor.cluster_tags.get(cluster_id, [])[:2])
            label = f'{topic_words} ({topic_tags})' if topic_tags else topic_words
            
            ax1.plot(cluster_data['date'], cluster_data['total_views'], 
                    marker='o', linewidth=2, color=colors[i], label=label[:30])
        
        ax1.set_title('Video Views Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Views')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Comment Volume Over Time
        ax2 = axes[0, 1]
        for i, cluster_id in enumerate(top_clusters):
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_words = ', '.join(self.predictor.cluster_topics.get(cluster_id, [])[:3])
            
            ax2.plot(cluster_data['date'], cluster_data['comment_count'], 
                    marker='s', linewidth=2, linestyle='--', color=colors[i], label=topic_words[:30])
        
        ax2.set_title('Comment Volume Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Comment Count')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Combined Trending Scores
        ax3 = axes[1, 0]
        for i, cluster_id in enumerate(top_clusters):
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_words = ', '.join(self.predictor.cluster_topics.get(cluster_id, [])[:3])
            
            ax3.plot(cluster_data['date'], cluster_data['combined_trending_score'], 
                    marker='D', linewidth=3, color=colors[i], label=topic_words[:30])
        
        ax3.set_title('Combined Trending Scores (Video+Tag Weighted)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Combined Trending Score')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Tag Relevance Over Time
        ax4 = axes[1, 1]
        for i, cluster_id in enumerate(top_clusters):
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_tags = ', '.join(self.predictor.cluster_tags.get(cluster_id, [])[:2])
            
            ax4.plot(cluster_data['date'], cluster_data['avg_tag_relevance'], 
                    marker='^', linewidth=2, color=colors[i], label=topic_tags[:30] or f'Cluster {cluster_id}')
        
        ax4.set_title('Tag Relevance Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Average Tag Relevance')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Video vs Comment Engagement Scatter
        ax5 = axes[2, 0]
        engagement_data = []
        for cluster_id in top_clusters:
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_words = ', '.join(self.predictor.cluster_topics.get(cluster_id, [])[:2])
            
            for _, row in cluster_data.iterrows():
                engagement_data.append({
                    'video_engagement': row['avg_engagement_rate'],
                    'comment_engagement': row['comment_count'],
                    'cluster': topic_words[:20]
                })
        
        if engagement_data:
            eng_df = pd.DataFrame(engagement_data)
            sns.scatterplot(data=eng_df, x='video_engagement', y='comment_engagement', 
                          hue='cluster', s=100, alpha=0.7, ax=ax5)
        
        ax5.set_title('Video vs Comment Engagement', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Video Engagement Rate')
        ax5.set_ylabel('Comment Count')
        
        # Plot 6: Sentiment Trends
        ax6 = axes[2, 1]
        for i, cluster_id in enumerate(top_clusters):
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_words = ', '.join(self.predictor.cluster_topics.get(cluster_id, [])[:2])
            
            ax6.plot(cluster_data['date'], cluster_data['avg_sentiment'], 
                    marker='o', linewidth=2, color=colors[i], label=topic_words[:30])
        
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.set_title('Sentiment Trends', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Average Sentiment')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.tick_params(axis='x', rotation=45)
        
        # Plot 7: Video Performance Metrics Heatmap
        ax7 = axes[3, 0]
        performance_data = []
        for cluster_id in top_clusters:
            cluster_data = self.predictor.combined_trend_data[self.predictor.combined_trend_data['cluster'] == cluster_id]
            topic_words = ', '.join(self.predictor.cluster_topics.get(cluster_id, [])[:2])
            
            performance_data.append({
                'Cluster': topic_words[:15],
                'Views': cluster_data['total_views'].sum(),
                'Likes': cluster_data['video_likes'].sum(),
                'Comments': cluster_data['comment_count'].sum(),
                'Engagement': cluster_data['avg_engagement_rate'].mean()
            })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            perf_df_norm = perf_df.set_index('Cluster')
            perf_df_norm = (perf_df_norm - perf_df_norm.min()) / (perf_df_norm.max() - perf_df_norm.min())
            
            sns.heatmap(perf_df_norm.T, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax7)
        
        ax7.set_title('Video Performance Metrics (Normalized)', fontsize=14, fontweight='bold')
        
        # Plot 8: Tag Popularity Distribution
        ax8 = axes[3, 1]
        if self.predictor.popular_tags is not None and not self.predictor.popular_tags.empty:
            top_tags = self.predictor.popular_tags.head(10)
            ax8.barh(range(len(top_tags)), top_tags['tag_popularity_score'], color='skyblue')
            ax8.set_yticks(range(len(top_tags)))
            ax8.set_yticklabels(top_tags.index, fontsize=10)
            ax8.set_xlabel('Tag Popularity Score')
            ax8.set_title('Top 10 Tag Popularity Distribution', fontsize=14, fontweight='bold')
        
        # Plot 9: Growth Rate Comparison
        ax9 = axes[4, 0]
        trending_topics = self.predictor.identify_trending_topics()[:top_n_clusters]
        if trending_topics:
            growth_data = {
                'Topic': [', '.join(t['topic_words'][:2]) for t in trending_topics],
                'Video Growth': [t['video_views_growth'] for t in trending_topics],
                'Comment Growth': [t['comment_growth'] for t in trending_topics],
                'Tag Growth': [t['tag_relevance_growth'] for t in trending_topics]
            }
            
            growth_df = pd.DataFrame(growth_data)
            growth_df_melted = growth_df.melt(id_vars='Topic', var_name='Growth Type', value_name='Growth Rate')
            
            sns.barplot(data=growth_df_melted, x='Topic', y='Growth Rate', hue='Growth Type', ax=ax9)
            ax9.tick_params(axis='x', rotation=45)
        
        ax9.set_title('Growth Rate Comparison by Type', fontsize=14, fontweight='bold')
        
        # Plot 10: Trending Categories Distribution
        ax10 = axes[4, 1]
        if trending_topics:
            categories = [t['trending_category'] for t in trending_topics]
            category_counts = pd.Series(categories).value_counts()
            
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
            wedges, texts, autotexts = ax10.pie(category_counts.values, labels=category_counts.index, 
                                              autopct='%1.1f%%', colors=colors_pie)
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
        
        ax10.set_title('Trending Categories Distribution', fontsize=14, fontweight='bold')
        
        # Adjust layout and show
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    def visualize_generational_trends(self) -> None:
        """Create visualizations for generational trend analysis."""
        if self.predictor.generational_trends is None:
            self.predictor.identify_generational_trending_topics()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Generational Trend Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Colors for generations
        gen_colors = {
            'gen_z': '#FF6B6B',
            'millennial': '#4ECDC4', 
            'gen_x': '#45B7D1',
            'boomer': '#96CEB4',
            'neutral': '#FFEAA7'
        }
        
        # Plot 1: Generation Distribution Across All Clusters
        ax1 = axes[0, 0]
        if self.predictor.generational_clusters:
            gen_counts = {}
            for cluster_data in self.predictor.generational_clusters.values():
                dominant_gen = cluster_data['dominant_generation']
                gen_counts[dominant_gen] = gen_counts.get(dominant_gen, 0) + 1
            
            colors = [gen_colors.get(gen, 'gray') for gen in gen_counts.keys()]
            bars = ax1.bar(gen_counts.keys(), gen_counts.values(), color=colors)
            ax1.set_title('Dominant Generation by Cluster', fontweight='bold')
            ax1.set_ylabel('Number of Clusters')
            
            # Add value labels
            for bar, value in zip(bars, gen_counts.values()):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(value), ha='center', va='bottom')
        
        # Plot 2: Comment Volume by Generation
        ax2 = axes[0, 1]
        gen_comment_volumes = {}
        for gen, trends in self.predictor.generational_trends.items():
            total_comments = sum(trend['recent_comment_volume'] for trend in trends)
            if total_comments > 0:
                gen_comment_volumes[gen] = total_comments
        
        if gen_comment_volumes:
            colors = [gen_colors.get(gen, 'gray') for gen in gen_comment_volumes.keys()]
            ax2.pie(gen_comment_volumes.values(), labels=gen_comment_volumes.keys(), 
                    colors=colors, autopct='%1.1f%%')
            ax2.set_title('Comment Volume Distribution', fontweight='bold')
        
        # Plot 3: Average Sentiment by Generation
        ax3 = axes[0, 2]
        gen_sentiments = {}
        for gen, trends in self.predictor.generational_trends.items():
            if trends:
                avg_sentiment = np.mean([trend['avg_sentiment'] for trend in trends if trend['avg_sentiment'] != 0])
                if not np.isnan(avg_sentiment):
                    gen_sentiments[gen] = avg_sentiment
        
        if gen_sentiments:
            colors = [gen_colors.get(gen, 'gray') for gen in gen_sentiments.keys()]
            bars = ax3.bar(gen_sentiments.keys(), gen_sentiments.values(), color=colors)
            ax3.set_title('Average Sentiment by Generation', fontweight='bold')
            ax3.set_ylabel('Sentiment Score')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, value in zip(bars, gen_sentiments.values()):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if value >= 0 else bar.get_height() - 0.01,
                        f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
        
        # Plot 4: Top Topics by Generation (Gen Z)
        ax4 = axes[1, 0]
        gen_z_trends = self.predictor.generational_trends.get('gen_z', [])[:5]
        if gen_z_trends:
            topics = [', '.join(trend['topic_words'][:2]) for trend in gen_z_trends]
            volumes = [trend['recent_comment_volume'] for trend in gen_z_trends]
            
            ax4.barh(range(len(topics)), volumes, color=gen_colors['gen_z'])
            ax4.set_yticks(range(len(topics)))
            ax4.set_yticklabels(topics, fontsize=10)
            ax4.set_title('Top Gen Z Topics', fontweight='bold')
            ax4.set_xlabel('Recent Comment Volume')
        
        # Plot 5: Top Topics by Generation (Millennial)
        ax5 = axes[1, 1]
        millennial_trends = self.predictor.generational_trends.get('millennial', [])[:5]
        if millennial_trends:
            topics = [', '.join(trend['topic_words'][:2]) for trend in millennial_trends]
            volumes = [trend['recent_comment_volume'] for trend in millennial_trends]
            
            ax5.barh(range(len(topics)), volumes, color=gen_colors['millennial'])
            ax5.set_yticks(range(len(topics)))
            ax5.set_yticklabels(topics, fontsize=10)
            ax5.set_title('Top Millennial Topics', fontweight='bold')
            ax5.set_xlabel('Recent Comment Volume')
        
        # Plot 6: Generational Language Intensity Heatmap
        ax6 = axes[1, 2]
        if self.predictor.generational_clusters:
            # Create heatmap data
            heatmap_data = []
            cluster_labels = []
            
            for cluster_id, analysis in list(self.predictor.generational_clusters.items())[:10]:  # Top 10 clusters
                scores = [analysis['avg_generational_scores'][gen] for gen in ['gen_z', 'millennial', 'gen_x', 'boomer']]
                heatmap_data.append(scores)
                topic_label = ', '.join(analysis['topic_words'][:2])
                cluster_labels.append(f"C{cluster_id}: {topic_label}"[:25])
            
            if heatmap_data:
                sns.heatmap(heatmap_data, 
                           xticklabels=['Gen Z', 'Millennial', 'Gen X', 'Boomer'],
                           yticklabels=cluster_labels,
                           annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6)
                ax6.set_title('Generational Language Intensity', fontweight='bold')
                ax6.set_xlabel('Generation')
                ax6.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        plt.show()

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

    def plot_generational_growth_by_clusters(self, generation: str = 'gen_z', 
                                           forecast_horizons: List[int] = [20, 30, 60]) -> None:
        """
        Plot growth rates for a specific generation across clusters and forecast horizons.
        
        Args:
            generation: Target generation ('gen_z', 'millennial', 'gen_x', 'boomer')
            forecast_horizons: List of forecast horizons to analyze
        """
        if not hasattr(self.predictor, 'generational_clusters') or not self.predictor.generational_clusters:
            print("No generational cluster data available. Running analysis...")
            self.predictor.analyze_generational_trends_by_cluster()
        
        # Get clusters dominated by the specified generation
        generation_clusters = []
        for cluster_id, analysis in self.predictor.generational_clusters.items():
            if analysis['dominant_generation'] == generation:
                generation_clusters.append({
                    'cluster_id': cluster_id,
                    'topic_words': analysis['topic_words'][:3],
                    'topic_tags': analysis['topic_tags'][:2],
                    'generation_confidence': analysis['dominant_generation_score'],
                    'cluster_size': analysis.get('cluster_size', 0)
                })
        
        if not generation_clusters:
            print(f"No clusters found for generation: {generation}")
            return
        
        # Sort by cluster size (popularity)
        generation_clusters.sort(key=lambda x: x['cluster_size'], reverse=True)
        top_clusters = generation_clusters[:15]  # Limit to top 15 for readability
        
        # Create subplots for different forecast horizons
        fig, axes = plt.subplots(1, len(forecast_horizons), figsize=(6*len(forecast_horizons), 8))
        if len(forecast_horizons) == 1:
            axes = [axes]
        
        fig.suptitle(f'{generation.replace("_", " ").title()} Growth Rates by Cluster Category', 
                     fontsize=16, fontweight='bold')
        
        # Generate forecasts for each horizon if not already available
        if not hasattr(self.predictor, 'multi_horizon_results'):
            print("Generating multi-horizon forecasts...")
            self.predictor.multi_horizon_results = self.predictor.generate_multi_horizon_forecasts(forecast_horizons)
        
        for idx, horizon in enumerate(forecast_horizons):
            ax = axes[idx]
            
            cluster_names = []
            growth_rates = []
            colors = []
            
            for cluster_data in top_clusters:
                cluster_id = cluster_data['cluster_id']
                
                # Get growth rate from forecast results
                if (hasattr(self.predictor, 'multi_horizon_results') and 
                    horizon in self.predictor.multi_horizon_results and
                    cluster_id in self.predictor.multi_horizon_results[horizon]['metrics']['cluster_metrics']):
                    
                    growth_rate = self.predictor.multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['growth_rate']
                    growth_rates.append(growth_rate)
                    
                    # Color based on growth rate
                    if growth_rate > 0.1:
                        colors.append('green')
                    elif growth_rate > 0:
                        colors.append('lightgreen')
                    elif growth_rate > -0.05:
                        colors.append('orange')
                    else:
                        colors.append('red')
                else:
                    growth_rates.append(0)
                    colors.append('gray')
                
                # Create cluster label
                topic_str = ', '.join(cluster_data['topic_words'])
                tag_str = ', '.join(cluster_data['topic_tags']) if cluster_data['topic_tags'] else ''
                label = f"C{cluster_id}: {topic_str}"
                if tag_str:
                    label += f"\n({tag_str})"
                cluster_names.append(label)
            
            # Create bar plot
            bars = ax.bar(range(len(cluster_names)), growth_rates, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_title(f'{horizon}-Day Forecast', fontweight='bold')
            ax.set_xlabel('Cluster Categories')
            ax.set_ylabel('Growth Rate')
            ax.set_xticks(range(len(cluster_names)))
            ax.set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, growth_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, 
                       height + 0.005 if height >= 0 else height - 0.01,
                       f'{value:.2f}', ha='center', 
                       va='bottom' if height >= 0 else 'top', 
                       fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def visualize_all_generations_forecast(self, forecast_horizons: List[int] = [20, 30, 60]) -> None:
        """Create comprehensive visualization showing all generations' forecast performance."""
        generational_forecasts = self.predictor.generate_generational_forecasts(forecast_horizons)
        
        generations = ['gen_z', 'millennial', 'gen_x', 'boomer']
        gen_colors = {
            'gen_z': '#FF6B6B',
            'millennial': '#4ECDC4', 
            'gen_x': '#45B7D1',
            'boomer': '#96CEB4'
        }
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Generational Trend Forecasting Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Average Growth Rate by Generation and Horizon
        ax1 = axes[0, 0]
        width = 0.2
        x = np.arange(len(forecast_horizons))
        
        for i, generation in enumerate(generations):
            if generation in generational_forecasts:
                avg_growth_rates = []
                for horizon in forecast_horizons:
                    avg_growth_rates.append(generational_forecasts[generation][horizon]['avg_growth_rate'])
                
                ax1.bar(x + i * width, avg_growth_rates, width, 
                       label=generation.replace('_', ' ').title(), 
                       color=gen_colors[generation], alpha=0.8)
        
        ax1.set_xlabel('Forecast Horizon (Days)')
        ax1.set_ylabel('Average Growth Rate')
        ax1.set_title('Average Growth Rate by Generation', fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(forecast_horizons)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 2: Number of Trending Topics by Generation
        ax2 = axes[0, 1]
        trending_counts = {gen: [] for gen in generations}
        
        for generation in generations:
            for horizon in forecast_horizons:
                count = len(generational_forecasts[generation][horizon]['trending_topics'])
                trending_counts[generation].append(count)
        
        for i, generation in enumerate(generations):
            ax2.plot(forecast_horizons, trending_counts[generation], 
                    marker='o', linewidth=2, markersize=8,
                    color=gen_colors[generation], 
                    label=generation.replace('_', ' ').title())
        
        ax2.set_xlabel('Forecast Horizon (Days)')
        ax2.set_ylabel('Number of Trending Topics')
        ax2.set_title('Trending Topics Count by Generation', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gen Z specific cluster growth (as requested)
        ax3 = axes[1, 0]
        if 'gen_z' in generational_forecasts:
            # Get Gen Z clusters for the middle forecast horizon
            mid_horizon = forecast_horizons[len(forecast_horizons)//2]
            gen_z_clusters = generational_forecasts['gen_z'][mid_horizon]['clusters'][:10]  # Top 10
            
            if gen_z_clusters:
                cluster_names = []
                growth_rates = []
                colors = []
                
                for cluster in gen_z_clusters:
                    topic_str = ', '.join(cluster['topic_words'])
                    cluster_names.append(f"C{cluster['cluster_id']}: {topic_str}")
                    growth_rates.append(cluster['growth_rate'])
                    
                    if cluster['growth_rate'] > 0.05:
                        colors.append('green')
                    elif cluster['growth_rate'] > 0:
                        colors.append('lightgreen')
                    else:
                        colors.append('red')
                
                bars = ax3.barh(range(len(cluster_names)), growth_rates, color=colors, alpha=0.7)
                ax3.set_yticks(range(len(cluster_names)))
                ax3.set_yticklabels(cluster_names, fontsize=9)
                ax3.set_xlabel('Growth Rate')
                ax3.set_title(f'Gen Z Cluster Growth ({mid_horizon}-Day Forecast)', fontweight='bold')
                ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Generation Comparison Heatmap
        ax4 = axes[1, 1]
        heatmap_data = []
        generation_labels = []
        
        for generation in generations:
            if generation in generational_forecasts:
                row_data = []
                for horizon in forecast_horizons:
                    avg_growth = generational_forecasts[generation][horizon]['avg_growth_rate']
                    row_data.append(avg_growth)
                heatmap_data.append(row_data)
                generation_labels.append(generation.replace('_', ' ').title())
        
        if heatmap_data:
            im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
            ax4.set_xticks(range(len(forecast_horizons)))
            ax4.set_xticklabels(forecast_horizons)
            ax4.set_yticks(range(len(generation_labels)))
            ax4.set_yticklabels(generation_labels)
            ax4.set_xlabel('Forecast Horizon (Days)')
            ax4.set_title('Growth Rate Heatmap', fontweight='bold')
            
            # Add text annotations
            for i in range(len(generation_labels)):
                for j in range(len(forecast_horizons)):
                    text = ax4.text(j, i, f'{heatmap_data[i][j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax4, label='Growth Rate')
        
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

    def plot_tag_trend_analysis(self, top_n_tags: int = 20) -> None:
        """
        Plot comprehensive tag trend analysis.
        
        Args:
            top_n_tags (int): Number of top tags to analyze
        """
        if not hasattr(self.predictor, 'popular_tags') or self.predictor.popular_tags is None:
            print("No tag analysis data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tag Trend Analysis Dashboard', fontsize=16, fontweight='bold')
        
        top_tags = self.predictor.popular_tags.head(top_n_tags)
        
        # Plot 1: Tag Popularity Scores
        ax1 = axes[0, 0]
        y_pos = np.arange(len(top_tags.head(15)))  # Show top 15
        ax1.barh(y_pos, top_tags.head(15)['tag_popularity_score'], color='skyblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_tags.head(15).index, fontsize=9)
        ax1.set_xlabel('Popularity Score')
        ax1.set_title('Top 15 Tag Popularity Scores', fontweight='bold')
        
        # Plot 2: Tag Frequency vs Views
        ax2 = axes[0, 1]
        scatter = ax2.scatter(top_tags['frequency'], top_tags['total_views'], 
                            c=top_tags['tag_popularity_score'], cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Tag Frequency (Number of Videos)')
        ax2.set_ylabel('Total Views')
        ax2.set_title('Tag Frequency vs Total Views', fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Popularity Score')
        
        # Plot 3: Average Views per Video for Tags
        ax3 = axes[1, 0]
        ax3.bar(range(len(top_tags.head(10))), top_tags.head(10)['avg_views_per_video'], 
                color='orange', alpha=0.7)
        ax3.set_xticks(range(len(top_tags.head(10))))
        ax3.set_xticklabels(top_tags.head(10).index, rotation=45, ha='right')
        ax3.set_ylabel('Average Views per Video')
        ax3.set_title('Top 10 Tags by Average Views per Video', fontweight='bold')
        
        # Plot 4: Tag Performance Metrics Heatmap
        ax4 = axes[1, 1]
        tag_metrics = top_tags.head(10)[['frequency_score', 'views_score', 'likes_score']]
        sns.heatmap(tag_metrics.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Tag Performance Metrics (Normalized)', fontweight='bold')
        ax4.set_xlabel('Tags')
        
        plt.tight_layout()
        plt.show()

    def create_summary_dashboard(self) -> None:
        """Create a comprehensive summary dashboard of all analyses."""
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Comprehensive Trend Analysis Summary Dashboard', fontsize=20, fontweight='bold')
        
        # Basic data summary
        ax1 = axes[0, 0]
        summary_data = {
            'Comments': len(self.predictor.comments_df) if self.predictor.comments_df is not None else 0,
            'Videos': len(self.predictor.videos_df) if self.predictor.videos_df is not None else 0,
            'Clusters': len(set(self.predictor.clusters)) if self.predictor.clusters is not None else 0,
            'Tags': len(self.predictor.popular_tags) if self.predictor.popular_tags is not None else 0
        }
        
        ax1.bar(summary_data.keys(), summary_data.values(), color=['blue', 'green', 'orange', 'red'])
        ax1.set_title('Data Summary', fontweight='bold')
        ax1.set_ylabel('Count')
        
        # Add value labels on bars
        for i, (key, value) in enumerate(summary_data.items()):
            ax1.text(i, value + max(summary_data.values()) * 0.01, f'{value:,}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Sentiment distribution
        ax2 = axes[0, 1]
        if self.predictor.comments_df is not None and 'compound' in self.predictor.comments_df.columns:
            sentiment_counts = {
                'Positive': len(self.predictor.comments_df[self.predictor.comments_df['compound'] > 0.1]),
                'Neutral': len(self.predictor.comments_df[
                    (self.predictor.comments_df['compound'] >= -0.1) & 
                    (self.predictor.comments_df['compound'] <= 0.1)
                ]),
                'Negative': len(self.predictor.comments_df[self.predictor.comments_df['compound'] < -0.1])
            }
            
            colors = ['green', 'gray', 'red']
            ax2.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), 
                colors=colors, autopct='%1.1f%%')
            ax2.set_title('Overall Sentiment Distribution', fontweight='bold')
        
        # Top performing clusters
        ax3 = axes[0, 2]
        if not self.predictor.combined_trend_data.empty:
            cluster_performance = self.predictor.combined_trend_data.groupby('cluster')['combined_trending_score'].mean().sort_values(ascending=False).head(10)
            
            y_pos = range(len(cluster_performance))
            cluster_labels = [f"C{cid}: {', '.join(self.predictor.cluster_topics.get(cid, [])[:2])}" 
                            for cid in cluster_performance.index]
            
            ax3.barh(y_pos, cluster_performance.values, color='purple')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(cluster_labels, fontsize=8)
            ax3.set_xlabel('Average Trending Score')
            ax3.set_title('Top 10 Performing Clusters', fontweight='bold')
        
        # Generation distribution
        ax4 = axes[1, 0]
        if hasattr(self.predictor, 'generational_clusters') and self.predictor.generational_clusters:
            gen_counts = {}
            for cluster_data in self.predictor.generational_clusters.values():
                dominant_gen = cluster_data['dominant_generation']
                gen_counts[dominant_gen] = gen_counts.get(dominant_gen, 0) + 1
            
            gen_colors = {'gen_z': '#FF6B6B', 'millennial': '#4ECDC4', 
                        'gen_x': '#45B7D1', 'boomer': '#96CEB4', 'neutral': '#FFEAA7'}
            colors = [gen_colors.get(gen, 'gray') for gen in gen_counts.keys()]
            
            ax4.bar(gen_counts.keys(), gen_counts.values(), color=colors)
            ax4.set_title('Generation Distribution', fontweight='bold')
            ax4.set_ylabel('Number of Clusters')
            ax4.tick_params(axis='x', rotation=45)
        
        # Video performance metrics
        ax5 = axes[1, 1]
        if self.predictor.videos_df is not None:
            metrics = {
                'Total Views': self.predictor.videos_df['viewCount'].sum(),
                'Total Likes': self.predictor.videos_df['likeCount'].sum(),
                'Total Comments': self.predictor.videos_df['commentCount'].sum()
            }
            
            # Normalize to millions for readability
            normalized_metrics = {k: v / 1000000 for k, v in metrics.items()}
            
            ax5.bar(normalized_metrics.keys(), normalized_metrics.values(), 
                color=['red', 'blue', 'green'])
            ax5.set_title('Video Performance Metrics (Millions)', fontweight='bold')
            ax5.set_ylabel('Count (Millions)')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, (key, value) in enumerate(normalized_metrics.items()):
                ax5.text(i, value + max(normalized_metrics.values()) * 0.01, 
                        f'{value:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Tag cloud representation (top tags)
        ax6 = axes[1, 2]
        if self.predictor.popular_tags is not None and not self.predictor.popular_tags.empty:
            top_15_tags = self.predictor.popular_tags.head(15)
            sizes = top_15_tags['tag_popularity_score'] * 100  # Scale for visibility
            
            # Create a simple scatter plot as tag cloud
            x_pos = np.random.rand(len(top_15_tags))
            y_pos = np.random.rand(len(top_15_tags))
            
            scatter = ax6.scatter(x_pos, y_pos, s=sizes, alpha=0.6, c=sizes, cmap='viridis')
            
            # Add tag labels
            for i, (tag, _) in enumerate(top_15_tags.iterrows()):
                ax6.annotate(tag, (x_pos[i], y_pos[i]), fontsize=8, ha='center')
            
            ax6.set_title('Top Tags (Size = Popularity)', fontweight='bold')
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.set_xticks([])
            ax6.set_yticks([])
        
        # Trending categories (if available)
        ax7 = axes[2, 0]
        trending_topics = self.predictor.identify_trending_topics()
        if trending_topics:
            categories = [t['trending_category'] for t in trending_topics]
            category_counts = pd.Series(categories).value_counts()
            
            ax7.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            ax7.set_title('Trending Categories', fontweight='bold')
        
        # Time series summary
        ax8 = axes[2, 1]
        if not self.predictor.combined_trend_data.empty:
            daily_trends = self.predictor.combined_trend_data.groupby('date')['combined_trending_score'].mean()
            ax8.plot(daily_trends.index, daily_trends.values, linewidth=2, color='purple')
            ax8.set_title('Overall Trending Score Over Time', fontweight='bold')
            ax8.set_xlabel('Date')
            ax8.set_ylabel('Average Trending Score')
            ax8.tick_params(axis='x', rotation=45)
        
        # Model performance summary
        ax9 = axes[2, 2]
        model_info = {
            'Video Weight': self.predictor.video_weight,
            'Comment Weight': self.predictor.comment_weight,
            'Tag Weight': self.predictor.tag_weight
        }
        
        ax9.bar(model_info.keys(), model_info.values(), color=['orange', 'blue', 'green'])
        ax9.set_title('Model Weighting Factors', fontweight='bold')
        ax9.set_ylabel('Weight')
        ax9.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (key, value) in enumerate(model_info.items()):
            ax9.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def export_visualizations(self, output_dir: str = "./visualizations"):
        """
        Export all visualizations as image files.
        
        Args:
            output_dir (str): Directory to save visualization images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting visualizations to {output_dir}...")
        
        # Set up for saving plots
        original_backend = plt.get_backend()
        plt.switch_backend('Agg')  # Non-interactive backend for saving
        
        try:
            # Enhanced trends
            self.visualize_enhanced_trends()
            plt.savefig(os.path.join(output_dir, "enhanced_trends_dashboard.png"), 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generational trends
            if hasattr(self.predictor, 'generational_trends'):
                self.visualize_generational_trends()
                plt.savefig(os.path.join(output_dir, "generational_trends_dashboard.png"), 
                        dpi=300, bbox_inches='tight')
                plt.close()
            
            # Tag analysis
            self.plot_tag_trend_analysis()
            plt.savefig(os.path.join(output_dir, "tag_analysis_dashboard.png"), 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            # Summary dashboard
            self.create_summary_dashboard()
            plt.savefig(os.path.join(output_dir, "summary_dashboard.png"), 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            print("All visualizations exported successfully!")
            
        except Exception as e:
            print(f"Error exporting visualizations: {e}")
        
        finally:
            plt.switch_backend(original_backend)  # Restore original backend


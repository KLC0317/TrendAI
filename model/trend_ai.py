"""Core TrendAI class with basic functionality."""

import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from generational_analyzer import EnhancedGenerationalLanguageAnalyzer
from config import *

warnings.filterwarnings('ignore')


class TrendAI:
    """
    A comprehensive trend prediction model for YouTube beauty industry data.
    Enhanced to integrate video metrics and tags with higher weighting.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the trend predictor with SBERT model and other components.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
        """
        print("Initializing Enhanced YouTube Trend Predictor with Tags Integration...")
        self.sbert_model = SentenceTransformer(model_name)
        self.kmeans_index = None  # FAISS K-means index
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.comments_df = None
        self.videos_df = None
        self.embeddings = None
        self.clusters = None
        self.trend_data = None
        self.video_trend_data = None
        self.combined_trend_data = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        
        # Enhanced weighting factors including tags
        self.video_weight = 0.6   # 60% weight for video metrics
        self.comment_weight = 0.25  # 25% weight for comment metrics
        self.tag_weight = 0.15    # 15% weight for tag relevance
        
        # Tag processing attributes
        self.tag_trends = None
        self.tag_clusters = None
        self.popular_tags = None

        self.generational_analyzer = EnhancedGenerationalLanguageAnalyzer
        self.generational_clusters = None
        self.generational_trends = None
        
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

    def load_data(self, comment_files: List[str], video_file: str) -> None:
        """
        Load and combine comment and video data from CSV files.
        
        Args:
            comment_files (List[str]): List of comment CSV file paths
            video_file (str): Path to video CSV file
        """
        print("Loading data...")
        
        # Load and combine comment files
        comment_dfs = []
        for file in comment_files:
            try:
                df = pd.read_csv(file)
                comment_dfs.append(df)
                print(f"Loaded {len(df)} comments from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if comment_dfs:
            self.comments_df = pd.concat(comment_dfs, ignore_index=True)
            print(f"Total comments loaded: {len(self.comments_df)}")
        
        # Load video data
        try:
            self.videos_df = pd.read_csv(video_file)
            print(f"Loaded {len(self.videos_df)} videos from {video_file}")
            
            # Log video data structure for debugging
            print("Video data columns:", self.videos_df.columns.tolist())
            print("Sample video data:")
            print(self.videos_df.head())
            
            # Check for tags column
            if 'tags' in self.videos_df.columns:
                print("Tags column found - will integrate tags into analysis")
                tag_sample = self.videos_df['tags'].dropna().head(3)
                print("Sample tags:", tag_sample.tolist())
            else:
                print("Warning: No 'tags' column found in video data")
                
        except Exception as e:
            print(f"Error loading video file: {e}")

    def preprocess_data(self) -> None:
        """
        Clean and preprocess the loaded data including tag processing.
        Enhanced to handle video tags properly.
        """
        print("Preprocessing data with tag integration...")
        
        if self.comments_df is not None:
            # Convert timestamp columns
            self.comments_df['publishedAt'] = pd.to_datetime(self.comments_df['publishedAt'])
            self.comments_df['updatedAt'] = pd.to_datetime(self.comments_df['updatedAt'])
            
            # Clean text data
            self.comments_df['cleaned_text'] = self.comments_df['textOriginal'].apply(self._clean_text)
            
            # Remove empty or very short comments
            self.comments_df = self.comments_df[
                (self.comments_df['cleaned_text'].str.len() > 10) &
                (self.comments_df['cleaned_text'].notna())
            ].reset_index(drop=True)
            
            # Add time-based features
            self.comments_df['date'] = self.comments_df['publishedAt'].dt.date
            self.comments_df['hour'] = self.comments_df['publishedAt'].dt.hour
            self.comments_df['day_of_week'] = self.comments_df['publishedAt'].dt.dayofweek
            
            print(f"Comments after preprocessing: {len(self.comments_df)}")
        
        if self.videos_df is not None:
            self.videos_df['publishedAt'] = pd.to_datetime(self.videos_df['publishedAt'])
            self.videos_df['date'] = self.videos_df['publishedAt'].dt.date
            
            # Clean video descriptions and titles
            self.videos_df['cleaned_title'] = self.videos_df['title'].apply(self._clean_text)
            self.videos_df['cleaned_description'] = self.videos_df['description'].apply(self._clean_text)
            
            # Enhanced tag processing
            if 'tags' in self.videos_df.columns:
                self.videos_df['processed_tags'] = self.videos_df['tags'].apply(self._process_tags)
                self.videos_df['tag_count'] = self.videos_df['processed_tags'].apply(len)
                self.videos_df['tag_text'] = self.videos_df['processed_tags'].apply(
                    lambda x: ' '.join(x) if x else ''
                )
                
                # Analyze tag popularity
                self._analyze_tag_popularity()
                
                # Calculate tag relevance scores
                self._calculate_tag_relevance_scores()
            else:
                self.videos_df['processed_tags'] = [[] for _ in range(len(self.videos_df))]
                self.videos_df['tag_count'] = 0
                self.videos_df['tag_text'] = ''
                self.videos_df['tag_relevance_score'] = 0
            
            # Enhanced video metrics preprocessing
            # Convert string numbers to integers if needed
            numeric_columns = ['viewCount', 'likeCount', 'commentCount']
            for col in numeric_columns:
                if col in self.videos_df.columns:
                    self.videos_df[col] = pd.to_numeric(self.videos_df[col], errors='coerce').fillna(0)
            
            # Calculate video performance metrics
            self.videos_df['engagement_rate'] = (
                (self.videos_df.get('likeCount', 0) + self.videos_df.get('commentCount', 0)) / 
                np.maximum(self.videos_df.get('viewCount', 1), 1)
            )
            
            # Enhanced video trending score including tags
            max_views = self.videos_df.get('viewCount', pd.Series([1])).max()
            max_likes = self.videos_df.get('likeCount', pd.Series([1])).max()
            
            if max_views > 0 and max_likes > 0:
                self.videos_df['view_score'] = self.videos_df.get('viewCount', 0) / max_views
                self.videos_df['like_score'] = self.videos_df.get('likeCount', 0) / max_likes
                
                # Recency score (more recent videos get higher scores)
                current_time = pd.Timestamp.now().tz_localize(None)
                published_times = pd.to_datetime(self.videos_df['publishedAt']).dt.tz_localize(None)
                days_since_publish = (current_time - published_times).dt.days
                max_days = days_since_publish.max() if len(days_since_publish) > 0 else 1
                self.videos_df['recency_score'] = 1 - (days_since_publish / max(max_days, 1))
                
                # Enhanced trending score including tag relevance
                self.videos_df['trending_score'] = (
                    0.35 * self.videos_df['view_score'] +
                    0.35 * self.videos_df['like_score'] +
                    0.15 * self.videos_df['recency_score'] +
                    0.15 * self.videos_df.get('tag_relevance_score', 0)
                )
            else:
                self.videos_df['trending_score'] = 0
            
            print(f"Videos after preprocessing: {len(self.videos_df)}")
            print(f"Video trending scores range: {self.videos_df['trending_score'].min():.3f} - {self.videos_df['trending_score'].max():.3f}")
            
            if 'tags' in self.videos_df.columns:
                print(f"Average tags per video: {self.videos_df['tag_count'].mean():.1f}")
                print(f"Videos with tags: {(self.videos_df['tag_count'] > 0).sum()}/{len(self.videos_df)}")

    def _process_tags(self, tags) -> List[str]:
        """
        Process and clean video tags.
        
        Args:
            tags: Raw tags data (could be string, list, or None)
            
        Returns:
            List[str]: Processed list of tags
        """
        if pd.isna(tags) or tags == '' or tags is None:
            return []
        
        if isinstance(tags, str):
            # Handle different tag formats
            # Common separators: comma, semicolon, pipe
            if ',' in tags:
                tag_list = tags.split(',')
            elif ';' in tags:
                tag_list = tags.split(';')
            elif '|' in tags:
                tag_list = tags.split('|')
            else:
                # If no separator, treat as single tag or space-separated
                tag_list = [tags] if ' ' not in tags else tags.split()
        elif isinstance(tags, list):
            tag_list = tags
        else:
            # Convert to string and process
            tag_list = str(tags).split(',')
        
        # Clean and normalize tags
        processed_tags = []
        for tag in tag_list:
            if isinstance(tag, str):
                # Remove quotes and extra whitespace
                clean_tag = tag.strip().strip('"\'').lower()
                # Remove special characters but keep spaces and hyphens
                clean_tag = re.sub(r'[^\w\s\-]', '', clean_tag)
                if clean_tag and len(clean_tag) > 1:  # Keep tags with more than 1 character
                    processed_tags.append(clean_tag)
        
        return processed_tags

    def _analyze_tag_popularity(self) -> None:
        """
        Analyze tag popularity and trends over time.
        """
        print("Analyzing tag popularity and trends...")
        
        # Collect all tags with their video metadata
        all_tags = []
        for _, video in self.videos_df.iterrows():
            for tag in video['processed_tags']:
                all_tags.append({
                    'tag': tag,
                    'videoId': video['videoId'],
                    'date': video['date'],
                    'viewCount': video.get('viewCount', 0),
                    'likeCount': video.get('likeCount', 0),
                    'publishedAt': video['publishedAt']
                })
        
        if all_tags:
            tag_df = pd.DataFrame(all_tags)
            
            # Calculate tag popularity metrics
            self.popular_tags = tag_df.groupby('tag').agg({
                'videoId': 'count',     # frequency
                'viewCount': 'sum',     # total views
                'likeCount': 'sum'      # total likes
            }).rename(columns={
                'videoId': 'frequency',
                'viewCount': 'total_views',
                'likeCount': 'total_likes'
            })
            
            # Calculate tag trending score
            self.popular_tags['avg_views_per_video'] = (
                self.popular_tags['total_views'] / self.popular_tags['frequency']
            )
            self.popular_tags['avg_likes_per_video'] = (
                self.popular_tags['total_likes'] / self.popular_tags['frequency']
            )
            
            # Normalize scores
            max_freq = self.popular_tags['frequency'].max()
            max_views = self.popular_tags['total_views'].max()
            max_likes = self.popular_tags['total_likes'].max()
            
            if max_freq > 0 and max_views > 0 and max_likes > 0:
                self.popular_tags['frequency_score'] = self.popular_tags['frequency'] / max_freq
                self.popular_tags['views_score'] = self.popular_tags['total_views'] / max_views
                self.popular_tags['likes_score'] = self.popular_tags['total_likes'] / max_likes
                
                # Combined tag popularity score
                self.popular_tags['tag_popularity_score'] = (
                    0.4 * self.popular_tags['frequency_score'] +
                    0.4 * self.popular_tags['views_score'] +
                    0.2 * self.popular_tags['likes_score']
                )
            else:
                self.popular_tags['tag_popularity_score'] = 0
            
            # Sort by popularity
            self.popular_tags = self.popular_tags.sort_values('tag_popularity_score', ascending=False)
            
            print(f"Analyzed {len(self.popular_tags)} unique tags")
            print("Top 10 most popular tags:")
            for tag, data in self.popular_tags.head(10).iterrows():
                print(f"  {tag}: {data['frequency']} videos, {data['total_views']:,} total views")
        else:
            self.popular_tags = pd.DataFrame()

    def _calculate_tag_relevance_scores(self) -> None:
        """
        Calculate relevance scores for videos based on their tags.
        """
        if self.popular_tags is not None and not self.popular_tags.empty:
            tag_scores = self.popular_tags['tag_popularity_score'].to_dict()
            
            def calculate_video_tag_score(tags):
                if not tags:
                    return 0
                scores = [tag_scores.get(tag, 0) for tag in tags]
                return np.mean(scores) if scores else 0
            
            self.videos_df['tag_relevance_score'] = self.videos_df['processed_tags'].apply(
                calculate_video_tag_score
            )
        else:
            self.videos_df['tag_relevance_score'] = 0

    def _clean_text(self, text: str) -> str:
        """
        Clean text while preserving emojis and meaningful slang.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs but keep the text structure
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation but keep some for sentiment
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!!', text)
        text = re.sub(r'[?]{2,}', '??', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep emojis and basic punctuation
        # Remove only clearly problematic characters
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF,.!?;:\'"()-]', '', text)
        
        return text.strip()

    def generate_embeddings(self, text_column: str = 'cleaned_text') -> np.ndarray:
        """
        Generate SBERT embeddings for the comment text.
        Enhanced to include video titles, descriptions, and tags.
        
        Args:
            text_column (str): Column name containing text to embed
            
        Returns:
            np.ndarray: Matrix of embeddings
        """
        print("Generating SBERT embeddings with tag integration...")
        
        if self.comments_df is None:
            raise ValueError("Comments data not loaded")
        
        texts = self.comments_df[text_column].tolist()
        self.embeddings = self.sbert_model.encode(texts, show_progress_bar=True)
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def perform_clustering(self, n_clusters: int = 20, niter: int = 50, verbose: bool = True) -> np.ndarray:
        """
        Perform FAISS K-Means clustering on the embeddings.
        
        Args:
            n_clusters (int): Number of clusters to create
            niter (int): Number of iterations for K-means
            verbose (bool): Whether to print progress
            
        Returns:
            np.ndarray: Cluster labels
        """
        print(f"Performing FAISS K-Means clustering with {n_clusters} clusters...")
        
        if self.embeddings is None:
            raise ValueError("Embeddings not generated")
        
        # Ensure embeddings are in the right format for FAISS (float32)
        embeddings_float32 = self.embeddings.astype(np.float32)
        
        # Get embedding dimension
        d = embeddings_float32.shape[1]
        
        # Initialize FAISS K-means
        self.kmeans_index = faiss.Kmeans(
            d=d,
            k=n_clusters,
            niter=niter,
            verbose=verbose,
            spherical=False,  # use Euclidean distance (not cosine)
            gpu=False  # Set to False for compatibility
        )
        
        # Train the K-means model
        print("Training K-means...")
        self.kmeans_index.train(embeddings_float32)
        
        # Get cluster assignments
        _, cluster_assignments = self.kmeans_index.index.search(embeddings_float32, 1)
        self.clusters = cluster_assignments.flatten()
        
        # Add cluster labels to dataframe
        self.comments_df['cluster'] = self.clusters
        
        n_clusters_found = len(set(self.clusters))
        print(f"Found {n_clusters_found} clusters")
        print(f"Cluster distribution:")
        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} comments")
        
        return self.clusters

    def map_videos_to_clusters(self) -> None:
        """
        Map videos to comment clusters based on content similarity including tags.
        This creates the link between video performance and comment topics.
        """
        print("Mapping videos to comment clusters (including tags)...")
        
        if self.videos_df is None or self.comments_df is None:
            raise ValueError("Both video and comment data must be loaded")
        
        # Generate embeddings for video content including tags
        video_texts = []
        for _, video in self.videos_df.iterrows():
            # Combine title, description, and tags for better topic matching
            # Give tags more weight by repeating them
            tag_text = ' '.join(video.get('processed_tags', [])) * 2  # Double weight for tags
            combined_text = f"{video.get('cleaned_title', '')} {video.get('cleaned_description', '')} {tag_text}"
            video_texts.append(combined_text if combined_text.strip() else video.get('title', ''))
        
        if not video_texts:
            print("No video text found for clustering")
            return
        
        # Generate embeddings for videos
        video_embeddings = self.sbert_model.encode(video_texts, show_progress_bar=True)
        
        # Find closest cluster for each video
        video_embeddings_float32 = video_embeddings.astype(np.float32)
        _, video_cluster_assignments = self.kmeans_index.index.search(video_embeddings_float32, 1)
        
        # Add cluster assignments to videos dataframe
        self.videos_df['cluster'] = video_cluster_assignments.flatten()
        
        print("Video-to-cluster mapping completed")
        cluster_video_counts = self.videos_df['cluster'].value_counts().sort_index()
        print("Videos per cluster:")
        for cluster_id, count in cluster_video_counts.items():
            print(f"  Cluster {cluster_id}: {count} videos")

    def analyze_sentiment(self) -> None:
        """
        Perform sentiment analysis on comments.
        """
        print("Analyzing sentiment...")
        
        def get_sentiment_scores(text):
            """Get sentiment scores using VADER."""
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound'], scores['pos'], scores['neu'], scores['neg']
        
        # Apply sentiment analysis
        sentiment_data = self.comments_df['cleaned_text'].apply(get_sentiment_scores)
        sentiment_df = pd.DataFrame(sentiment_data.tolist(), 
                                  columns=['compound', 'positive', 'neutral', 'negative'])
        
        # Add sentiment columns to main dataframe
        self.comments_df = pd.concat([self.comments_df, sentiment_df], axis=1)
        
        # Create sentiment categories
        self.comments_df['sentiment_label'] = pd.cut(
            self.comments_df['compound'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        print("Sentiment analysis completed")

    def extract_cluster_topics(self, top_n_words: int = 10) -> Dict[int, List[str]]:
        """
        Extract representative topics for each cluster including tags.
        Enhanced to heavily weight video titles and tags for better topic identification.
        
        Args:
            top_n_words (int): Number of top words to extract per cluster
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster IDs to top words and tags
        """
        print("Extracting cluster topics with tag integration...")
        
        cluster_topics = {}
        cluster_tags = {}
        
        for cluster_id in sorted(set(self.clusters)):
            # Get comment texts for this cluster
            cluster_comments = self.comments_df[
                self.comments_df['cluster'] == cluster_id
            ]['cleaned_text'].tolist()
            
            # Get video titles and tags for this cluster
            cluster_videos = self.videos_df[
                self.videos_df['cluster'] == cluster_id
            ] if 'cluster' in self.videos_df.columns else pd.DataFrame()
            
            video_titles = cluster_videos['cleaned_title'].tolist() if not cluster_videos.empty else []
            
            # Collect tags for this cluster
            cluster_video_tags = []
            if not cluster_videos.empty:
                for _, video in cluster_videos.iterrows():
                    cluster_video_tags.extend(video.get('processed_tags', []))
            
            # Analyze tag frequency for this cluster
            if cluster_video_tags:
                tag_counter = Counter(cluster_video_tags)
                top_cluster_tags = [tag for tag, count in tag_counter.most_common(5)]
                cluster_tags[cluster_id] = top_cluster_tags
            else:
                cluster_tags[cluster_id] = []
            
            # Combine texts with heavy weighting for titles and tags
            all_texts = (
                cluster_comments + 
                video_titles * 4 +      # 4x weight for video titles
                cluster_video_tags * 3  # 3x weight for tags
            )
            
            all_text = ' '.join(all_texts).lower()
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
            
            # Enhanced stop words list
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was',
                'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new',
                'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way', 'she', 'too', 'any',
                'use', 'your', 'here', 'this', 'that', 'with', 'have', 'from', 'they', 'know', 'want',
                'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'could', 'like', 'will',
                'said', 'would', 'make', 'just', 'into', 'over', 'think', 'also', 'back', 'after',
                'first', 'well', 'work', 'life', 'only', 'look', 'year', 'more', 'where', 'what',
                'than', 'love', 'really', 'great', 'video', 'youtube', 'channel', 'subscribe', 'comment',
                'please', 'thank', 'thanks', 'watch', 'follow', 'instagram'
            }
            
            filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
            
            # Count word frequencies
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
            cluster_topics[cluster_id] = [word for word, count in top_words]
        
        self.cluster_topics = cluster_topics
        self.cluster_tags = cluster_tags
        return cluster_topics

    def analyze_generational_patterns(self) -> None:
        """
        Analyze generational language patterns in comments and videos.
        """
        print("Analyzing generational language patterns...")
        
        # Initialize the analyzer if not already done
        if not hasattr(self, 'generational_analyzer'):
            self.generational_analyzer = EnhancedGenerationalLanguageAnalyzer
        
        if self.comments_df is not None:
            print("Analyzing comments for generational patterns...")
            # Analyze comments
            self.comments_df['generational_scores'] = self.comments_df['cleaned_text'].apply(
                self.generational_analyzer.analyze_generational_language
            )
            
            # Extract individual generation scores
            for generation in ['gen_z', 'millennial', 'gen_x', 'boomer']:
                self.comments_df[f'{generation}_score'] = self.comments_df['generational_scores'].apply(
                    lambda x: x.get(generation, 0) if isinstance(x, dict) else 0
                )
            
            # Classify predominant generation
            self.comments_df['dominant_generation'] = self.comments_df['cleaned_text'].apply(
                self.generational_analyzer.classify_generation
            )
            
            print("Comments generational analysis completed")
        
        if self.videos_df is not None:
            print("Analyzing videos for generational patterns...")
            # Analyze video titles and descriptions
            combined_video_text = (
                self.videos_df.get('cleaned_title', '').fillna('') + ' ' + 
                self.videos_df.get('cleaned_description', '').fillna('')
            )
            
            self.videos_df['generational_scores'] = combined_video_text.apply(
                self.generational_analyzer.analyze_generational_language
            )
            
            # Extract individual generation scores
            for generation in ['gen_z', 'millennial', 'gen_x', 'boomer']:
                self.videos_df[f'{generation}_score'] = self.videos_df['generational_scores'].apply(
                    lambda x: x.get(generation, 0) if isinstance(x, dict) else 0
                )
            
            self.videos_df['dominant_generation'] = combined_video_text.apply(
                self.generational_analyzer.classify_generation
            )
            
            print("Videos generational analysis completed")

    
    def analyze_generational_trends_by_cluster(self) -> Dict:
        """
        Analyze generational trends for each cluster.
        
        Returns:
            Dict: Generational analysis by cluster
        """
        if self.clusters is None or self.comments_df is None:
            print("Clustering must be performed first")
            return {}
        
        generational_cluster_analysis = {}
        
        for cluster_id in sorted(set(self.clusters)):
            cluster_comments = self.comments_df[self.comments_df['cluster'] == cluster_id]
            cluster_videos = self.videos_df[self.videos_df['cluster'] == cluster_id] if 'cluster' in self.videos_df.columns else pd.DataFrame()
            
            if len(cluster_comments) == 0:
                continue
            
            # Analyze generational distribution in comments
            generation_distribution = cluster_comments['dominant_generation'].value_counts(normalize=True)
            
            # Calculate average generational scores
            avg_gen_scores = {}
            for generation in ['gen_z', 'millennial', 'gen_x', 'boomer']:
                avg_gen_scores[generation] = cluster_comments[f'{generation}_score'].mean()
            
            # Identify dominant generation for this cluster
            dominant_gen = max(avg_gen_scores.items(), key=lambda x: x[1])
            
            # Analyze video performance by generation
            video_performance_by_gen = {}
            if not cluster_videos.empty:
                for generation in ['gen_z', 'millennial', 'gen_x', 'boomer', 'neutral']:
                    gen_videos = cluster_videos[cluster_videos['dominant_generation'] == generation]
                    if len(gen_videos) > 0:
                        video_performance_by_gen[generation] = {
                            'count': len(gen_videos),
                            'avg_views': gen_videos['viewCount'].mean(),
                            'avg_likes': gen_videos['likeCount'].mean(),
                            'avg_engagement': gen_videos['engagement_rate'].mean()
                        }
            
            generational_cluster_analysis[cluster_id] = {
                'topic_words': self.cluster_topics.get(cluster_id, []),
                'topic_tags': self.cluster_tags.get(cluster_id, []),
                'dominant_generation': dominant_gen[0],
                'dominant_generation_score': dominant_gen[1],
                'generation_distribution': generation_distribution.to_dict(),
                'avg_generational_scores': avg_gen_scores,
                'video_performance_by_generation': video_performance_by_gen,
                'total_comments': len(cluster_comments),
                'total_videos': len(cluster_videos)
            }
        
        self.generational_clusters = generational_cluster_analysis
        return generational_cluster_analysis

    def prepare_time_series_data(self, time_window: str = 'D') -> pd.DataFrame:
        """
        Prepare enhanced time series data combining video, comment, and tag metrics.
        
        Args:
            time_window (str): Time aggregation window ('D' for daily, 'W' for weekly)
            
        Returns:
            pd.DataFrame: Enhanced time series data including tag metrics
        """
        print(f"Preparing enhanced time series data with tags and {time_window} aggregation...")
        
        # Create comment-based time series
        comment_time_series = []
        for cluster_id in sorted(set(self.clusters)):
            cluster_data = self.comments_df[self.comments_df['cluster'] == cluster_id]
            
            # Aggregate by time window
            time_series = cluster_data.groupby(
                pd.Grouper(key='publishedAt', freq=time_window)
            ).agg({
                'commentId': 'count',
                'likeCount': 'sum',
                'compound': 'mean',
                'positive': 'mean',
                'negative': 'mean'
            }).reset_index()
            
            time_series['cluster'] = cluster_id
            time_series.columns = ['date', 'comment_count', 'comment_likes', 
                                 'avg_sentiment', 'avg_positive', 'avg_negative', 'cluster']
            comment_time_series.append(time_series)
        
        comment_trend_data = pd.concat(comment_time_series, ignore_index=True) if comment_time_series else pd.DataFrame()
        
        # Create enhanced video-based time series including tag metrics
        video_time_series = []
        if 'cluster' in self.videos_df.columns:
            for cluster_id in sorted(set(self.clusters)):
                cluster_videos = self.videos_df[self.videos_df['cluster'] == cluster_id]
                
                if len(cluster_videos) > 0:
                    # Aggregate video metrics by time window
                    video_ts = cluster_videos.groupby(
                        pd.Grouper(key='publishedAt', freq=time_window)
                    ).agg({
                        'videoId': 'count',
                        'viewCount': 'sum',
                        'likeCount': 'sum',
                        'commentCount': 'sum',
                        'trending_score': 'mean',
                        'engagement_rate': 'mean',
                        'tag_count': 'mean',
                        'tag_relevance_score': 'mean'
                    }).reset_index()
                    
                    video_ts['cluster'] = cluster_id
                    video_ts.columns = ['date', 'video_count', 'total_views', 'video_likes',
                                      'video_comments', 'avg_trending_score', 'avg_engagement_rate',
                                      'avg_tag_count', 'avg_tag_relevance', 'cluster']
                    video_time_series.append(video_ts)
        
        video_trend_data = pd.concat(video_time_series, ignore_index=True) if video_time_series else pd.DataFrame()
        
        # Combine comment and video data
        if not comment_trend_data.empty and not video_trend_data.empty:
            # Merge on date and cluster
            combined_data = pd.merge(
                comment_trend_data,
                video_trend_data,
                on=['date', 'cluster'],
                how='outer'
            ).fillna(0)
        elif not comment_trend_data.empty:
            combined_data = comment_trend_data.copy()
            # Add empty video columns
            video_cols = ['video_count', 'total_views', 'video_likes', 'video_comments',
                         'avg_trending_score', 'avg_engagement_rate', 'avg_tag_count', 'avg_tag_relevance']
            for col in video_cols:
                combined_data[col] = 0
        else:
            combined_data = pd.DataFrame()
        
        if not combined_data.empty:
            # Enhanced combined trending score with tag integration
            combined_data['combined_trending_score'] = (
                self.video_weight * (
                    combined_data['avg_trending_score'] * 0.4 +
                    (combined_data['total_views'] / combined_data['total_views'].max() 
                     if combined_data['total_views'].max() > 0 else 0) * 0.3 +
                    (combined_data['video_likes'] / combined_data['video_likes'].max() 
                     if combined_data['video_likes'].max() > 0 else 0) * 0.3
                ) +
                self.comment_weight * (
                    (combined_data['comment_count'] / combined_data['comment_count'].max() 
                     if combined_data['comment_count'].max() > 0 else 0) * 0.5 +
                    combined_data['avg_sentiment'] * 0.3 +
                    (combined_data['comment_likes'] / combined_data['comment_likes'].max() 
                     if combined_data['comment_likes'].max() > 0 else 0) * 0.2
                ) +
                self.tag_weight * (
                    combined_data['avg_tag_relevance'] * 0.7 +
                    (combined_data['avg_tag_count'] / combined_data['avg_tag_count'].max() 
                     if combined_data['avg_tag_count'].max() > 0 else 0) * 0.3
                )
            )
            
            # Fill missing dates with zero values
            date_range = pd.date_range(
                start=combined_data['date'].min(),
                end=combined_data['date'].max(),
                freq=time_window
            )
            
            complete_data = []
            for cluster_id in combined_data['cluster'].unique():
                cluster_df = pd.DataFrame({'date': date_range, 'cluster': cluster_id})
                cluster_trend = combined_data[combined_data['cluster'] == cluster_id]
                merged = cluster_df.merge(cluster_trend, on=['date', 'cluster'], how='left')
                merged = merged.fillna(0)
                complete_data.append(merged)
            
            self.combined_trend_data = pd.concat(complete_data, ignore_index=True)
        else:
            self.combined_trend_data = pd.DataFrame()
        
        return self.combined_trend_data

    def identify_trending_topics(self, window_days: int = 30, growth_threshold: float = 0.05) -> List[Dict]:
        """
        Enhanced trending topic identification considering video performance and tag relevance.
        
        Args:
            window_days (int): Number of recent days to analyze
            growth_threshold (float): Minimum growth rate to consider trending
            
        Returns:
            List[Dict]: List of trending topics with comprehensive metadata including tags
        """
        print("Identifying trending topics with video and tag emphasis...")
        
        if self.combined_trend_data.empty:
            print("No combined trend data available")
            return []
        
        trending_topics = []
        recent_date = self.combined_trend_data['date'].max() - timedelta(days=window_days)
        
        for cluster_id in self.combined_trend_data['cluster'].unique():
            cluster_data = self.combined_trend_data[self.combined_trend_data['cluster'] == cluster_id]
            recent_data = cluster_data[cluster_data['date'] >= recent_date]
            older_data = cluster_data[cluster_data['date'] < recent_date]
            
            if len(recent_data) == 0 or len(older_data) == 0:
                continue
            
            # Calculate growth for different metrics including tags
            metrics = {
                'combined_score': 'combined_trending_score',
                'video_views': 'total_views',
                'video_likes': 'video_likes',
                'comment_count': 'comment_count',
                'tag_relevance': 'avg_tag_relevance'
            }
            
            growth_rates = {}
            for metric_name, metric_col in metrics.items():
                if metric_col in recent_data.columns:
                    recent_avg = recent_data[metric_col].mean()
                    older_avg = older_data[metric_col].mean()
                    
                    if older_avg > 0:
                        growth_rates[metric_name] = (recent_avg - older_avg) / older_avg
                    else:
                        growth_rates[metric_name] = float('inf') if recent_avg > 0 else 0
                else:
                    growth_rates[metric_name] = 0
            
            # Use combined score as primary growth indicator
            primary_growth = growth_rates.get('combined_score', 0)
            
            if primary_growth >= growth_threshold:
                topic_words = self.cluster_topics.get(cluster_id, [])
                topic_tags = self.cluster_tags.get(cluster_id, [])
                
                # Get recent video performance
                recent_videos = self.videos_df[
                    (self.videos_df['cluster'] == cluster_id) &
                    (self.videos_df['date'] >= recent_date.date())
                ] if 'cluster' in self.videos_df.columns else pd.DataFrame()
                
                trending_topics.append({
                    'cluster_id': int(cluster_id),
                    'topic_words': topic_words,
                    'topic_tags': topic_tags,
                    'combined_growth_rate': primary_growth,
                    'video_views_growth': growth_rates.get('video_views', 0),
                    'video_likes_growth': growth_rates.get('video_likes', 0),
                    'comment_growth': growth_rates.get('comment_count', 0),
                    'tag_relevance_growth': growth_rates.get('tag_relevance', 0),
                    'recent_video_count': len(recent_videos),
                    'recent_total_views': int(recent_data['total_views'].sum()),
                    'recent_video_likes': int(recent_data['video_likes'].sum()),
                    'recent_comment_count': int(recent_data['comment_count'].sum()),
                    'avg_sentiment': float(recent_data['avg_sentiment'].mean()),
                    'avg_trending_score': float(recent_data['avg_trending_score'].mean()),
                    'avg_engagement_rate': float(recent_data['avg_engagement_rate'].mean()),
                    'avg_tag_count': float(recent_data['avg_tag_count'].mean()),
                    'avg_tag_relevance': float(recent_data['avg_tag_relevance'].mean()),
                    'trending_category': self._categorize_trend(primary_growth, growth_rates)
                })
        
        # Sort by combined growth rate (video and tag weighted)
        trending_topics.sort(key=lambda x: x['combined_growth_rate'], reverse=True)
        return trending_topics

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


    def _categorize_trend(self, primary_growth: float, growth_rates: Dict) -> str:
        """
        Categorize the type of trend based on growth patterns including tags.
        
        Args:
            primary_growth (float): Primary growth rate
            growth_rates (Dict): Dictionary of growth rates for different metrics
            
        Returns:
            str: Trend category
        """
        video_growth = max(growth_rates.get('video_views', 0), growth_rates.get('video_likes', 0))
        comment_growth = growth_rates.get('comment_count', 0)
        tag_growth = growth_rates.get('tag_relevance', 0)
        
        if video_growth > 0.3 and comment_growth > 0.2 and tag_growth > 0.2:
            return "viral"
        elif video_growth > 0.2 and tag_growth > 0.15:
            return "video_trending"
        elif comment_growth > 0.2:
            return "discussion_trending"
        elif tag_growth > 0.2:
            return "tag_trending"
        elif primary_growth > 0.1:
            return "emerging"
        else:
            return "stable_growth"

    def build_enhanced_lstm_model(self, sequence_length: int = 7) -> None:
        """
        Build enhanced LSTM model that includes video metrics and tag features.
        
        Args:
            sequence_length (int): Number of time steps to look back
        """
        print("Building enhanced LSTM model with video and tag features...")
        
        # Enhanced feature set including tag metrics
        self.features = [
            'comment_count', 'comment_likes', 'avg_sentiment', 'video_count',
            'total_views', 'video_likes', 'avg_trending_score', 'avg_engagement_rate',
            'avg_tag_count', 'avg_tag_relevance', 'combined_trending_score'
        ]
        self.sequence_length = sequence_length
        
        # Build enhanced model architecture
        self.lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, len(self.features))),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("Enhanced LSTM model with tag features built successfully")

    def analyze_tag_trends(self) -> Dict:
        """
        Analyze trending tags and their evolution over time.
        
        Returns:
            Dict: Tag trend analysis results
        """
        print("Analyzing tag trends over time...")
        
        if self.popular_tags is None or self.popular_tags.empty:
            return {}
        
        # Get tag trends over time
        tag_time_series = []
        
        # Create time series for each popular tag
        for tag in self.popular_tags.head(20).index:  # Top 20 tags
            tag_videos = []
            for _, video in self.videos_df.iterrows():
                if tag in video.get('processed_tags', []):
                    tag_videos.append({
                        'date': video['date'],
                        'viewCount': video.get('viewCount', 0),
                        'likeCount': video.get('likeCount', 0),
                        'tag': tag
                    })
            
            if tag_videos:
                tag_df = pd.DataFrame(tag_videos)
                tag_ts = tag_df.groupby('date').agg({
                    'viewCount': 'sum',
                    'likeCount': 'sum'
                }).reset_index()
                tag_ts['tag'] = tag
                tag_time_series.append(tag_ts)
        
        tag_trends = {}
        if tag_time_series:
            # Calculate growth rates for tags
            for tag_data in tag_time_series:
                tag = tag_data['tag'].iloc[0]
                if len(tag_data) >= 2:
                    recent_views = tag_data['viewCount'].tail(7).mean()  # Last week average
                    older_views = tag_data['viewCount'].head(7).mean()   # First week average
                    
                    if older_views > 0:
                        growth_rate = (recent_views - older_views) / older_views
                    else:
                        growth_rate = float('inf') if recent_views > 0 else 0
                    
                    tag_trends[tag] = {
                        'growth_rate': growth_rate,
                        'total_views': tag_data['viewCount'].sum(),
                        'total_likes': tag_data['likeCount'].sum(),
                        'video_count': self.popular_tags.loc[tag, 'frequency'],
                        'popularity_score': self.popular_tags.loc[tag, 'tag_popularity_score']
                    }
        
        # Sort by growth rate
        sorted_tag_trends = dict(sorted(tag_trends.items(), key=lambda x: x[1]['growth_rate'], reverse=True))
        return sorted_tag_trends

    def analyze_generational_trends_by_cluster(self) -> None:
        """
        Analyze generational trends for each cluster.
        """
        if self.comments_df is None or 'cluster' not in self.comments_df.columns:
            print("Comment data or clustering results not available")
            return
        
        print("Analyzing generational trends by cluster...")
        
        # First, ensure we have the generational analyzer
        if not hasattr(self, 'generational_analyzer'):
            self.generational_analyzer = EnhancedGenerationalLanguageAnalyzer
        
        # Perform generational analysis if not already done
        if 'dominant_generation' not in self.comments_df.columns:
            print("Performing generational language analysis...")
            self.analyze_generational_patterns()
        
        # Ensure we have the required columns
        required_columns = ['dominant_generation', 'gen_z_score', 'millennial_score', 'gen_x_score', 'boomer_score']
        missing_columns = [col for col in required_columns if col not in self.comments_df.columns]
        
        if missing_columns:
            print(f"Missing columns: {missing_columns}. Performing generational analysis...")
            self.analyze_generational_patterns()
        
        clusters = self.comments_df['cluster'].unique()
        self.generational_clusters = {}
        
        for cluster_id in clusters:
            cluster_comments = self.comments_df[self.comments_df['cluster'] == cluster_id].copy()
            
            if len(cluster_comments) < 10:  # Skip small clusters
                continue
            
            try:
                # Generational distribution analysis
                generation_distribution = cluster_comments['dominant_generation'].value_counts(normalize=True)
                
                if len(generation_distribution) == 0:
                    continue
                    
                dominant_generation = generation_distribution.index[0]
                dominant_generation_score = generation_distribution.iloc[0]
                
                # Calculate average generational scores for this cluster
                avg_generational_scores = {
                    'gen_z': cluster_comments['gen_z_score'].mean(),
                    'millennial': cluster_comments['millennial_score'].mean(),
                    'gen_x': cluster_comments['gen_x_score'].mean(),
                    'boomer': cluster_comments['boomer_score'].mean()
                }
                
                # Get topic information
                topic_words = self.cluster_topics.get(cluster_id, [])[:10]
                topic_tags = self.cluster_tags.get(cluster_id, [])[:5]
                
                # Video performance analysis by generation
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
                    'dominant_generation_score': dominant_generation_score,
                    'generation_distribution': generation_distribution.to_dict(),
                    'avg_generational_scores': avg_generational_scores,
                    'topic_words': topic_words,
                    'topic_tags': topic_tags,
                    'video_performance_by_generation': video_performance_by_generation,
                    'cluster_size': len(cluster_comments)
                }
                
            except Exception as e:
                print(f"Error analyzing cluster {cluster_id}: {e}")
                continue
        
        print(f"Generational analysis completed for {len(self.generational_clusters)} clusters")

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


    def generate_enhanced_trend_report(self) -> Dict:
        """
        Generate comprehensive trend analysis report with video and tag emphasis.
        
        Returns:
            Dict: Enhanced trend analysis report including tag insights
        """
        print("Generating enhanced trend analysis report with tag integration...")
        
        # Get trending topics with video and tag emphasis
        trending_topics = self.identify_trending_topics()
        
        # Analyze tag trends
        tag_trends = self.analyze_tag_trends()
        
        # Calculate overall statistics
        total_comments = len(self.comments_df) if self.comments_df is not None else 0
        total_videos = len(self.videos_df) if self.videos_df is not None else 0
        total_clusters = len(set(self.clusters)) if self.clusters is not None else 0
        total_unique_tags = len(self.popular_tags) if self.popular_tags is not None else 0
        
        # Video performance statistics
        if self.videos_df is not None and not self.videos_df.empty:
            total_views = self.videos_df.get('viewCount', pd.Series([0])).sum()
            total_video_likes = self.videos_df.get('likeCount', pd.Series([0])).sum()
            avg_engagement_rate = self.videos_df.get('engagement_rate', pd.Series([0])).mean()
            avg_tags_per_video = self.videos_df.get('tag_count', pd.Series([0])).mean()
            
            top_performing_videos = self.videos_df.nlargest(5, 'trending_score')[
                ['title', 'viewCount', 'likeCount', 'trending_score', 'processed_tags']
            ].to_dict('records')
        else:
            total_views = 0
            total_video_likes = 0
            avg_engagement_rate = 0
            avg_tags_per_video = 0
            top_performing_videos = []
        
        # Comment statistics
        if self.comments_df is not None and not self.comments_df.empty:
            avg_sentiment = self.comments_df['compound'].mean()
        else:
            avg_sentiment = 0
        
        # Get most engaging clusters (video and tag weighted)
        if not self.combined_trend_data.empty:
            cluster_performance = self.combined_trend_data.groupby('cluster').agg({
                'combined_trending_score': 'sum',
                'total_views': 'sum',
                'video_likes': 'sum',
                'comment_count': 'sum',
                'avg_sentiment': 'mean',
                'avg_tag_relevance': 'mean'
            }).sort_values('combined_trending_score', ascending=False)
        else:
            cluster_performance = pd.DataFrame()
        
        report = {
            'analysis_summary': {
                'total_comments_analyzed': total_comments,
                'total_videos_analyzed': total_videos,
                'total_topics_identified': total_clusters,
                'total_unique_tags': total_unique_tags,
                'total_video_views': int(total_views),
                'total_video_likes': int(total_video_likes),
                'avg_video_engagement_rate': float(avg_engagement_rate),
                'avg_tags_per_video': float(avg_tags_per_video),
                'overall_sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                'video_weight_factor': self.video_weight,
                'comment_weight_factor': self.comment_weight,
                'tag_weight_factor': self.tag_weight,
                'analysis_period': {
                    'start_date': str(self.comments_df['publishedAt'].min().date()) if self.comments_df is not None and not self.comments_df.empty else 'N/A',
                    'end_date': str(self.comments_df['publishedAt'].max().date()) if self.comments_df is not None and not self.comments_df.empty else 'N/A'
                }
            },
            'trending_topics': trending_topics[:10],  # Top 10 trending with video and tag emphasis
            'trending_tags': [
                {
                    'tag': tag,
                    'growth_rate': data['growth_rate'],
                    'total_views': data['total_views'],
                    'video_count': data['video_count'],
                    'popularity_score': data['popularity_score']
                }
                for tag, data in list(tag_trends.items())[:10]
            ],
            'top_performing_videos': top_performing_videos,
            'top_engaging_clusters': [
                {
                    'cluster_id': int(cluster_id),
                    'topic_words': self.cluster_topics.get(cluster_id, [])[:5],
                    'topic_tags': self.cluster_tags.get(cluster_id, [])[:3],
                    'combined_score': float(row['combined_trending_score']),
                    'total_views': int(row['total_views']),
                    'total_video_likes': int(row['video_likes']),
                    'total_comments': int(row['comment_count']),
                    'avg_sentiment': float(row['avg_sentiment']),
                    'avg_tag_relevance': float(row['avg_tag_relevance'])
                }
                for cluster_id, row in cluster_performance.head(10).iterrows()
            ] if not cluster_performance.empty else [],
            'video_performance_insights': {
                'highest_engagement_cluster': int(cluster_performance.index[0]) if not cluster_performance.empty else None,
                'most_viewed_topic': self.cluster_topics.get(int(cluster_performance.index[0]), [])[:3] if not cluster_performance.empty else [],
                'most_viewed_tags': self.cluster_tags.get(int(cluster_performance.index[0]), [])[:3] if not cluster_performance.empty else [],
                'trend_categories': {
                    category: len([t for t in trending_topics if t.get('trending_category') == category])
                    for category in ['viral', 'video_trending', 'discussion_trending', 'tag_trending', 'emerging', 'stable_growth']
                }
            },
            'tag_insights': {
                'most_popular_tags': list(self.popular_tags.head(10).index) if self.popular_tags is not None else [],
                'fastest_growing_tags': list(tag_trends.keys())[:5],
                'tag_coverage': (self.videos_df['tag_count'] > 0).sum() / len(self.videos_df) if self.videos_df is not None and not self.videos_df.empty else 0
            }
        }
        
         # Add generational analysis
        if not hasattr(self, 'generational_trends') or self.generational_trends is None:
            self.identify_generational_trending_topics()
        
        # Add generational insights to the report
        report['generational_insights'] = {
            'trending_by_generation': {},
            'generation_distribution': {},
            'top_generational_topics': {},
            'generational_sentiment': {}
        }
        
        for generation, trends in self.generational_trends.items():
            if trends:
                report['generational_insights']['trending_by_generation'][generation] = trends[:5]
                report['generational_insights']['top_generational_topics'][generation] = [
                    {
                        'topic_words': trend['topic_words'],
                        'topic_tags': trend['topic_tags'],
                        'comment_volume': trend['recent_comment_volume'],
                        'sentiment': trend['avg_sentiment']
                    }
                    for trend in trends[:3]
                ]
        
        return report

    def run_enhanced_pipeline(self, comment_files: List[str], video_file: str, 
                            n_clusters: int = None, forecast_days: int = 30) -> Dict:
        """
        Run the complete enhanced trend analysis pipeline with video and tag emphasis.
        
        Args:
            comment_files (List[str]): List of comment CSV files
            video_file (str): Video CSV file path
            n_clusters (int): Number of clusters (if None, will optimize)
            forecast_days (int): Number of days to forecast
            
        Returns:
            Dict: Complete enhanced analysis results including tag insights
        """
        print("Starting enhanced trend analysis pipeline with video and tag emphasis...")
        
        # Step 1: Load and preprocess data (including tags)
        self.load_data(comment_files, video_file)
        self.preprocess_data()
        
        # Step 2: Generate embeddings
        self.generate_embeddings()
        
        # Step 3: Optimize cluster number if not provided
        if n_clusters is None:
            n_clusters = self.optimize_cluster_number()
        
        # Step 4: Perform clustering
        self.perform_clustering(n_clusters=n_clusters)
        
        # Step 5: Map videos to clusters (including tags)
        self.map_videos_to_clusters()
        
        # Step 6: Analyze sentiment
        self.analyze_sentiment()
        
        # Step 7: Extract topics (enhanced with video titles and tags)
        self.extract_cluster_topics()
        
        # Step 8: Prepare enhanced time series data (including tag metrics)
        self.prepare_time_series_data()
        
        # Step 9: Build enhanced LSTM model (including tag features)
        if not self.combined_trend_data.empty:
            self.build_enhanced_lstm_model()
        
        # Step 10: Generate enhanced visualizations
        self.visualize_enhanced_trends()
        
        # Step 11: Generate comprehensive report
        report = self.generate_enhanced_trend_report()
        
        print("Enhanced pipeline completed successfully!")
        print(f"Video weight factor: {self.video_weight}")
        print(f"Comment weight factor: {self.comment_weight}")
        print(f"Tag weight factor: {self.tag_weight}")
        
        return report

    def plot_generational_growth_by_clusters(self, generation: str = 'gen_z', 
                                       forecast_horizons: List[int] = [20, 30, 60]) -> None:
        """
        Plot growth rates for a specific generation across clusters and forecast horizons.
        
        Args:
            generation: Target generation ('gen_z', 'millennial', 'gen_x', 'boomer')
            forecast_horizons: List of forecast horizons to analyze
        """
        if not hasattr(self, 'generational_clusters') or not self.generational_clusters:
            print("No generational cluster data available. Running analysis...")
            self.analyze_generational_trends_by_cluster()
        
        # Get clusters dominated by the specified generation
        generation_clusters = []
        for cluster_id, analysis in self.generational_clusters.items():
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
        if not hasattr(self, 'multi_horizon_results'):
            print("Generating multi-horizon forecasts...")
            self.multi_horizon_results = self.generate_multi_horizon_forecasts(forecast_horizons)
        
        for idx, horizon in enumerate(forecast_horizons):
            ax = axes[idx]
            
            cluster_names = []
            growth_rates = []
            colors = []
            
            for cluster_data in top_clusters:
                cluster_id = cluster_data['cluster_id']
                
                # Get growth rate from forecast results
                if (hasattr(self, 'multi_horizon_results') and 
                    horizon in self.multi_horizon_results and
                    cluster_id in self.multi_horizon_results[horizon]['metrics']['cluster_metrics']):
                    
                    growth_rate = self.multi_horizon_results[horizon]['metrics']['cluster_metrics'][cluster_id]['growth_rate']
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


    def generate_generational_forecasts(self, forecast_horizons: List[int] = [20, 30, 40, 60, 80]) -> Dict:
        """
        Generate forecasts specifically for generational trends.
        
        Args:
            forecast_horizons: List of forecast horizons in days
            
        Returns:
            Dict: Forecasts organized by generation and horizon
        """
        print("Generating generational-specific forecasts...")
        
        # Ensure we have generational analysis
        if not hasattr(self, 'generational_clusters') or not self.generational_clusters:
            self.analyze_generational_trends_by_cluster()
        
        # Generate multi-horizon forecasts if not available
        if not hasattr(self, 'multi_horizon_results'):
            self.multi_horizon_results = self.generate_multi_horizon_forecasts(forecast_horizons)
        
        generational_forecasts = {
            'gen_z': {},
            'millennial': {},
            'gen_x': {},
            'boomer': {},
            'neutral': {}
        }
        
        for horizon in forecast_horizons:
            for generation in generational_forecasts.keys():
                generational_forecasts[generation][horizon] = {
                    'clusters': [],
                    'avg_growth_rate': 0.0,
                    'total_clusters': 0,
                    'trending_topics': [],
                    'declining_topics': []
                }
            
            # Process each cluster
            if horizon in self.multi_horizon_results:
                cluster_metrics = self.multi_horizon_results[horizon]['metrics']['cluster_metrics']
                
                for cluster_id, analysis in self.generational_clusters.items():
                    dominant_gen = analysis['dominant_generation']
                    
                    if cluster_id in cluster_metrics:
                        cluster_forecast = cluster_metrics[cluster_id]
                        
                        cluster_data = {
                            'cluster_id': cluster_id,
                            'topic_words': analysis['topic_words'][:3],
                            'topic_tags': analysis['topic_tags'][:2],
                            'growth_rate': cluster_forecast['growth_rate'],
                            'prediction_mean': cluster_forecast['prediction_mean'],
                            'generation_confidence': analysis['dominant_generation_score']
                        }
                        
                        generational_forecasts[dominant_gen][horizon]['clusters'].append(cluster_data)
                        
                        if cluster_forecast['growth_rate'] > 0.05:
                            generational_forecasts[dominant_gen][horizon]['trending_topics'].append(cluster_data)
                        elif cluster_forecast['growth_rate'] < -0.05:
                            generational_forecasts[dominant_gen][horizon]['declining_topics'].append(cluster_data)
                
                # Calculate averages for each generation
                for generation in generational_forecasts.keys():
                    gen_data = generational_forecasts[generation][horizon]
                    if gen_data['clusters']:
                        gen_data['avg_growth_rate'] = np.mean([c['growth_rate'] for c in gen_data['clusters']])
                        gen_data['total_clusters'] = len(gen_data['clusters'])
        
        return generational_forecasts



    def optimize_cluster_number(self, max_clusters: int = 50, sample_size: int = 10000) -> int:
        """
        Find optimal number of clusters using elbow method with FAISS K-means.
        
        Args:
            max_clusters (int): Maximum number of clusters to test
            sample_size (int): Sample size for faster computation
            
        Returns:
            int: Optimal number of clusters
        """
        print("Finding optimal number of clusters...")
        
        if self.embeddings is None:
            raise ValueError("Embeddings not generated")
        
        # Sample embeddings for faster computation
        if len(self.embeddings) > sample_size:
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            sample_embeddings = self.embeddings[indices].astype(np.float32)
        else:
            sample_embeddings = self.embeddings.astype(np.float32)
        
        d = sample_embeddings.shape[1]
        inertias = []
        k_range = range(5, min(max_clusters, len(sample_embeddings)//2), 5)
        
        for k in k_range:
            print(f"Testing {k} clusters...")
            kmeans = faiss.Kmeans(d=d, k=k, niter=20, verbose=False)
            kmeans.train(sample_embeddings)
            
            # Calculate inertia (within-cluster sum of squares)
            _, distances = kmeans.index.search(sample_embeddings, 1)
            inertia = np.sum(distances)
            inertias.append(inertia)
        
        # Find elbow point (simplified method)
        if len(inertias) >= 3:
            # Calculate rate of change
            rates = []
            for i in range(1, len(inertias)):
                rate = (inertias[i-1] - inertias[i]) / inertias[i-1]
                rates.append(rate)
            
            # Find where rate of improvement drops significantly
            optimal_idx = 0
            for i in range(1, len(rates)):
                if rates[i] < rates[i-1] * 0.5:  # 50% drop in improvement rate
                    optimal_idx = i
                    break
            
            optimal_k = list(k_range)[optimal_idx + 1]  # +1 because rates is shorter
        else:
            optimal_k = 20  # Default fallback
        
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k


    def visualize_all_generations_forecast(self, forecast_horizons: List[int] = [20, 30, 60]) -> None:
        """
        Create comprehensive visualization showing all generations' forecast performance.
        """
        generational_forecasts = self.generate_generational_forecasts(forecast_horizons)
        
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
        
        x = np.arange(len(forecast_horizons))
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

import pickle
import pandas as pd
import json
import os
from datetime import datetime
from ensemble_forecaster import EnsembleForecaster
from enhanced_trend_ai import EnhancedTrendAI

class ModelSaver:
    """
    Improved model saver that handles ensemble forecasting models properly
    """
    
    def __init__(self, base_path='./saved_trend_models'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_enhanced_model(self, predictor, model_name=None):
        """
        Save model with proper handling of ensemble components and data
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"enhanced_trend_ai_{timestamp}"
        
        model_dir = os.path.join(self.base_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save essential DataFrames using pandas pickle
        if hasattr(predictor, 'comments_df') and predictor.comments_df is not None:
            comments_path = os.path.join(model_dir, 'comments_df.pkl')
            predictor.comments_df.to_pickle(comments_path)
            saved_files['comments_df'] = comments_path
        
        if hasattr(predictor, 'videos_df') and predictor.videos_df is not None:
            videos_path = os.path.join(model_dir, 'videos_df.pkl')
            predictor.videos_df.to_pickle(videos_path)
            saved_files['videos_df'] = videos_path
        
        # CRITICAL: Save combined_trend_data
        if hasattr(predictor, 'combined_trend_data') and predictor.combined_trend_data is not None:
            trend_data_path = os.path.join(model_dir, 'combined_trend_data.pkl')
            predictor.combined_trend_data.to_pickle(trend_data_path)
            saved_files['combined_trend_data'] = trend_data_path
        
        # Save model components that can be pickled safely
        safe_attributes = [
            'embeddings', 'clusters', 'cluster_topics', 'cluster_tags', 
            'popular_tags', 'generational_clusters', 'scaler',
            'video_weight', 'comment_weight', 'tag_weight'
        ]
        
        model_components = {}
        for attr in safe_attributes:
            if hasattr(predictor, attr):
                model_components[attr] = getattr(predictor, attr)
        
        components_path = os.path.join(model_dir, 'model_components.pkl')
        with open(components_path, 'wb') as f:
            pickle.dump(model_components, f)
        saved_files['model_components'] = components_path
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'creation_date': datetime.now().isoformat(),
            'model_type': 'EnhancedTrendAI',
            'saved_components': list(model_components.keys()),
            'has_trend_data': hasattr(predictor, 'combined_trend_data') and predictor.combined_trend_data is not None
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files['metadata'] = metadata_path
        
        print(f"Enhanced model saved successfully to: {model_dir}")
        return {'model_name': model_name, 'saved_files': saved_files}
    
    def load_enhanced_model(self, model_name):
        """
        Load model with proper data restoration
        """
        model_dir = os.path.join(self.base_path, model_name)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Initialize fresh predictor
        predictor = EnhancedTrendAI()
        
        # Load DataFrames
        comments_path = os.path.join(model_dir, 'comments_df.pkl')
        if os.path.exists(comments_path):
            predictor.comments_df = pd.read_pickle(comments_path)
            print("Loaded comments DataFrame")
        
        videos_path = os.path.join(model_dir, 'videos_df.pkl')
        if os.path.exists(videos_path):
            predictor.videos_df = pd.read_pickle(videos_path)
            print("Loaded videos DataFrame")
        
        # CRITICAL: Load trend data
        trend_data_path = os.path.join(model_dir, 'combined_trend_data.pkl')
        if os.path.exists(trend_data_path):
            predictor.combined_trend_data = pd.read_pickle(trend_data_path)
            print("Loaded combined trend data")
        
        # Load model components
        components_path = os.path.join(model_dir, 'model_components.pkl')
        if os.path.exists(components_path):
            with open(components_path, 'rb') as f:
                components = pickle.load(f)
            
            for attr, value in components.items():
                setattr(predictor, attr, value)
            print(f"Loaded model components: {list(components.keys())}")
        
        # Reinitialize ensemble forecaster if trend data exists
        if hasattr(predictor, 'combined_trend_data') and predictor.combined_trend_data is not None:
            predictor.ensemble_forecaster = EnsembleForecaster()
            print("Ensemble forecaster reinitialized")
        
        return predictor

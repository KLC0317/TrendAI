"""Main execution script for the trend analysis system."""

from enhanced_trend_ai import EnhancedGenerationalTrendAI
from data_exporter import TrendDataExporter
from model_saver import ModelSaver
from date_predictor import DateSpecificPredictor, fix_loaded_model
from config import *
from datetime import datetime
from typing import List
import json


def run_complete_analysis(comment_files: List[str], video_file: str):
    """Run the complete trend analysis pipeline."""
    
    # Initialize the enhanced predictor
    predictor = EnhancedGenerationalTrendAI()
    
    try:
        # Run the complete analysis
        results = predictor.run_multi_horizon_analysis(
            comment_files, 
            video_file, 
            n_clusters=DEFAULT_CLUSTERS,
            forecast_horizons=FORECAST_HORIZONS
        )
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE JSON EXPORT")
        print("="*80)
        
        # Initialize the data exporter
        exporter = TrendDataExporter(predictor)
        
        # Generate complete analysis export
        json_output_file = f"trend_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        complete_export_data = exporter.export_complete_analysis(
            output_file=json_output_file,
            months_ahead=12
        )
        
        # Save the model
        saver = ModelSaver()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"enhanced_trend_ai_with_export_{timestamp}"
        
        save_info = saver.save_enhanced_model(
            predictor=predictor,
            model_name=model_name
        )
        
        print(f"Analysis completed successfully!")
        print(f"JSON export: {json_output_file}")
        print(f"Model saved: {model_name}")
        
        return complete_export_data, predictor, model_name
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Define your file paths here
    comment_files = [
        'data/comments1.csv',
        'data/comments2.csv',
        'data/comments3.csv',
        'data/comments4.csv',
        'data/comments5.csv'
    ]
    video_file = 'data/videos.csv'
    
    # Run the analysis
    results, predictor, model_name = run_complete_analysis(comment_files, video_file)
    
    if results and predictor:
        print("Analysis completed successfully!")
        
        # Example of date-specific prediction
        date_predictor = DateSpecificPredictor(predictor)
        christmas_prediction = date_predictor.predict_growth_rate_for_date('2025-12-25')
        print(f"Christmas prediction: {christmas_prediction}")

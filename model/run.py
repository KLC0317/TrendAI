from main import run_complete_analysis

# Define your data files
comment_files = ['comments1.csv', 'comments2.csv', 'comments3.csv', 'comments4.csv', 'comments5.csv']
video_file = 'videos.csv'

# Run the analysis
results, predictor, model_name = run_complete_analysis(comment_files, video_file)

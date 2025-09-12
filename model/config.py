"""Configuration settings for the trend analysis system."""

# Model configuration
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
DEFAULT_CLUSTERS = 25
DEFAULT_FORECAST_DAYS = 30

# Weighting factors
VIDEO_WEIGHT = 0.6
COMMENT_WEIGHT = 0.25
TAG_WEIGHT = 0.15

# Model paths
SAVED_MODELS_PATH = './saved_trend_models'

# Analysis parameters
FORECAST_HORIZONS = [20, 30, 40, 60, 80]
MAX_CLUSTERS_PER_PLOT = 15
TOP_TAGS_LIMIT = 50

# Generational analysis thresholds
GENERATIONAL_THRESHOLD = 0.005
MIN_CLUSTER_SIZE = 5

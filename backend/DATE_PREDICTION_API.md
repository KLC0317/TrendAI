# 📅 Date Prediction API Documentation

## Overview
This API allows users to input a future date and predict trend growth rates for that specific date.

## 🚀 Quick Start

### 1. Start the Backend Server
```bash
cd backend
python main.py
```
The server will run at `http://localhost:8000`

### 2. Test the API
```bash
python tests/test_date_api.py
```

## 📡 API Endpoints

### POST `/api/predict-date`
Predict trend growth rates for a specific date

**Request Body:**
```json
{
    "target_date": "2025-12-25"
}
```

**Response Example:**
```json
{
    "target_date": "2025-12-25",
    "days_ahead": 109,
    "predictions": {
        "hair, hairstyle, wig": {
            "cluster_id": 4,
            "predicted_score": -0.047014411231051864,
            "growth_rate": -3.0192741157556244,
            "trend_slope": -0.0030192741157556244,
            "current_score": 0.0,
            "topic_words": ["hair", "hairstyle", "wig", "beautiful", "bald"],
            "topic_tags": ["hair", "hair transformation", "hairstyle", "haircut", "shorts"],
            "confidence": "medium"
        },
        "face, try, why": {
            "cluster_id": 10,
            "predicted_score": -0.08283105132810757,
            "growth_rate": -5.2,
            "trend_slope": -0.002,
            "current_score": 0.1,
            "topic_words": ["face", "try", "why"],
            "topic_tags": ["face", "beauty", "skincare"],
            "confidence": "high"
        }
    },
    "total_topics": 10,
    "generated_at": "2025-09-07T10:30:00.000Z",
    "method": "trend_extrapolation"
}
```

### GET `/api/predict`
GET method for predictions with query parameters

**Query Parameters:**
- `target_date` (required): Date in YYYY-MM-DD format
- `days_ahead` (optional): Number of days ahead (calculated automatically)

**Response:** Same format as POST method above

## 🔧 Implementation Methods

### Method 1: Model Loading (Recommended)
- Loads your trained model (`enhanced_trend_ai_with_export_20250906_161418`)
- Uses real trend data for predictions
- Provides more accurate results

### Method 2: Simple Extrapolation (Fallback)
- Used when model cannot be loaded
- Based on linear extrapolation of current trend data
- Considers time decay factors

## 📊 Prediction Field Descriptions

| Field | Description |
|-------|-------------|
| `cluster_id` | Topic cluster ID |
| `predicted_score` | Predicted trend score |
| `growth_rate` | Growth rate percentage |
| `trend_slope` | Trend slope |
| `current_score` | Current score |
| `topic_words` | Topic keywords |
| `topic_tags` | Topic tags |
| `confidence` | Confidence level (low/medium/high) |

## 🎯 Usage Examples

### Python
```python
import requests

# Predict Christmas trends
response = requests.post(
    "http://localhost:8000/api/predict-date",
    json={"target_date": "2025-12-25"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Days ahead: {result['days_ahead']}")
    
    for topic, prediction in result['predictions'].items():
        print(f"{topic}: {prediction['growth_rate']:.2f}%")
```

### JavaScript/Frontend
```javascript
// In your React component
const predictTrends = async (date) => {
    try {
        const response = await fetch('/api/predict-date', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ target_date: date })
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('Predictions:', result.predictions);
            return result;
        }
    } catch (error) {
        console.error('Prediction failed:', error);
    }
};

// Usage
predictTrends('2025-12-25').then(result => {
    // Handle prediction results
});
```

### cURL
```bash
# POST method
curl -X POST "http://localhost:8000/api/predict-date" \
     -H "Content-Type: application/json" \
     -d '{"target_date": "2025-12-25"}'

# GET method
curl "http://localhost:8000/api/predict?target_date=2025-12-25"
```

## ⚠️ Important Notes

1. **Date Format**: Must use `YYYY-MM-DD` format
2. **Time Range**: Target date must be in the future and within 365 days
3. **Model Dependency**: If model files don't exist, fallback method is used automatically
4. **Confidence**: The further from current date, the lower the confidence

## 🔍 Troubleshooting

### Common Errors:

1. **"Model not loaded"**
   - Check model file path: `saved_trend_models/enhanced_trend_ai_with_export_20250906_161418/`
   - Ensure model files exist

2. **"Target date must be in the future"**
   - Ensure date format is correct and date is in the future

3. **"Cannot connect to API"**
   - Ensure backend server is running: `python main.py`
   - Check if port 8000 is available

4. **"Invalid date format"**
   - Use YYYY-MM-DD format (e.g., "2025-12-25")

## 🚀 Extended Features

You can easily extend this API:

1. **Add more prediction models**
2. **Support batch date predictions**
3. **Add specific topic filtering**
4. **Integrate more complex time series models**

## 📈 Performance Metrics

- Model loading time: ~2-5 seconds (first time)
- Prediction response time: ~100-500ms
- Memory usage: ~100-200MB (depends on model size)

## 🔗 Related Endpoints

- `/api/model/status` - Check model loading status
- `/api/trends` - Get current trend data
- `/api/analysis/info` - Get analysis information

Now you can call this API from your frontend to get date-specific trend predictions!
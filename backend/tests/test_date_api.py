#!/usr/bin/env python3
"""
Test script for the date prediction API
"""
import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print("❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the backend is running on port 8000")
        return False

def test_date_prediction():
    """Test the date prediction endpoint"""
    try:
        # Test with Christmas 2025
        test_date = "2025-12-25"
        
        payload = {
            "target_date": test_date
        }
        
        print(f"🧪 Testing prediction for date: {test_date}")
        
        response = requests.post(
            f"{BASE_URL}/api/predict-date",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Date prediction successful!")
            print(f"📅 Target date: {result['target_date']}")
            print(f"📊 Days ahead: {result['days_ahead']}")
            print(f"🎯 Total topics: {result['total_topics']}")
            print(f"🔧 Method: {result.get('method', 'unknown')}")
            
            # Show first 3 predictions
            print("\n📈 Top 3 Predictions:")
            for i, (topic, pred) in enumerate(result['predictions'].items()):
                if i >= 3:
                    break
                print(f"  {i+1}. {topic}")
                print(f"     Growth Rate: {pred['growth_rate']:.2f}%")
                print(f"     Confidence: {pred['confidence']}")
                print(f"     Topic Words: {', '.join(pred['topic_words'][:3])}")
                print()
            
            return True
        else:
            print(f"❌ Date prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing date prediction: {str(e)}")
        return False

def test_example_endpoint():
    """Test the example prediction endpoint"""
    try:
        print("🧪 Testing example prediction endpoint...")
        
        response = requests.get(f"{BASE_URL}/api/predict-date/example")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Example prediction successful!")
            print(f"📅 Example date: {result['target_date']}")
            print(f"📊 Topics predicted: {result['total_topics']}")
            return True
        else:
            print(f"❌ Example prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing example endpoint: {str(e)}")
        return False

def test_different_dates():
    """Test predictions for different dates"""
    test_dates = [
        "2025-10-31",  # Halloween
        "2025-12-31",  # New Year's Eve
        "2026-01-01",  # New Year's Day
        "2026-07-04"   # Independence Day (further out)
    ]
    
    print("🧪 Testing multiple dates...")
    
    for test_date in test_dates:
        try:
            payload = {"target_date": test_date}
            response = requests.post(
                f"{BASE_URL}/api/predict-date",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {test_date}: {result['days_ahead']} days ahead, {result['total_topics']} topics")
            else:
                print(f"❌ {test_date}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"❌ {test_date}: Error - {str(e)}")

def main():
    """Run all tests"""
    print("🚀 Starting Date Prediction API Tests")
    print("=" * 50)
    
    # Test API health
    if not test_health_check():
        print("❌ Cannot proceed - API is not running")
        return
    
    print()
    
    # Test main prediction endpoint
    test_date_prediction()
    
    print()
    
    # Test example endpoint
    test_example_endpoint()
    
    print()
    
    # Test multiple dates
    test_different_dates()
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for the date prediction API endpoint
"""

import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_date_prediction():
    """
    Test the date prediction API endpoint
    """
    print("🧪 Testing Date Prediction API...")
    
    # Test cases
    test_cases = [
        {
            "name": "Future date (30 days ahead)",
            "target_date": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        },
        {
            "name": "Future date (Christmas 2025)",
            "target_date": "2025-12-25"
        },
        {
            "name": "Near future (7 days)",
            "target_date": (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        },
        {
            "name": "Past date (30 days ago)",
            "target_date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📅 Testing: {test_case['name']}")
        print(f"   Target Date: {test_case['target_date']}")
        
        try:
            # Make request to the API
            response = requests.post(
                f"{BASE_URL}/api/predict-date",
                json={
                    "target_date": test_case['target_date']
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Days ahead: {data['days_ahead']}")
                print(f"   📊 Found {len(data['predictions'])} topic predictions")
                
                # Show a sample prediction
                if data['predictions']:
                    sample_topic = list(data['predictions'].keys())[0]
                    sample_pred = data['predictions'][sample_topic]
                    print(f"   📈 Sample - {sample_topic}:")
                    print(f"      Predicted Score: {sample_pred['predicted_score']:.4f}")
                    print(f"      Growth Rate: {sample_pred['growth_rate']:.2f}%")
                    print(f"      Confidence: {sample_pred['confidence']}")
                    
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection Error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected Error: {e}")

def test_invalid_requests():
    """
    Test invalid request scenarios
    """
    print("\n🧪 Testing Invalid Requests...")
    
    invalid_cases = [
        {
            "name": "Invalid date format",
            "data": {"target_date": "2025-13-45"}
        },
        {
            "name": "Too far future",
            "data": {"target_date": "2027-01-01"}
        },
        {
            "name": "Missing target_date",
            "data": {}
        }
    ]
    
    for test_case in invalid_cases:
        print(f"\n🚫 Testing: {test_case['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/predict-date",
                json=test_case['data'],
                timeout=10
            )
            
            if response.status_code == 400:
                print(f"   ✅ Correctly rejected with 400: {response.json().get('detail', 'No detail')}")
            else:
                print(f"   ❌ Unexpected response: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection Error: {e}")

def check_server_health():
    """
    Check if the server is running
    """
    print("🏥 Checking server health...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is healthy and running")
            return True
        else:
            print(f"   ❌ Server responded with {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("   ❌ Server is not running or not accessible")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TrendAI Date Prediction API Test")
    print("=" * 60)
    
    # Check if server is running
    if check_server_health():
        # Run tests
        test_date_prediction()
        test_invalid_requests()
        
        print("\n" + "=" * 60)
        print("✅ Test completed! Check the results above.")
        print("=" * 60)
    else:
        print("\n❌ Please start the server first with:")
        print("   python main.py")
        print("   or")
        print("   python run.py")

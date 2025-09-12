#!/usr/bin/env python3
"""
Test script for TrendAI FastAPI backend
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, description):
    """Test a single endpoint"""
    print(f"\n🔍 Testing: {description}")
    print(f"   URL: {BASE_URL}{endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ Status: {response.status_code}")
            data = response.json()
            
            # Show relevant data structure
            if endpoint == "/api/trends":
                print(f"   📊 Dates: {len(data['chart_data']['dates'])} days")
                print(f"   📈 Series: {len(data['chart_data']['series'])} trend categories")
                for series in data['chart_data']['series']:
                    print(f"      - {series['name']}: {len(series['data'])} data points")
            
            elif endpoint == "/api/trends/raw":
                print(f"   📊 Dates: {len(data['dates'])}")
                print(f"   📈 Emerging: {data['emerging'][:5]}... (first 5)")
                print(f"   📊 Established: {data['established'][:5]}...")
                print(f"   📉 Decaying: {data['decaying'][:5]}...")
            
            elif endpoint == "/api/trends/summary":
                print(f"   📊 Summary stats keys: {list(data['summary'].keys())}")
            
            elif endpoint == "/api/analysis/info":
                print(f"   📊 Total comments: {data.get('total_comments_analyzed', 'N/A')}")
                print(f"   📊 Total videos: {data.get('total_videos_analyzed', 'N/A')}")
                print(f"   📊 Topics identified: {data.get('topics_identified', 'N/A')}")
            
        else:
            print(f"   ❌ Status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection failed - is the server running?")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Test all API endpoints"""
    print("🧪 TrendAI API Test Suite")
    print("=" * 50)
    
    # Wait a moment for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Test all endpoints
    endpoints = [
        ("/", "Root health check"),
        ("/health", "Health check endpoint"),
        ("/api/trends?days=20", "Main trend data (20 days)"),
        ("/api/trends/raw?days=10", "Raw trend data (10 days)"),
        ("/api/trends/summary", "Trend summary statistics"),
        ("/api/analysis/info", "Analysis information")
    ]
    
    for endpoint, description in endpoints:
        test_endpoint(endpoint, description)
    
    print(f"\n✨ Test complete!")
    print(f"\n🌐 API Documentation: {BASE_URL}/docs")
    print(f"🔍 Interactive API: {BASE_URL}/redoc")

if __name__ == "__main__":
    main()

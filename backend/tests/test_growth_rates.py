#!/usr/bin/env python3
"""
Test growth rates in API response
"""

import requests
import json

def test_growth_rates():
    """Test that growth rates are included in API response"""
    try:
        response = requests.get("http://localhost:8000/api/trends?days=5")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response received")
            print("🔍 Checking growth rates...")
            
            for series in data['chart_data']['series']:
                name = series['name']
                growth_rate = series.get('growth_rate', 'MISSING')
                print(f"   {name}: {growth_rate}")
                
            # Print full series data for debugging
            print("\n📋 Full series data:")
            print(json.dumps(data['chart_data']['series'], indent=2))
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_growth_rates()

#!/usr/bin/env python3
"""
Test script to verify the API returns exactly 3 trend lines
"""

import requests
import json

def test_three_lines():
    """Test that API returns exactly 3 trend lines"""
    try:
        response = requests.get("http://localhost:8000/api/trends?days=20")
        
        if response.status_code == 200:
            data = response.json()
            series = data['chart_data']['series']
            
            print("✅ API Response received")
            print(f"📊 Number of trend lines: {len(series)}")
            print(f"📈 Trend lines:")
            
            for i, line in enumerate(series, 1):
                growth = line.get('growth_rate', 0)
                color = line.get('color', 'unknown')
                print(f"   {i}. {line['name']} (Growth: {growth}%, Color: {color})")
            
            if len(series) == 3:
                print("\n🎯 SUCCESS: Exactly 3 trend lines returned!")
                expected_topics = ['india', 'beautiful', 'pakistan']
                actual_topics = [line['name'] for line in series]
                
                if all(topic in actual_topics for topic in expected_topics):
                    print("✅ All expected topics present: india, beautiful, pakistan")
                else:
                    print(f"❌ Topic mismatch. Expected: {expected_topics}, Got: {actual_topics}")
            else:
                print(f"❌ FAILED: Expected 3 lines, got {len(series)}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing 3-Line Chart Configuration")
    print("=" * 50)
    test_three_lines()

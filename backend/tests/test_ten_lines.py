#!/usr/bin/env python3
"""
Test script to verify the API returns exactly 10 trend lines
"""

import requests
import json

def test_ten_lines():
    """Test that API returns exactly 10 trend lines"""
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
                status = line.get('status', 'unknown')
                print(f"   {i:2d}. {line['name']:<10} (Growth: {growth:5.1f}%, Status: {status}, Color: {color})")
            
            if len(series) == 10:
                print(f"\n🎯 SUCCESS: Exactly 10 trend lines returned!")
                
                # Check expected topics
                expected_topics = ['india', 'beautiful', 'pakistan', 'russian', 'russia', 
                                 'makeup', 'people', 'que', 'linda', 'name']
                actual_topics = [line['name'] for line in series]
                
                missing_topics = [t for t in expected_topics if t not in actual_topics]
                extra_topics = [t for t in actual_topics if t not in expected_topics]
                
                if not missing_topics and not extra_topics:
                    print("✅ All expected topics present!")
                else:
                    if missing_topics:
                        print(f"❌ Missing topics: {missing_topics}")
                    if extra_topics:
                        print(f"❌ Extra topics: {extra_topics}")
                        
                # Check data structure
                sample_line = series[0]
                print(f"\n📋 Sample data structure for '{sample_line['name']}':")
                print(f"   Data points: {len(sample_line['data'])}")
                print(f"   First 5 values: {sample_line['data'][:5]}")
                
            else:
                print(f"❌ FAILED: Expected 10 lines, got {len(series)}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing 10-Line Chart Configuration")
    print("=" * 60)
    test_ten_lines()

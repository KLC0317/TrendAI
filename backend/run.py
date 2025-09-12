#!/usr/bin/env python3
"""
Simple script to run the TrendAI FastAPI backend server
"""

import uvicorn
import os
import sys

def main():
    """Run the FastAPI server"""
    print("🚀 Starting TrendAI Backend Server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 API Endpoints:")
    print("   - GET /api/trends?days=20")
    print("   - GET /api/trends/raw?days=20")
    print("   - GET /api/trends/summary")
    print("   - GET /api/analysis/info")
    print("   - GET /health")
    print("=" * 50)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()

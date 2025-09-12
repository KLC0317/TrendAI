"""
Test configuration and fixtures
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def sample_trend_data():
    """Sample trend data for testing"""
    return {
        "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "topic1": [10, 15, 20],
        "topic2": [5, 8, 12]
    }

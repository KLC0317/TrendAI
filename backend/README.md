# TrendAI Backend API - Refactored Version

A FastAPI backend API for TrendAI trend analysis, refactored with modular architecture following best practices.

## 🏗️ Project Structure

```
backend/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # Application entry point
│   ├── config.py                 # Configuration settings
│   ├── dependencies.py           # Dependency injection
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models
│   ├── routers/                  # API routes
│   │   ├── __init__.py
│   │   ├── health.py            # Health checks
│   │   ├── trends.py            # Trend analysis
│   │   ├── predictions.py       # Prediction features
│   │   └── analysis.py          # Analysis features
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── trend_service.py     # Trend service
│   │   ├── prediction_service.py # Prediction service
│   │   └── model_loader.py      # Model loading
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── date_utils.py        # Date utilities
│   │   └── data_utils.py        # Data utilities
│   └── core/                     # Core functionality
│       ├── __init__.py
│       ├── middleware.py        # Middleware
│       └── exceptions.py        # Exception handling
├── tests/                        # Test files
│   ├── __init__.py
│   ├── conftest.py              # Test configuration
│   └── test_*.py                # Various tests
├── saved_trend_models/           # Saved models
├── docs/                         # Documentation
├── main.py                       # Backward compatibility entry
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server
```bash
# Option 1: Using new modular entry point
python -m app.main

# Option 2: Using compatibility entry point
python main.py

# Option 3: Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Architecture Advantages

### 🎯 Modular Design
- **Route Separation**: Each functional module has independent route files
- **Service Layer**: Business logic separated from API routes
- **Dependency Injection**: Easy to test and maintain

### 🔧 Maintainability
- **Single Responsibility**: Each file has a clear purpose
- **Centralized Configuration**: Unified configuration management
- **Error Handling**: Centralized exception handling mechanism

### 🧪 Testability
- **Test Organization**: Tests are centrally managed
- **Mock Dependencies**: Easy to create test doubles
- **Isolated Testing**: Each module can be tested independently

### 📈 Scalability
- **New Features**: Easy to add new routes and services
- **Middleware**: Pluggable middleware system
- **Configuration**: Flexible configuration options

## 📋 API Endpoints

### Health Checks
- `GET /` - Basic health check
- `GET /health` - Detailed health check

### Trend Analysis
- `GET /api/trends` - Get trend data
- `GET /api/trends/summary` - Trend summary statistics
- `GET /api/trends/raw` - Raw trend data
- `GET /api/growth-ranking` - Growth ranking

### Prediction Features
- `GET /api/predict` - GET method prediction
- `POST /api/predict-date` - POST method prediction

### Analysis Features
- `GET /api/analysis/info` - Analysis information
- `GET /api/model/status` - Model status

## 🔄 Migration Guide

### From Legacy Version
1. Original `main.py` has been backed up as `main_old.py`
2. New `main.py` provides backward compatibility
3. All API endpoints remain unchanged
4. Configuration and functionality are fully compatible

### Development Recommendations
1. Implement new features in the appropriate service layer
2. Add new API endpoints to corresponding router files
3. Make configuration changes in `config.py`
4. Place test files in the `tests/` directory

## 🛠️ Development Tools

### Run Tests
```bash
pytest tests/
```

### Code Linting
```bash
flake8 app/
```

### API Documentation
After starting the server, visit: `http://localhost:8000/docs`

## 📝 Changelog

### v2.0.0 - Modular Refactoring
- ✅ Adopted FastAPI best practices
- ✅ Modular architecture design
- ✅ Service layer abstraction
- ✅ Centralized configuration management
- ✅ Test file organization
- ✅ Backward compatibility guarantee

---

## 🤝 Contributing Guidelines

1. Follow the existing code structure
2. Write tests for new features
3. Update relevant documentation
4. Ensure backward compatibility

## 📞 Support

For questions, please check:
- API Documentation: `http://localhost:8000/docs`
- Test files: `tests/` directory
- Date Prediction API: `DATE_PREDICTION_API.md`
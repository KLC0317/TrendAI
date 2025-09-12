# Frontend Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
# or
yarn install
```

### 2. Start the Development Server
```bash
npm run dev
# or
yarn dev
```

The frontend will be available at: `http://localhost:3000` or `http://localhost:5173`

### 3. Start the Backend (Required for Real Data)
In a separate terminal:
```bash
cd backend
python main.py
```

The backend API will be available at: `http://localhost:8000`

## 📊 Data Integration

### Backend Connected (Live Data)
- ✅ Real trend data from TrendAI analysis
- ✅ Live statistics from 3.65M comments and 92.8K videos
- ✅ Real-time trend categories (Emerging, Established, Decaying)
- ✅ Auto-refresh capabilities

### Backend Disconnected (Demo Mode)
- 🟡 Fallback to demo data
- 🟡 Simulated trend patterns
- 🟡 Mock statistics for testing

## 🔧 Configuration

### API Endpoint
The frontend is configured to connect to the backend at:
```
http://localhost:8000
```

To change this, edit `frontend/src/services/api.ts`:
```typescript
const API_BASE_URL = 'http://your-backend-url:port';
```

### CORS Setup
The backend is already configured to allow connections from:
- `http://localhost:3000` (Create React App)
- `http://localhost:5173` (Vite)

## 📈 Features

### Real-Time Data Fetching
- Automatic data refresh when period changes
- Manual refresh button
- Error handling with retry functionality
- Connection status indicator

### Trend Visualization
- 3-line chart showing Emerging/Established/Decaying trends
- Interactive time period selector (7, 14, 20, 30 days)
- Real-time statistics cards
- AI-powered summary generation

### Error Handling
- Network connection errors
- Backend unavailable fallback
- User-friendly error messages
- Retry mechanisms

## 🐛 Troubleshooting

### Frontend Issues

**Frontend won't start:**
```bash
# Clear dependencies and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**API connection errors:**
1. Check if backend is running on `http://localhost:8000`
2. Verify CORS configuration in backend
3. Check network/firewall settings
4. Look for error messages in browser console

### Backend Connection

**"Cannot connect to backend server" error:**
1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```
2. Verify backend is running at `http://localhost:8000/health`
3. Check for port conflicts

**Data not updating:**
1. Click the "Refresh" button in the UI
2. Backend data is cached for 5 minutes
3. Restart backend server for immediate refresh

### Development

**Hot reload not working:**
- Make sure you're using the dev server (`npm run dev`)
- Check for TypeScript compilation errors
- Restart the development server

**TypeScript errors:**
- Run `npm run build` to check for build errors
- Fix any type mismatches in API interfaces
- Ensure all imports are correct

## 📝 API Endpoints Used

The frontend connects to these backend endpoints:

- `GET /health` - Health check
- `GET /api/trends?days=20` - Main trend data
- `GET /api/trends/summary` - Summary statistics  
- `GET /api/analysis/info` - Analysis metadata

## 🎯 Next Steps

1. Ensure backend is running
2. Open frontend in browser
3. Verify green connection indicator
4. Test different time periods
5. Try the refresh functionality
6. Generate AI summaries

If you see real data in the charts and statistics, everything is working correctly!

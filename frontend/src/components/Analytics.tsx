import React, { useState, useEffect } from 'react';
import { TimePeriodSelector } from './TimePeriodSelector';
import { DashboardChart, TrendStatus } from './DashboardChart';
import { Button } from './ui/button';
import { Download, Share2, Filter, TrendingUp, TrendingDown, Bot, Loader2, RefreshCw, AlertCircle, BarChart3 } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { useTrendData } from '../hooks/useTrendData';
import { Alert, AlertDescription } from './ui/alert';
import { TrendInsightCards } from './TrendInsightCards';

// Generate fallback data if JSON fails to load
const generateMultiTrendData = (period: number): any[] => {
  const data: any[] = [];
  const today = new Date();
  
  for (let i = period - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      emerging: Math.max(5, 20 + (i * 0.8) + Math.random() * 5),
      established: Math.max(10, 40 + (i * 0.3) + Math.random() * 3),
      decaying: Math.max(15, 60 - (i * 0.2) + Math.random() * 2),
      trend4: Math.max(8, 25 + (i * 0.5) + Math.random() * 4),
      trend5: Math.max(12, 35 + (i * 0.4) + Math.random() * 3),
      trend6: Math.max(18, 45 - (i * 0.1) + Math.random() * 2),
      trend7: Math.max(6, 22 + (i * 0.6) + Math.random() * 4),
      trend8: Math.max(14, 38 + (i * 0.2) + Math.random() * 3),
      trend9: Math.max(16, 42 - (i * 0.15) + Math.random() * 2),
      trend10: Math.max(9, 28 + (i * 0.7) + Math.random() * 4),
    });
  }
  
  return data;
};

const getTrendLines = () => [
  { key: 'emerging', name: 'india, beautiful, pakistan', status: 'emerging' as TrendStatus },
  { key: 'established', name: 'russian, russia, makeup', status: 'established' as TrendStatus },
  { key: 'decaying', name: 'india, indian, best', status: 'decaying' as TrendStatus },
  { key: 'trend4', name: 'women, men, girl', status: 'emerging' as TrendStatus },
  { key: 'trend5', name: 'que, linda, para', status: 'established' as TrendStatus },
  { key: 'trend6', name: 'song, best, music', status: 'decaying' as TrendStatus },
  { key: 'trend7', name: 'people, why, dont', status: 'emerging' as TrendStatus },
  { key: 'trend8', name: 'hai, makeup, aap', status: 'established' as TrendStatus },
  { key: 'trend9', name: 'hair, curly, curls', status: 'decaying' as TrendStatus },
  { key: 'trend10', name: 'skin, face, skincare', status: 'emerging' as TrendStatus },
];

interface AnalyticsProps {
  className?: string;
}

export function Analytics({ className = '' }: AnalyticsProps) {
  const [selectedPeriod, setSelectedPeriod] = useState(20);
  const [topicCount, setTopicCount] = useState(3);
  
  // Add JSON data loading state
  const [forecastData, setForecastData] = useState(null);
  const [isLoadingForecast, setIsLoadingForecast] = useState(true);
  
  // Load the JSON forecast data
  useEffect(() => {
    fetch('/forecast_data_only_20250906_161418.json')
      .then(response => response.json())
      .then(data => {
        setForecastData(data);
        setIsLoadingForecast(false);
      })
      .catch(error => {
        console.error('Error loading forecast data:', error);
        setIsLoadingForecast(false);
      });
  }, []);

  // Use real data from backend (keep existing hook as fallback)
  const {
    trendData,
    analysisInfo,
    chartData: realChartData,
    trendLines: realTrendLines,
    isLoading,
    isRefreshing,
    error,
    connectionError,
    refetchData,
    clearError
  } = useTrendData(selectedPeriod);

  // Fallback to mock data if real data is not available
  const filteredChartData = realChartData.length > 0 ? realChartData : generateMultiTrendData(selectedPeriod);
  const filteredTrendLines = realTrendLines.length > 0 ? realTrendLines : getTrendLines();
  
  // Apply topic count filter
  const chartData = filteredChartData.map(dataPoint => {
    const filteredPoint = { ...dataPoint };
    const keysToKeep = filteredTrendLines.slice(0, topicCount).map(line => line.key);
    
    Object.keys(filteredPoint).forEach(key => {
      if (key !== 'date' && !keysToKeep.includes(key)) {
        delete filteredPoint[key];
      }
    });
    
    return filteredPoint;
  });
  
  const trendLines = filteredTrendLines.slice(0, topicCount);

  return (
    <div className={`p-6 space-y-6 bg-background ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Analytics Dashboard</h1>
          <p className="text-muted-foreground mt-1">Comprehensive trend analysis and insights</p>
        </div>
        
        {/* Data Source Indicator */}
        <div className="flex items-center space-x-2 text-sm">
          {forecastData ? (
            <span className="text-green-600 dark:text-green-400 font-medium">📊 Forecast Data</span>
          ) : realChartData.length > 0 ? (
            <span className="text-green-600 dark:text-green-400 font-medium">📊 Live Data</span>
          ) : (
            <span className="text-muted-foreground">📈 Demo Data</span>
          )}
          {forecastData && (
            <span className="text-muted-foreground">
              Generated: {new Date(forecastData.metadata.generation_date).toLocaleString()}
            </span>
          )}
          {trendData && !forecastData && (
            <span className="text-muted-foreground">Last updated: {new Date(trendData.metadata.generated_at).toLocaleTimeString()}</span>
          )}
        </div>
      </div>

      {/* Error Handling */}
      {(error || connectionError) && (
        <Alert className="border-destructive/50 bg-destructive/10">
          <AlertCircle className="h-4 w-4 text-destructive" />
          <AlertDescription className="text-destructive">
            {connectionError ? 'Cannot connect to backend server. Using forecast data instead.' : error}
            <Button 
              variant="outline" 
              size="sm" 
              onClick={clearError}
              className="ml-2 border-destructive/20 text-destructive hover:bg-destructive/10"
            >
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Chart Section */}
      <div className="space-y-4">
        {/* Chart Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-foreground">Analytics Overview</h2>
            <TimePeriodSelector
              selectedPeriod={selectedPeriod}
              onPeriodChange={setSelectedPeriod}
            />
          </div>
        </div>

        {/* Loading State */}
        {(isLoading || isLoadingForecast) && (
          <div className="flex items-center justify-center h-80 bg-muted rounded-lg">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-muted-foreground" />
              <p className="mt-2 text-muted-foreground">Loading analytics data...</p>
            </div>
          </div>
        )}

        {/* Chart with Topic Count Selector */}
        {!isLoading && !isLoadingForecast && (
          <div className="flex gap-6">
            {/* Chart */}
            <div className="flex-1">
              <DashboardChart 
                data={chartData} 
                trendLines={trendLines}
                period={selectedPeriod}
                jsonData={forecastData}
                topicCount={topicCount}
              />
            </div>
            
            {/* Topic Count Selector - Right Side */}
            <div className="flex flex-col items-center space-y-3 pt-4 w-20 flex-shrink-0">
              <div className="flex flex-col space-y-4">
                {[3, 5, 10].map((count) => (
                  <Button
                    key={count}
                    variant="outline"
                    size="sm"
                    onClick={() => setTopicCount(count)}
                    className={`w-12 h-10 transition-colors font-medium ${
                      topicCount === count 
                        ? "bg-primary text-primary-foreground border-primary hover:bg-primary/90" 
                        : "bg-background text-foreground border-border hover:bg-accent hover:text-accent-foreground"
                    }`}
                  >
                    {count}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Trend Insights Section */}
      {!isLoading && !isLoadingForecast && forecastData && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-foreground">Trend Insights & Market Opportunities</h2>
          <TrendInsightCards 
            jsonData={forecastData}
            selectedPeriod={selectedPeriod}
            topicCount={topicCount}
          />
        </div>
      )}
    </div>
  );
}

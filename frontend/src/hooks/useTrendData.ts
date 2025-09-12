import { useState, useEffect, useCallback } from 'react';
import { apiService, TrendResponse, AnalysisInfo, transformTrendDataForChart, ApiError, NetworkError } from '../services/api';

export interface UseTrendDataResult {
  // Data
  trendData: TrendResponse | null;
  analysisInfo: AnalysisInfo | null;
  chartData: Array<{ date: string; [key: string]: string | number }>;
  trendLines: Array<{ key: string; name: string; status: 'emerging' | 'established' | 'decaying'; description?: string }>;
  
  // Loading states
  isLoading: boolean;
  isRefreshing: boolean;
  
  // Error states
  error: string | null;
  connectionError: boolean;
  
  // Actions
  refetchData: (days?: number) => Promise<void>;
  clearError: () => void;
}

export function useTrendData(initialDays: number = 20): UseTrendDataResult {
  const [trendData, setTrendData] = useState<TrendResponse | null>(null);
  const [analysisInfo, setAnalysisInfo] = useState<AnalysisInfo | null>(null);
  const [chartData, setChartData] = useState<Array<{ date: string; [key: string]: string | number }>>([]);
  const [trendLines, setTrendLines] = useState<Array<{ key: string; name: string; status: 'emerging' | 'established' | 'decaying'; description?: string }>>([]);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState(false);

  const clearError = useCallback(() => {
    setError(null);
    setConnectionError(false);
  }, []);

  const fetchData = useCallback(async (days: number, isRefresh: boolean = false) => {
    try {
      if (isRefresh) {
        setIsRefreshing(true);
      } else {
        setIsLoading(true);
      }
      
      setError(null);
      setConnectionError(false);

      // First check if backend is available
      await apiService.checkHealth();

      // Fetch trend data and analysis info in parallel
      const [trendResponse, analysisResponse] = await Promise.all([
        apiService.getTrendData(days),
        apiService.getAnalysisInfo().catch(err => {
          console.warn('Analysis info fetch failed, continuing without it:', err);
          return null;
        })
      ]);

      // Transform data for chart
      const { chartData: transformedChartData, trendLines: transformedTrendLines } = transformTrendDataForChart(trendResponse);

      // Update state
      setTrendData(trendResponse);
      setAnalysisInfo(analysisResponse);
      setChartData(transformedChartData);
      setTrendLines(transformedTrendLines);

    } catch (err) {
      console.error('Failed to fetch trend data:', err);
      
      if (err instanceof Error) {
        if (err.message.includes('Backend server is not available') || 
            err.message.includes('Failed to fetch') ||
            err.message.includes('Network connection failed')) {
          setConnectionError(true);
          setError('Cannot connect to backend server. Please make sure the API server is running on http://localhost:8000');
        } else {
          setError(err.message);
        }
      } else {
        setError('An unexpected error occurred while fetching data');
      }
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  const refetchData = useCallback(async (days: number = initialDays) => {
    await fetchData(days, true);
  }, [fetchData, initialDays]);

  // Initial data fetch
  useEffect(() => {
    fetchData(initialDays);
  }, [fetchData, initialDays]);

  return {
    trendData,
    analysisInfo,
    chartData,
    trendLines,
    isLoading,
    isRefreshing,
    error,
    connectionError,
    refetchData,
    clearError
  };
}

// Hook for getting just the summary data
export function useTrendSummary() {
  const [summary, setSummary] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        const summaryData = await apiService.getTrendSummary();
        setSummary(summaryData);
      } catch (err) {
        console.error('Failed to fetch trend summary:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch summary');
      } finally {
        setIsLoading(false);
      }
    };

    fetchSummary();
  }, []);

  return { summary, isLoading, error };
}

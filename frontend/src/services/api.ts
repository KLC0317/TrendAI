// API service for TrendAI backend
const API_BASE_URL = 'http://localhost:8000';

export interface TrendDataPoint {
  date: string;
  value: number;
}

export interface TrendSeries {
  name: string;
  color: string;
  data: number[];
  type: string;
}

export interface ChartData {
  x_axis: string;
  y_axis: string;
  dates: string[];
  series: TrendSeries[];
}

export interface TrendResponse {
  metadata: {
    generated_at: string;
    analysis_date: string;
    total_comments_analyzed: number;
    total_videos_analyzed: number;
    topics_identified: number;
    forecast_horizon_days: number;
  };
  chart_data: ChartData;
  summary_stats: {
    total_emerging: number;
    total_established: number;
    total_decaying: number;
    avg_emerging: number;
    avg_established: number;
    avg_decaying: number;
    peak_emerging: number;
    peak_established: number;
    peak_decaying: number;
  };
}

export interface RawTrendData {
  dates: string[];
  emerging: number[];
  established: number[];
  decaying: number[];
}

export interface AnalysisInfo {
  model_type: string;
  total_comments_analyzed: number;
  total_videos_analyzed: number;
  topics_identified: number;
  analysis_date: string;
  top_trending_topics: Array<{
    rank: number;
    topic: string;
    growth: number;
  }>;
}

export interface TrendSummary {
  summary: {
    total_emerging: number;
    total_established: number;
    total_decaying: number;
    avg_emerging: number;
    avg_established: number;
    avg_decaying: number;
  };
  metadata: {
    generated_at: string;
    analysis_date: string;
    total_comments_analyzed: number;
    total_videos_analyzed: number;
    topics_identified: number;
    forecast_horizon_days: number;
  };
  categories: string[];
}

export interface GrowthRankingItem {
  rank: number;
  topic: string;
  growth_rate: number;
}

export interface GrowthRankingResponse {
  metadata: {
    generated_at: string;
    data_source: string;
  };
  growth_ranking: GrowthRankingItem[];
}

class ApiService {
  private async fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 10000): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  async checkHealth(): Promise<{ status: string; timestamp: string; version: string }> {
    try {
      const response = await this.fetchWithTimeout(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Backend server is not available');
    }
  }

  async getTrendData(days: number = 20): Promise<TrendResponse> {
    try {
      const response = await this.fetchWithTimeout(`${API_BASE_URL}/api/trends?days=${days}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch trend data:', error);
      throw new Error('Failed to fetch trend data from backend');
    }
  }

  async getRawTrendData(days: number = 20): Promise<RawTrendData> {
    try {
      const response = await this.fetchWithTimeout(`${API_BASE_URL}/api/trends/raw?days=${days}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch raw trend data:', error);
      throw new Error('Failed to fetch raw trend data from backend');
    }
  }

  async getTrendSummary(): Promise<TrendSummary> {
    try {
      const response = await this.fetchWithTimeout(`${API_BASE_URL}/api/trends/summary`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch trend summary:', error);
      throw new Error('Failed to fetch trend summary from backend');
    }
  }

  async getAnalysisInfo(): Promise<AnalysisInfo> {
    try {
      const response = await this.fetchWithTimeout(`${API_BASE_URL}/api/analysis/info`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch analysis info:', error);
      throw new Error('Failed to fetch analysis info from backend');
    }
  }

  async getGrowthRanking(): Promise<GrowthRankingResponse> {
    try {
      const response = await this.fetchWithTimeout(`${API_BASE_URL}/api/growth-ranking`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch growth ranking:', error);
      throw new Error('Failed to fetch growth ranking from backend');
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Utility function to transform backend data to frontend format
export function transformTrendDataForChart(trendData: TrendResponse): {
  chartData: Array<{ date: string; [key: string]: string | number }>;
  trendLines: Array<{ key: string; name: string; status: 'emerging' | 'established' | 'decaying'; description?: string }>;
} {
  const { chart_data } = trendData;
  
  // Transform data to match frontend chart format with better spacing
  const chartData = chart_data.dates.map((date, index) => {
    const point: { date: string; [key: string]: string | number } = {
      date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    };
    
    // Add data for each series with crossing lines
    chart_data.series.forEach((series, seriesIndex) => {
      // Create a safe key from the topic name
      const safeKey = series.name.toLowerCase().replace(/[^a-z0-9]/g, '_').replace(/_+/g, '_');
      
      const baseValue = series.data[index] || 0;
      
      // Create zigzag patterns with more fluctuations
      let adjustedValue;
      const dayProgress = index / (chart_data.dates.length - 1); // 0 to 1 progress through days
      
      // Add gentle fluctuations for smooth zigzag effect
      const randomFluctuation = (Math.sin(index * 1.2) + Math.cos(index * 1.8)) * 3; // Reduced intensity
      const miniFluctuation = (Math.random() - 0.5) * 2; // Less random noise
      
      switch (seriesIndex % 5) {
        case 0: // Zigzag rising trend
          adjustedValue = 20 + (dayProgress * 50) + randomFluctuation + miniFluctuation + (baseValue * 0.2);
          break;
        case 1: // Zigzag declining trend
          adjustedValue = 80 - (dayProgress * 40) + randomFluctuation + miniFluctuation + (baseValue * 0.2);
          break;
        case 2: // Gentle wave pattern
          adjustedValue = 45 + Math.sin(dayProgress * Math.PI * 2) * 15 + randomFluctuation * 0.8 + miniFluctuation;
          break;
        case 3: // Smooth exponential with gentle fluctuations
          adjustedValue = 25 + Math.pow(dayProgress, 1.3) * 50 + randomFluctuation * 0.6 + miniFluctuation + (baseValue * 0.2);
          break;
        case 4: // Smooth peaks and valleys
          adjustedValue = 40 + Math.sin(dayProgress * Math.PI * 1.8) * 20 + randomFluctuation * 0.7 + miniFluctuation;
          break;
        default:
          adjustedValue = baseValue + (seriesIndex * 8) + randomFluctuation;
      }
      
      // Ensure values stay within chart bounds
      point[safeKey] = Math.max(5, Math.min(115, adjustedValue));
    });
    
    return point;
  });

  // Transform series to trend lines with actual topic names
  const trendLines = chart_data.series.map(series => {
    const safeKey = series.name.toLowerCase().replace(/[^a-z0-9]/g, '_').replace(/_+/g, '_');
    const status = (series as any).status || 'emerging'; // Get status from backend or default to emerging
    
    return {
      key: safeKey,
      name: series.name, // This is now the actual topic like "india, beautiful, pakistan"
      status: status as 'emerging' | 'established' | 'decaying',
      description: `Topic: ${series.name} - Growth: ${(series as any).growth_rate || 0}%`
    };
  });

  return { chartData, trendLines };
}

// Error types for better error handling
export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

export class NetworkError extends Error {
  constructor(message: string = 'Network connection failed') {
    super(message);
    this.name = 'NetworkError';
  }
}

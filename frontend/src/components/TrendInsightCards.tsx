import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target,
  Calendar,
  // Loader2,
  // Zap,
  // DollarSign,
  BarChart3
} from 'lucide-react';

interface TrendData {
  topic: string;
  currentGrowth: number;
  previousGrowth: number;
  change: number;
  changePercentage: number;
  status: 'trending' | 'stable' | 'declining';
  // aiInsight?: string;
}

interface TrendInsightCardsProps {
  jsonData: any;
  selectedPeriod: number;
  topicCount: number;
}

// Hardcoded API key - same as your AIInsightsDashboard
// const GEMINI_API_KEY = 'AIzaSyD1ezjuKYkM5PhxMHxkOjqP9S3eNYUljUo';

// Enhanced AI Analysis function with more detailed insights
// const generateMarketingInsight = async (trendData: TrendData, comparisonDays: number): Promise<string> => {
//   if (!GEMINI_API_KEY) {
//     return `${trendData.status === 'trending' ? 'Capitalize on' : 'Monitor'} this ${trendData.status} trend for strategic positioning with targeted campaigns and budget allocation.`;
//   }

//   const prompt = `
// You are a senior L'Oréal marketing strategist. Analyze this beauty trend data and provide a comprehensive marketing insight in exactly 75 words:

// Trend: "${trendData.topic}"
// Current Growth: ${trendData.currentGrowth.toFixed(1)}%
// Previous Growth (${comparisonDays} days ago): ${trendData.previousGrowth.toFixed(1)}%
// Change: ${trendData.change > 0 ? '+' : ''}${trendData.change.toFixed(1)}% (${trendData.changePercentage > 0 ? '+' : ''}${trendData.changePercentage.toFixed(1)}%)
// Status: ${trendData.status}

// Your analysis should include:
// 1. Strategic action (launch, pause, pivot, or monitor)
// 2. Target demographic considerations
// 3. Product positioning implications
// 4. Budget allocation recommendations
// 5. Timeline for implementation

// Provide actionable marketing strategy for L'Oréal beauty division in exactly 75 words. Focus on revenue optimization, market timing, and competitive advantage.
//   `;

//   try {
//     const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GEMINI_API_KEY}`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         contents: [{
//           parts: [{
//             text: prompt
//           }]
//         }],
//         generationConfig: {
//           temperature: 0.7,
//           topK: 40,
//           topP: 0.95,
//           maxOutputTokens: 200,
//         }
//       })
//     });

//     if (!response.ok) {
//       throw new Error('Failed to get AI insight');
//     }

//     const data = await response.json();
//     return data.candidates?.[0]?.content?.parts?.[0]?.text || `Leverage ${trendData.topic} trends with targeted campaigns. Focus on demographic alignment and product positioning. Consider budget reallocation based on growth momentum. Implement strategic timing for maximum market penetration and competitive advantage. Monitor performance metrics for optimization opportunities.`;
//   } catch (error) {
//     console.error('AI insight generation failed:', error);
//     // Enhanced fallback insights based on trend status
//     switch (trendData.status) {
//       case 'trending':
//         return `Launch accelerated campaigns for "${trendData.topic}" targeting Gen Z and millennials. Increase budget allocation by 30%. Focus on social media and influencer partnerships. Develop complementary product lines. Implement within 2 weeks to capture momentum. Leverage user-generated content and authentic testimonials for maximum engagement and conversion rates.`;
//       case 'declining':
//         return `Pivot from "${trendData.topic}" trends immediately. Reallocate 40% budget to emerging opportunities. Target loyal customer base with retention strategies. Consider clearance promotions for existing inventory. Focus on data-driven alternatives. Implement new positioning within 3 weeks. Analyze competitor strategies and identify untapped market segments for recovery.`;
//       default:
//         return `Monitor "${trendData.topic}" stability with steady investment approach. Maintain current budget levels. Target broad demographics with consistent messaging. Focus on brand loyalty and customer lifetime value. Implement gradual optimization strategies. Consider seasonal adjustments and regional variations. Perfect for sustained marketing efforts and predictable ROI generation.`;
//     }
//   }
// };

const analyzeTrendData = (jsonData: any, selectedPeriod: number, comparisonDays: number, topicCount: number): TrendData[] => {
  if (!jsonData || !jsonData.topics) return [];

  const topics = Object.keys(jsonData.topics).slice(0, topicCount);
  const trendData: TrendData[] = [];

  topics.forEach(topicName => {
    const topicData = jsonData.topics[topicName];
    
    // Get current period data (last few days of selected period)
    const currentPeriodStart = Math.max(1, selectedPeriod - comparisonDays + 1);
    const currentPeriodEnd = selectedPeriod;
    
    // Get previous period data
    const previousPeriodStart = Math.max(1, currentPeriodStart - comparisonDays);
    const previousPeriodEnd = currentPeriodStart - 1;

    // Calculate average growth for current period
    const currentPeriodData = topicData.daily_forecasts.filter(
      (forecast: any) => forecast.day >= currentPeriodStart && forecast.day <= currentPeriodEnd
    );
    const currentGrowth = currentPeriodData.length > 0 
      ? currentPeriodData.reduce((sum: number, forecast: any) => sum + forecast.growth_rate, 0) / currentPeriodData.length * 100
      : 0;

    // Calculate average growth for previous period
    const previousPeriodData = topicData.daily_forecasts.filter(
      (forecast: any) => forecast.day >= previousPeriodStart && forecast.day <= previousPeriodEnd
    );
    const previousGrowth = previousPeriodData.length > 0
      ? previousPeriodData.reduce((sum: number, forecast: any) => sum + forecast.growth_rate, 0) / previousPeriodData.length * 100
      : 0;

    const change = currentGrowth - previousGrowth;
    const changePercentage = previousGrowth !== 0 ? (change / previousGrowth) * 100 : 0;

    // Determine status
    let status: 'trending' | 'stable' | 'declining';
    if (changePercentage > 10) status = 'trending';
    else if (changePercentage < -10) status = 'declining';
    else status = 'stable';

    trendData.push({
      topic: topicName,
      currentGrowth,
      previousGrowth,
      change,
      changePercentage,
      status
    });
  });

  // Sort by absolute change percentage (most significant changes first)
  return trendData.sort((a, b) => Math.abs(b.changePercentage) - Math.abs(a.changePercentage));
};

// Helper function to determine which comparison periods should be available
const getAvailableComparisonPeriods = (selectedPeriod: number) => {
  const periods = [
    { value: 1, label: '1 Day' },
    { value: 7, label: '1 Week' },
    { value: 14, label: '2 Weeks' },
    { value: 21, label: '3 Weeks' },
    { value: 28, label: '4 Weeks' }
  ];

  // Filter out periods that would result in insufficient data
  // We need at least the comparison period worth of previous data
  return periods.filter(period => {
    const minRequiredPeriod = period.value * 2; // Current period + previous period
    return selectedPeriod >= minRequiredPeriod;
  });
};

export function TrendInsightCards({ jsonData, selectedPeriod, topicCount }: TrendInsightCardsProps) {
  const [comparisonPeriod, setComparisonPeriod] = useState(3); // Default 3 days
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  // const [loadingInsights, setLoadingInsights] = useState<{[key: string]: boolean}>({});

  // Get available comparison periods based on selected period
  const availableComparisonPeriods = getAvailableComparisonPeriods(selectedPeriod);

  // Update comparison period if current selection is not available
  useEffect(() => {
    if (availableComparisonPeriods.length > 0) {
      const isCurrentPeriodAvailable = availableComparisonPeriods.some(p => p.value === comparisonPeriod);
      if (!isCurrentPeriodAvailable) {
        // Set to the smallest available period
        setComparisonPeriod(availableComparisonPeriods[0].value);
      }
    }
  }, [selectedPeriod, availableComparisonPeriods, comparisonPeriod]);

  // Calculate trend data when inputs change
  useEffect(() => {
    if (jsonData) {
      const data = analyzeTrendData(jsonData, selectedPeriod, comparisonPeriod, topicCount);
      setTrendData(data);
    }
  }, [jsonData, selectedPeriod, comparisonPeriod, topicCount]);

  // Generate AI insights for top trends
  // const generateInsight = async (trend: TrendData, index: number) => {
  //   setLoadingInsights(prev => ({ ...prev, [index]: true }));
    
  //   try {
  //     const insight = await generateMarketingInsight(trend, comparisonPeriod);
  //     setTrendData(prev => prev.map((t, i) => 
  //       i === index ? { ...t, aiInsight: insight } : t
  //     ));
  //   } catch (error) {
  //     console.error('Failed to generate insight:', error);
  //   } finally {
  //     setLoadingInsights(prev => ({ ...prev, [index]: false }));
  //   }
  // };

  // Auto-generate insights for top 3 trends on data change
  // useEffect(() => {
  //   if (trendData.length > 0) {
  //     trendData.slice(0, 3).forEach((trend, index) => {
  //       if (!trend.aiInsight) {
  //         generateInsight(trend, index);
  //       }
  //     });
  //   }
  // }, [trendData.length]); // Only depend on length to avoid infinite loops

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'trending':
        return {
          bg: 'bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30',
          border: 'border-green-200 dark:border-green-700',
          badge: 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200',
          icon: <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
        };
      case 'declining':
        return {
          bg: 'bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-900/30 dark:to-pink-900/30',
          border: 'border-red-200 dark:border-red-700',
          badge: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200',
          icon: <TrendingDown className="w-4 h-4 text-red-600 dark:text-red-400" />
        };
      default:
        return {
          bg: 'bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/30 dark:to-cyan-900/30',
          border: 'border-blue-200 dark:border-blue-700',
          badge: 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200',
          icon: <Activity className="w-4 h-4 text-blue-600 dark:text-blue-400" />
        };
    }
  };

  const getComparisonPeriodLabel = (days: number) => {
    switch (days) {
      case 1: return '1 Day';
      case 7: return '1 Week';
      case 14: return '2 Weeks';
      case 21: return '3 Weeks';
      case 28: return '4 Weeks';
      default: return `${days} Days`;
    }
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-foreground flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400" />
            Trend Performance Analysis
          </h3>
          <Badge variant="outline" className="text-xs">
            Showing top {Math.min(9, trendData.length)} trends
          </Badge>
        </div>

        <div className="flex items-center space-x-3">
          <Calendar className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Compare with:</span>
          <Select 
            value={comparisonPeriod.toString()} 
            onValueChange={(value) => setComparisonPeriod(parseInt(value))}
            disabled={availableComparisonPeriods.length === 0}
          >
            <SelectTrigger className="w-32 h-8 text-xs border-border bg-background">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-background border-border">
              {availableComparisonPeriods.map(period => (
                <SelectItem 
                  key={period.value} 
                  value={period.value.toString()} 
                  className="text-foreground hover:bg-accent"
                >
                  {period.label}
                </SelectItem>
              ))}
              {availableComparisonPeriods.length === 0 && (
                <SelectItem value="0" disabled className="text-muted-foreground">
                  No comparison periods available
                </SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Trend Boxes Grid - Custom width containers instead of full-width cards */}
      <div className="flex flex-wrap gap-4">
        {trendData.slice(0, 9).map((trend, index) => {
          const statusStyle = getStatusColor(trend.status);
          // const isLoading = loadingInsights[index];
          
          return (
            <div 
              key={trend.topic} 
              className={`
                ${statusStyle.bg} ${statusStyle.border} 
                border-2 rounded-lg shadow-sm hover:shadow-lg transition-all duration-300 
                w-80 flex-shrink-0 p-4
              `}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  {statusStyle.icon}
                  <Badge className={`${statusStyle.badge} text-xs font-medium px-2 py-0.5`}>
                    {trend.status.charAt(0).toUpperCase() + trend.status.slice(1)}
                  </Badge>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-bold ${trend.change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {trend.change >= 0 ? '+' : ''}{trend.change.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">
                    vs {getComparisonPeriodLabel(comparisonPeriod)}
                  </div>
                </div>
              </div>

              {/* Title */}
              <h4 className="text-sm font-semibold text-foreground mb-3 leading-tight" title={trend.topic}>
                {trend.topic.length > 35 ? `${trend.topic.substring(0, 35)}...` : trend.topic}
              </h4>
              
              {/* Data Visualization */}
              <div className="space-y-2 mb-4">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">Current</span>
                  <span className="font-medium text-foreground">{trend.currentGrowth.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">Previous</span>
                  <span className="font-medium text-foreground">{trend.previousGrowth.toFixed(1)}%</span>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                  <div 
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      trend.status === 'trending' ? 'bg-gradient-to-r from-green-400 to-green-600' :
                      trend.status === 'declining' ? 'bg-gradient-to-r from-red-400 to-red-600' :
                      'bg-gradient-to-r from-blue-400 to-blue-600'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(5, trend.currentGrowth))}%` }}
                  />
                </div>
              </div>

              {/* AI Insight - COMMENTED OUT */}
              {/* <div className="p-3 bg-white/60 dark:bg-gray-800/60 rounded-md border border-gray-200/50 dark:border-gray-600/50">
                <div className="flex items-start space-x-2">
                  <div className="flex-shrink-0 mt-0.5">
                    {isLoading ? (
                      <Loader2 className="w-3 h-3 animate-spin text-purple-600 dark:text-purple-400" />
                    ) : (
                      <DollarSign className="w-3 h-3 text-purple-600 dark:text-purple-400" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-xs font-medium text-purple-800 dark:text-purple-200 mb-1">
                      Strategic Analysis
                    </p>
                    {isLoading ? (
                      <div className="space-y-1">
                        <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
                        <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-4/5"></div>
                        <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-3/5"></div>
                      </div>
                    ) : (
                      <p className="text-xs text-foreground leading-relaxed">
                        {trend.aiInsight || 'Generating strategic analysis...'}
                      </p>
                    )}
                  </div>
                </div>
                {!isLoading && !trend.aiInsight && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => generateInsight(trend, index)}
                    className="w-full mt-2 h-5 text-xs text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/30"
                  >
                    <Zap className="w-2.5 h-2.5 mr-1" />
                    Generate Analysis
                  </Button>
                )}
              </div> */}
            </div>
          );
        })}
      </div>

      {trendData.length === 0 && jsonData && (
        <div className="p-8 text-center border border-border rounded-lg">
          <Target className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
          <p className="text-muted-foreground">No trend data available for the selected period.</p>
        </div>
      )}

      {availableComparisonPeriods.length === 0 && jsonData && (
        <div className="p-4 text-center border border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20 dark:border-yellow-800 rounded-lg">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            Selected period ({selectedPeriod} days) is too short for meaningful trend comparison. 
            Please select a longer period to enable trend analysis.
          </p>
        </div>
      )}
    </div>
  );
}

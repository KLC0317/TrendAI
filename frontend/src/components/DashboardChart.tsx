import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';


export type TrendStatus = 'emerging' | 'established' | 'decaying';

export const TREND_COLORS = {
  emerging: '#8B5CF6',
  established: '#3498DB', 
  decaying: '#6B7280'    
} as const;


export const TREND_LABELS = {
  emerging: 'Emerging',
  established: 'Established',
  decaying: 'Decaying'
} as const;

interface TrendLine {
  key: string;
  name: string;
  status: TrendStatus;
  description?: string;
}

interface ChartDataPoint {
  date: string;
  [key: string]: string | number;
}

interface DashboardChartProps {
  data: ChartDataPoint[];
  trendLines: TrendLine[];
  period: number;
  className?: string;
  jsonData?: any;
  topicCount?: number;
}

// Extract data from the JSON structure
const extractDataFromJSON = (jsonData: any, selectedPeriod: number, topicCount: number = 10) => {
  if (!jsonData || !jsonData.topics || !jsonData.metadata) {
    return { chartData: [], trendLines: [] };
  }

  const topics = Object.keys(jsonData.topics).slice(0, topicCount);
  const chartData: ChartDataPoint[] = [];
  const trendLines: TrendLine[] = [];

  // Get the generation date as start date
  const generationDate = new Date(jsonData.metadata.generation_date);

  // Create trend lines based on growth rates
  topics.forEach((topicName, index) => {
    const topicData = jsonData.topics[topicName];
    const avgGrowthRate = topicData.average_growth_rate;
    
    let status: TrendStatus;
    if (avgGrowthRate > 0.8) {
      status = 'emerging';
    } else if (avgGrowthRate > 0.4) {
      status = 'established';
    } else {
      status = 'decaying';
    }
    
    trendLines.push({
      key: `topic_${index}`,
      name: topicName,
      status: status,
      description: `Growth rate: ${(avgGrowthRate * 100).toFixed(1)}%`
    });
  });

  // Create chart data points based on selected period
  const startDay = 1;
  const endDay = Math.min(selectedPeriod, 100);

  for (let day = startDay; day <= endDay; day++) {
    const currentDate = new Date(generationDate);
    currentDate.setDate(currentDate.getDate() + day - 1);
    
    const dataPoint: ChartDataPoint = {
      // Change this line to use actual formatted date
      date: currentDate.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: selectedPeriod > 60 ? 'numeric' : undefined // Show year only for longer periods
      })
    };

    topics.forEach((topicName, topicIndex) => {
      const topicData = jsonData.topics[topicName];
      let growthRate = 0;

      // Find exact match for this day
      const exactMatch = topicData.daily_forecasts.find((forecast: any) => forecast.day === day);
      
      if (exactMatch) {
        growthRate = exactMatch.growth_rate * 100;
      } else {
        // Interpolation logic remains the same
        const availableDays = topicData.daily_forecasts.map((f: any) => f.day).sort((a: number, b: number) => a - b);
        
        let lowerDay = null;
        let upperDay = null;
        
        for (let i = 0; i < availableDays.length - 1; i++) {
          if (availableDays[i] <= day && availableDays[i + 1] >= day) {
            lowerDay = availableDays[i];
            upperDay = availableDays[i + 1];
            break;
          }
        }
        
        if (lowerDay && upperDay) {
          const lowerForecast = topicData.daily_forecasts.find((f: any) => f.day === lowerDay);
          const upperForecast = topicData.daily_forecasts.find((f: any) => f.day === upperDay);
          
          if (lowerForecast && upperForecast) {
            const ratio = (day - lowerDay) / (upperDay - lowerDay);
            growthRate = (lowerForecast.growth_rate + 
                         (upperForecast.growth_rate - lowerForecast.growth_rate) * ratio) * 100;
          }
        } else if (day < availableDays[0]) {
          const firstForecast = topicData.daily_forecasts.find((f: any) => f.day === availableDays[0]);
          growthRate = firstForecast ? firstForecast.growth_rate * 100 : 0;
        } else if (day > availableDays[availableDays.length - 1]) {
          const lastForecast = topicData.daily_forecasts.find((f: any) => f.day === availableDays[availableDays.length - 1]);
          growthRate = lastForecast ? lastForecast.growth_rate * 100 : 0;
        }
      }

      dataPoint[`topic_${topicIndex}`] = Math.max(0, growthRate);
    });

    chartData.push(dataPoint);
  }

  return { chartData, trendLines };
};


// Custom hook to get theme-aware colors
function useThemeColors() {
  const [colors, setColors] = React.useState({
    border: '#e5e5e5',
    mutedForeground: '#6b7280',
    background: '#ffffff'
  });

  React.useEffect(() => {
    const updateColors = () => {
      const isDark = document.documentElement.classList.contains('dark');
      
      setColors({
        border: isDark ? '#374151' : '#e5e5e5',
        mutedForeground: isDark ? '#9ca3af' : '#6b7280',
        background: isDark ? '#1f2937' : '#ffffff'
      });
    };

    updateColors();

    // Watch for theme changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && 
            (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme')) {
          updateColors();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme']
    });

    return () => observer.disconnect();
  }, []);

  return colors;
}

// 自定义Tooltip组件
function CustomTooltip({ active, payload, label }: any) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-popover border border-border rounded-lg shadow-lg p-4 max-w-xs">
        <p className="font-medium text-popover-foreground mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center justify-between mb-1">
            <div className="flex items-center">
              <div
                className="w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-muted-foreground truncate max-w-[120px]" title={entry.name}>
                {entry.name.length > 15 ? `${entry.name.substring(0, 15)}...` : entry.name}
              </span>
            </div>
            <span className="text-sm font-medium text-popover-foreground ml-2">
              {typeof entry.value === 'number' ? `${entry.value.toFixed(1)}%` : entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}

// 自定义Legend组件
function CustomLegend({ trendLines }: { trendLines: TrendLine[] }) {
  const categoryColors = [
    '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
    '#06b6d4', '#84cc16', '#f97316', '#a855f7', '#10b981'
  ];

  return (
    <div className="flex flex-wrap gap-3 justify-center mb-4">
      {trendLines.map((line, index) => {
        const lineColor = categoryColors[index % categoryColors.length];

        return (
          <div key={line.key} className="flex items-center space-x-2 max-w-[200px]">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: lineColor }}
            />
            <span 
              className="text-xs text-foreground truncate" 
              title={`${line.name} - ${line.description || ''}`}
            >
              {line.name.length > 25 ? `${line.name.substring(0, 25)}...` : line.name}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function DashboardChart({ 
  data, 
  trendLines, 
  period, 
  className = '', 
  jsonData, 
  topicCount = 10 
}: DashboardChartProps) {
  const themeColors = useThemeColors();

  // Extract data from JSON if provided
  const { chartData: extractedData, trendLines: extractedTrendLines } = React.useMemo(() => {
    if (jsonData) {
      return extractDataFromJSON(jsonData, period, topicCount);
    }
    return { chartData: data, trendLines: trendLines };
  }, [jsonData, period, topicCount, data, trendLines]);

  const finalData = extractedData.length > 0 ? extractedData : data;
  const finalTrendLines = extractedTrendLines.length > 0 ? extractedTrendLines : trendLines;

  // Get generation date for display
  const generationDate = jsonData?.metadata?.generation_date ? 
    new Date(jsonData.metadata.generation_date).toLocaleDateString() : 
    new Date().toLocaleDateString();

  return (
    <div className={`bg-card border border-border rounded-lg p-6 ${className}`}>
      {/* Chart Header */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-card-foreground mb-1">Trend Analytics Comparison</h3>
        <p className="text-sm text-muted-foreground mb-2">
          Multi-trend comparison over {period} days - Showing {finalTrendLines.length} topics
        </p>
        {jsonData && (
          <p className="text-xs text-muted-foreground">
            Data generated: {generationDate} | Forecast range: {jsonData.metadata?.forecast_range_days} days
          </p>
        )}
      </div>

      {/* Custom Legend */}
      <CustomLegend trendLines={finalTrendLines} />

      {/* Chart Container */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={finalData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={themeColors.border}
              horizontal={true}
              vertical={false}
            />
            <XAxis
              dataKey="date"
              stroke={themeColors.mutedForeground}
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tick={{ fill: themeColors.mutedForeground, fontSize: 12 }}
            />
            <YAxis
              stroke={themeColors.mutedForeground}
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value.toFixed(0)}%`}
              domain={[0, 'dataMax + 20']}
              tick={{ fill: themeColors.mutedForeground, fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Render lines based on extracted data */}
            {finalTrendLines.map((line, index) => {
              const categoryColors = [
                '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
                '#06b6d4', '#84cc16', '#f97316', '#a855f7', '#10b981'
              ];

              const lineColor = categoryColors[index % categoryColors.length];

              return (
                <Line
                  key={line.key}
                  type="monotone"
                  dataKey={line.key}
                  name={line.name}
                  stroke={lineColor}
                  strokeWidth={2.5}
                  dot={{
                    fill: 'transparent',
                    stroke: 'transparent',
                    strokeWidth: 0,
                    r: 0
                  }}
                  activeDot={{
                    r: 5,
                    fill: lineColor,
                    strokeWidth: 2,
                    stroke: themeColors.background
                  }}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

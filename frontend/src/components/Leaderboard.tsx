import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { TrendingUp, Loader2, Play, X, ArrowUp, ArrowDown, ChevronDown } from 'lucide-react';
import { MonthSelector } from './MonthSelector';

interface TrendRanking {
  rank: number;
  topic: string;
  growthRate: number;
  previousRank?: number;
}

interface TagItem {
  rank: number;
  tag: string;
}

interface VideoItem {
  id: string;
  title: string;
  thumbnail: string;
  views: string;
  duration: string;
  url: string;
}

interface LeaderboardProps {
  className?: string;
}

export function Leaderboard({ className = '' }: LeaderboardProps) {
  const [leaderboardData, setLeaderboardData] = useState<TrendRanking[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<string>('');
  const [selectedVideo, setSelectedVideo] = useState<VideoItem | null>(null);
  const [showVideoModal, setShowVideoModal] = useState(false);
  const [trendsLimit, setTrendsLimit] = useState<10 | 25>(10);
  const [showTrendsDropdown, setShowTrendsDropdown] = useState(false);
  const trendsDropdownRef = useRef<HTMLDivElement>(null);

  const [tagsData] = useState<TagItem[]>([
    { rank: 1, tag: 'косметика' },
    { rank: 2, tag: 'уход за волосами' },
    { rank: 3, tag: 'уход' },
    { rank: 4, tag: 'mone professional' },
    { rank: 5, tag: 'средства для волос' },
    { rank: 6, tag: 'makeup challenge' },
    { rank: 7, tag: 'makeup shorts' },
    { rank: 8, tag: 'dating' },
    { rank: 9, tag: 'women' },
    { rank: 10, tag: 'dating tips' },
  ]);

  const [videosData] = useState<VideoItem[]>([
    { 
      id: 'aYrnObMCnyI', 
      title: 'YouTube Shorts Video 1', 
      thumbnail: 'https://img.youtube.com/vi/aYrnObMCnyI/maxresdefault.jpg', 
      views: '1.2M', 
      duration: '0:30',
      url: 'https://youtube.com/shorts/aYrnObMCnyI?si=5A0tcMvaBLqcydCr'
    },
    { 
      id: 'rLQYad1Z1Pg', 
      title: 'YouTube Shorts Video 2', 
      thumbnail: 'https://img.youtube.com/vi/rLQYad1Z1Pg/maxresdefault.jpg', 
      views: '850K', 
      duration: '0:45',
      url: 'https://youtube.com/shorts/rLQYad1Z1Pg?si=2bd_bva-e5MM-gQ5'
    },
  ]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (trendsDropdownRef.current && !trendsDropdownRef.current.contains(event.target as Node)) {
        setShowTrendsDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Function to get ranking change component with enhanced animations
  const getRankingChangeComponent = (currentRank: number, previousRank?: number) => {
    if (!previousRank) {
      return null;
    }
    
    const change = previousRank - currentRank; // Positive = improvement, Negative = decline
    
    if (change > 0) {
      // Rank improved (moved up)
      return (
        <div className="flex items-center ml-2 text-green-500 dark:text-green-400 animate-pulse">
          <div className="relative">
            <ArrowUp className="w-4 h-4 animate-bounce" />
            <div className="absolute inset-0 animate-ping">
              <ArrowUp className="w-4 h-4 opacity-75" />
            </div>
          </div>
          <span className="text-xs font-semibold ml-1 animate-fade-in">+{change}</span>
        </div>
      );
    } else if (change < 0) {
      // Rank dropped (moved down) 
      return (
        <div className="flex items-center ml-2 text-red-500 dark:text-red-400 animate-pulse">
          <div className="relative">
            <ArrowDown className="w-4 h-4 animate-bounce" />
            <div className="absolute inset-0 animate-ping">
              <ArrowDown className="w-4 h-4 opacity-75" />
            </div>
          </div>
          <span className="text-xs font-semibold ml-1 animate-fade-in">{change}</span>
        </div>
      );
    }
    
    return null; // No change, show nothing
  };

  useEffect(() => {
    const fetchDefaultMonthData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const currentDate = new Date();
        // Set to next month instead of current month
        const nextMonthDate = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1);
        const nextMonth = `${nextMonthDate.getFullYear()}-${String(nextMonthDate.getMonth() + 1).padStart(2, '0')}`;
        setSelectedMonth(nextMonth);

        const response = await fetch('/top_25_trends_extraction.json');
        const data = await response.json();

        // Since we're defaulting to next month, use 60-day forecast with 30-day as previous
        const currentForecastData = data['60_day_forecast'];
        const previousForecastData = data['30_day_forecast'];

        const transformedData: TrendRanking[] = currentForecastData.top_25_trends.map((trend: any) => {
          let previousRank = undefined;
          
          if (previousForecastData && previousForecastData.top_25_trends) {
            // Find the same topic in previous forecast data by matching topic
            const previousTrend = previousForecastData.top_25_trends.find(
              (prevTrend: any) => prevTrend.topic === trend.topic
            );
            previousRank = previousTrend?.rank;
          }

          return {
            rank: trend.rank,
            topic: trend.topic,
            growthRate: trend.growth_rate * 100,
            previousRank: previousRank
          };
        });

        setLeaderboardData(transformedData);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : 'Failed to fetch default month data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDefaultMonthData();
  }, []);

  const handleMonthChange = async (month: string) => {
    setSelectedMonth(month);
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch('/top_25_trends_extraction.json');
      const data = await response.json();

      const currentDate = new Date();
      const selectedDate = new Date(month + '-01');
      const monthDiff = selectedDate.getMonth() - currentDate.getMonth();

      let currentForecastData, previousForecastData;

      if (monthDiff === 0) {
        // Current month - show 30-day forecast with no previous comparison
        currentForecastData = data['30_day_forecast'];
        previousForecastData = null;
      } else if (monthDiff === 1) {
        // Next month - show 60-day forecast compared to 30-day forecast
        currentForecastData = data['60_day_forecast'];
        previousForecastData = data['30_day_forecast'];
      } else if (monthDiff === 2) {
        // Month after - show 90-day forecast compared to 60-day forecast
        currentForecastData = data['90_day_forecast'];
        previousForecastData = data['60_day_forecast'];
      } else {
        // Fallback to 30-day forecast
        currentForecastData = data['30_day_forecast'];
        previousForecastData = null;
      }

      const transformedData: TrendRanking[] = currentForecastData.top_25_trends.map((trend: any) => {
        let previousRank = undefined;
        
        if (previousForecastData) {
          // Find the same topic in previous forecast data by matching topic
          const previousTrend = previousForecastData.top_25_trends.find(
            (prevTrend: any) => prevTrend.topic === trend.topic
          );
          previousRank = previousTrend?.rank;
        }

        return {
          rank: trend.rank,
          topic: trend.topic,
          growthRate: trend.growth_rate * 100,
          previousRank: previousRank
        };
      });

      setLeaderboardData(transformedData);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Failed to fetch month data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVideoClick = (video: VideoItem) => {
    setSelectedVideo(video);
    setShowVideoModal(true);
  };

  const closeVideoModal = () => {
    setShowVideoModal(false);
    setSelectedVideo(null);
  };

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeVideoModal();
      }
    };

    if (showVideoModal) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [showVideoModal]);

  // Handle trends limit selection
  const handleTrendsLimitSelect = (limit: 10 | 25) => {
    setTrendsLimit(limit);
    setShowTrendsDropdown(false);
  };

  // Get displayed trends based on selected limit
  const displayedTrends = leaderboardData.slice(0, trendsLimit);

  return (
    <>
      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: scale(0.9);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.6s ease-out;
        }
      `}</style>
      
      <div className={`w-full h-full flex gap-6 p-4 ${className}`}>
        <Card className="flex-1 min-w-0 h-full flex flex-col">
          <CardHeader className="pb-4 flex-shrink-0">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center text-xl font-semibold">
                <TrendingUp className="mr-2 h-5 w-5 text-primary" />
                Trends Leaderboard
              </CardTitle>
              <div className="flex items-center gap-3">
                {/* Top 10/25 Dropdown */}
                <div className="relative" ref={trendsDropdownRef}>
                  <button
                    onClick={() => setShowTrendsDropdown(!showTrendsDropdown)}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium bg-background border border-border rounded-md hover:bg-accent focus:outline-none focus:ring-2 focus:ring-ring focus:border-input transition-colors"
                  >
                    Top {trendsLimit}
                    <ChevronDown className={`w-4 h-4 transition-transform ${showTrendsDropdown ? 'rotate-180' : ''}`} />
                  </button>
                  
                  {showTrendsDropdown && (
                    <div className="absolute right-0 mt-2 w-32 bg-popover border border-border rounded-md shadow-lg z-10">
                      <div className="py-1">
                        <button
                          onClick={() => handleTrendsLimitSelect(10)}
                          className={`w-full text-left px-4 py-2 text-sm hover:bg-accent transition-colors ${
                            trendsLimit === 10 ? 'bg-accent text-accent-foreground font-medium' : ''
                          }`}
                        >
                          Top 10
                        </button>
                        <button
                          onClick={() => handleTrendsLimitSelect(25)}
                          className={`w-full text-left px-4 py-2 text-sm hover:bg-accent transition-colors ${
                            trendsLimit === 25 ? 'bg-accent text-accent-foreground font-medium' : ''
                          }`}
                        >
                          Top 25
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                <MonthSelector selectedMonth={selectedMonth} onMonthChange={handleMonthChange} />
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0 flex-1 overflow-hidden">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Loading trends...</span>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center py-12 text-destructive">
                <span>{error}</span>
              </div>
            ) : (
              <div className="h-full overflow-auto">
                <table className="w-full">
                  <thead className="sticky top-0 bg-muted/50">
                    <tr className="border-b border-border">
                      <th className="text-left px-6 py-3 text-sm font-medium text-muted-foreground uppercase tracking-wide w-32">
                        Rank
                      </th>
                      <th className="text-left px-6 py-3 text-sm font-medium text-muted-foreground uppercase tracking-wide">
                        Topic
                      </th>
                      <th className="text-center px-4 py-3 text-sm font-medium text-muted-foreground uppercase tracking-wide w-32">
                        Growth Rate
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayedTrends.map((trend, index) => (
                      <tr 
                        key={trend.rank} 
                        className={`transition-colors hover:bg-accent/50 border-b border-border ${
                          index % 2 === 0 ? 'bg-background' : 'bg-muted/30'
                        }`}
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center">
                            <span className="text-sm font-semibold">
                              #{trend.rank}
                            </span>
                            {getRankingChangeComponent(trend.rank, trend.previousRank)}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm font-medium capitalize">
                          {trend.topic}
                        </td>
                        <td className="px-4 py-4 text-sm font-semibold text-center">
                          {trend.growthRate.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right sidebar with tags and videos */}
        <div className="w-80 h-full flex flex-col gap-4 flex-shrink-0">
          {/* Tags section - reduced height */}
          <Card className="flex flex-col" style={{ height: '280px' }}>
            <CardHeader className="flex-shrink-0 pb-3">
              <CardTitle className="text-lg font-semibold">Top 10 Tags</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden px-4 pb-4">
              <div className="h-full overflow-auto space-y-2">
                {tagsData.map((tagItem) => (
                  <div
                    key={tagItem.rank}
                    className="flex items-center py-2 px-3 bg-muted/50 rounded-md hover:bg-muted transition-colors cursor-pointer"
                  >
                    <span className="text-xs font-medium text-muted-foreground mr-3 w-6 flex-shrink-0">
                      #{tagItem.rank}
                    </span>
                    <span className="text-sm font-medium truncate">
                      {tagItem.tag}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Videos section - now takes up more space */}
          <Card className="flex-1 flex flex-col">
            <CardHeader className="flex-shrink-0 pb-3">
              <CardTitle className="text-lg font-semibold">Trending Videos</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden px-4 pb-4">
              <div className="h-full overflow-auto space-y-3">
                {videosData.map((video) => (
                  <div
                    key={video.id}
                    onClick={() => handleVideoClick(video)}
                    className="group cursor-pointer bg-muted/50 rounded-lg p-3 hover:bg-muted transition-colors"
                  >
                    <div className="relative mb-2">
                      <img
                        src={video.thumbnail}
                        alt={video.title}
                        className="w-full h-32 object-cover rounded-md"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = `https://img.youtube.com/vi/${video.id}/hqdefault.jpg`;
                        }}
                      />
                      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
                        <Play className="w-8 h-8 text-white" />
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                        {video.duration}
                      </div>
                    </div>
                    <h3 className="text-sm font-medium line-clamp-2 mb-1">
                      {video.title}
                    </h3>
                    <p className="text-xs text-muted-foreground">{video.views} views</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Video Modal */}
        {showVideoModal && selectedVideo && (
          <div className="fixed inset-0 z-50 flex items-center justify-center">
            <div 
              className="absolute inset-0 bg-black bg-opacity-75"
              onClick={closeVideoModal}
            />
            
            <div className="relative bg-background rounded-lg shadow-2xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden border border-border">
              <div className="flex items-center justify-between p-4 border-b border-border">
                <h3 className="text-lg font-semibold truncate pr-4">
                  {selectedVideo.title}
                </h3>
                <button
                  onClick={closeVideoModal}
                  className="flex-shrink-0 p-1 hover:bg-accent rounded-full transition-colors"
                >
                  <X className="w-6 h-6 text-muted-foreground" />
                </button>
              </div>
              
              <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
                <iframe
                  className="absolute top-0 left-0 w-full h-full"
                  src={`https://www.youtube.com/embed/${selectedVideo.id}?autoplay=1&rel=0`}
                  title={selectedVideo.title}
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
              
              <div className="p-4 border-t border-border">
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>{selectedVideo.views} views</span>
                  <span>Duration: {selectedVideo.duration}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

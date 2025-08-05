import React, { useState, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  MagnifyingGlassIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  CalendarDaysIcon,
  HeartIcon,
  StarIcon,
  XMarkIcon,
  ChartBarSquareIcon,
  ClockIcon,
  BuildingOfficeIcon,
  MapPinIcon,
  ChatBubbleLeftRightIcon,
  PaperAirplaneIcon
} from '@heroicons/react/24/outline';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  Filler,
  ChartData
} from 'chart.js';
import { getMarketTrends, ragChat, MarketTrends } from '../services/api';

// Dark background plugin for chart area
const chartAreaBackgroundPlugin = {
  id: 'chartAreaBackground',
  beforeDraw: (chart: any) => {
    const { ctx, chartArea } = chart;
    if (!chartArea) return;
    ctx.save();
    ctx.fillStyle = '#18181B';
    ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
    ctx.restore();
  },
};

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  chartAreaBackgroundPlugin
);

interface LocationData {
  id: string;
  name: string;
  trends: MarketTrends | null;
  isLoading: boolean;
  error: Error | null;
}

interface PropPulseScore {
  score: number;
  letterGrade: string;
  components: {
    roi: number;
    momentum: number;
    volatility: number;
    risk: number;
  };
}

type ChartMetric = 'median_price' | 'price_per_sqft' | 'roi_forecast' | 'days_on_market' | 'inventory_turnover' | 'price_momentum' | 'volatility';

const ComparativeAnalysis: React.FC = () => {
  const [locations, setLocations] = useState<LocationData[]>([]);
  const [searchInputs, setSearchInputs] = useState(['', '']); // Two search inputs
  const [timeRange, setTimeRange] = useState<'1m' | '3m' | '6m' | '1y' | '5y'>('6m');
  const [selectedMetric, setSelectedMetric] = useState<ChartMetric>('median_price');
  const [viewMode, setViewMode] = useState<'chart' | 'metrics'>('chart');

  // Add a new location
  const addLocation = useCallback((index: number) => {
    const input = searchInputs[index].trim();
    if (input && !locations.some(loc => loc.name === input)) {
      setLocations(prev => [...prev, {
        id: Date.now().toString(),
        name: input,
        trends: null,
        isLoading: false,
        error: null
      }]);
      // Clear only the used input
      setSearchInputs(prev => prev.map((val, i) => i === index ? '' : val));
    }
  }, [searchInputs, locations]);

  // Remove a location
  const removeLocation = useCallback((id: string) => {
    setLocations(prev => prev.filter(loc => loc.id !== id));
  }, []);

  // Handle Enter key press
  const handleKeyPress = (e: React.KeyboardEvent, index: number) => {
    if (e.key === 'Enter') {
      addLocation(index);
    }
  };

  // Handle search input change
  const handleSearchInputChange = (value: string, index: number) => {
    setSearchInputs(prev => prev.map((val, i) => i === index ? value : val));
  };

  // Fixed queries for two locations
  const query1 = useQuery<MarketTrends>({
    queryKey: ['marketTrends', locations[0]?.name, timeRange],
    queryFn: () => getMarketTrends(locations[0].name),
    enabled: !!locations[0]?.name && !locations[0]?.trends && !locations[0]?.error,
  });

  const query2 = useQuery<MarketTrends>({
    queryKey: ['marketTrends', locations[1]?.name, timeRange],
    queryFn: () => getMarketTrends(locations[1].name),
    enabled: !!locations[1]?.name && !locations[1]?.trends && !locations[1]?.error,
  });

  // Update locations when queries change
  React.useEffect(() => {
    if (query1.data || query1.error) {
      setLocations(prev => prev.map((loc, i) => 
        i === 0 ? {
          ...loc,
          trends: query1.data || null,
          error: query1.error || null,
          isLoading: false
        } : loc
      ));
    }
  }, [query1.data, query1.error]);

  React.useEffect(() => {
    if (query2.data || query2.error) {
      setLocations(prev => prev.map((loc, i) => 
        i === 1 ? {
          ...loc,
          trends: query2.data || null,
          error: query2.error || null,
          isLoading: false
        } : loc
      ));
    }
  }, [query2.data, query2.error]);

  // Update chart options to handle different value types
  const getChartOptions = (): ChartOptions<'line'> => {
    const baseOptions: ChartOptions<'line'> = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: `${selectedMetric.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} Over Time`,
        },
      },
    };

    // Customize y-axis based on metric type
    switch (selectedMetric) {
      case 'roi_forecast':
        return {
          ...baseOptions,
          scales: {
            y: {
              type: 'linear',
              beginAtZero: false,
              ticks: {
                callback: function(tickValue) {
                  return `$${Number(tickValue).toLocaleString(undefined, {maximumFractionDigits: 0})}`;
                },
              },
            },
          },
        };
      case 'inventory_turnover':
        return {
          ...baseOptions,
          scales: {
            y: {
              type: 'linear',
              beginAtZero: true,
              ticks: {
                callback: function(tickValue) {
                  return `${Number(tickValue).toFixed(2)}x`;
                },
              },
            },
          },
        };
      case 'volatility':
        return {
          ...baseOptions,
          scales: {
            y: {
              type: 'linear',
              beginAtZero: true,
              ticks: {
                callback: function(tickValue) {
                  return `${Number(tickValue).toFixed(2)}%`;
                },
              },
            },
          },
        };
      default:
        return {
          ...baseOptions,
          scales: {
            y: {
              type: 'linear',
              beginAtZero: false,
              ticks: {
                callback: function(tickValue) {
                  return `$${Number(tickValue).toLocaleString(undefined, {maximumFractionDigits: 0})}`;
                },
              },
            },
          },
        };
    }
  };

  // Enhanced comparison chart data generation
  const generateComparisonChartData = () => {
    const getRangeMonths = (range: string) => {
      switch (range) {
        case '1m':
          return 1;
        case '3m':
          return 3;
        case '6m':
          return 6;
        case '1y':
          return 12;
        case '5y':
        default:
          return 60; // assume monthly data up to 5 years
      }
    };
    const rangeMonths = getRangeMonths(timeRange);

    const chartData = {
      labels: [] as string[],
      datasets: locations.map((location, index) => {
        const colors = [
          'rgb(59, 130, 246)',   // Blue
          '#87CEEB',   // Powder Blue (formerly Green)
          'rgb(245, 158, 11)',   // Yellow
          'rgb(239, 68, 68)',    // Red
          'rgb(139, 92, 246)',   // Purple
        ];

        let values: number[] = [];
        let dates: string[] = [];

        if (location.trends?.market_data?.historical_data) {
          switch (selectedMetric) {
            case 'median_price':
              values = location.trends.market_data.historical_data.median_list_price?.values || [];
              dates = location.trends.market_data.historical_data.median_list_price?.dates || [];
              break;
            case 'price_per_sqft':
              values = location.trends.market_data.historical_data.price_per_sqft?.values || [];
              dates = location.trends.market_data.historical_data.price_per_sqft?.dates || [];
              break;
            case 'roi_forecast':
              if (location.trends.market_analysis?.forecast) {
                values = [
                  location.trends.market_analysis.forecast.short_term_forecast.value,
                  location.trends.market_analysis.forecast.medium_term_forecast.value,
                  location.trends.market_analysis.forecast.long_term_forecast.value
                ];
                dates = ['Short-term', 'Medium-term', 'Long-term'];
              }
              break;
            case 'days_on_market':
              // Use historical median DOM data if available, otherwise use current value
              if (location.trends.market_data.historical_data.median_dom?.values) {
                values = location.trends.market_data.historical_data.median_dom.values;
                dates = location.trends.market_data.historical_data.median_dom.dates;
              } else {
                values = [location.trends.market_data.current_metrics?.avg_days_on_market || 0];
                dates = ['Current'];
              }
              break;
            case 'inventory_turnover': {
              // Inventory Turnover = 12 / (Median DOM / 30)
              const doms = location.trends.market_data.historical_data.median_dom?.values;
              dates = location.trends.market_data.historical_data.median_dom?.dates || [];
              if (doms && doms.length) {
                values = doms.map(dom => dom > 0 ? 12 / (dom / 30) : 0);
              } else {
                values = [location.trends.market_analysis?.market_health?.inventory_turnover || 0];
                dates = ['Current'];
              }
              break;
            }
            case 'price_momentum':
              // Calculate price momentum trend using historical data
              if (location.trends.market_data.historical_data.median_list_price?.values) {
                const prices = location.trends.market_data.historical_data.median_list_price.values;
                values = prices.map((price, i) => {
                  if (i === 0) return 0;
                  return ((price - prices[i - 1]) / prices[i - 1]) * 100;
                });
                dates = location.trends.market_data.historical_data.median_list_price.dates;
              } else {
                values = [location.trends.market_analysis?.market_health?.price_momentum || 0];
                dates = ['Current'];
              }
              break;
            case 'volatility': {
              // Volatility = rolling coefficient of variation (as percent)
              const prices = location.trends.market_data.historical_data.median_list_price?.values;
              dates = location.trends.market_data.historical_data.median_list_price?.dates || [];
              if (prices && prices.length) {
                const windowSize = 3;
                values = prices.map((_, i) => {
                  if (i < windowSize - 1) return 0;
                  const window = prices.slice(i - windowSize + 1, i + 1);
                  const mean = window.reduce((a, b) => a + b, 0) / windowSize;
                  const variance = window.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / windowSize;
                  return mean > 0 ? (Math.sqrt(variance) / mean) * 100 : 0; // percent
                });
              } else {
                values = [(location.trends.market_analysis?.volatility?.coefficient_of_variation || 0) * 100];
                dates = ['Current'];
              }
              break;
            }
          }
        }

        // Slice to respect selected time range (for metrics with monthly arrays)
        if (dates.length > 0 && values.length === dates.length && ['median_price','price_per_sqft','days_on_market','inventory_turnover','price_momentum','volatility'].includes(selectedMetric)) {
          values = values.slice(-rangeMonths);
          dates = dates.slice(-rangeMonths);
        }

        return {
          label: location.name,
          data: values,
          borderColor: colors[index % colors.length],
          backgroundColor: colors[index % colors.length].replace('rgb', 'rgba').replace(')', ', 0.5)'),
          tension: 0.4,
          fill: false,
        };
      }),
    };

    // Set labels based on the first location's dates
    const firstLocation = locations[0];
    if (firstLocation?.trends?.market_data?.historical_data) {
      switch (selectedMetric) {
        case 'median_price':
        case 'price_per_sqft':
        case 'days_on_market':
        case 'inventory_turnover':
        case 'price_momentum':
        case 'volatility':
          const rawDates = firstLocation.trends.market_data.historical_data.median_list_price?.dates || [];
          const slicedDates = rawDates.slice(-rangeMonths);
          chartData.labels = slicedDates.map((date: string) =>
            new Date(Math.floor(parseInt(date) / 100), (parseInt(date) % 100) - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
          );
          break;
        case 'roi_forecast':
          chartData.labels = ['Short-term', 'Medium-term', 'Long-term'];
          break;
        default:
          chartData.labels = ['Current'];
      }
    }

    return chartData;
  };

  // Add PropPulse Score calculation
  const calculatePropPulseScore = (marketAnalysis: any): PropPulseScore | null => {
    if (!marketAnalysis?.price_trends || !marketAnalysis?.market_health || !marketAnalysis?.volatility) {
      return null;
    }
    const { price_trends, market_health, volatility } = marketAnalysis;
    // ROI
    const roiScore = Math.min(100, Math.max(0, ((price_trends.short_term_trend + price_trends.medium_term_trend) / 2) * 10));
    // Momentum
    const momentumScore = Math.min(100, Math.max(0, market_health.price_momentum * 5));
    // Volatility (lower is better)
    const volatilityScore = Math.max(0, 100 - (volatility.coefficient_of_variation * 100));
    // Inventory Turnover (higher is better)
    const invTurn = Math.min(100, Math.max(0, (market_health.inventory_turnover / 5) * 100));
    // Market Balance: Seller's Market = 100, Balanced = 70, Buyer's Market = 40
    let marketBalanceScore = 70;
    if (market_health.market_balance === "Seller's Market") marketBalanceScore = 100;
    else if (market_health.market_balance === "Buyer's Market") marketBalanceScore = 40;
    // New risk score: lower volatility, higher inv turnover, and seller's market = lower risk
    // Risk = 100 - (0.5 * volatilityScore + 0.3 * invTurn + 0.2 * marketBalanceScore)
    const riskScore = Math.max(0, 100 - (0.5 * volatilityScore + 0.3 * invTurn + 0.2 * marketBalanceScore));
    // Weighted score
    const weightedScore = Math.min(100, Math.max(0,
      (roiScore * 0.4) + (momentumScore * 0.25) + (volatilityScore * 0.2) + (riskScore * 0.15)
    ));
    const getLetterGrade = (score: number) => {
      if (score >= 90) return 'A+';
      if (score >= 85) return 'A';
      if (score >= 80) return 'A-';
      if (score >= 75) return 'B+';
      if (score >= 70) return 'B';
      if (score >= 65) return 'B-';
      if (score >= 60) return 'C+';
      if (score >= 55) return 'C';
      if (score >= 50) return 'C-';
      if (score >= 45) return 'D+';
      if (score >= 40) return 'D';
      if (score >= 35) return 'D-';
      return 'F';
    };
    return {
      score: Number(weightedScore.toFixed(1)),
      letterGrade: getLetterGrade(weightedScore),
      components: {
        roi: Number(roiScore.toFixed(1)),
        momentum: Number(momentumScore.toFixed(1)),
        volatility: Number(volatilityScore.toFixed(1)),
        risk: Number(riskScore.toFixed(1)),
      },
    };
  };

  // --- RAG CHATBOT UI ---
  const [chatHistory, setChatHistory] = React.useState<{role: 'user'|'ai', message: string}[]>([]);
  const [chatInput, setChatInput] = React.useState('');
  const [isChatLoading, setIsChatLoading] = React.useState(false);

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    setChatHistory(prev => [...prev, { role: 'user', message: chatInput }]);
    setIsChatLoading(true);
    
    // Gather context from compared locations
    const context = locations.map(loc => {
      if (!loc.trends) return '';
      return `Location: ${loc.name}\nMedian Price: $${loc.trends.market_data?.current_metrics?.median_price?.toLocaleString()}\nPrice Momentum: ${loc.trends.market_analysis?.market_health?.price_momentum}%\nVolatility: ${loc.trends.market_analysis?.volatility?.coefficient_of_variation}\nInventory Turnover: ${loc.trends.market_analysis?.market_health?.inventory_turnover}\nROI Forecast (Medium): $${loc.trends.market_analysis?.forecast?.medium_term_forecast?.value?.toLocaleString()}`;
    }).join('\n---\n');
    
    try {
      const data = await ragChat(
        chatInput,
        context,
        chatHistory.filter(h => h.role === 'user').map(h => h.message)
      );
      setChatHistory(prev => [...prev, { role: 'ai', message: data.answer }]);
    } catch (err) {
      console.error('Chat error:', err);
      setChatHistory(prev => [...prev, { role: 'ai', message: 'Sorry, I could not answer your question (server error).' }]);
    }
    
    setIsChatLoading(false);
    setChatInput('');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" style={{ background: '#121212' }}>
      {/* Header with gradient background */}
      <div className="bg-gradient-to-r from-[#3B4D5E] to-[#4F627A] rounded-xl p-8 mb-8 text-white">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold mb-2">Comparative Market Analysis</h1>
            <p className="text-primary-100">Compare multiple locations side by side</p>
          </div>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as '1m' | '3m' | '6m' | '1y' | '5y')}
            className="px-4 py-2 bg-[#18181B] text-white border border-[#232336] rounded-lg focus:ring-2 focus:ring-[#87CEEB] focus:border-[#87CEEB] min-w-fit pr-28"
          >
            <option value="1m">1 Month</option>
            <option value="3m">3 Months</option>
            <option value="6m">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="5y">5 Years</option>
          </select>
        </div>
      </div>

      {/* Search Inputs with enhanced styling */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {searchInputs.map((input, index) => (
          <div key={index} className="relative rounded-lg shadow-sm border border-[#232336] p-2" style={{ background: '#0D0D0D' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => handleSearchInputChange(e.target.value, index)}
              onKeyPress={(e) => handleKeyPress(e, index)}
              placeholder={`Enter ${index + 1}${index === 0 ? 'st' : 'nd'} location (City, State or ZIP)`}
              className="w-full px-4 py-2 border-0 focus:ring-0 focus:outline-none bg-transparent text-white placeholder:text-gray-400"
            />
            <button
              onClick={() => addLocation(index)}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-[#87CEEB]"
            >
              <MagnifyingGlassIcon className="h-5 w-5" />
            </button>
          </div>
        ))}
      </div>

      {/* Location Tags with enhanced styling */}
      <div className="flex flex-wrap gap-2 mb-6">
        {locations.map(location => (
          <div
            key={location.id}
            className="flex items-center gap-2 px-4 py-2 bg-[#18181B] text-white rounded-full border border-[#232336]"
          >
            <span className="font-medium">{location.name}</span>
            <button
              onClick={() => removeLocation(location.id)}
              className="hover:text-[#87CEEB]"
            >
              <XMarkIcon className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>

      {/* View Mode Toggle */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex rounded-lg border border-[#232336] p-1" style={{ background: '#0D0D0D' }}>
          <button
            onClick={() => setViewMode('chart')}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              viewMode === 'chart'
                ? 'bg-[#87CEEB] text-white'
                : 'text-gray-300 hover:text-white'
            }`}
          >
            Chart View
          </button>
          <button
            onClick={() => setViewMode('metrics')}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              viewMode === 'metrics'
                ? 'bg-[#87CEEB] text-white'
                : 'text-gray-300 hover:text-white'
            }`}
          >
            Metrics View
          </button>
        </div>
      </div>

      {/* Metric Selection with enhanced styling */}
      <div className="flex flex-wrap justify-center gap-2 mb-6 p-2 rounded-lg border border-[#232336]" style={{ background: '#0D0D0D' }}>
        {[
          { value: 'median_price', label: 'Median Price' },
          { value: 'price_per_sqft', label: 'Price per Sq Ft' },
          { value: 'roi_forecast', label: 'ROI Forecast' },
          { value: 'days_on_market', label: 'Days on Market' },
          { value: 'inventory_turnover', label: 'Inventory Turnover' },
          { value: 'price_momentum', label: 'Price Momentum' },
          { value: 'volatility', label: 'Volatility' },
        ].map(metric => (
          <button
            key={metric.value}
            onClick={() => setSelectedMetric(metric.value as ChartMetric)}
            className={`px-4 py-2 rounded-full transition-colors duration-200 text-sm font-medium border ${
              selectedMetric === metric.value
                ? 'bg-[#87CEEB] text-white border-[#232336]'
                : 'bg-[#18181B] text-gray-300 border-[#232336] hover:bg-[#222222]'
            }`}
          >
            {metric.label}
          </button>
        ))}
      </div>

      {/* Comparison Chart */}
      {locations.length > 0 && viewMode === 'chart' && (
        <div className="p-6 rounded-lg shadow-sm border border-[#232336]" style={{ background: '#18181B' }}>
          <div className="h-[400px]">
            <Line data={generateComparisonChartData()} options={getChartOptions()} />
          </div>
        </div>
      )}

      {/* Location Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        {locations.map(location => (
          <div key={location.id} className="p-6 rounded-xl shadow-sm border border-[#232336]" style={{ background: '#0D0D0D' }}>
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-bold text-white">{location.name}</h3>
              <button
                onClick={() => removeLocation(location.id)}
                className="text-gray-400 hover:text-[#87CEEB]"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            {/* PropPulse Score */}
            {location.trends?.market_analysis && (
              <div className="mb-6 p-4 rounded-lg border border-[#232336]" style={{ background: '#18181B' }}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-lg font-semibold text-white">PropPulse Score</h4>
                  <StarIcon className="h-6 w-6 text-[#87CEEB]" />
                </div>
                {(() => {
                  const score = calculatePropPulseScore(location.trends?.market_analysis);
                  if (!score) return <p className="text-gray-400">Score not available</p>;
                  return (
                    <div className="flex items-center space-x-4">
                      <div className="text-4xl font-bold text-white">{score.letterGrade}</div>
                      <div className="flex-1">
                        <div className="h-2 bg-[#232336] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#87CEEB] rounded-full"
                            style={{ width: `${score.score}%` }}
                          />
                        </div>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-gray-400">ROI:</span>{' '}
                            <span className="font-semibold text-white">{score.components.roi.toFixed(1)}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Momentum:</span>{' '}
                            <span className="font-semibold text-white">{score.components.momentum.toFixed(1)}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Volatility:</span>{' '}
                            <span className="font-semibold text-white">{score.components.volatility.toFixed(1)}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Risk:</span>{' '}
                            <span className="font-semibold text-white">{score.components.risk.toFixed(1)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* Market Trends */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 rounded-lg" style={{ background: '#18181B' }}>
                <span className="text-gray-400 text-sm">Price Momentum</span>
                <p className="font-semibold text-lg text-white">
                  {(location.trends?.market_analysis?.market_health?.price_momentum || 0).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: '#18181B' }}>
                <span className="text-gray-400 text-sm">Volatility</span>
                <p className="font-semibold text-lg text-white">
                  {location.trends?.market_analysis?.volatility?.volatility_level || 'N/A'}
                </p>
              </div>
            </div>

            {/* ROI Forecasts */}
            <div className="p-4 rounded-lg" style={{ background: '#18181B' }}>
              <h4 className="text-white font-medium mb-2">ROI Forecasts</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Short-term</span>
                  <span className="font-semibold text-white">
                    ${(location.trends?.market_analysis?.forecast?.short_term_forecast?.value || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Medium-term</span>
                  <span className="font-semibold text-white">
                    ${(location.trends?.market_analysis?.forecast?.medium_term_forecast?.value || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Long-term</span>
                  <span className="font-semibold text-white">
                    ${(location.trends?.market_analysis?.forecast?.long_term_forecast?.value || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ML Insights Section */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold text-white mb-6">ML Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Price Trends */}
          <div className="p-6 rounded-xl shadow-md border border-[#232336]" style={{ background: '#0D0D0D' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Price Trends</h3>
              <ChartBarSquareIcon className="h-8 w-8 text-[#87CEEB]" />
            </div>
            {locations.map(location => (
              <div key={location.id} className="mb-4 last:mb-0">
                <h4 className="text-sm font-medium text-gray-200 mb-2">{location.name}</h4>
                {location.trends?.market_analysis?.price_trends ? (
                  <div className="space-y-2 p-3 rounded-lg" style={{ background: '#18181B' }}>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Short-term</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.price_trends.short_term_trend > 0 ? '+' : ''}
                        {location.trends.market_analysis.price_trends.short_term_trend.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Medium-term</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.price_trends.medium_term_trend > 0 ? '+' : ''}
                        {location.trends.market_analysis.price_trends.medium_term_trend.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">YoY Change</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.price_trends.yoy_change > 0 ? '+' : ''}
                        {location.trends.market_analysis.price_trends.yoy_change.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">No price trend data available</p>
                )}
              </div>
            ))}
          </div>

          {/* Market Health */}
          <div className="p-6 rounded-xl shadow-md border border-[#232336]" style={{ background: '#0D0D0D' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Market Health</h3>
              <HeartIcon className="h-8 w-8 text-[#87CEEB]" />
            </div>
            {locations.map(location => (
              <div key={location.id} className="mb-4 last:mb-0">
                <h4 className="text-sm font-medium text-gray-200 mb-2">{location.name}</h4>
                {location.trends?.market_analysis?.market_health ? (
                  <div className="space-y-2 p-3 rounded-lg" style={{ background: '#18181B' }}>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Market Balance</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.market_health.market_balance}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Inventory Turnover</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.market_health.inventory_turnover.toFixed(1)}x
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Price Momentum</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.market_health.price_momentum.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">No market health data available</p>
                )}
              </div>
            ))}
          </div>

          {/* Risk Analysis */}
          <div className="p-6 rounded-xl shadow-md border border-[#232336]" style={{ background: '#0D0D0D' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Risk Analysis</h3>
              <ExclamationTriangleIcon className="h-8 w-8 text-[#87CEEB]" />
            </div>
            {locations.map(location => (
              <div key={location.id} className="mb-4 last:mb-0">
                <h4 className="text-sm font-medium text-gray-200 mb-2">{location.name}</h4>
                {location.trends?.market_analysis?.volatility ? (
                  <div className="space-y-2 p-3 rounded-lg" style={{ background: '#18181B' }}>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Volatility Level</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.volatility.volatility_level}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Variation Coefficient</span>
                      <span className="font-semibold text-white">
                        {location.trends.market_analysis.volatility.coefficient_of_variation.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Price Range</span>
                      <span className="font-semibold text-white">
                        ${location.trends.market_analysis.volatility.price_range.toLocaleString()}
                      </span>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">No risk analysis data available</p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* AI-Powered Comparative Summary */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold text-white mb-6">AI-Powered Comparative Summary</h2>
        <div className="p-8 rounded-xl shadow-md border border-[#232336]" style={{ background: '#0D0D0D' }}>
          {locations.length >= 2 ? (
            <div className="space-y-6">
              {/* Market Dynamics Comparison */}
              <div className="p-6 rounded-lg" style={{ background: '#18181B' }}>
                <h3 className="text-lg font-semibold text-white mb-4">Market Dynamics Comparison</h3>
                <div className="prose prose-sm max-w-none text-gray-200">
                  {(() => {
                    const strongestMomentum = locations.reduce((prev, current) => {
                      const prevMomentum = prev.trends?.market_analysis?.market_health?.price_momentum || 0;
                      const currentMomentum = current.trends?.market_analysis?.market_health?.price_momentum || 0;
                      return currentMomentum > prevMomentum ? current : prev;
                    });
                    const lowestVolatility = locations.reduce((prev, current) => {
                      const prevVol = prev.trends?.market_analysis?.volatility?.coefficient_of_variation || 1;
                      const currentVol = current.trends?.market_analysis?.volatility?.coefficient_of_variation || 1;
                      return currentVol < prevVol ? current : prev;
                    });
                    return (
                      <p>
                        {strongestMomentum.name} shows the strongest price momentum at{' '}
                        {strongestMomentum.trends?.market_analysis?.market_health?.price_momentum.toFixed(1)}%,
                        while {lowestVolatility.name} demonstrates the most stable market conditions with a
                        variation coefficient of {lowestVolatility.trends?.market_analysis?.volatility?.coefficient_of_variation.toFixed(2)}.
                        This suggests {strongestMomentum.name} may offer better short-term appreciation potential,
                        while {lowestVolatility.name} could be more suitable for risk-averse investors.
                      </p>
                    );
                  })()}
                </div>
              </div>

              {/* Investment Strategy Insights */}
              <div className="p-6 rounded-lg" style={{ background: '#18181B' }}>
                <h3 className="text-lg font-semibold text-white mb-4">Investment Strategy Insights</h3>
                <div className="prose prose-sm max-w-none text-gray-200">
                  {(() => {
                    const bestROI = locations.reduce((prev, current) => {
                      const prevROI = prev.trends?.market_analysis?.forecast?.medium_term_forecast?.value || 0;
                      const currentROI = current.trends?.market_analysis?.forecast?.medium_term_forecast?.value || 0;
                      return currentROI > prevROI ? current : prev;
                    });
                    const bestInventory = locations.reduce((prev, current) => {
                      const prevInv = prev.trends?.market_analysis?.market_health?.inventory_turnover || 0;
                      const currentInv = current.trends?.market_analysis?.market_health?.inventory_turnover || 0;
                      return currentInv > prevInv ? current : prev;
                    });
                    return (
                      <p>
                        Based on the analysis, {bestROI.name} presents the highest medium-term ROI potential at{' '}
                        ${bestROI.trends?.market_analysis?.forecast?.medium_term_forecast?.value.toLocaleString()},
                        while {bestInventory.name} shows the most active market with an inventory turnover rate of{' '}
                        {bestInventory.trends?.market_analysis?.market_health?.inventory_turnover.toFixed(1)}x.
                        This combination suggests that {bestROI.name} may be better suited for long-term appreciation strategies,
                        while {bestInventory.name} could be more favorable for active investors seeking quicker returns.
                      </p>
                    );
                  })()}
                </div>
              </div>

              {/* Risk-Reward Analysis */}
              <div className="p-6 rounded-lg" style={{ background: '#18181B' }}>
                <h3 className="text-lg font-semibold text-white mb-4">Risk-Reward Analysis</h3>
                <div className="prose prose-sm max-w-none text-gray-200">
                  {(() => {
                    const highestScore = locations.reduce((prev, current) => {
                      const prevScore = calculatePropPulseScore(prev.trends?.market_analysis)?.score || 0;
                      const currentScore = calculatePropPulseScore(current.trends?.market_analysis)?.score || 0;
                      return currentScore > prevScore ? current : prev;
                    });
                    const lowestScore = locations.reduce((prev, current) => {
                      const prevScore = calculatePropPulseScore(prev.trends?.market_analysis)?.score || 0;
                      const currentScore = calculatePropPulseScore(current.trends?.market_analysis)?.score || 0;
                      return currentScore < prevScore ? current : prev;
                    });
                    return (
                      <p>
                        The PropPulse Score analysis reveals that {highestScore.name} has the highest overall market score
                        of {calculatePropPulseScore(highestScore.trends?.market_analysis)?.letterGrade}, indicating a more
                        balanced risk-reward profile. In contrast, {lowestScore.name} has a lower score of{' '}
                        {calculatePropPulseScore(lowestScore.trends?.market_analysis)?.letterGrade}, suggesting higher
                        risk factors that investors should carefully consider. This differential highlights the importance
                        of aligning investment strategies with risk tolerance levels.
                      </p>
                    );
                  })()}
                </div>
              </div>

              {/* Strategic Recommendations */}
              <div className="p-6 rounded-lg" style={{ background: '#18181B' }}>
                <h3 className="text-lg font-semibold text-white mb-4">Strategic Recommendations</h3>
                <div className="prose prose-sm max-w-none text-gray-200">
                  <ul className="list-disc list-inside space-y-2">
                    {(() => {
                      const recommendations = [];
                      const strongestMomentum = locations.reduce((prev, current) => {
                        const prevMomentum = prev.trends?.market_analysis?.market_health?.price_momentum || 0;
                        const currentMomentum = current.trends?.market_analysis?.market_health?.price_momentum || 0;
                        return currentMomentum > prevMomentum ? current : prev;
                      });
                      const lowestVolatility = locations.reduce((prev, current) => {
                        const prevVol = prev.trends?.market_analysis?.volatility?.coefficient_of_variation || 1;
                        const currentVol = current.trends?.market_analysis?.volatility?.coefficient_of_variation || 1;
                        return currentVol < prevVol ? current : prev;
                      });

                      recommendations.push(
                        <li key="momentum">
                          Consider {strongestMomentum.name} for short-term investment strategies, as its strong price
                          momentum of {strongestMomentum.trends?.market_analysis?.market_health?.price_momentum.toFixed(1)}%
                          suggests potential for quick appreciation.
                        </li>
                      );

                      recommendations.push(
                        <li key="stability">
                          {lowestVolatility.name} may be more suitable for conservative investors, offering greater
                          market stability with a variation coefficient of {lowestVolatility.trends?.market_analysis?.volatility?.coefficient_of_variation.toFixed(2)}.
                        </li>
                      );

                      const bestROI = locations.reduce((prev, current) => {
                        const prevROI = prev.trends?.market_analysis?.forecast?.medium_term_forecast?.value || 0;
                        const currentROI = current.trends?.market_analysis?.forecast?.medium_term_forecast?.value || 0;
                        return currentROI > prevROI ? current : prev;
                      });

                      recommendations.push(
                        <li key="roi">
                          For long-term investments, {bestROI.name} shows the highest ROI potential at{' '}
                          ${bestROI.trends?.market_analysis?.forecast?.medium_term_forecast?.value.toLocaleString()},
                          making it attractive for buy-and-hold strategies.
                        </li>
                      );

                      return recommendations;
                    })()}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-400">Add at least two locations to generate a comparative analysis.</p>
          )}
        </div>
      </div>

      {/* Empty State */}
      {locations.length === 0 && (
        <div className="text-center py-12 bg-[#0D0D0D] rounded-xl shadow-sm border border-[#232336]">
          <p className="text-gray-300">Add locations to compare market data</p>
          <p className="text-gray-400 text-sm mt-2">Enter a location above to get started</p>
        </div>
      )}

      {/* RAG Chatbot UI */}
      <div className="mt-16 max-w-3xl mx-auto">
        <div className="rounded-xl shadow-lg border border-[#232336] p-6" style={{ background: '#0D0D0D' }}>
          <h2 className="text-xl font-bold mb-4 text-white">Ask the AI about these markets</h2>
          <div className="h-64 overflow-y-auto mb-4 rounded p-3" style={{ background: '#18181B' }}>
            {chatHistory.length === 0 && <div className="text-gray-400">Ask a question about the compared markets...</div>}
            {chatHistory.map((msg, i) => (
              <div key={i} className={`mb-2 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}> 
                <span className={`inline-block px-3 py-2 rounded-lg ${msg.role === 'user' ? 'bg-[#87CEEB] text-white' : 'bg-[#232336] text-gray-100'}`}>{msg.message}</span>
              </div>
            ))}
            {isChatLoading && <div className="text-gray-400">AI is thinking...</div>}
          </div>
          <form onSubmit={handleChatSubmit} className="flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              className="flex-1 px-4 py-2 border border-[#232336] rounded-lg focus:ring-2 focus:ring-[#87CEEB] focus:border-[#87CEEB] bg-[#18181B] text-white placeholder:text-gray-400"
              placeholder="Type your question..."
              disabled={isChatLoading}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-[#87CEEB] text-white rounded-lg font-semibold disabled:opacity-50"
              disabled={isChatLoading}
            >Ask</button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ComparativeAnalysis; 
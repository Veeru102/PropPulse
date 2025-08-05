import React, { useState, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowTrendingUpIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  MagnifyingGlassIcon,
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
import { getMarketTrends, MarketTrends } from '../services/api';
import clsx from 'clsx';

// Icons for ML Insights
import {
  ChartBarSquareIcon,
  HeartIcon,
  CalendarDaysIcon,
  ExclamationTriangleIcon,
  AdjustmentsHorizontalIcon,
  StarIcon,
} from '@heroicons/react/24/outline';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

type ChartMetric = 'median_price' | 'price_per_sqft' | 'roi_forecast';

type ScoringComponent = 'roi' | 'momentum' | 'volatility' | 'risk';

interface ScoringWeights {
  roi: number;
  momentum: number;
  volatility: number;
  risk: number;
}

interface ScoreComponents {
  roi: number;
  momentum: number;
  volatility: number;
  risk: number;
}

interface PropPulseScore {
  score: number;
  letterGrade: string;
  components: ScoreComponents;
}

// Define default scoring weights as a constant
const DEFAULT_SCORING_WEIGHTS: ScoringWeights = {
  roi: 40,
  momentum: 25,
  volatility: 20,
  risk: 15
};

// Function to get score color with gradient logic
const getScoreColor = (s: number) => {
  // Function to linearly interpolate between two colors in RGB space
  const lerpColor = (color1: [number, number, number], color2: [number, number, number], factor: number): string => {
    const result = color1.map((c, i) => Math.round(c + factor * (color2[i] - c)));
    return `rgb(${result[0]}, ${result[1]}, ${result[2]})`;
  };

  if (s >= 90) { // A+ scores: #87CEEB (light blue) to a slightly darker blue
    const startColor: [number, number, number] = [135, 206, 235]; // #87CEEB
    const endColor: [number, number, number] = [100, 180, 200]; // A slightly darker blue
    const factor = (s - 90) / 10; // 0 for 90, 1 for 100
    return lerpColor(startColor, endColor, factor);
  } 
  if (s >= 80) return '#87CEEB'; // A
  if (s >= 70) return 'rgb(59, 130, 246)'; // B
  if (s >= 60) return '#A7F3D0'; // C
  if (s >= 50) return '#F59E0B'; // D
  return '#EF4444'; // F
};

// Animated circular score dial for PropPulse score
const ScoreDial: React.FC<{ score: number }> = ({ score }) => {
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const dialColor = getScoreColor(score);
  const strokeWidth = 12;

  // Calculate the position of the indicator (tip of the dial)
  const angle = (score / 100) * 360; // Angle in degrees
  const x = 70 + radius * Math.cos((angle - 90) * (Math.PI / 180)); // Convert to radians and adjust for -90 rotation
  const y = 70 + radius * Math.sin((angle - 90) * (Math.PI / 180));

  // New calculations for needle start and end points
  const needleInnerRadius = radius - strokeWidth / 2 - 5; // Starts inside the track
  const needleOuterRadius = radius + strokeWidth / 2 + 5; // Extends outside the track

  const needleX1 = 70 + needleInnerRadius * Math.cos((angle - 90) * (Math.PI / 180));
  const needleY1 = 70 + needleInnerRadius * Math.sin((angle - 90) * (Math.PI / 180));
  const needleX2 = 70 + needleOuterRadius * Math.cos((angle - 90) * (Math.PI / 180));
  const needleY2 = 70 + needleOuterRadius * Math.sin((angle - 90) * (Math.PI / 180));

  // Generate tick marks
  const ticks = Array.from({ length: 40 }).map((_, i) => {
    const tickAngle = (i * 9 - 90) * (Math.PI / 180); // 40 ticks, 9 degrees apart
    const isMajorTick = i % 5 === 0; // Every 5th tick is major (0, 5, 10, ..., 35)
    const tickInnerRadius = radius - strokeWidth / 2 - (isMajorTick ? 8 : 4); // Longer for major ticks
    const tickOuterRadius = radius - strokeWidth / 2; // Ends at the inner edge of the track

    const tx1 = 70 + tickInnerRadius * Math.cos(tickAngle);
    const ty1 = 70 + tickInnerRadius * Math.sin(tickAngle);
    const tx2 = 70 + tickOuterRadius * Math.cos(tickAngle);
    const ty2 = 70 + tickOuterRadius * Math.sin(tickAngle);

    return {
      x1: tx1,
      y1: ty1,
      x2: tx2,
      y2: ty2,
      strokeWidth: isMajorTick ? 2 : 1,
      stroke: isMajorTick ? '#555' : '#333',
    };
  });

  // Generate score labels
  const scoreLabels = [
    { value: 0, angle: -90 }, // Corresponds to 0% at the start of the dial (-90 degrees)
    { value: 25, angle: 0 },  // Corresponds to 25% (0 degrees)
    { value: 50, angle: 90 }, // Corresponds to 50% (90 degrees)
    { value: 75, angle: 180 }, // Corresponds to 75% (180 degrees)
    { value: 100, angle: 270 } // Corresponds to 100% (270 degrees)
  ].map(label => {
    const labelRadius = radius + strokeWidth / 2 + 10; // Position labels slightly outside the track
    const lx = 70 + labelRadius * Math.cos(label.angle * (Math.PI / 180));
    const ly = 70 + labelRadius * Math.sin(label.angle * (Math.PI / 180));
    return { x: lx, y: ly, value: label.value };
  });

  return (
    <svg width="140" height="140" className="-rotate-90">
      {/* Background circle for depth */}
      <circle
        cx="70"
        cy="70"
        r={radius + 5}
        stroke="none"
        fill="#121212"
      />
      {/* Inner shadow/gradient effect */}
      <circle
        cx="70"
        cy="70"
        r={radius + 2}
        stroke="none"
        fill="url(#dialGradient)"
      />
      <defs>
        <radialGradient id="dialGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
          <stop offset="0%" stopColor="#18181B" stopOpacity="0" /> {/* Darker center, fully transparent */}
          <stop offset="70%" stopColor="#222222" stopOpacity="0.5" /> {/* Mid-point, semi-transparent */}
          <stop offset="100%" stopColor="#232336" stopOpacity="1" /> {/* Outer edge, fully opaque */}
        </radialGradient>
        {/* Subtle drop shadow filter */}
        <filter id="dropshadow" height="130%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
          <feOffset dx="2" dy="2" result="shadow" />
          <feComponentTransfer in="shadow">
            <feFuncA type="linear" slope="0.5" />
          </feComponentTransfer>
          <feMerge>
            <feMergeNode />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        {/* Glow filter */}
        <filter id="glowFilter" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blurred" />
          <feComponentTransfer in="blurred" result="glowLayer">
            <feFuncA type="linear" slope="0.8" /> {/* Adjust glow intensity */}
          </feComponentTransfer>
          <feMerge>
            <feMergeNode in="glowLayer" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        {/* Glossy overlay gradient */}
        <radialGradient id="glossyOverlayGradient" cx="50%" cy="50%" r="50%" fx="60%" fy="40%">
          <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Tick Marks */}
      <g>
        {ticks.map((tick, index) => (
          <line
            key={index}
            x1={tick.x1}
            y1={tick.y1}
            x2={tick.x2}
            y2={tick.y2}
            stroke={tick.stroke}
            strokeWidth={tick.strokeWidth}
            strokeLinecap="round"
          />
        ))}
      </g>

      {/* Score Labels */}
      <g transform="rotate(90 70 70)"> {/* Rotate labels back to upright position */}
        {scoreLabels.map((label, index) => (
          <text
            key={index}
            x={label.x}
            y={label.y}
            textAnchor="middle"
            alignmentBaseline="middle"
            fontSize="10"
            fill="#888" // Subtle color for labels
            fontWeight="bold"
          >
            {label.value}
          </text>
        ))}
      </g>

      {/* Main Track */}
      <circle
        cx="70"
        cy="70"
        r={radius}
        stroke="#232336"
        strokeWidth={strokeWidth}
        fill="transparent"
      />
      {/* Glossy Overlay */}
      <circle
        cx="70"
        cy="70"
        r={radius}
        stroke="none"
        fill="url(#glossyOverlayGradient)"
      />
      {/* Animated progress GLOW layer */}
      <circle
        cx="70"
        cy="70"
        r={radius}
        stroke={dialColor}
        strokeWidth={strokeWidth + 4} // Slightly wider for glow
        fill="transparent"
        strokeDasharray={`${circumference} ${circumference}`}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 1s ease-out, stroke 0.5s ease-in-out' }}
        filter="url(#glowFilter)"
        opacity="0.6"
      />
      {/* Animated progress main layer */}
      <circle
        cx="70"
        cy="70"
        r={radius}
        stroke={dialColor}
        strokeWidth={strokeWidth}
        fill="transparent"
        strokeDasharray={`${circumference} ${circumference}`}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 1s ease-out, stroke 0.5s ease-in-out' }}
        filter="url(#dropshadow)"
      />
      {/* Indicator/Pointer as a line needle */}
      <line
        x1={needleX1}
        y1={needleY1}
        x2={needleX2}
        y2={needleY2}
        stroke={dialColor}
        strokeWidth={3}
        strokeLinecap="round"
        style={{ transition: 'x1 1s ease-out, y1 1s ease-out, x2 1s ease-out, y2 1s ease-out, stroke 0.5s ease-in-out' }}
        filter="url(#dropshadow)" // Apply drop shadow
      />
      {/* Central pivot for the needle */}
      <circle
        cx="70"
        cy="70"
        r={strokeWidth / 2}
        fill="#232336"
        stroke="#121212"
        strokeWidth="2"
      />
      {/* Inner reflective circle for the pivot */}
      <circle
        cx="70"
        cy="70"
        r={strokeWidth / 4} // Smaller inner circle
        fill="#4A4A5A" // Lighter grey for reflection
        stroke="none"
      />
      {/* Score text */}
      <g transform="rotate(90 70 70)">
        <text x="70" y="78" textAnchor="middle" fontSize="28" fontWeight="bold" fill={dialColor}>
          {score.toFixed(1)}
        </text>
      </g>
    </svg>
  );
};

const MarketAnalysis: React.FC = () => {
  const [searchInput, setSearchInput] = useState('');
  const [selectedLocation, setSelectedLocation] = useState('');
  const [timeRange, setTimeRange] = useState('6m');
  const [selectedMetric, setSelectedMetric] = useState<ChartMetric>('median_price');
  const [showScoringSettings, setShowScoringSettings] = useState(false);
  const [scoringWeights, setScoringWeights] = useState<ScoringWeights>({
    ...DEFAULT_SCORING_WEIGHTS
  });

  // Fetch market trends data
  const { data: trends, isLoading, error } = useQuery<MarketTrends>({
    queryKey: ['marketTrends', selectedLocation, timeRange],
    queryFn: () => getMarketTrends(selectedLocation),
    enabled: !!selectedLocation,
  });

  // Log the trends data for debugging
  console.log("MarketAnalysis component - trends state:", trends);
  console.log("MarketAnalysis component - current metrics:", trends?.market_data?.current_metrics);
  console.log("MarketAnalysis component - historical data:", trends?.market_data?.historical_data);

  // Add a check for trends being undefined at the start
  if (!trends && !isLoading && !error && selectedLocation) {
    console.warn("Trends data is undefined after fetch for", selectedLocation);
  }

  // Handle search
  const handleSearch = useCallback(() => {
    if (searchInput.trim()) {
      console.log("Searching for location:", searchInput.trim());
      setSelectedLocation(searchInput.trim());
    }
  }, [searchInput]);

  // Handle Enter key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // Format date from YYYYMM to readable format
  const formatDate = (dateNum: number) => {
    const year = Math.floor(dateNum / 100);
    const month = dateNum % 100;
    return new Date(year, month - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  // Update the chart options based on selected metric
  const getChartOptions = (): ChartOptions<'line'> => {
    const baseOptions: ChartOptions<'line'> = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
          labels: { color: '#fff' },
        },
        title: {
          display: true,
          text: `${selectedMetric.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} Over Time`,
          color: '#fff',
        },
        tooltip: {
          backgroundColor: '#222',
          titleColor: '#fff',
          bodyColor: '#fff',
        },
      },
      scales: {
        x: {
          ticks: { color: '#fff' },
          grid: { color: '#444' },
        },
        y: {
          type: 'linear',
          beginAtZero: false,
          ticks: {
            color: '#fff',
            callback: function(tickValue: number | string) {
              const value = Number(tickValue);
              if (selectedMetric === 'median_price') {
                return `$${(value / 1000).toFixed(0)}k`;
              }
              return `$${value.toFixed(0)}`;
            },
          },
          grid: { color: '#444' },
        },
      },
    };

    if (selectedMetric === 'roi_forecast') {
      return {
        ...baseOptions,
        plugins: {
          ...baseOptions.plugins,
          tooltip: {
            callbacks: {
              label: function(context) {
                const idx = context.dataIndex;
                let value = context.parsed.y;
                let confidence = 0;
                if (trends?.market_analysis?.forecast) {
                  if (idx === 0) confidence = trends.market_analysis.forecast.short_term_forecast.confidence;
                  if (idx === 1) confidence = trends.market_analysis.forecast.medium_term_forecast.confidence;
                  if (idx === 2) confidence = trends.market_analysis.forecast.long_term_forecast.confidence;
                }
                return `${context.label} forecast: $${value.toLocaleString(undefined, {maximumFractionDigits: 3})}\n(Confidence: ${confidence.toFixed(2)})`;
              }
            },
            backgroundColor: '#222',
            titleColor: '#fff',
            bodyColor: '#fff',
          }
        },
      };
    }

    return baseOptions;
  };

  // Generate price trend data based on historical data
  const generatePriceTrendData = () => {
    const values = trends?.market_data?.historical_data?.median_list_price?.values;
    const dates = trends?.market_data?.historical_data?.median_list_price?.dates;
    
    // Ensure historical data exists and has values before generating chart data
    if (!values || !dates || values.length === 0) {
      return {
        labels: [],
        datasets: [{
          label: 'No Data Available',
          data: [],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          tension: 0.4,
          fill: true,
        }],
      };
    }

    // Filter data points based on selected time range
    let filteredDates = [...dates];
    let filteredValues = [...values];
    
    if (timeRange === '1m') {
      filteredDates = dates.slice(-2);
      filteredValues = values.slice(-2);
    } else if (timeRange === '3m') {
      filteredDates = dates.slice(-4);
      filteredValues = values.slice(-4);
    } else if (timeRange === '6m') {
      filteredDates = dates.slice(-7);
      filteredValues = values.slice(-7);
    } else if (timeRange === '1y') {
      filteredDates = dates.slice(-13);
      filteredValues = values.slice(-13);
    }

    // Format dates for display
    const formattedDates = filteredDates.map(date => formatDate(parseInt(date)));

    return {
      labels: formattedDates,
      datasets: [{
        label: 'Median Home Price',
        data: filteredValues,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        tension: 0.4,
        fill: true,
      }],
    };
  };

  // Calculate market momentum based on recent price changes
  const calculateMarketMomentum = () => {
    if (!trends?.market_analysis?.price_trends?.short_term_trend) {
      return 0;
    }
    return trends.market_analysis.price_trends.short_term_trend;
  };

  // Generate chart data based on selected metric
  const generateChartData = (): ChartData<'line'> => {
    if (!trends?.market_data?.historical_data) {
      return {
        labels: [],
        datasets: [{
          label: 'No Data Available',
          data: [],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          tension: 0.4,
          fill: true,
        }],
      };
    }

    const { historical_data } = trends.market_data;
    let values: number[] = [];
    let label = '';
    let dates: string[] = [];

    switch (selectedMetric) {
      case 'median_price':
        if (!historical_data.median_list_price) {
          return {
            labels: [],
            datasets: [{
              label: 'No Price Data Available',
              data: [],
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.5)',
              tension: 0.4,
              fill: true,
            }],
          };
        }
        values = historical_data.median_list_price.values;
        dates = historical_data.median_list_price.dates;
        label = 'Median Home Price';
        break;
      case 'price_per_sqft':
        if (!historical_data.price_per_sqft) {
          return {
            labels: [],
            datasets: [{
              label: 'No Price per Sq Ft Data Available',
              data: [],
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.5)',
              tension: 0.4,
              fill: true,
            }],
          };
        }
        values = historical_data.price_per_sqft.values;
        dates = historical_data.price_per_sqft.dates;
        label = 'Price per Square Foot';
        break;
      case 'roi_forecast':
        // Use the forecast data from market analysis
        const forecast = trends.market_analysis?.forecast;
        if (!forecast) {
          return {
            labels: [],
            datasets: [{
              label: 'No Forecast Data Available',
              data: [],
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.5)',
              tension: 0.4,
              fill: true,
            }],
          };
        }
        values = [
          forecast.short_term_forecast.value,
          forecast.medium_term_forecast.value,
          forecast.long_term_forecast.value
        ];
        dates = ['Short-term', 'Medium-term', 'Long-term'];
        label = 'ROI Forecast';
        break;
    }

    // Filter data points based on selected time range
    let filteredValues = [...values];
    let filteredDates = [...dates];
    
    if (timeRange === '1m') {
      filteredDates = dates.slice(-2);
      filteredValues = values.slice(-2);
    } else if (timeRange === '3m') {
      filteredDates = dates.slice(-4);
      filteredValues = values.slice(-4);
    } else if (timeRange === '6m') {
      filteredDates = dates.slice(-7);
      filteredValues = values.slice(-7);
    } else if (timeRange === '1y') {
      filteredDates = dates.slice(-13);
      filteredValues = values.slice(-13);
    }

    // Format dates for display
    const formattedDates = selectedMetric === 'roi_forecast' 
      ? filteredDates 
      : filteredDates.map(date => formatDate(parseInt(date)));

    return {
      labels: formattedDates,
      datasets: [{
        label,
        data: filteredValues,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        tension: 0.4,
        fill: true,
      }],
    };
  };

  // Update the forecast display section
  const renderForecast = () => {
    if (!trends?.market_analysis?.forecast) {
      return (
        <div className="text-gray-500">No forecast data available</div>
      );
    }
    const { short_term_forecast, medium_term_forecast, long_term_forecast } = trends.market_analysis.forecast;
    return (
      <div className="space-y-2">
        <div>
          <span className="font-medium">Short-term forecast:</span> <span className="font-bold text-lg">${short_term_forecast.value.toLocaleString(undefined, {maximumFractionDigits: 3})}</span>
          <div className="text-gray-500 text-sm">(Confidence: {(short_term_forecast.confidence * 100).toFixed(2)}%)</div>
        </div>
        <div>
          <span className="font-medium">Medium-term forecast:</span> <span className="font-bold text-lg">${medium_term_forecast.value.toLocaleString(undefined, {maximumFractionDigits: 3})}</span>
          <div className="text-gray-500 text-sm">(Confidence: {(medium_term_forecast.confidence * 100).toFixed(2)}%)</div>
        </div>
        <div>
          <span className="font-medium">Long-term forecast:</span> <span className="font-bold text-lg">${long_term_forecast.value.toLocaleString(undefined, {maximumFractionDigits: 3})}</span>
          <div className="text-gray-500 text-sm">(Confidence: {(long_term_forecast.confidence * 100).toFixed(2)}%)</div>
        </div>
      </div>
    );
  };

  // Helper to get momentum label
  const getMomentumLabel = (value: number) => {
    if (value > 1) return 'Accelerating';
    if (value < -1) return 'Decelerating';
    return 'Stable';
  };

  // Handle weight changes
  const handleWeightChange = (component: ScoringComponent, value: number) => {
    // Calculate the total of other weights
    const otherWeights = Object.entries(scoringWeights)
      .filter(([key]) => key !== component)
      .reduce((sum, [_, weight]) => sum + weight, 0);
    
    // If the new total would exceed 100%, scale down other weights proportionally
    if (value + otherWeights > 100) {
      const scaleFactor = (100 - value) / otherWeights;
      setScoringWeights(prev => {
        const newWeights = { ...prev };
        Object.keys(newWeights).forEach(key => {
          if (key !== component) {
            newWeights[key as ScoringComponent] = Math.round(newWeights[key as ScoringComponent] * scaleFactor);
          }
        });
        newWeights[component] = value;
        return newWeights;
      });
    } else {
      setScoringWeights(prev => ({
        ...prev,
        [component]: value
      }));
    }
  };

  // Calculate PropPulse Score
  const calculatePropPulseScore = (): PropPulseScore | null => {
    if (!trends?.market_analysis?.price_trends || 
        !trends?.market_analysis?.market_health || 
        !trends?.market_analysis?.volatility) {
      return null;
    }

    const { price_trends, market_health, volatility } = trends.market_analysis;
    
    // Normalize ROI (0-100)
    const roiScore = Math.min(100, Math.max(0, 
      ((price_trends.short_term_trend + price_trends.medium_term_trend) / 2) * 10
    ));

    // Normalize Momentum (0-100) - Much more stringent calculation
    const momentumFactors = {
      priceMomentum: market_health.price_momentum,
      inventoryTurnover: market_health.inventory_turnover,
      marketBalance: market_health.market_balance === 'Seller\'s Market' ? 1 : 0.5,
      trendStrength: price_trends.trend_strength === 'Strong' ? 1 : 
                    price_trends.trend_strength === 'Moderate' ? 0.7 : 0.4,
      yoyChange: price_trends.yoy_change,
      shortTermTrend: price_trends.short_term_trend,
      mediumTermTrend: price_trends.medium_term_trend
    };

    // Calculate individual component scores with stricter thresholds
    const componentScores = {
      // Price momentum components (40% total)
      rawMomentum: Math.min(1, Math.max(0, momentumFactors.priceMomentum / 2)) * 0.15, // 15%
      yoyChange: Math.min(1, Math.max(0, momentumFactors.yoyChange / 30)) * 0.15, // 15% - requires 30% YoY growth for full points
      shortTermTrend: Math.min(1, Math.max(0, momentumFactors.shortTermTrend / 15)) * 0.05, // 5% - requires 15% growth for full points
      mediumTermTrend: Math.min(1, Math.max(0, momentumFactors.mediumTermTrend / 20)) * 0.05, // 5% - requires 20% growth for full points

      // Market activity components (35% total)
      inventoryTurnover: Math.min(1, momentumFactors.inventoryTurnover / 4) * 0.25, // 25% - requires 4x turnover for full points
      marketBalance: momentumFactors.marketBalance * 0.10, // 10%

      // Trend strength (25% total)
      trendStrength: momentumFactors.trendStrength * 0.25 // 25%
    };

    // Calculate base momentum score (0-100)
    const baseMomentumScore = Object.values(componentScores).reduce((sum, score) => sum + score, 0) * 100;

    // Apply extremely aggressive scaling to make the scores more differentiated
    const momentumScore = Math.min(100, Math.max(0, 
      Math.pow(baseMomentumScore / 100, 2.5) * 100 // Power of 2.5 for even more aggressive scaling
    ));

    // Normalize Volatility (0-100, inverted because lower volatility is better)
    const volatilityScore = Math.min(100, Math.max(0, 
      (1 - volatility.coefficient_of_variation) * 100
    ));

    // Calculate Risk Score (0-100)
    const riskFactors = {
      marketBalance: market_health.market_balance === 'Seller\'s Market' ? 80 : 60,
      inventoryTurnover: Math.min(100, market_health.inventory_turnover * 40),
      volatilityLevel: volatility.volatility_level === 'Low' ? 80 : 
                      volatility.volatility_level === 'Medium' ? 60 : 40
    };
    const riskScore = (riskFactors.marketBalance + riskFactors.inventoryTurnover + riskFactors.volatilityLevel) / 3;

    // Calculate weighted score with normalized weights
    const totalWeight = Object.values(scoringWeights).reduce((sum, weight) => sum + weight, 0);
    const normalizedWeights = {
      roi: scoringWeights.roi / totalWeight,
      momentum: scoringWeights.momentum / totalWeight,
      volatility: scoringWeights.volatility / totalWeight,
      risk: scoringWeights.risk / totalWeight
    };

    const weightedScore = Math.min(100, Math.max(0,
      (
        (roiScore * normalizedWeights.roi) +
        (momentumScore * normalizedWeights.momentum) +
        (volatilityScore * normalizedWeights.volatility) +
        (riskScore * normalizedWeights.risk)
      )
    ));

    // Convert to letter grade
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
        risk: Number(riskScore.toFixed(1))
      }
    };
  };

  // Add helper calculations for enhanced ML insights
  const calculatedInventoryTurnover = useMemo(() => {
    const dom = trends?.market_data?.current_metrics?.avg_days_on_market;
    if (dom && dom > 0) {
      return 12 / (dom / 30);
    }
    return trends?.market_analysis?.market_health?.inventory_turnover || 0;
  }, [trends]);

  const calculateYOYChange = useMemo(() => {
    const values = trends?.market_data?.historical_data?.median_list_price?.values;
    if (values && values.length >= 13) {
      const latest = values[values.length - 1];
      const yearAgo = values[values.length - 13];
      if (yearAgo > 0) {
        return ((latest - yearAgo) / yearAgo) * 100;
      }
    }
    return trends?.market_analysis?.price_trends?.yoy_change || 0;
  }, [trends]);

  const getMarketStrengthLabel = (): { label: string; color: string } => {
    if (!trends?.market_analysis?.market_health) return { label: 'N/A', color: '#9CA3AF' }; // gray-400

    const { price_momentum } = trends.market_analysis.market_health;
    const inv = calculatedInventoryTurnover;

    // A combination score might be better to evaluate overall strength
    // Let's create a score for market strength based on momentum and inventory turnover
    // Max price_momentum can be 10 (as per example, if 5 is strong, 10 can be very strong)
    // Max inv_turnover can be 5 (if 3 is strong, 5 can be very strong)
    const momentumScore = Math.min(Math.max(price_momentum, -10) / 10, 1); // Normalize to -1 to 1, then to 0-1 range (add 1 and divide by 2 for overall range)
    const invScore = Math.min(Math.max(inv, 0) / 5, 1); // Normalize to 0-1

    const combinedScore = (momentumScore * 0.6) + (invScore * 0.4); // Weights: momentum is slightly more important

    if (combinedScore >= 0.8) return { label: 'Very Strong', color: '#4CAF50' }; // Green for very strong
    if (combinedScore >= 0.6) return { label: 'Strong', color: '#8BC34A' };    // Lighter green for strong
    if (combinedScore >= 0.4) return { label: 'Moderate', color: '#FFC107' };  // Orange-yellow for moderate
    if (combinedScore >= 0.2) return { label: 'Weak', color: '#FF9800' };      // Orange for weak
    return { label: 'Very Weak', color: '#F44336' }; // Red for very weak
  };

  const getGrowthPotentialLabel = (): { label: string; color: string } => {
    if (!trends?.market_analysis?.price_trends) return { label: 'N/A', color: '#9CA3AF' }; // gray-400

    const { short_term_trend, medium_term_trend, yoy_change } = trends.market_analysis.price_trends;
    const avgTrend = (short_term_trend + medium_term_trend) / 2;

    // Combine average trend and YoY change for a comprehensive growth score
    // Max avgTrend can be 10 (e.g. 10% average growth)
    // Max yoy_change can be 20 (e.g. 20% YoY change)
    const avgTrendScore = Math.min(Math.max(avgTrend, -10) / 10, 1); // Normalize to -1 to 1, then to 0-1 range
    const yoyScore = Math.min(Math.max(yoy_change, -10) / 20, 1); // Normalize to -0.5 to 1, then to 0-1 range

    const combinedGrowthScore = (avgTrendScore * 0.7) + (yoyScore * 0.3); // Avg trend is more critical for potential

    if (combinedGrowthScore >= 0.8) return { label: 'Very Strong', color: '#4CAF50' };
    if (combinedGrowthScore >= 0.6) return { label: 'Strong', color: '#8BC34A' };
    if (combinedGrowthScore >= 0.4) return { label: 'Moderate', color: '#FFC107' };
    if (combinedGrowthScore >= 0.2) return { label: 'Weak', color: '#FF9800' };
    return { label: 'Very Weak', color: '#F44336' };
  };

  const getSeasonalStrengthLabel = (value: number) => {
    if (value >= 40) return 'Very Strong';
    if (value >= 25) return 'Strong';
    if (value >= 10) return 'Moderate';
    if (value >= 1) return 'Weak';
    if (value > 0) return 'Very Weak';
    return 'Negligible';
  };

  const getVolatilityLevelLabel = (coefficientOfVariation: number): { label: string; color: string } => {
    // Normalize coefficientOfVariation: lower is better
    const normalizedVol = Math.min(Math.max(coefficientOfVariation, 0), 0.5);

    // Invert the score for color mapping (lower volatility = higher score 
    const invertedScore = 1 - (normalizedVol / 0.5); // 0.5 is max 

    if (invertedScore >= 0.8) return { label: 'Very Low', color: '#4CAF50' }; // Green
    if (invertedScore >= 0.6) return { label: 'Low', color: '#8BC34A' }; // Lighter green
    if (invertedScore >= 0.4) return { label: 'Moderate', color: '#FFC107' }; // Orange-yellow
    if (invertedScore >= 0.2) return { label: 'High', color: '#FF9800' }; // Orange
    return { label: 'Very High', color: '#F44336' }; // Red
  };

  return (
    <div className="min-h-screen max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" style={{ background: '#121212', color: '#fff' }}>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-white">Market Analysis</h1>
        <div className="flex gap-4">
          <div className="relative">
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter City, State or ZIP Code"
              className="w-64 px-4 py-2 border border-[#232336] rounded-lg focus:ring-2 focus:ring-[#87CEEB] focus:border-[#87CEEB]" style={{ background: '#222222', color: '#fff' }}
            />
            <button
              onClick={handleSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-[#87CEEB]"
            >
              <MagnifyingGlassIcon className="h-5 w-5" />
            </button>
          </div>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-4 py-2 border border-[#232336] rounded-lg focus:ring-2 focus:ring-[#87CEEB] focus:border-[#87CEEB]" style={{ background: '#222222', color: '#fff' }}
          >
            <option value="1m">1 Month</option>
            <option value="3m">3 Months</option>
            <option value="6m">6 Months</option>
            <option value="1y">1 Year</option>
          </select>
        </div>
      </div>

      {isLoading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#87CEEB] mx-auto"></div>
          <p className="mt-4 text-gray-300">Loading market data...</p>
        </div>
      )}

      {/* Display generic error message if a fetch error occurs */}
      {error && (
        <div className="bg-red-900 border border-red-700 text-red-300 px-4 py-3 rounded-lg mb-6">
          Error loading market data: {error.message}
        </div>
      )}

      {/* Display data only if trends and current_metrics are available */}
      {!isLoading && !error && trends?.market_data?.current_metrics && selectedLocation && (
        <>
          {/* Primary Tier - Critical Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Median Price Card - Primary */}
            <div className="p-8 rounded-xl shadow-lg border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white">Median Price</h3>
                <CurrencyDollarIcon className="h-10 w-10 text-[#87CEEB]" />
              </div>
              <p className="text-4xl font-bold text-white mb-2">
                ${Math.round(trends.market_data.current_metrics.median_price / 1000) * 1000}
              </p>
              <p className="text-lg text-gray-300">Current market value</p>
            </div>

            {/* Market Momentum Card - Primary */}
            <div className="p-8 rounded-xl shadow-lg border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white">Market Momentum</h3>
                <ArrowTrendingUpIcon className="h-10 w-10 text-[#87CEEB]" />
              </div>
              <div className="flex flex-col items-start">
                <span className="text-4xl font-extrabold text-[#87CEEB] leading-tight">{getMomentumLabel(calculateMarketMomentum())}</span>
                <span className="text-base text-gray-300 mt-1">{calculateMarketMomentum() > 0 ? '+' : ''}{calculateMarketMomentum().toFixed(1)}% recent change</span>
              </div>
            </div>

            {/* Forecast Card - Primary */}
            <div className="p-8 rounded-xl shadow-lg border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white">Market Forecast</h3>
                <ChartBarIcon className="h-10 w-10 text-[#87CEEB]" />
              </div>
              {trends?.market_analysis?.forecast ? (
                <div className="space-y-2">
                  <div>
                    <span className="font-medium">Short-term forecast:</span> <span className="font-bold text-lg">${trends.market_analysis.forecast.short_term_forecast.value.toLocaleString(undefined, {maximumFractionDigits: 3})}</span>
                    <div className="text-gray-300 text-sm">(Confidence: {(trends.market_analysis.forecast.short_term_forecast.confidence * 100).toFixed(2)}%)</div>
                  </div>
                  <div>
                    <span className="font-medium">Medium-term forecast:</span> <span className="font-bold text-lg">${trends.market_analysis.forecast.medium_term_forecast.value.toLocaleString(undefined, {maximumFractionDigits: 3})}</span>
                    <div className="text-gray-300 text-sm">(Confidence: {(trends.market_analysis.forecast.medium_term_forecast.confidence * 100).toFixed(2)}%)</div>
                  </div>
                  <div>
                    <span className="font-medium">Long-term forecast:</span> <span className="font-bold text-lg">${trends.market_analysis.forecast.long_term_forecast.value.toLocaleString(undefined, {maximumFractionDigits: 3})}</span>
                    <div className="text-gray-300 text-sm">(Confidence: {(trends.market_analysis.forecast.long_term_forecast.confidence * 100).toFixed(2)}%)</div>
                  </div>
                </div>
              ) : renderForecast()}
            </div>
          </div>

          {/* Chart Toggle Buttons and Chart */}
          <div className="flex flex-col items-center mb-8">
            <div className="flex justify-center space-x-2 mb-4">
              <button
                onClick={() => setSelectedMetric('median_price')}
                className={clsx(
                  'px-6 py-2 rounded-full transition-colors duration-200 font-semibold shadow-sm border-2 focus:outline-none',
                  selectedMetric === 'median_price' ? 'bg-[#87CEEB] text-white border-[#87CEEB]' : 'bg-[#222222] text-gray-300 border-[#232336] hover:bg-gray-700'
                )}
              >
                Median Price
              </button>
              <button
                onClick={() => setSelectedMetric('price_per_sqft')}
                className={clsx(
                  'px-6 py-2 rounded-full transition-colors duration-200 font-semibold shadow-sm border-2 focus:outline-none',
                  selectedMetric === 'price_per_sqft' ? 'bg-[#87CEEB] text-white border-[#87CEEB]' : 'bg-[#222222] text-gray-300 border-[#232336] hover:bg-gray-700'
                )}
              >
                Price per Sq Ft
              </button>
              <button
                onClick={() => setSelectedMetric('roi_forecast')}
                className={clsx(
                  'px-6 py-2 rounded-full transition-colors duration-200 font-semibold shadow-sm border-2 focus:outline-none',
                  selectedMetric === 'roi_forecast' ? 'bg-[#87CEEB] text-white border-[#87CEEB]' : 'bg-[#222222] text-gray-300 border-[#232336] hover:bg-gray-700'
                )}
              >
                ROI Forecast
              </button>
            </div>
            {!isLoading && !error && trends?.market_data?.historical_data && (
              <div className="p-6 rounded-lg shadow-sm border border-[#232336] w-full max-w-4xl" style={{ background: '#222222' }}>
                <div className="h-[400px]">
                  <Line
                    data={generateChartData()}
                    options={getChartOptions()}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Secondary Tier - Important Context */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Days on Market Card - Secondary */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">Days on Market</h3>
                <ClockIcon className="h-8 w-8 text-white" />
              </div>
              <p className="text-3xl font-bold text-white mb-2">
                {trends.market_data.current_metrics.avg_days_on_market}
              </p>
              <p className="text-sm text-gray-300">Average listing time</p>
            </div>

            {/* Price per Sq Ft Card - Secondary */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">Price per Sq Ft</h3>
                <CurrencyDollarIcon className="h-8 w-8 text-white" />
              </div>
              <p className="text-3xl font-bold text-white mb-2">
                ${Math.round(trends.market_data.current_metrics.avg_price_per_sqft)}
              </p>
              <p className="text-sm text-gray-300">Median price per square foot</p>
            </div>

            {/* Volatility Card - Secondary */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">Market Volatility</h3>
                <ExclamationTriangleIcon className="h-8 w-8 text-white" />
              </div>
              {trends?.market_analysis?.volatility ? (
                <div className="space-y-2">
                  <p className="text-3xl font-bold text-white mb-2 capitalize">
                    {trends.market_analysis.volatility.volatility_level.toLowerCase()}
                  </p>
                  <p className="text-sm text-gray-300">
                    Variation coefficient: {trends.market_analysis.volatility.coefficient_of_variation.toFixed(2)}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-gray-300">Volatility data unavailable</p>
              )}
            </div>
          </div>

          {/* Tertiary Tier - Additional Insights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Seasonality Card - Tertiary */}
            <div className="p-5 rounded-lg shadow-sm border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-white">Seasonality</h3>
                <CalendarDaysIcon className="h-6 w-6 text-white" />
              </div>
              {trends?.market_analysis?.seasonality ? (
                <div className="space-y-2">
                  <p className="text-sm text-gray-300">
                    Strongest: {new Date(0, trends.market_analysis.seasonality.strongest_month.month - 1).toLocaleString('default', { month: 'long' })}
                  </p>
                  <p className="text-sm text-gray-300">
                    Pattern: {trends.market_analysis.seasonality.seasonal_pattern}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-gray-300">Seasonality data unavailable</p>
              )}
            </div>

            {/* YOY Change Card - Tertiary */}
            <div className="p-5 rounded-lg shadow-sm border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-white">Year-over-Year</h3>
                <ChartBarSquareIcon className="h-6 w-6 text-white" />
              </div>
              {trends?.market_analysis?.price_trends ? (
                <div className="space-y-2">
                  <p className="text-sm text-gray-300">
                    Price Change: {trends.market_analysis.price_trends.yoy_change > 0 ? '+' : ''}{trends.market_analysis.price_trends.yoy_change.toFixed(1)}%
                  </p>
                  <p className="text-sm text-gray-300">
                    Trend Strength: {trends.market_analysis.price_trends.trend_strength}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-gray-300">YOY data unavailable</p>
              )}
            </div>

            {/* Inventory Turnover Card - Tertiary */}
            <div className="p-5 rounded-lg shadow-sm border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-white">Inventory Turnover</h3>
                <ChartBarIcon className="h-6 w-6 text-white" />
              </div>
              {trends?.market_analysis?.market_health ? (
                <div className="space-y-2">
                  <p className="text-sm text-gray-300">
                    Rate: {calculatedInventoryTurnover.toFixed(2)}
                  </p>
                  <p className="text-sm text-gray-300">
                    Market Balance: {trends.market_analysis.market_health.market_balance}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-gray-300">Inventory data unavailable</p>
              )}
            </div>
          </div>

          {/* Market Insights and Recommendations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Market Insights Card */}
            <div className="p-6 rounded-lg shadow-sm border border-[#232336]" style={{ background: '#222222' }}>
              <h3 className="text-lg font-semibold text-white mb-4">Market Insights</h3>
              {trends?.market_analysis?.market_health && trends?.market_analysis?.price_trends && trends?.market_data?.current_metrics ? (
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-white mb-2">Market Dynamics</h4>
                    <p className="text-gray-300">
                      {trends.market_analysis.market_health.market_balance === 'Seller\'s Market' ?
                        `The ${selectedLocation} market favors sellers, with properties typically spending ${trends.market_data.current_metrics.avg_days_on_market} days on market.` :
                        `The ${selectedLocation} market shows balanced conditions, with properties typically spending ${trends.market_data.current_metrics.avg_days_on_market} days on market.`}
                      {trends.market_analysis.market_health.inventory_turnover > 1.5 ?
                        ' High inventory turnover indicates strong buyer demand and competitive pricing.' :
                        trends.market_analysis.market_health.inventory_turnover > 1 ?
                        ' Moderate inventory turnover suggests a healthy balance between supply and demand.' :
                        ' Slower inventory turnover points to a more measured market pace with less pressure on buyers.'}
                    </p>
                  </div>

                  <div>
                    <h4 className="font-medium text-white mb-2">Price Trends Overview</h4>
                    <p className="text-gray-300">
                      {trends.market_analysis.price_trends.yoy_change < 0 && trends.market_analysis.price_trends.short_term_trend > 0 ?
                        `While prices are down ${Math.abs(trends.market_analysis.price_trends.yoy_change).toFixed(1)}% from last year, recent trends show a ${trends.market_analysis.price_trends.trend_strength.toLowerCase()} recovery, with the median price reaching $${Math.round(trends.market_data.current_metrics.median_price / 1000) * 1000} and price-per-square-foot at $${Math.round(trends.market_data.current_metrics.avg_price_per_sqft)}.` :
                        `The median price has reached $${Math.round(trends.market_data.current_metrics.median_price / 1000) * 1000}, reflecting a ${trends.market_analysis.price_trends.trend_strength.toLowerCase()} ${trends.market_analysis.price_trends.short_term_trend > 0 ? 'upward' : 'downward'} trend. Prices are ${trends.market_analysis.price_trends.yoy_change > 0 ? 'up' : 'down'} ${Math.abs(trends.market_analysis.price_trends.yoy_change).toFixed(1)}% year-over-year, with price-per-square-foot at $${Math.round(trends.market_data.current_metrics.avg_price_per_sqft)}.`}
                    </p>
                  </div>

                  {trends.market_analysis.volatility && (
                    <div>
                      <h4 className="font-medium text-white mb-2">Market Volatility Impact</h4>
                      <p className="text-gray-300">
                        {trends.market_analysis.volatility.coefficient_of_variation > 0.15 ?
                          `The market shows significant price fluctuations (variation coefficient: ${trends.market_analysis.volatility.coefficient_of_variation.toFixed(2)}), suggesting careful timing is crucial for optimal entry points.` :
                          `Price stability is ${trends.market_analysis.volatility.coefficient_of_variation > 0.1 ? 'moderate' : 'strong'} (variation coefficient: ${trends.market_analysis.volatility.coefficient_of_variation.toFixed(2)}), providing a more predictable environment for market participants.`}
                      </p>
                    </div>
                  )}

                  <div>
                    <h4 className="font-medium text-white mb-2">Overall Market Posture</h4>
                    <p className="text-gray-300 font-medium">
                      {trends.market_analysis.market_health.market_balance === 'Seller\'s Market' ?
                        `${selectedLocation} presents a dynamic opportunity with strong price momentum—ideal for sellers and investors seeking short-term appreciation.` :
                        `${selectedLocation} offers a stable environment with moderate price growth—suitable for both buyers and long-term investors.`}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-gray-300">
                    Based on the current market conditions, we recommend:
                  </p>
                  <ul className="list-disc list-inside space-y-2 text-gray-300">
                    <li>
                      {calculateMarketMomentum() > 0 ? 'Consider buying soon as prices are trending upward' : 'The market is stable, good time for strategic investments'}
                    </li>
                    <li>
                      {trends.market_data.current_metrics.avg_days_on_market < 30
                        ? 'Properties are selling quickly, be ready to make competitive offers'
                        : 'More time available for property evaluation and negotiation'}
                    </li>
                    <li>
                      {trends.market_data.current_metrics.avg_price_per_sqft > 200
                        ? 'Focus on properties with high appreciation potential'
                        : 'Look for properties with good value per square foot'}
                    </li>
                  </ul>
                </div>
              )}
            </div>

            {/* Market Recommendations Card */}
            <div className="p-6 rounded-lg shadow-sm border border-[#232336]" style={{ background: '#222222' }}>
              <h3 className="text-xl font-bold text-white mb-4 text-center">Market Recommendations</h3>
              {trends?.market_analysis?.market_health && trends?.market_analysis?.seasonality ? (
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-white mb-2">Timing Strategy</h4>
                    <ul className="list-disc list-inside space-y-2 text-base text-gray-300">
                      <li>
                        {trends.market_analysis.seasonality.seasonal_strength > 20 ?
                          `Target ${new Date(0, trends.market_analysis.seasonality.strongest_month.month - 1).toLocaleString('default', { month: 'long' })} for optimal pricing, when historical averages show prices around $${Math.round(trends.market_analysis.seasonality.strongest_month.average_price / 1000) * 1000}` :
                          `Consider flexible timing as seasonal patterns are weak, though ${new Date(0, trends.market_analysis.seasonality.strongest_month.month - 1).toLocaleString('default', { month: 'long' })} typically offers slightly better pricing at $${Math.round(trends.market_analysis.seasonality.strongest_month.average_price / 1000) * 1000}`}
                      </li>
                      {trends.market_analysis.forecast && (
                        <li>
                          {trends.market_analysis.forecast.short_term_forecast.confidence > 0.7 ?
                            `${trends.market_analysis.forecast.short_term_forecast.value > trends.market_data.current_metrics.median_price ?
                              `Consider accelerating purchases to capture projected price increases to $${Math.round(trends.market_analysis.forecast.short_term_forecast.value / 1000) * 1000} (${trends.market_analysis.forecast.short_term_forecast.confidence.toFixed(1)}% confidence)` :
                              `Consider deferring purchases to align with the projected short-term dip to $${Math.round(trends.market_analysis.forecast.short_term_forecast.value / 1000) * 1000} (${trends.market_analysis.forecast.short_term_forecast.confidence.toFixed(1)}% confidence)`}` :
                            'Monitor market conditions closely as forecast confidence remains moderate'}
                        </li>
                      )}
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium text-white mb-2 mt-4">Investment Approach</h4>
                    <ul className="list-disc list-inside space-y-2 text-base text-gray-300">
                      <li>
                        {trends.market_analysis.market_health.market_balance === `Seller's Market` ?
                          `Secure pre-approval and prepare competitive offers within 48 hours of listing, as high demand requires quick action` :
                          `Leverage extended negotiation windows of 7-10 days for favorable terms and price adjustments`
                        }
                      </li>
                      <li>
                        {trends.market_analysis.market_health.price_momentum > 0.1 ?
                          `Focus on properties with strong appreciation potential, targeting 5-7% annual growth based on recent price momentum` :
                          `Seek value-add opportunities in stable neighborhoods with 3-5% appreciation potential, supported by current market stability`
                        }
                      </li>
                      <li>
                        {trends.market_analysis.market_health.inventory_turnover > 1.5 ?
                          `For investors: Consider quick-turn strategies with 3-6 month holding periods, capitalizing on high market velocity` :
                          `For investors: Focus on long-term holds with 5+ year appreciation targets, taking advantage of stable market conditions`
                        }
                      </li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium text-white mb-2 mt-4">Risk Management</h4>
                    <ul className="list-disc list-inside space-y-2 text-base text-gray-300">
                      {trends.market_analysis.volatility && (
                        <li>
                          {trends.market_analysis.volatility.volatility_level === 'High' ?
                            `Implement strict due diligence with 10-14 day inspection periods and price protection clauses to mitigate market volatility` :
                            `Maintain standard 7-day inspection periods while exploring a broader range of opportunities in this stable market`
                          }
                        </li>
                      )}
                      <li>
                        {trends.market_analysis.market_health.inventory_turnover > 1.5 ?
                          `Prepare for 5-7 day response times on high-demand properties, with financing pre-approval in place` :
                          `Utilize 10-14 day evaluation windows for thorough property assessment and negotiation`
                        }
                      </li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium text-white mb-2 mt-4">Overall Outlook</h4>
                    <p className="text-base text-gray-300 font-medium">
                      {trends.market_analysis.market_health.market_balance === `Seller's Market` ?
                        `${selectedLocation} presents a dynamic opportunity for both investors and homebuyers, with strong price momentum and clear seasonal patterns supporting confident decision-making.` :
                        `${selectedLocation} offers a stable environment for strategic investments, with moderate volatility and predictable market cycles enabling well-timed entries.`
                      }
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-base text-gray-300">
                    Based on the current market conditions, we recommend:
                  </p>
                  <ul className="list-disc list-inside space-y-2 text-base text-gray-300">
                    <li>
                      {calculateMarketMomentum() > 0 ? 'Consider buying soon as prices are trending upward' : 'The market is stable, good time for strategic investments'}
                    </li>
                    <li>
                      {trends.market_data.current_metrics.avg_days_on_market < 30
                        ? 'Properties are selling quickly, be ready to make competitive offers'
                        : 'More time available for property evaluation and negotiation'}
                    </li>
                    <li>
                      {trends.market_data.current_metrics.avg_price_per_sqft > 200
                        ? 'Focus on properties with high appreciation potential'
                        : 'Look for properties with good value per square foot'}
                    </li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Display message when current metrics are loaded but historical data is not */}
      {!isLoading && !error && trends?.market_data?.current_metrics && !trends?.market_data?.historical_data?.median_list_price?.values?.length && selectedLocation && (
        <div className="text-center py-6">
          <p className="text-gray-400">Current market metrics loaded, but historical trend data is not available for this location.</p>
        </div>
      )}

      {/* Display message when no data is available or location is not selected */}
      {!isLoading && !error && (!trends || !trends.market_data?.current_metrics) && selectedLocation && (
        <div className="text-center py-12">
          <p className="text-gray-400">No market data available for this location.</p>
          <p className="text-gray-400">Please try searching for a different location or check the backend logs for issues.</p>
        </div>
      )}

      {!isLoading && !error && !selectedLocation && (
        <div className="text-center py-12">
          <p className="text-gray-400">Enter a location to view market analysis</p>
          <p className="text-gray-200 text-sm mt-2">You can enter either a ZIP code or City, State (e.g., "Austin, TX")</p>
        </div>
      )}

      {/* New section for ML Insights */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-white mb-6">ML Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Price Trends */}
          <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Price Trends</h3>
              <ChartBarSquareIcon className="h-8 w-8 text-indigo-300" />
            </div>
            {trends?.market_analysis?.price_trends ? (
              <div className="space-y-3">
                <p className="text-gray-200">
                  <span className="font-medium">Short-term trend:</span> <span className="font-bold text-lg text-green-300">{trends.market_analysis.price_trends.short_term_trend.toFixed(2)}%</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Medium-term trend:</span> <span className="font-bold text-lg text-blue-300">{trends.market_analysis.price_trends.medium_term_trend.toFixed(2)}%</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Long-term trend:</span> <span className="font-bold text-lg text-red-300">{trends.market_analysis.price_trends.long_term_trend.toFixed(2)}%</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Year-over-year change:</span> <span className="font-bold text-lg">{calculateYOYChange > 0 ? '+' : ''}{calculateYOYChange.toFixed(2)}%</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Trend strength:</span> <span className="font-bold text-lg text-purple-300">{trends.market_analysis.price_trends.trend_strength}</span>
                </p>
              </div>
            ) : (
              <p className="text-gray-400">No price trend insights available.</p>
            )}
          </div>

          {/* Market Health */}
          <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Market Health</h3>
              <HeartIcon className="h-8 w-8 text-red-300" />
            </div>
            {trends?.market_analysis?.market_health ? (
              <div className="space-y-3">
                <p className="text-gray-200">
                  <span className="font-medium">Price momentum:</span> <span className="font-bold text-lg">{trends.market_analysis.market_health.price_momentum.toFixed(2)}</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Inventory turnover:</span> <span className="font-bold text-lg">{calculatedInventoryTurnover.toFixed(2)}</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Market balance:</span> <span className="font-bold text-lg text-green-300">{trends.market_analysis.market_health.market_balance}</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Overall health:</span> <span className="font-bold text-lg text-green-300">{trends.market_analysis.market_health.overall_health}</span>
                </p>
              </div>
            ) : (
              <p className="text-gray-400">No market health insights available.</p>
            )}
          </div>

          {/* Seasonality */}
          <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Seasonality</h3>
              <CalendarDaysIcon className="h-8 w-8 text-orange-300" />
            </div>
            {trends?.market_analysis?.seasonality ? (
              <div className="space-y-3">
                <p className="text-gray-200">
                  <span className="font-medium">Seasonal pattern:</span> <span className="font-bold text-lg">{trends.market_analysis.seasonality.seasonal_pattern}</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Strongest month:</span> <span className="font-bold text-lg">{new Date(0, trends.market_analysis.seasonality.strongest_month.month - 1).toLocaleString('default', { month: 'long' })} (${trends.market_analysis.seasonality.strongest_month.average_price.toLocaleString()})</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Weakest month:</span> <span className="font-bold text-lg">{new Date(0, trends.market_analysis.seasonality.weakest_month.month - 1).toLocaleString('default', { month: 'long' })} (${trends.market_analysis.seasonality.weakest_month.average_price.toLocaleString()})</span>
                </p>
                <p className="text-gray-200">
                  <span className="font-medium">Seasonal strength:</span> <span className="font-bold text-lg">{getSeasonalStrengthLabel(trends.market_analysis.seasonality.seasonal_strength)}</span>
                </p>
              </div>
            ) : (
              <p className="text-gray-400">No seasonality insights available.</p>
            )}
          </div>

          {/* Volatility and Investment Potential Row */}
          <div className="lg:col-span-3 flex justify-center gap-6">
            {/* Volatility */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336] w-full max-w-md" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">Volatility</h3>
                <ExclamationTriangleIcon className="h-8 w-8 text-yellow-300" />
              </div>
              {trends?.market_analysis?.volatility ? (
                <div className="space-y-3">
                  <p className="text-gray-200">
                    <span className="font-medium">Standard deviation:</span> <span className="font-bold text-lg">{trends.market_analysis.volatility.standard_deviation.toLocaleString()}</span>
                  </p>
                  <p className="text-gray-200">
                    <span className="font-medium">Coefficient of variation:</span> <span className="font-bold text-lg">{trends.market_analysis.volatility.coefficient_of_variation.toFixed(2)}</span>
                  </p>
                  <p className="text-gray-200">
                    <span className="font-medium">Price range:</span> <span className="font-bold text-lg">{trends.market_analysis.volatility.price_range.toLocaleString()}</span>
                  </p>
                  <p className="text-gray-200">
                    <span className="font-medium">Volatility level:</span> <span className="font-bold text-lg" style={{ color: getVolatilityLevelLabel(trends.market_analysis.volatility.coefficient_of_variation).color }}>{getVolatilityLevelLabel(trends.market_analysis.volatility.coefficient_of_variation).label}</span>
                  </p>
                </div>
              ) : (
                <p className="text-gray-400">No volatility insights available.</p>
              )}
            </div>

            {/* Investment Potential */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336] w-full max-w-md" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">Investment Potential</h3>
                <CurrencyDollarIcon className="h-8 w-8 text-blue-300" />
              </div>
              {trends?.market_analysis?.price_trends && trends?.market_analysis?.market_health && trends?.market_data?.current_metrics ? (
                <div className="space-y-3">
                  <div className="relative group">
                    <p className="text-gray-200">
                      <span className="font-medium">ROI potential:</span> <span className="font-bold text-lg text-green-300">{((trends.market_analysis.price_trends.short_term_trend + trends.market_analysis.price_trends.medium_term_trend) / 2).toFixed(2)}%</span>
                      <span className="ml-1 text-gray-400 cursor-help">ⓘ</span>
                    </p>
                    <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                      <p>ROI Potential is calculated as the average of short-term and medium-term price trends, representing expected annual appreciation. This metric assumes:</p>
                      <ul className="list-disc list-inside mt-1">
                        <li>No major market disruptions</li>
                        <li>Current market conditions persist</li>
                        <li>Property is held for at least 1 year</li>
                      </ul>
                    </div>
                  </div>
                  <div className="relative group">
                    <p className="text-gray-200">
                      <span className="font-medium">Price per sqft:</span> <span className="font-bold text-lg">${trends.market_data.current_metrics.avg_price_per_sqft}</span>
                      <span className="ml-1 text-gray-400 cursor-help">ⓘ</span>
                    </p>
                    <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                      <p>Average price per square foot helps evaluate property value relative to size. Lower values may indicate better investment opportunities.</p>
                    </div>
                  </div>
                  <div className="relative group">
                    <p className="text-gray-200">
                      <span className="font-medium">Market strength:</span> <span className="font-bold text-lg" style={{ color: getMarketStrengthLabel().color }}>{getMarketStrengthLabel().label}</span>
                      <span className="ml-1 text-gray-400 cursor-help">ⓘ</span>
                    </p>
                    <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                      <p>Market strength is determined by the balance between supply and demand. Strong markets typically have faster price appreciation.</p>
                    </div>
                  </div>
                  <div className="relative group">
                    <p className="text-gray-200">
                      <span className="font-medium">Growth potential:</span> <span className="font-bold text-lg" style={{ color: getGrowthPotentialLabel().color }}>{getGrowthPotentialLabel().label}</span>
                      <span className="ml-1 text-gray-400 cursor-help">ⓘ</span>
                    </p>
                    <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                      <p>Growth potential indicates the likelihood of sustained price appreciation based on historical trends and market conditions.</p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">No investment potential insights available.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Deal Scoring Engine Section */}
      <div className="mt-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">Deal Scoring Engine</h2>
          <div className="flex gap-4">
            <button
              onClick={() => setShowScoringSettings(!showScoringSettings)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gray-800 border border-gray-700 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              <AdjustmentsHorizontalIcon className="h-5 w-5" />
              Customize Weights
            </button>
            {showScoringSettings && (
              <button
                onClick={() => setScoringWeights(DEFAULT_SCORING_WEIGHTS)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gray-800 border border-gray-700 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                Reset to Default
              </button>
            )}
          </div>
        </div>

        {showScoringSettings && (
          <div className="p-6 rounded-lg shadow-md border border-[#232336] mb-6" style={{ background: '#222222' }}>
            <h3 className="text-lg font-semibold text-white mb-4">Customize Scoring Weights</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {(Object.entries(scoringWeights) as [ScoringComponent, number][]).map(([component, weight]) => (
                <div key={component} className="space-y-2">
                  <label className="block text-sm font-medium text-gray-200 capitalize">
                    {component} Weight
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={weight}
                    onChange={(e) => handleWeightChange(component, parseInt(e.target.value))}
                    className="w-full accent-primary-500"
                  />
                  <div className="flex justify-between text-sm text-gray-300">
                    <span>0%</span>
                    <span>{weight}%</span>
                    <span>100%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {calculatePropPulseScore() && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Overall Score Card */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">PropPulse Score</h3>
                {calculatePropPulseScore()?.letterGrade && (
                  <span className="text-3xl font-bold" style={{ color: getScoreColor(calculatePropPulseScore()!.score) }}>
                    {calculatePropPulseScore()?.letterGrade}
                  </span>
                )}
              </div>
              <div className="flex flex-col items-center justify-center">
                {calculatePropPulseScore() && (
                  <ScoreDial score={calculatePropPulseScore()!.score} />
                )}
              </div>
              <div className="mt-4 text-sm text-gray-300">
                <p>Based on current market conditions and your selected weights:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>ROI: {calculatePropPulseScore()?.components.roi.toFixed(1)}</li>
                  <li>Momentum: {calculatePropPulseScore()?.components.momentum.toFixed(1)}</li>
                  <li>Volatility: {calculatePropPulseScore()?.components.volatility.toFixed(1)}</li>
                  <li>Risk: {calculatePropPulseScore()?.components.risk.toFixed(1)}</li>
                </ul>
              </div>
            </div>

            {/* Score Breakdown Card */}
            <div className="p-6 rounded-lg shadow-md border border-[#232336]" style={{ background: '#222222' }}>
              <h3 className="text-xl font-semibold text-white mb-4">Score Breakdown</h3>
              <div className="space-y-4">
                {(Object.entries(scoringWeights) as [ScoringComponent, number][]).map(([component, weight]) => {
                  const score = calculatePropPulseScore()?.components[component] || 0;
                  return (
                    <div key={component} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium capitalize text-gray-200">{component}</span>
                        <span className="text-gray-300">{weight}% weight</span>
                      </div>
                      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full" style={{ width: `${score * (weight / 100)}%`, background: '#87CEEB' }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketAnalysis;
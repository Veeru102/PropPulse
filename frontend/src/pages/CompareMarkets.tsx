import React, { useState } from 'react';
import { darkThemeStyles } from '../styles/darkTheme';
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
} from 'chart.js';

// Add a plugin to give the chart area a dark background that matches the page theme
const chartAreaBackgroundPlugin = {
  id: 'chartAreaBackground',
  beforeDraw: (chart: any) => {
    const { ctx, chartArea } = chart as any;
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
  chartAreaBackgroundPlugin
);

interface MarketData {
  name: string;
  medianPrice: number;
  inventory: number;
  daysOnMarket: number;
  pricePerSqFt: number;
  yearOverYearGrowth: number;
}

const CompareMarkets: React.FC = () => {
  const [selectedMarket1, setSelectedMarket1] = useState<string>('');
  const [selectedMarket2, setSelectedMarket2] = useState<string>('');
  const [timeRange, setTimeRange] = useState<'1m' | '3m' | '6m' | '1y' | '5y'>('6m');

  // Mock data for demonstration
  const markets: MarketData[] = [
    {
      name: 'Los Angeles, CA',
      medianPrice: 1195000,
      inventory: 12500,
      daysOnMarket: 45,
      pricePerSqFt: 688,
      yearOverYearGrowth: 7.8,
    },
    {
      name: 'San Francisco, CA',
      medianPrice: 1450000,
      inventory: 8900,
      daysOnMarket: 38,
      pricePerSqFt: 892,
      yearOverYearGrowth: 6.5,
    },
    {
      name: 'Austin, TX',
      medianPrice: 715000,
      inventory: 15600,
      daysOnMarket: 52,
      pricePerSqFt: 375,
      yearOverYearGrowth: 8.2,
    },
  ];

  // Mock time series data for 5 years (60 months)
  const allLabels = [
    ...Array.from({ length: 60 }, (_, i) => {
      const date = new Date();
      date.setMonth(date.getMonth() - (59 - i));
      return date.toLocaleString('default', { month: 'short', year: '2-digit' });
    })
  ];
  const allData1 = [
    ...Array.from({ length: 60 }, (_, i) => 700000 + i * 8000 + Math.round(Math.sin(i / 6) * 20000))
  ];
  const allData2 = [
    ...Array.from({ length: 60 }, (_, i) => 900000 + i * 6000 + Math.round(Math.cos(i / 7) * 15000))
  ];

  // Filter data based on time range
  let rangeMonths: number;
  switch (timeRange) {
    case '1m':
      rangeMonths = 1;
      break;
    case '3m':
      rangeMonths = 3;
      break;
    case '6m':
      rangeMonths = 6;
      break;
    case '1y':
      rangeMonths = 12;
      break;
    case '5y':
    default:
      rangeMonths = 60;
  }
  const labels = allLabels.slice(-rangeMonths);
  const data1 = allData1.slice(-rangeMonths);
  const data2 = allData2.slice(-rangeMonths);

  const chartData = {
    labels,
    datasets: [
      {
        label: selectedMarket1,
        data: data1,
        borderColor: 'rgb(56, 189, 248)',
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 3,
      },
      {
        label: selectedMarket2,
        data: data2,
        borderColor: 'rgb(168, 85, 247)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#E5E7EB',
          font: {
            family: 'Inter, system-ui, sans-serif',
            weight: 500,
          },
          padding: 20,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(24, 24, 27, 0.9)',
        titleColor: '#F9FAFB',
        bodyColor: '#E5E7EB',
        borderColor: '#3F3F46',
        borderWidth: 1,
        padding: 12,
        bodyFont: {
          family: 'Inter, system-ui, sans-serif',
        },
        titleFont: {
          family: 'Inter, system-ui, sans-serif',
          weight: 600,
        },
      },
    },
    layout: {
      padding: 24,
    },
    scales: {
      y: {
        grid: {
          color: 'rgba(63, 63, 70, 0.5)',
          drawBorder: false,
        },
        ticks: {
          color: '#D1D5DB',
          font: {
            family: 'Inter, system-ui, sans-serif',
          },
          padding: 10,
        },
        border: {
          display: false,
        },
        backgroundColor: '#18181B', // Chart area background
      },
      x: {
        grid: {
          color: 'rgba(63, 63, 70, 0.5)',
          drawBorder: false,
        },
        ticks: {
          color: '#D1D5DB',
          font: {
            family: 'Inter, system-ui, sans-serif',
          },
          padding: 10,
        },
        border: {
          display: false,
        },
        backgroundColor: '#18181B',
      },
    },
    backgroundColor: '#18181B',
  };

  const getMarketData = (marketName: string): MarketData | undefined => {
    return markets.find(market => market.name === marketName);
  };

  const market1Data = getMarketData(selectedMarket1);
  const market2Data = getMarketData(selectedMarket2);

  return (
    <div className={darkThemeStyles.pageContainer}>
      {/* Header with gradient background */}
      <div className="bg-gradient-to-r from-[#1E1E2E] to-[#2D2D44] rounded-xl p-6 mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">Comparative Market Analysis</h1>
        <p className="text-gray-300">Compare multiple locations side by side</p>
      </div>

      <div className={`${darkThemeStyles.card} mb-6`}>
        {/* Market Selection & Time Range */}
        <div className="p-6 border-b border-[#27272A] flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full md:w-auto">
            <div>
              <label className={darkThemeStyles.labelText}>Market 1</label>
              <select
                className={darkThemeStyles.searchInput}
                value={selectedMarket1}
                onChange={(e) => setSelectedMarket1(e.target.value)}
              >
                <option value="">Select Market</option>
                {markets.map(market => (
                  <option key={market.name} value={market.name}>
                    {market.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className={darkThemeStyles.labelText}>Market 2</label>
              <select
                className={darkThemeStyles.searchInput}
                value={selectedMarket2}
                onChange={(e) => setSelectedMarket2(e.target.value)}
              >
                <option value="">Select Market</option>
                {markets.map(market => (
                  <option key={market.name} value={market.name}>
                    {market.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          {/* Time Range Dropdown */}
          <div className="flex items-center gap-2">
            <label className={darkThemeStyles.labelText}>Time Range</label>
            <select
              className={darkThemeStyles.searchInput + ' w-36'}
              value={timeRange}
              onChange={e => setTimeRange(e.target.value as '1m' | '3m' | '6m' | '1y' | '5y')}
            >
              <option value="1m">1 Month</option>
              <option value="3m">3 Months</option>
              <option value="6m">6 Months</option>
              <option value="1y">1 Year</option>
              <option value="5y">5 Years</option>
            </select>
          </div>
        </div>

        {/* Chart Section */}
        {(selectedMarket1 || selectedMarket2) && (
          <div className="p-6" style={{ background: '#18181B', borderRadius: '1rem' }}>
            <div className="h-[400px] w-full">
              <Line data={chartData} options={chartOptions} />
            </div>
          </div>
        )}
      </div>

      {/* Market Metrics Comparison */}
      {(market1Data || market2Data) && (
        <div className={`${darkThemeStyles.card} mb-6`}>
          <div className="p-6">
            <h2 className="text-xl font-semibold text-white mb-6">City Metrics Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Median Price */}
              <div className={`${darkThemeStyles.statsCard} ${darkThemeStyles.gradientBg}`}>
                <h3 className={darkThemeStyles.subheading}>Median Price</h3>
                <div className="mt-2 space-y-2">
                  {market1Data && (
                    <div>
                      <span className={darkThemeStyles.labelText}>{market1Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        ${market1Data.medianPrice.toLocaleString()}
                      </p>
                    </div>
                  )}
                  {market2Data && (
                    <div className="mt-4">
                      <span className={darkThemeStyles.labelText}>{market2Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        ${market2Data.medianPrice.toLocaleString()}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Price per Sq Ft */}
              <div className={`${darkThemeStyles.statsCard} ${darkThemeStyles.gradientBg}`}>
                <h3 className={darkThemeStyles.subheading}>Price per Sq Ft</h3>
                <div className="mt-2 space-y-2">
                  {market1Data && (
                    <div>
                      <span className={darkThemeStyles.labelText}>{market1Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        ${market1Data.pricePerSqFt}
                      </p>
                    </div>
                  )}
                  {market2Data && (
                    <div className="mt-4">
                      <span className={darkThemeStyles.labelText}>{market2Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        ${market2Data.pricePerSqFt}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Days on Market */}
              <div className={`${darkThemeStyles.statsCard} ${darkThemeStyles.gradientBg}`}>
                <h3 className={darkThemeStyles.subheading}>Days on Market</h3>
                <div className="mt-2 space-y-2">
                  {market1Data && (
                    <div>
                      <span className={darkThemeStyles.labelText}>{market1Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        {market1Data.daysOnMarket} days
                      </p>
                    </div>
                  )}
                  {market2Data && (
                    <div className="mt-4">
                      <span className={darkThemeStyles.labelText}>{market2Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        {market2Data.daysOnMarket} days
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Year over Year Growth */}
              <div className={`${darkThemeStyles.statsCard} ${darkThemeStyles.gradientBg}`}>
                <h3 className={darkThemeStyles.subheading}>Year over Year Growth</h3>
                <div className="mt-2 space-y-2">
                  {market1Data && (
                    <div>
                      <span className={darkThemeStyles.labelText}>{market1Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        {market1Data.yearOverYearGrowth}%
                      </p>
                    </div>
                  )}
                  {market2Data && (
                    <div className="mt-4">
                      <span className={darkThemeStyles.labelText}>{market2Data.name}</span>
                      <p className={darkThemeStyles.valueText}>
                        {market2Data.yearOverYearGrowth}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Analysis Section */}
      {(market1Data && market2Data) && (
        <div className={`${darkThemeStyles.card} mb-6`}>
          <div className="p-6">
            <h2 className="text-xl font-semibold text-white mb-4">AI-Powered Comparative Summary</h2>
            <div className={`${darkThemeStyles.gradientBg} rounded-lg p-6 border border-[#27272A]`}>
              <p className="text-gray-100 leading-relaxed">
                Based on the analysis of {market1Data.name} and {market2Data.name}, our AI suggests that...
                {/* Add your AI analysis content here */}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ML Insights Section */}
      <div className={darkThemeStyles.card}>
        <div className="p-6">
          <h2 className="text-xl font-semibold text-white mb-4">ML Insights</h2>
          <div className={`flex gap-4 ${darkThemeStyles.gradientBg} rounded-lg p-6 border border-[#27272A]`}>
            <input
              type="text"
              placeholder="Ask a question about these markets..."
              className={darkThemeStyles.searchInput}
            />
            <button className={darkThemeStyles.button}>
              Ask AI
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CompareMarkets; 
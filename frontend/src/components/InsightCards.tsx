import React from 'react';
import { 
  ChartBarIcon, 
  CurrencyDollarIcon, 
  ExclamationTriangleIcon, 
  HomeIcon 
} from '@heroicons/react/24/outline';
import { darkThemeStyles } from '../styles/darkTheme';

interface InsightCardProps {
  title: string;
  icon: React.ReactNode;
  insights: string[];
  metrics?: { label: string; value: string | number; color?: string; infoButton?: React.ReactNode }[];
  badges?: { label: string; color: string }[];
}

export const InsightCard: React.FC<InsightCardProps> = ({ title, icon, insights, metrics, badges }) => {
  return (
    <div className="bg-[#1A1A1A] rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <div className="text-[#93C5FD]">{icon}</div>
      </div>
      
      <div className="space-y-2">
        {insights.map((insight, index) => (
          <p key={index} className="text-sm text-white">{insight}</p>
        ))}
      </div>

      {metrics && (
        <div className="mt-3 space-y-2">
          {metrics.map((metric, index) => (
            <div key={index} className="flex justify-between items-center">
              <span className="text-sm text-white">{metric.label}</span>
              <span className={`text-sm font-medium ${metric.color || 'text-white'} flex items-center`}>
                {metric.value}
                {metric.infoButton}
              </span>
            </div>
          ))}
        </div>
      )}

      {badges && (
        <div className="mt-3 flex flex-wrap gap-2">
          {badges.map((badge, index) => (
            <span
              key={index}
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badge.color}`}
            >
              {badge.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

interface MarketSnapshotCardProps {
  marketHealth: any;
  priceTrends: any;
  volatility: any;
}

const getOverallHealthTextColor = (healthLevel: string) => {
  if (healthLevel === 'Strong') return 'text-[#22C55E]';
  if (healthLevel === 'Moderate') return 'text-[#3B82F6]';
  if (healthLevel === 'Weak') return 'text-[#EF4444]';
  return 'text-gray-400';
};

const getMarketBalanceBadgeColor = (balance: string) => {
  if (balance === "Seller's Market") return 'bg-[#22C55E] text-white';
  if (balance === 'Balanced Market') return 'bg-[#3B82F6] text-white';
  return 'bg-[#EF4444] text-white'; // Buyer's Market
};

const getVolatilityLevelBadgeColor = (level: string) => {
  if (level === 'Low' || level === 'Very Low') return 'bg-[#22C55E] text-white';
  if (level === 'Moderate') return 'bg-[#3B82F6] text-white';
  return 'bg-[#EF4444] text-white'; // High / Very High
};

const getOverallRiskBadgeColor = (overallRisk: number) => {
  if (overallRisk > 7) return 'bg-[#22C55E] text-white'; // Low Risk
  if (overallRisk > 4) return 'bg-[#3B82F6] text-white'; // Moderate Risk
  return 'bg-[#EF4444] text-white'; // High Risk
};

export const getOverallHealth5CategoryColor = (healthScore: number) => {
  if (healthScore >= 0.85) return 'text-[#10B981]'; // Very Strong
  if (healthScore >= 0.7) return 'text-[#34D399]'; // Strong
  if (healthScore >= 0.4) return 'text-[#3B82F6]'; // Moderate
  if (healthScore >= 0.2) return 'text-[#FBBF24]'; // Weak
  return 'text-[#EF4444]'; // Very Weak
};

export const getPriceTrend5CategoryColor = (trendValue: number) => {
  if (trendValue >= 10) return 'text-[#10B981]'; // Very Strong Growth
  if (trendValue >= 5) return 'text-[#34D399]'; // Strong Growth
  if (trendValue >= 0) return 'text-[#3B82F6]'; // Stable / Neutral
  if (trendValue >= -5) return 'text-[#FBBF24]'; // Weak Decline
  return 'text-[#EF4444]'; // Strong Decline
};

export const getVolatilityLevel5CategoryColor = (level: string) => {
  if (level === 'Very Low') return 'bg-[#10B981] text-white';
  if (level === 'Low') return 'bg-[#34D399] text-white';
  if (level === 'Moderate') return 'bg-[#3B82F6] text-white';
  if (level === 'High') return 'bg-[#FBBF24] text-white';
  return 'bg-[#EF4444] text-white'; // Very High
};

export const getGrowthBadge5CategoryColor = (momentum: number) => {
  if (momentum >= 0.15) return 'bg-[#10B981] text-white'; // Very Strong Growth
  if (momentum >= 0.05) return 'bg-[#34D399] text-white'; // Strong Growth
  if (momentum >= 0) return 'bg-[#3B82F6] text-white'; // Stable Growth
  if (momentum >= -0.05) return 'bg-[#FBBF24] text-white'; // Weak Decline
  return 'bg-[#EF4444] text-white'; // Strong Decline
};

export const getRoiBadge5CategoryColor = (roiValue: number) => {
  if (roiValue >= 10) return 'bg-[#10B981] text-white'; // Very High ROI
  if (roiValue >= 5) return 'bg-[#34D399] text-white'; // High ROI
  if (roiValue >= 0) return 'bg-[#3B82F6] text-white'; // Neutral ROI
  if (roiValue >= -5) return 'bg-[#FBBF24] text-white'; // Low ROI
  return 'bg-[#EF4444] text-white'; // Very Low ROI
};

export const getOverallRisk5CategoryColor = (overallRisk: number) => {
  if (overallRisk >= 8.5) return 'bg-[#10B981] text-white'; // Very Low Risk
  if (overallRisk >= 7) return 'bg-[#34D399] text-white'; // Low Risk
  if (overallRisk >= 4) return 'bg-[#3B82F6] text-white'; // Moderate Risk
  if (overallRisk >= 2) return 'bg-[#FBBF24] text-white'; // High Risk
  return 'bg-[#EF4444] text-white'; // Very High Risk
};

export const getRentalViability5CategoryColor = (priceToRent: number) => {
  if (priceToRent < 10) return 'bg-[#10B981] text-white'; // Very High Viability
  if (priceToRent < 15) return 'bg-[#34D399] text-white'; // High Viability
  if (priceToRent < 20) return 'bg-[#3B82F6] text-white'; // Neutral
  if (priceToRent < 25) return 'bg-[#FBBF24] text-white'; // Low Viability
  return 'bg-[#EF4444] text-white'; // Very Low Viability
};

export const getDemand5CategoryColor = (inventoryTurnover: number) => {
  if (inventoryTurnover > 2.0) return 'bg-[#10B981] text-white'; // Very High Demand
  if (inventoryTurnover > 1.5) return 'bg-[#34D399] text-white'; // High Demand
  if (inventoryTurnover > 1.0) return 'bg-[#3B82F6] text-white'; // Moderate Demand
  if (inventoryTurnover > 0.5) return 'bg-[#FBBF24] text-white'; // Low Demand
  return 'bg-[#EF4444] text-white'; // Very Low Demand
};

export const MarketSnapshotCard: React.FC<MarketSnapshotCardProps> = ({
  marketHealth,
  priceTrends,
  volatility
}) => {
  const insights = [
    `${marketHealth.market_balance} with ${
      typeof marketHealth.inventory_turnover === 'number'
        ? marketHealth.inventory_turnover.toFixed(1)
        : marketHealth.inventory_turnover || 'N/A'
    }x inventory turnover`,
    `Price momentum at ${
      typeof priceTrends.short_term_trend === 'number'
        ? priceTrends.short_term_trend.toFixed(1)
        : priceTrends.short_term_trend || 'N/A'
    }% with ${volatility.volatility_level} volatility`
  ];

  const metrics = [
    { label: 'Market Health', value: marketHealth.overall_health, color: getOverallHealth5CategoryColor(marketHealth.overall_health) },
    { label: 'Price Trend', value: `${
      typeof priceTrends.yoy_change === 'number'
        ? (priceTrends.yoy_change > 0 ? '+' : '') + priceTrends.yoy_change.toFixed(1)
        : priceTrends.yoy_change || 'N/A'
    }%`, color: getPriceTrend5CategoryColor(typeof priceTrends.yoy_change === 'number' ? priceTrends.yoy_change : 0) }
  ];

  const badges = [
    { label: marketHealth.market_balance, color: getMarketBalanceBadgeColor(marketHealth.market_balance) },
    { label: volatility.volatility_level, color: getVolatilityLevel5CategoryColor(volatility.volatility_level) }
  ];

  return (
    <InsightCard
      title="Market Snapshot"
      icon={<ChartBarIcon className="h-6 w-6" />}
      insights={insights}
      metrics={metrics}
      badges={badges}
    />
  );
};

interface InvestmentStrategyCardProps {
  forecast: any;
  marketHealth: any;
  investmentMetrics: any;
  priceTrends: any;
  showCashOnCashInfo?: boolean;
}

const getGrowthBadgeColor = (momentum: number) => {
  if (momentum > 0.1) return 'bg-[#22C55E] text-white'; // Strong Growth
  if (momentum > 0) return 'bg-[#3B82F6] text-white'; // Stable Growth
  return 'bg-[#EF4444] text-white'; // Decline
};

const getRoiBadgeColor = (roiValue: number) => {
  if (roiValue >= 0) return 'bg-[#22C55E] text-white';
  return 'bg-[#EF4444] text-white';
};

export const InvestmentStrategyCard: React.FC<InvestmentStrategyCardProps> = ({
  forecast,
  marketHealth,
  investmentMetrics,
  priceTrends,
  showCashOnCashInfo
}) => {
  const insights = [
    `Projected ${priceTrends.yoy_change > 0 ? 'growth' : 'decline'} of ${Math.abs(priceTrends.yoy_change).toFixed(1)}% in next 6 months`,
    `Cap rate of ${investmentMetrics.cap_rate.toFixed(2)}% with ${investmentMetrics.cash_on_cash.toFixed(2)}% cash on cash return`
  ];

  // Info button for Cash-on-Cash
  const cashOnCashInfoButton = showCashOnCashInfo ? (
    <span className="ml-2 group relative">
      <button className="text-blue-600 hover:text-blue-800 text-xs" tabIndex={0}>
        ⓘ
      </button>
      <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-3 bg-white text-gray-800 text-xs rounded-lg opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-200 z-20 pointer-events-none">
        Cash-on-Cash is based on:<br />
        • 20% down payment<br />
        • 7% interest rate<br />
        • 0.8% monthly rent<br />
        • 3% annual operating expenses
      </div>
    </span>
  ) : null;

  const metrics = [
    { label: 'Cap Rate', value: `${investmentMetrics.cap_rate.toFixed(2)}%`, color: 'text-[#93C5FD]' },
    { label: 'Cash on Cash', value: `${investmentMetrics.cash_on_cash.toFixed(2)}%`, color: 'text-[#93C5FD]', infoButton: cashOnCashInfoButton }
  ];

  const badges = [
    { label: marketHealth.price_momentum > 0.1 ? 'Strong Growth' : 'Stable Growth', color: getGrowthBadge5CategoryColor(marketHealth.price_momentum) },
    { label: `ROI ${priceTrends.yoy_change.toFixed(1)}%`, color: getRoiBadge5CategoryColor(priceTrends.yoy_change) }
  ];

  return (
    <InsightCard
      title="Investment Strategy"
      icon={<CurrencyDollarIcon className="h-6 w-6" />}
      insights={insights}
      metrics={metrics}
      badges={badges}
    />
  );
};

interface RiskOverviewCardProps {
  riskMetrics: any;
  marketHealth: any;
  volatility: any;
}

export const RiskOverviewCard: React.FC<RiskOverviewCardProps> = ({
  riskMetrics,
  marketHealth,
  volatility
}) => {
  const insights = [
    `Overall risk score of ${
      !isNaN(Number(riskMetrics.overall_risk))
        ? Number(riskMetrics.overall_risk).toFixed(1)
        : riskMetrics.overall_risk || 'N/A'
    }/10`,
    `${marketHealth.market_balance} conditions with ${volatility.volatility_level} price volatility`
  ];

  const metrics = [
    {
      label: 'Market Risk',
      value: (!isNaN(Number(riskMetrics.market_risk))
        ? Number(riskMetrics.market_risk).toFixed(1)
        : riskMetrics.market_risk || 'N/A') + '/10',
      color: 'text-orange-600'
    },
    {
      label: 'Location Risk',
      value: (!isNaN(Number(riskMetrics.location_risk))
        ? Number(riskMetrics.location_risk).toFixed(1)
        : riskMetrics.location_risk || 'N/A') + '/10',
      color: 'text-red-600'
    }
  ];

  const badges = [
    { label: (!isNaN(Number(riskMetrics.overall_risk)) && Number(riskMetrics.overall_risk) > 7) ? 'Low Risk' : (!isNaN(Number(riskMetrics.overall_risk)) && Number(riskMetrics.overall_risk) > 4) ? 'Moderate Risk' : 'High Risk', color: getOverallRisk5CategoryColor(Number(riskMetrics.overall_risk)) },
    { label: volatility.volatility_level, color: getVolatilityLevel5CategoryColor(volatility.volatility_level) }
  ];

  return (
    <InsightCard
      title="Risk Overview"
      icon={<ExclamationTriangleIcon className="h-6 w-6" />}
      insights={insights}
      metrics={metrics}
      badges={badges}
    />
  );
};

interface RentalOutlookCardProps {
  investmentMetrics: any;
  marketHealth: any;
  forecast: any;
}

const getRentalViabilityBadgeColor = (priceToRent: number) => {
  if (priceToRent < 15) return 'bg-[#22C55E] text-white';
  return 'bg-[#EF4444] text-white';
};

const getRentalDemandInfo = (inventoryTurnover: number) => {
  const demandInfo = getDemand5CategoryColor(inventoryTurnover);
  if (inventoryTurnover > 1.5) {
    return { label: 'High Demand', color: demandInfo };
  } else if (inventoryTurnover > 1.0) {
    return { label: 'Moderate Demand', color: demandInfo };
  } else {
    return { label: 'Low Demand', color: demandInfo };
  }
};

export const RentalOutlookCard: React.FC<RentalOutlookCardProps> = ({
  investmentMetrics,
  marketHealth,
  forecast
}) => {
  const insights = [
    `Price to rent ratio of ${investmentMetrics.price_to_rent.toFixed(2)}x`,
    `Rental demand ${marketHealth.inventory_turnover > 1.5 ? 'strong' : 'moderate'} with ${marketHealth.market_balance} conditions`
  ];

  const metrics = [
    { label: 'Price/Rent', value: `${investmentMetrics.price_to_rent.toFixed(2)}x`, color: 'text-[#93C5FD]' },
    { label: 'Rental Yield', value: `${(investmentMetrics.cap_rate * 0.8).toFixed(2)}%`, color: 'text-[#93C5FD]' }
  ];

  const badges = [
    { label: investmentMetrics.price_to_rent < 15 ? 'Rental Viable' : 'Rental Risk', color: getRentalViability5CategoryColor(investmentMetrics.price_to_rent) },
    getRentalDemandInfo(marketHealth.inventory_turnover)
  ];

  return (
    <InsightCard
      title="Rental Outlook"
      icon={<HomeIcon className="h-6 w-6" />}
      insights={insights}
      metrics={metrics}
      badges={badges}
    />
  );
}; 
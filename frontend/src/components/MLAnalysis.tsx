import React, { useState } from 'react';
import { darkThemeStyles } from '../styles/darkTheme';
import { Property, getCrewAIAnalysis } from '../services/api';
import { 
  MarketSnapshotCard, 
  InvestmentStrategyCard, 
  RiskOverviewCard, 
  RentalOutlookCard,
  InsightCard,
  getOverallHealth5CategoryColor,
  getPriceTrend5CategoryColor,
  getVolatilityLevel5CategoryColor,
  getGrowthBadge5CategoryColor,
  getRoiBadge5CategoryColor,
  getOverallRisk5CategoryColor,
  getRentalViability5CategoryColor,
  getDemand5CategoryColor
} from './InsightCards';
import {
  ChartBarSquareIcon,
  HeartIcon,
  CalendarDaysIcon,
  ExclamationTriangleIcon,
  AdjustmentsHorizontalIcon,
  StarIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface MarketMetrics {
  market_health: number;
  market_momentum: number;
  market_stability: number;
}

interface RiskMetrics {
  market_risk: number;
  property_risk: number;
  location_risk: number;
  overall_risk: number;
}

interface InvestmentMetrics {
  cap_rate: number;
  cash_on_cash: number;
  price_to_rent: number;
  dom_ratio: number;
}

interface MLAnalysisProps {
  property: Property;
  market_metrics?: MarketMetrics;
  risk_metrics?: RiskMetrics;
  investment_metrics?: InvestmentMetrics;
}

// ---------------- Investment Assumption Defaults ----------------
export const DEFAULT_ASSUMPTIONS = {
  appreciationRate: 3,          // % annual price appreciation
  rentGrowthRate: 3,            // % annual rent growth (used for IRR)
  monthlyRentPct: 0.8,          // Monthly rent as % of property value (e.g., 0.8 = 0.8%)
  propertyTaxRate: 1.5,         // % of price per year
  insuranceRate: 0.5,           // % of price per year
  maintenanceRate: 1,           // % of price per year
  managementRate: 10,           // % of rent
  ltvRatio: 80,                 // Loan-to-value ratio (%)
  interestRate: 7,              // Mortgage interest rate (%)
  mortgageTerm: 30,             // Years
} as const;

const MLAnalysis: React.FC<MLAnalysisProps> = ({ property }) => {
  const [financingMethod, setFinancingMethod] = useState<'cash' | 'loan'>('loan');
  const [assumptions, setAssumptions] = useState<typeof DEFAULT_ASSUMPTIONS>({ ...DEFAULT_ASSUMPTIONS });

  // Convenience aliases (kept for readability in existing code)
  const ltvRatio = assumptions.ltvRatio;
  const interestRate = assumptions.interestRate;

  const handleAssumptionChange = (key: keyof typeof DEFAULT_ASSUMPTIONS, value: number) => {
    setAssumptions(prev => ({ ...prev, [key]: value }));
  };

  const resetAssumptions = () => setAssumptions({ ...DEFAULT_ASSUMPTIONS });

  const isChanged = (key: keyof typeof DEFAULT_ASSUMPTIONS) => assumptions[key] !== DEFAULT_ASSUMPTIONS[key];

  const [analysisTab, setAnalysisTab] = useState<'full' | 'rental' | 'flip'>('full');
  const [aiSummaries, setAiSummaries] = useState<{ [key: string]: string }>({});
  const [loadingAI, setLoadingAI] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  
  const {
    predicted_value,
    value_confidence,
    investment_confidence,
    feature_importance,
    risk_metrics,
    market_metrics,
    base_metrics,
    investment_metrics,
  } = property;

  // Debug: Log risk metrics for each property
  console.log('Risk metrics for property', property.property_id, property.risk_metrics);

  // Helper to extract and highlight missing data/uncertainty from AI output
  const extractMissingData = (text: string) => {
    const missingMatch = text.match(/Missing Data:(.*)/i);
    if (missingMatch) {
      return missingMatch[1].trim();
    }
    return null;
  };

  // Fetch CrewAI/LLM summary when tab or property changes
  React.useEffect(() => {
    const key = `${property.property_id}_${analysisTab}`;
    if (aiSummaries[key] || loadingAI) return;
    setLoadingAI(true);
    setAiError(null);
    getCrewAIAnalysis(property.property_id, analysisTab)
      .then(res => {
        // If backend returns a string or fallback, handle gracefully
        let summary = res.summary;
        if (typeof summary !== 'string') summary = JSON.stringify(summary);
        setAiSummaries(prev => ({ ...prev, [key]: summary }));
        setLoadingAI(false);
      })
      .catch(err => {
        setAiError('AI insights are temporarily unavailable. Please try again later or review the metrics above for guidance.');
        setLoadingAI(false);
      });
  }, [property.property_id, analysisTab]);

  // Render a colored progress bar with the numeric value shown for clarity and tooltip
  const renderProgressBar = (value: number, invert: boolean = false, tooltipText?: string) => {
    // Scale the value to a percentage (0-100)
    const scaledValue = Math.max(0, Math.min(100, value * 100));
    const pct = invert ? (100 - scaledValue) / 100 : scaledValue / 100;

    const getBarColor = (percentage: number, inverted: boolean) => {
      const colors = [
        '#EF4444', // Red
        '#F97316', // Orange
        '#FACC15', // Yellow
        '#22C55E', // Green
        '#60A5FA'  // Blue
      ];

      let startColor, endColor;
      // Adjust step for 5 categories (0-20, 20-40, 40-60, 60-80, 80-100)
      let step = 0.20; 

      // Ensure percentage is clamped between 0 and 1
      const clampedPercentage = Math.max(0, Math.min(1, percentage));
      const segment = Math.floor(clampedPercentage / step);
      const segmentPct = (clampedPercentage % step) / step;

      if (inverted) {
        startColor = colors[Math.min(colors.length - 1 - segment, colors.length - 1)];
        endColor = colors[Math.min(colors.length - 1 - (segment + 1), colors.length - 1)];
      } else {
        startColor = colors[Math.min(segment, colors.length - 1)];
        endColor = colors[Math.min(segment + 1, colors.length - 1)];
      }
      
      // Simple linear interpolation for gradient segment
      const lerpColor = (c1: string, c2: string, factor: number) => {
        const hexToRgb = (hex: string) => {
          const bigint = parseInt(hex.slice(1), 16);
          const r = (bigint >> 16) & 255;
          const g = (bigint >> 8) & 255;
          const b = bigint & 255;
          return [r, g, b];
        };
        const rgbToHex = (r: number, g: number, b: number) => '#' + [r,g,b].map(x => Math.round(x).toString(16).padStart(2, '0')).join('');

        const rgb1 = hexToRgb(c1);
        const rgb2 = hexToRgb(c2);

        const r = rgb1[0] + factor * (rgb2[0] - rgb1[0]);
        const g = rgb1[1] + factor * (rgb2[1] - rgb1[1]);
        const b = rgb1[2] + factor * (rgb2[2] - rgb1[2]);

        return rgbToHex(r, g, b);
      };
      
      const interpolatedColor = lerpColor(startColor, endColor, segmentPct);

      return interpolatedColor;
    };

    return (
      <div className="flex items-center gap-2 group relative">
        <div className="w-full bg-[#2A2A2A] rounded-full h-2.5 relative">
          <div
            className="h-2.5 rounded-full"
            style={{
              width: `${scaledValue}%`,
              backgroundColor: getBarColor(pct, invert)
            }}
          ></div>
        </div>
        <span className="text-xs text-white whitespace-nowrap" style={{minWidth:'50px'}}>
          {scaledValue.toFixed(2)}%
        </span>
        {tooltipText && (
          <div className="absolute left-0 bottom-full mb-2 w-80 p-3 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20 pointer-events-none">
            <div className="whitespace-pre-line">{tooltipText}</div>
          </div>
        )}
      </div>
    );
  };

  // Calculate cashflow metrics
  const calculateCashflow = () => {
    if (!property.price || !investment_metrics) return null;

    const price = property.price;
    const monthlyRent = price * (assumptions.monthlyRentPct / 100);

    // Operating expenses based on user assumptions
    const propertyTax = (price * assumptions.propertyTaxRate / 100) / 12;
    const insurance = (price * assumptions.insuranceRate / 100) / 12;
    const maintenance = (price * assumptions.maintenanceRate / 100) / 12;
    const management = assumptions.managementRate / 100;
    const totalExpenses = propertyTax + insurance + maintenance + management;

    let monthlyMortgage = 0;
    let downPayment = 0;
    let loanAmount = 0;

    if (financingMethod === 'loan') {
      downPayment = price * ((100 - ltvRatio) / 100);
      loanAmount = price - downPayment;
      const monthlyRate = interestRate / 100 / 12;
      const numPayments = 30 * 12;
      monthlyMortgage = (loanAmount * monthlyRate * Math.pow(1 + monthlyRate, numPayments)) / 
                       (Math.pow(1 + monthlyRate, numPayments) - 1);
    }

    const netCashflow = monthlyRent - totalExpenses - monthlyMortgage;

    return {
      monthlyRent,
      monthlyMortgage,
      totalExpenses,
      netCashflow,
      downPayment,
      loanAmount,
      breakdown: {
        propertyTax,
        insurance,
        maintenance,
        management
      }
    };
  };

  // Calculate forecast
  const calculateForecast = () => {
    if (!property.price || !market_metrics) return null;

    const currentPrice = property.price;
    let annualGrowthRate: number | null = null;
    if (isChanged('appreciationRate')) {
      annualGrowthRate = assumptions.appreciationRate / 100;
    }

    // Try to use YOY change from market_trends if available
    if (annualGrowthRate === null && property.market_trends && (property.market_trends as any).market_analysis && (property.market_trends as any).market_analysis.price_trends) {
      const yoy = (property.market_trends as any).market_analysis.price_trends.yoy_change;
      if (typeof yoy === 'number') {
        annualGrowthRate = yoy / 100; // Convert percent to decimal
      }
    }

    // Fallback to weighted formula if YOY not available
    if (annualGrowthRate === null) {
      const marketMomentum = market_metrics.market_momentum || 0.25;
      const marketHealth = market_metrics.market_health || 0.35;
      // More sophisticated fallback: allow negative growth
      const baseGrowthRate = 0.03; // 3% base appreciation
      const momentumBonus = (marketMomentum - 0.5) * 0.08; // Up to 4% bonus/penalty
      const healthBonus = (marketHealth - 0.5) * 0.06; // Up to 3% bonus/penalty
      annualGrowthRate = baseGrowthRate + momentumBonus + healthBonus;
    }

    const forecastedPrice = currentPrice * (1 + annualGrowthRate);
    const priceChange = forecastedPrice - currentPrice;
    const percentageChange = (priceChange / currentPrice) * 100;

    // Confidence based on market stability and data quality
    const marketStability = market_metrics.market_stability || 0.45;
    const stabilityFactor = marketStability;
    const confidence = Math.min(95, Math.max(45, stabilityFactor * 85 + 10));

    return {
      currentPrice,
      forecastedPrice,
      priceChange,
      percentageChange,
      confidence,
      annualGrowthRate: annualGrowthRate * 100
    };
  };

  const renderFeatureImportance = () => {
    if (!feature_importance) return null;
    return (
      <div className="space-y-4 p-6" style={{ background: '#1A1A1A' }}>
        <h3 className="text-lg font-semibold text-[#E4E4E7]">Feature Importance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Value Model</h4>
            {Object.entries(feature_importance.value_model)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 5)
              .map(([feature, importance]) => (
                <div key={feature} className="mt-2">
                  <div className="flex justify-between text-sm">
                    <span>{feature}</span>
                    <span>{(importance * 100).toFixed(2)}%</span>
                  </div>
                  {renderProgressBar(importance)}
                </div>
              ))}
          </div>
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Investment Model</h4>
            {Object.entries(feature_importance.investment_model)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 5)
              .map(([feature, importance]) => (
                <div key={feature} className="mt-2">
                  <div className="flex justify-between text-sm">
                    <span>{feature}</span>
                    <span>{(importance * 100).toFixed(2)}%</span>
                  </div>
                  {renderProgressBar(importance)}
                </div>
              ))}
          </div>
        </div>
      </div>
    );
  };

  const renderRiskMetrics = () => {
    if (!risk_metrics) return null;
    
    const tooltips = {
      market_risk: `Market Risk Assessment:\n• Inventory levels and turnover rates\n• Price reduction vs increase trends\n• Days on market compared to optimal (≤45 days)\n• Market volatility and price stability\n\nLower scores indicate higher market risk.`,
      property_risk: `Property Risk Assessment:\n• Property age (optimal: <30 years)\n• Size appropriateness (optimal: 1,500-3,000 sqft)\n• Price positioning relative to market\n• Property condition and renovation needs\n\nLower scores indicate higher property-specific risk.`,
      location_risk: `Location Risk Assessment:\n• Crime rates (optimal: <100 per 100k)\n• School ratings (optimal: 8+ rating)\n• Walkability scores (optimal: 70+ score)\n• Economic indicators (unemployment <8%)\n• Natural disaster/flood zone exposure\n\nLower scores indicate higher location-based risk.`,
      overall_risk: `Overall Risk Assessment:\n• Weighted combination of all risk factors\n• Market Risk: 40% weight\n• Property Risk: 30% weight\n• Location Risk: 30% weight\n\nThis represents the comprehensive risk profile of the investment opportunity.`
    };
    
    return (
      <div className="space-y-4 p-6" style={{ background: '#1A1A1A' }}>
        <h3 className="text-lg font-semibold text-[#E4E4E7]">Risk Assessment</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Market Risk</h4>
            {renderProgressBar(risk_metrics.market_risk, true, tooltips.market_risk)}
          </div>
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Property Risk</h4>
            {renderProgressBar(risk_metrics.property_risk, true, tooltips.property_risk)}
          </div>
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Location Risk</h4>
            {renderProgressBar(risk_metrics.location_risk, true, tooltips.location_risk)}
          </div>
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Overall Risk</h4>
            {renderProgressBar(risk_metrics.overall_risk, true, tooltips.overall_risk)}
          </div>
        </div>
      </div>
    );
  };

  const renderMarketMetrics = () => {
    if (!market_metrics) return null;
    
    const tooltips = {
      market_health: `Market Health Assessment:\n• Population growth trends (requires >0% for good scores)\n• Employment growth rates (requires >0% for good scores)\n• Income growth patterns (weighted 40%)\n• Penalties applied for negative growth in any area\n\nHigh scores require consistent positive growth across all metrics.`,
      market_momentum: `Market Momentum Assessment:\n• 1-year price changes (weighted 50%)\n• 3-year price trends (weighted 30%)\n• 5-year historical patterns (weighted 20%)\n• Consistency bonus for sustained positive trends\n• Penalties for recent price declines\n\nRequires significant price appreciation for high scores.`,
      market_stability: `Market Stability Assessment:\n• Price volatility (must be <25% for good scores)\n• Inventory volatility (must be <35% for good scores)\n• Days on market (optimal: ≤20 days)\n• Market activity ratios (price increases vs reductions)\n\nHigh scores require low volatility and quick market turnover.`
    };
    
    return (
      <div className="space-y-4 p-6" style={{ background: '#1A1A1A' }}>
        <h3 className="text-lg font-semibold text-[#E4E4E7]">Market Health</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <h4 className="text-sm font-medium" style={{ color: getOverallHealth5CategoryColor(market_metrics.market_health) }}>Market Health</h4>
            {renderProgressBar(market_metrics.market_health, false, tooltips.market_health)}
          </div>
          <div>
            <h4 className="text-sm font-medium text-[#93C5FD]">Market Momentum</h4>
            {renderProgressBar(market_metrics.market_momentum, false, tooltips.market_momentum)}
          </div>
          <div>
            <h4 className="text-sm font-medium text-[#A3A3A3]">Market Stability</h4>
            {renderProgressBar(market_metrics.market_stability, false, tooltips.market_stability)}
          </div>
        </div>
      </div>
    );
  };

  const renderBaseMetrics = () => {
    if (!base_metrics) return null;
    return (
      <div className="space-y-4 p-6" style={{ background: '#1A1A1A' }}>
        <h3 className="text-lg font-semibold text-[#E4E4E7]">Property Metrics</h3>
        <dl className="grid grid-cols-2 gap-4">
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Price per Sq Ft</dt>
            <dd className="text-lg font-semibold">${base_metrics.price_per_sqft.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Price to Median</dt>
            <dd className="text-lg font-semibold">{base_metrics.price_to_median.toFixed(2)}x</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Sq Ft per Bed</dt>
            <dd className="text-lg font-semibold">{base_metrics.sqft_per_bed.toFixed(1)}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Beds/Baths Ratio</dt>
            <dd className="text-lg font-semibold">{base_metrics.beds_baths_ratio.toFixed(1)}</dd>
          </div>
        </dl>
      </div>
    );
  };

  const renderInvestmentMetrics = () => {
    if (!investment_metrics) return null;
    return (
      <div className="space-y-4 p-6" style={{ background: '#1A1A1A' }}>
        <h3 className="text-lg font-semibold text-[#E4E4E7]">Investment Metrics</h3>
        <dl className="grid grid-cols-2 gap-4">
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Cap Rate</dt>
            <dd className="text-lg font-semibold">{investment_metrics.cap_rate.toFixed(2)}%</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Cash on Cash</dt>
            <dd className="text-lg font-semibold">{investment_metrics.cash_on_cash.toFixed(2)}%</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">Price to Rent</dt>
            <dd className="text-lg font-semibold">{investment_metrics.price_to_rent.toFixed(2)}x</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-[#A3A3A3]">DOM Ratio</dt>
            <dd className="text-lg font-semibold">{investment_metrics.dom_ratio.toFixed(3)}x</dd>
          </div>
        </dl>
      </div>
    );
  };

  // Add new state for assumption panel visibility
  const [showAssumptions, setShowAssumptions] = useState(false);
  const [highlightedAssumption, setHighlightedAssumption] = useState<keyof typeof DEFAULT_ASSUMPTIONS | null>(null);

  // Add helper to get assumption tooltip text
  const getAssumptionTooltip = (metric: 'irr' | 'coc' | 'forecast') => {
    const forecast = calculateForecast();
    switch (metric) {
      case 'irr':
        return `IRR is based on:\n• 5-year hold period\n• ${assumptions.appreciationRate}% annual appreciation\n• ${assumptions.monthlyRentPct}% monthly rent\n• ${assumptions.propertyTaxRate + assumptions.insuranceRate + assumptions.maintenanceRate}% annual operating expenses`;
      case 'coc':
        return `Cash-on-Cash is based on:\n• ${100 - assumptions.ltvRatio}% down payment\n• ${assumptions.interestRate}% interest rate\n• ${assumptions.monthlyRentPct}% monthly rent\n• ${assumptions.propertyTaxRate + assumptions.insuranceRate + assumptions.maintenanceRate}% annual operating expenses`;
      case 'forecast':
        return `Price forecast is based on:\n• ${isChanged('appreciationRate') ? Number(assumptions.appreciationRate).toFixed(2) : forecast?.annualGrowthRate ? Number(forecast.annualGrowthRate).toFixed(2) : '3.00'}% annual appreciation\n• Market momentum: ${market_metrics?.market_momentum ? (market_metrics.market_momentum * 100).toFixed(1) : 'N/A'}%\n• Market health: ${market_metrics?.market_health ? (market_metrics.market_health * 100).toFixed(1) : 'N/A'}%`;
    }
  };

  // Add helper to get relevant assumptions for each metric
  const getRelevantAssumptions = (metric: 'irr' | 'coc' | 'forecast'): (keyof typeof DEFAULT_ASSUMPTIONS)[] => {
    switch (metric) {
      case 'irr':
        return ['appreciationRate', 'monthlyRentPct', 'propertyTaxRate', 'insuranceRate', 'maintenanceRate', 'managementRate'];
      case 'coc':
        return ['ltvRatio', 'interestRate', 'monthlyRentPct', 'propertyTaxRate', 'insuranceRate', 'maintenanceRate', 'managementRate'];
      case 'forecast':
        return ['appreciationRate'];
    }
  };

  // Unified AI analysis renderer
  const renderAIAnalysis = () => {
    const key = `${property.property_id}_${analysisTab}`;
    const summary = aiSummaries[key];
    const missingData = summary ? extractMissingData(summary) : null;
    const isFallback = summary && (summary.includes('AI insights are temporarily unavailable') || summary.length < 40);

    if (loadingAI) {
      return <div className="text-[#A3A3A3]">Loading AI insights...</div>;
    }

    if (aiError) {
      return <div className="text-[#FF6B6B]">{aiError}</div>;
    }

    if (!summary) {
      return <div className="text-[#A3A3A3]">No AI analysis available.</div>;
    }

    // --- Robust, data-driven transformation ---
    // Use actual property data, and show 'Data unavailable' if missing
    const getMarketBalance = (health: number | undefined) => {
      if (typeof health !== 'number') return 'Data unavailable';
      if (health > 0.7) return "Seller's Market";
      if (health > 0.4) return 'Balanced Market';
      return "Buyer's Market";
    };
    const getOverallHealth = (health: number | undefined) => {
      if (typeof health !== 'number') return 'Data unavailable';
      if (health > 0.7) return 'Strong';
      if (health > 0.4) return 'Moderate';
      return 'Weak';
    };
    const getVolatilityLevel = (stability: number | undefined) => {
      if (typeof stability !== 'number') return 'Data unavailable';
      if (stability > 0.7) return 'Low';
      if (stability > 0.4) return 'Moderate';
      return 'High';
    };

    // Use actual values or show 'N/A' if missing
    const marketHealth = market_metrics?.market_health;
    const marketMomentum = market_metrics?.market_momentum;
    const marketStability = market_metrics?.market_stability;

    // risk metrics logic
    function isDefaultRiskMetrics(risk: any) {

      if (!risk) return true;
      const vals = [risk.market_risk, risk.property_risk, risk.location_risk, risk.overall_risk];
      if (vals.every(v => v === undefined || v === null)) return true;
      // If all are numbers and match the static pattern, treat as default
      if (
        vals[0] === 0.4 && vals[1] === 0.5 && vals[2] === 0.6 && vals[3] === 0.5
      ) return true;
      return false;
    }
    let riskMetricsForCard = { market_risk: 'N/A', property_risk: 'N/A', location_risk: 'N/A', overall_risk: 'N/A' };
    if (property.risk_metrics && !isDefaultRiskMetrics(property.risk_metrics)) {
      riskMetricsForCard = {
        market_risk: typeof property.risk_metrics.market_risk === 'number' ? property.risk_metrics.market_risk.toString() : 'N/A',
        property_risk: typeof property.risk_metrics.property_risk === 'number' ? property.risk_metrics.property_risk.toString() : 'N/A',
        location_risk: typeof property.risk_metrics.location_risk === 'number' ? property.risk_metrics.location_risk.toString() : 'N/A',
        overall_risk: typeof property.risk_metrics.overall_risk === 'number' ? property.risk_metrics.overall_risk.toString() : 'N/A',
      };
    }

    // --- Only Full Analysis tab ---
    const transformedMarketHealth = {
      market_balance: getMarketBalance(marketHealth),
      inventory_turnover: typeof marketMomentum === 'number' ? marketMomentum.toFixed(2) : 'N/A',
      overall_health: getOverallHealth(marketHealth),
      price_momentum: typeof marketMomentum === 'number' ? marketMomentum : 'N/A',
    };
    const transformedPriceTrends = {
      short_term_trend: typeof marketMomentum === 'number' ? marketMomentum * 100 : 'N/A',
      yoy_change: typeof marketMomentum === 'number' ? marketMomentum * 100 : 'N/A',
    };
    const transformedVolatility = {
      volatility_level: getVolatilityLevel(marketStability),
      coefficient_of_variation: typeof marketStability === 'number' ? (1 - marketStability).toFixed(2) : 'N/A',
    };
    const transformedForecast = {
      short_term_forecast: {
        value: typeof marketMomentum === 'number' ? marketMomentum * 100 : 'N/A',
        confidence: typeof marketStability === 'number' ? Math.min(95, Math.max(45, marketStability * 85 + 10)) : 'N/A',
      },
      medium_term_forecast: {
        value: typeof marketMomentum === 'number' ? marketMomentum * 120 : 'N/A',
        confidence: typeof marketStability === 'number' ? Math.min(90, Math.max(40, marketStability * 80 + 10)) : 'N/A',
      },
    };
    const investmentMetricsForCard = investment_metrics || {};

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <MarketSnapshotCard
            marketHealth={transformedMarketHealth}
            priceTrends={transformedPriceTrends}
            volatility={transformedVolatility}
          />
          <InvestmentStrategyCard
            forecast={transformedForecast}
            marketHealth={transformedMarketHealth}
            investmentMetrics={investmentMetricsForCard}
            showCashOnCashInfo
          />
          <LiquidityExitCard
            domRatio={investment_metrics?.dom_ratio || property.investment_metrics?.dom_ratio || 1}
            daysOnMarket={
              // Try to get actual days on market from various sources
              (property as any).days_on_market ||
              (property.base_metrics as any)?.days_on_market ||
              (investment_metrics?.dom_ratio && investment_metrics.dom_ratio > 0 ? Math.round(investment_metrics.dom_ratio * 30) : 0) ||
              0
            }
            marketHealth={transformedMarketHealth}
            volatility={transformedVolatility}
          />
          <RentalOutlookCard
            investmentMetrics={investmentMetricsForCard}
            marketHealth={transformedMarketHealth}
            forecast={transformedForecast}
          />
        </div>

        {missingData && (
          <div className="bg-[#2A2A2A] border-l-4 border-[#60A5FA] p-3 rounded text-[#E4E4E7]">
            <strong>Missing Data:</strong> {missingData}
          </div>
        )}
      </div>
    );
  };

  // Helper: Calculate IRR
  function calculateIRR(cashFlows: number[], guess = 0.1): number | null {
    // Newton-Raphson method for IRR
    const maxIter = 1000;
    const tol = 1e-6;
    let rate = guess;
    for (let iter = 0; iter < maxIter; iter++) {
      let npv = 0;
      let dNpv = 0;
      for (let t = 0; t < cashFlows.length; t++) {
        npv += cashFlows[t] / Math.pow(1 + rate, t);
        if (t > 0) {
          dNpv -= t * cashFlows[t] / Math.pow(1 + rate, t + 1);
        }
      }
      if (Math.abs(npv) < tol) return rate;
      rate = rate - npv / dNpv;
      if (rate < -0.99) return null;
    }
    return null;
  }

  // Long-Term IRR Projection Calculation
  const getLongTermIRR = () => {
    if (!property.price || !investment_metrics || !market_metrics) return null;
    const years = 5;
    const price = property.price;
    const downPayment = price * 0.2; // Assume 20% down
    const loanAmount = price - downPayment;
    const interestRateValue = interestRate / 100;
    const monthlyRate = interestRateValue / 12;
    const numPayments = 30 * 12;
    const monthlyMortgage = (loanAmount * monthlyRate * Math.pow(1 + monthlyRate, numPayments)) /
      (Math.pow(1 + monthlyRate, numPayments) - 1);
    const annualRent = price * (assumptions.monthlyRentPct / 100) * 12;
    const annualExpenses = (price * assumptions.propertyTaxRate / 100) +
                          (price * assumptions.insuranceRate / 100) +
                          (price * assumptions.maintenanceRate / 100) +
                          (annualRent * assumptions.managementRate / 100);

    // Appreciation uses user assumption (fallback to forecast if unchanged)
    let appreciation = assumptions.appreciationRate / 100;
    if (!isChanged('appreciationRate')) {
    const forecast = calculateForecast();
    if (forecast && forecast.annualGrowthRate) {
      appreciation = forecast.annualGrowthRate / 100;
      }
    }

    // Cash flows: Year 0 = -downPayment, Years 1-5 = net cashflow, Year 5 = sale proceeds
    let cashFlows = [-downPayment];
    let currentRent = annualRent;
    let currentPrice = price;
    for (let year = 1; year <= years; year++) {
      const netCashflow = currentRent - annualExpenses - (monthlyMortgage * 12);
      if (year < years) {
        cashFlows.push(netCashflow);
      } else {
        // Final year: add sale proceeds
        currentPrice = currentPrice * Math.pow(1 + appreciation, years);
        const saleProceeds = currentPrice - (loanAmount - (monthlyMortgage * 12 * years));
        cashFlows.push(netCashflow + saleProceeds);
      }
      currentRent *= (1 + appreciation);
    }
    const irr = calculateIRR(cashFlows);
    return irr !== null ? (irr * 100) : null;
  };

  // Hold Time Recommendation Calculation
  const getHoldTimeRecommendation = () => {
    if (!property.price || !investment_metrics || !market_metrics) return null;
    const irr = getLongTermIRR();
    const years = 5;
    const forecast = calculateForecast();
    const appreciation = forecast ? forecast.annualGrowthRate : 0;
    const dom_ratio = investment_metrics.dom_ratio || 0;
    const risk = risk_metrics?.overall_risk;
    let explanation = '';

    // Logic: If IRR is high (>8%) and appreciation is positive, recommend hold
    if (irr !== null && irr > 8 && appreciation > 0) {
      return { rec: 'Strong hold recommended: high IRR and positive price forecast.', explanation: '' };
    }
    // If IRR is high but appreciation is negative, explain why
    if (irr !== null && irr > 8 && appreciation <= 0) {
      explanation = 'IRR is high due to strong rental income, but price is forecast to decline. Consider risk of capital loss.';
      return { rec: 'Caution: High IRR, but negative price forecast.', explanation };
    }
    // If IRR is low (<5%) or risk is high, recommend against hold
    if ((irr !== null && irr < 5) || (risk !== undefined && risk > 7)) {
      explanation = 'Low IRR and/or high risk make this property unsuitable for holding.';
      return { rec: 'Not recommended to hold.', explanation };
    }
    // If appreciation is negative, recommend against hold
    if (appreciation < 0) {
      explanation = 'Price is forecast to decline, which may offset rental returns.';
      return { rec: 'Not recommended to hold.', explanation };
    }
    // Default: recommend hold for moderate IRR and positive appreciation
    return { rec: 'Consider holding for moderate returns.', explanation: '' };
  };

  const renderLongTermIRRProjection = () => {
    const irr = getLongTermIRR();
    return (
      <div className="p-6" style={{ background: '#1A1A1A' }}>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-[#E4E4E7]">Long-Term IRR Projection</h3>
          <div 
            className="relative group"
            onMouseEnter={() => setHighlightedAssumption('appreciationRate')}
            onMouseLeave={() => setHighlightedAssumption(null)}
          >
            <button 
              className="text-[#60A5FA] hover:text-[#60A5FA] text-sm flex items-center"
              onClick={() => setShowAssumptions(true)}
            >
              <span className="mr-1">ⓘ</span> Based on assumptions
            </button>
            <div className="absolute right-0 bottom-full mb-2 w-80 p-3 bg-[#1A1A1A] text-[#E4E4E7] text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20">
              <div className="whitespace-pre-line">{getAssumptionTooltip('irr')}</div>
            </div>
          </div>
        </div>
        {irr !== null ? (
          <div className="text-2xl font-bold text-[#60A5FA]">{irr.toFixed(2)}% IRR (5-year hold)</div>
        ) : (
          <p className="text-sm text-[#A3A3A3]">Insufficient data for IRR calculation.</p>
        )}
        <p className="text-sm text-[#A3A3A3] mt-2">Projected IRR if held for 5 years, including rent growth and forecasted appreciation.</p>
      </div>
    );
  };

  const renderHoldTimeRecommendation = () => {
    const recObj = getHoldTimeRecommendation();
    
    const getRecommendationColor = (recommendation: string) => {
      if (recommendation.includes('Strong hold recommended')) return 'text-[#22C55E]';
      if (recommendation.includes('Not recommended to hold')) return 'text-[#EF4444]';
      return 'text-[#60A5FA]'; // Default for 'Consider holding'
    };

    return (
      <div className="p-6" style={{ background: '#1A1A1A' }}>
        <h3 className="text-xl font-bold text-[#E4E4E7] mb-6">Hold Time Recommendation</h3>
        <div className={`text-lg font-semibold ${recObj ? getRecommendationColor(recObj.rec) : 'text-[#A3A3A3]'}`}>{recObj ? recObj.rec : 'Insufficient data for recommendation.'}</div>
        {recObj && recObj.explanation && (
          <div className="text-sm text-[#E4E4E7] mt-2">{recObj.explanation}</div>
        )}
        <p className="text-sm text-[#A3A3A3] mt-2">Recommendation based on IRR, price forecast, and risk profile.</p>
      </div>
    );
  };

  // Calculate breakeven rent
  const calculateBreakevenRent = () => {
    if (!property.price) return null;

    const price = property.price;
    const downPayment = price * ((100 - ltvRatio) / 100);
    const loanAmount = price - downPayment;
    const monthlyRate = interestRate / 100 / 12;
    const numPayments = 30 * 12;
    const monthlyMortgage = (loanAmount * monthlyRate * Math.pow(1 + monthlyRate, numPayments)) / 
                           (Math.pow(1 + monthlyRate, numPayments) - 1);

    // Operating expenses
    const propertyTax = (price * assumptions.propertyTaxRate / 100) / 12;
    const insurance = (price * assumptions.insuranceRate / 100) / 12;
    const maintenance = (price * assumptions.maintenanceRate / 100) / 12;
    const management = assumptions.managementRate / 100;

    // Solve for rent where net cashflow = 0
    // netCashflow = rent - (rent * management) - monthlyMortgage - propertyTax - insurance - maintenance
    // 0 = rent * (1 - management) - monthlyMortgage - propertyTax - insurance - maintenance
    // rent = (monthlyMortgage + propertyTax + insurance + maintenance) / (1 - management)
    const breakevenRent = (monthlyMortgage + propertyTax + insurance + maintenance) / (1 - management);
    
    return breakevenRent;
  };

  // Calculate current rent (using 0.8% rule)
  const calculateCurrentRent = () => {
    if (!property.price) return null;
    return property.price * (assumptions.monthlyRentPct / 100);
  };

  // Calculate breakeven progress
  const calculateBreakevenProgress = () => {
    const breakevenRent = calculateBreakevenRent();
    const currentRent = calculateCurrentRent();
    if (!breakevenRent || !currentRent) return null;
    
    const progress = (currentRent / breakevenRent) * 100;
    let color = 'bg-[#FF6B6B]';
    if (progress >= 100) {
      color = 'bg-[#60A5FA]';
    } else if (progress >= 90) {
      color = 'bg-[#E4E4E7]';
    }
    
    return { progress, color };
  };

  const renderEnhancedCashflowAnalysis = () => {
    const cashflow = calculateCashflow();
    if (!cashflow) return null;

    const breakevenRent = calculateBreakevenRent();
    const currentRent = calculateCurrentRent();
    const breakevenProgress = calculateBreakevenProgress();

    // Helper function to get gradient color based on percentage
    const getProgressGradient = (progress: number) => {
      if (progress <= 50) return '#EF4444';  // Red
      if (progress <= 75) return '#F97316';  // Orange
      if (progress <= 90) return '#FACC15';  // Yellow
      return '#22C55E';  // Green
    };

    // Helper function to get cashflow color
    const getCashflowColor = (amount: number) => {
      if (amount <= -500) return '#EF4444';  // Strong negative - Red
      if (amount < 0) return '#F97316';      // Mild negative - Orange
      if (amount === 0) return '#FACC15';    // Break-even - Yellow
      if (amount <= 500) return '#86EFAC';   // Small positive - Light Green
      return '#22C55E';                      // Strong positive - Green
    };

    return (
      <div className="p-6" style={{ background: '#1A1A1A' }}>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-[#E4E4E7]">Enhanced Cashflow Analysis</h3>
          <div 
            className="relative group"
            onMouseEnter={() => setHighlightedAssumption('ltvRatio')}
            onMouseLeave={() => setHighlightedAssumption(null)}
          >
            <button 
              className="text-[#60A5FA] hover:text-[#60A5FA] text-sm flex items-center"
              onClick={() => setShowAssumptions(true)}
            >
              <span className="mr-1">ⓘ</span> Based on assumptions
            </button>
            <div className="absolute right-0 bottom-full mb-2 w-80 p-3 bg-[#1A1A1A] text-[#E4E4E7] text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20">
              <div className="whitespace-pre-line">{getAssumptionTooltip('coc')}</div>
            </div>
          </div>
        </div>

        {/* Sensitivity Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="rounded-lg p-4" style={{ background: '#1A1A1A' }}>
            <label className="block text-sm font-medium text-white mb-2">
              LTV Ratio: {ltvRatio}%
            </label>
            <input
              type="range"
              min="50"
              max="100"
              value={ltvRatio}
              onChange={(e) => handleAssumptionChange('ltvRatio', Number(e.target.value))}
              className="w-full h-2 bg-[#2A2A2A] rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#60A5FA]"
            />
          </div>
          <div className="rounded-lg p-4" style={{ background: '#1A1A1A' }}>
            <label className="block text-sm font-medium text-white mb-2">
              Interest Rate: {interestRate}%
            </label>
            <input
              type="range"
              min="5"
              max="9"
              step="0.1"
              value={interestRate}
              onChange={(e) => handleAssumptionChange('interestRate', Number(e.target.value))}
              className="w-full h-2 bg-[#2A2A2A] rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#60A5FA]"
            />
          </div>
        </div>

        {/* Breakeven Analysis */}
        {breakevenRent && currentRent && breakevenProgress && (
          <div className="p-4 mb-6" style={{ background: '#1A1A1A' }}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-3">Breakeven Analysis</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-white">Breakeven Rent</span>
                <span className="font-semibold text-white">${breakevenRent.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-white">Current Rent</span>
                <span className="font-semibold text-white">${currentRent.toLocaleString()}</span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-white">Progress to Breakeven</span>
                  <span className="font-medium text-white">{breakevenProgress.progress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-[#2A2A2A] rounded-full h-2.5">
                  <div
                    className="h-2.5 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${Math.min(breakevenProgress.progress, 100)}%`,
                      background: `linear-gradient(90deg, ${getProgressGradient(breakevenProgress.progress)} 0%, ${getProgressGradient(Math.min(breakevenProgress.progress + 20, 100))} 100%)`
                    }}
                  ></div>
                </div>
              </div>
              <div className="text-xs text-[#A3A3A3] mt-2">
                Note: Breakeven rent is a point-in-time calculation and does not account for future rent growth.
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Income */}
          <div className="rounded-lg p-4" style={{ background: '#1A1A1A' }}>
            <h4 className="text-lg font-semibold text-[#22C55E] mb-3">Monthly Income</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-white">Rental Income</span>
                <span className="font-semibold text-white">${cashflow.monthlyRent.toLocaleString()}</span>
              </div>
              <div className="border-t border-[#2A2A2A] pt-2">
                <div className="flex justify-between">
                  <span className="font-bold text-[#22C55E]">Total Income</span>
                  <span className="font-bold text-[#22C55E]">${cashflow.monthlyRent.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Expenses */}
          <div className="rounded-lg p-4" style={{ background: '#1A1A1A' }}>
            <h4 className="text-lg font-semibold text-[#EF4444] mb-3">Monthly Expenses</h4>
            <div className="space-y-2 text-sm">
              {financingMethod === 'loan' && (
                <div className="flex justify-between">
                  <span className="text-white">Mortgage ({ltvRatio}% LTV, {interestRate}%)</span>
                  <span className="font-semibold text-white">${cashflow.monthlyMortgage.toLocaleString()}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-white">Property Tax</span>
                <span className="text-white">${cashflow.breakdown.propertyTax.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white">Insurance</span>
                <span className="text-white">${cashflow.breakdown.insurance.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white">Maintenance</span>
                <span className="text-white">${cashflow.breakdown.maintenance.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white">Management</span>
                <span className="text-white">${cashflow.breakdown.management.toLocaleString()}</span>
              </div>
              <div className="border-t border-[#2A2A2A] pt-2">
                <div className="flex justify-between">
                  <span className="font-bold text-[#EF4444]">Total Expenses</span>
                  <span className="font-bold text-[#EF4444]">${(cashflow.totalExpenses + cashflow.monthlyMortgage).toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Net Cashflow */}
          <div className="rounded-lg p-4" style={{ background: '#1A1A1A' }}>
            <h4 className="text-lg font-semibold text-[#60A5FA] mb-3">Net Cashflow</h4>
            <div className="space-y-3">
              <div className="text-3xl font-bold" style={{ color: getCashflowColor(cashflow.netCashflow) }}>
                {cashflow.netCashflow >= 0 ? '+' : ''}${cashflow.netCashflow.toLocaleString()}
              </div>
              <div className="text-sm">
                <div className="text-white">Annual: {cashflow.netCashflow >= 0 ? '+' : ''}${(cashflow.netCashflow * 12).toLocaleString()}</div>
                {financingMethod === 'loan' && (
                  <>
                    <div className="mt-2 pt-2 border-t border-[#2A2A2A]">
                      <div className="text-white">Down Payment: ${cashflow.downPayment.toLocaleString()}</div>
                      <div className="text-white">Loan Amount: ${cashflow.loanAmount.toLocaleString()}</div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderForecast = () => {
    const forecast = calculateForecast();
    if (!forecast) return null;

    return (
      <div className="p-6" style={{ background: '#1A1A1A' }}>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-[#E4E4E7]">12-Month Price Forecast</h3>
          <div 
            className="relative group"
            onMouseEnter={() => setHighlightedAssumption('appreciationRate')}
            onMouseLeave={() => setHighlightedAssumption(null)}
          >
            <button 
              className="text-[#60A5FA] hover:text-[#60A5FA] text-sm flex items-center"
              onClick={() => setShowAssumptions(true)}
            >
              <span className="mr-1">ⓘ</span> Based on assumptions
            </button>
            <div className="absolute right-0 bottom-full mb-2 w-80 p-3 bg-[#1A1A1A] text-[#E4E4E7] text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20">
              <div className="whitespace-pre-line">{getAssumptionTooltip('forecast')}</div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Current vs Forecasted */}
          <div className={`${darkThemeStyles.card} rounded-lg p-6 shadow-sm border border-[#A3A3A3]`}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-4">Price Projection</h4>
            <div className="space-y-4">
              <div>
                <div className="text-sm text-[#A3A3A3]">Current Price</div>
                <div className="text-2xl font-bold text-[#E4E4E7]">
                  ${forecast.currentPrice.toLocaleString()}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-[#1A1A1A] rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-[#60A5FA] to-[#60A5FA] h-2 rounded-full transition-all duration-1000"
                    style={{ width: '100%' }}
                  ></div>
                </div>
                <span className="text-sm text-[#A3A3A3]">12 months</span>
              </div>
              <div>
                <div className="text-sm text-[#A3A3A3]">Forecasted Price</div>
                <div className={`text-2xl font-bold ${forecast.forecastedPrice > forecast.currentPrice ? 'text-[#22C55E]' : forecast.forecastedPrice < forecast.currentPrice ? 'text-[#EF4444]' : 'text-[#60A5FA]'}`}>
                  ${forecast.forecastedPrice.toLocaleString()}
                </div>
                <div className={`text-lg font-semibold ${forecast.percentageChange >= 0 ? 'text-[#60A5FA]' : 'text-[#FF6B6B]'}`}>
                  {forecast.percentageChange >= 0 ? '+' : ''}{forecast.percentageChange.toFixed(1)}%
                  <span className="text-sm text-[#A3A3A3] ml-2">
                    ({forecast.percentageChange >= 0 ? '+' : ''}${forecast.priceChange.toLocaleString()})
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Forecast Details */}
          <div className={`${darkThemeStyles.card} rounded-lg p-6 shadow-sm border border-[#A3A3A3]`}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-4">Forecast Details</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-[#A3A3A3]">Annual Growth Rate</span>
                <span className={`font-semibold ${forecast.annualGrowthRate >= 0 ? 'text-[#60A5FA]' : 'text-[#FF6B6B]'}`}>{forecast.annualGrowthRate.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[#A3A3A3]">Confidence Level</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-[#1A1A1A] rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${forecast.confidence >= 70 ? 'bg-[#60A5FA]' : forecast.confidence >= 50 ? 'bg-[#E4E4E7]' : 'bg-[#FF6B6B]'}`}
                      style={{ width: `${forecast.confidence}%` }}
                    ></div>
                  </div>
                  <span className="font-semibold">{forecast.confidence.toFixed(0)}%</span>
                </div>
              </div>
              <div className="mt-4 p-3 bg-[#1A1A1A] rounded-lg">
                <div className="text-xs text-[#A3A3A3]">
                  <div className="font-medium mb-1">Forecast based on:</div>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Market momentum and health indicators</li>
                    <li>Historical price trends and patterns</li>
                    <li>Local market stability factors</li>
                    <li>Economic growth projections</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderInvestmentAssumptions = () => {
    const forecast = calculateForecast();
    const irr = getLongTermIRR();
    
    return (
      <div className={`${darkThemeStyles.card} p-6 border border-[#2A2A2A] mb-6`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-[#E4E4E7]">Investment Assumptions</h3>
          <button
            onClick={() => setShowAssumptions(!showAssumptions)}
            className="text-[#60A5FA] hover:text-[#60A5FA] text-sm flex items-center"
          >
            {showAssumptions ? 'Hide' : 'Edit'}
            <svg 
              className={`ml-1 w-4 h-4 transform transition-transform ${showAssumptions ? 'rotate-180' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {showAssumptions && (
          <>
            {/* Editable Assumptions Grid */}
            <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Interest Rate */}
              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('interestRate') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'interestRate' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-[#E4E4E7] flex items-center">
                    Interest Rate (%)
                    <span className="ml-1 text-[#E4E4E7]" title="Used in mortgage calculations for Cash-on-Cash, IRR, and Breakeven.">ⓘ</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    className="w-20 border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                    value={interestRate}
                    onChange={(e) => handleAssumptionChange('interestRate', Number(e.target.value))}
                  />
                </div>
                <input
                  type="range"
                  min="2"
                  max="12"
                  step="0.1"
                  value={interestRate}
                  onChange={(e) => handleAssumptionChange('interestRate', Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* LTV Ratio */}
              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('ltvRatio') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'ltvRatio' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-[#E4E4E7] flex items-center">
                    LTV Ratio (%)
                    <span className="ml-1 text-[#E4E4E7]" title="Determines down payment and loan amount for mortgage calculations.">ⓘ</span>
                  </label>
                  <input
                    type="number"
                    step="1"
                    className="w-20 border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                    value={ltvRatio}
                    onChange={(e) => handleAssumptionChange('ltvRatio', Number(e.target.value))}
                  />
                </div>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={ltvRatio}
                  onChange={(e) => handleAssumptionChange('ltvRatio', Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Appreciation Rate */}
              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('appreciationRate') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'appreciationRate' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <label className="text-sm font-medium text-[#E4E4E7] flex items-center mb-2">
                  Appreciation Rate (%/yr)
                  <span className="ml-1 text-[#E4E4E7]" title="Primary driver for IRR and Price Forecast. If unchanged, system forecast is used.">ⓘ</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                  value={assumptions.appreciationRate}
                  onChange={(e) => handleAssumptionChange('appreciationRate', Number(e.target.value))}
                />
              </div>

              {/* Monthly Rent Rule */}
              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('monthlyRentPct') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'monthlyRentPct' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <label className="text-sm font-medium text-[#E4E4E7] flex items-center mb-2">
                  Monthly Rent (% of value)
                  <span className="ml-1 text-[#E4E4E7]" title="Used in rental income calculations across metrics.">ⓘ</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                  value={assumptions.monthlyRentPct}
                  onChange={(e) => handleAssumptionChange('monthlyRentPct', Number(e.target.value))}
                />
              </div>

              {/* Expense Rates */}
              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('propertyTaxRate') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'propertyTaxRate' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <label className="text-sm font-medium text-[#E4E4E7] flex items-center mb-2">
                  Property Tax Rate (%/yr)
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                  value={assumptions.propertyTaxRate}
                  onChange={(e) => handleAssumptionChange('propertyTaxRate', Number(e.target.value))}
                />
              </div>

              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('insuranceRate') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'insuranceRate' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <label className="text-sm font-medium text-[#E4E4E7] flex items-center mb-2">
                  Insurance Rate (%/yr)
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                  value={assumptions.insuranceRate}
                  onChange={(e) => handleAssumptionChange('insuranceRate', Number(e.target.value))}
                />
              </div>

              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('maintenanceRate') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'maintenanceRate' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <label className="text-sm font-medium text-[#E4E4E7] flex items-center mb-2">
                  Maintenance Rate (%/yr)
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                  value={assumptions.maintenanceRate}
                  onChange={(e) => handleAssumptionChange('maintenanceRate', Number(e.target.value))}
                />
              </div>

              <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border transition-colors duration-200 ${
                isChanged('managementRate') ? 'border-[#60A5FA]' :
                highlightedAssumption === 'managementRate' ? 'border-[#60A5FA] bg-[#1A1A1A]' : ''
              }`}>
                <label className="text-sm font-medium text-[#E4E4E7] flex items-center mb-2">
                  Management Fee (% of rent)
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-[#E4E4E7] bg-transparent"
                  value={assumptions.managementRate}
                  onChange={(e) => handleAssumptionChange('managementRate', Number(e.target.value))}
                />
              </div>
            </div>

            {/* Reset Button */}
            <div className="mb-4 flex justify-end">
              <button
                onClick={resetAssumptions}
                className="text-sm text-[#60A5FA] hover:text-[#60A5FA] flex items-center"
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Reset to Defaults
              </button>
            </div>
          </>
        )}

        {/* Always visible summary */}
        <div className="space-y-4">
          <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border border-[#A3A3A3]`}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-2">IRR (5-year hold)</h4>
            <p className="text-sm text-[#A3A3A3] mb-2">Assumes:</p>
            <ul className="list-disc list-inside text-sm text-[#A3A3A3] space-y-1">
              <li>20% down payment</li>
              <li>30-year fixed mortgage at {interestRate}%</li>
              <li>Annual appreciation: {forecast ? forecast.annualGrowthRate.toFixed(1) : '3.0'}%</li>
              <li>Rent growth matches appreciation rate</li>
              <li>Operating expenses: 2.5% of property value annually</li>
            </ul>
          </div>

          <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border border-[#A3A3A3]`}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-2">Cash-on-Cash Return (Annual)</h4>
            <p className="text-sm text-[#A3A3A3] mb-2">Assumes:</p>
            <ul className="list-disc list-inside text-sm text-[#A3A3A3] space-y-1">
              <li>20% down payment</li>
              <li>30-year fixed mortgage at {interestRate}%</li>
              <li>Monthly rent: 0.8% of property value</li>
              <li>Operating expenses: 2.5% of property value annually</li>
              <li>No appreciation factored into calculation</li>
            </ul>
          </div>

          <div className={`${darkThemeStyles.card} rounded-lg p-4 shadow-sm border border-[#A3A3A3]`}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-2">12-Month Price Forecast</h4>
            <p className="text-sm text-[#A3A3A3] mb-2">Based on:</p>
            <ul className="list-disc list-inside text-sm text-[#A3A3A3] space-y-1">
              <li>Market momentum: {market_metrics?.market_momentum ? (market_metrics.market_momentum * 100).toFixed(1) : 'N/A'}%</li>
              <li>Market health: {market_metrics?.market_health ? (market_metrics.market_health * 100).toFixed(1) : 'N/A'}%</li>
              <li>Market stability: {market_metrics?.market_stability ? (market_metrics.market_stability * 100).toFixed(1) : 'N/A'}%</li>
              <li>YOY price trends: {property.market_trends && typeof property.market_trends === 'object' && 'market_analysis' in property.market_trends && property.market_trends.market_analysis?.price_trends?.yoy_change ? property.market_trends.market_analysis.price_trends.yoy_change.toFixed(1) : 'N/A'}%</li>
            </ul>
          </div>

          <div className={`${darkThemeStyles.card} p-4 rounded`}>
            <h4 className="text-lg font-semibold text-[#E4E4E7] mb-2">Why Different Metrics Show Different Results</h4>
            <p className="text-sm text-[#E4E4E7]">
              The metrics appear contradictory because they measure different aspects of the investment:
            </p>
            <ul className="list-disc list-inside text-sm text-[#E4E4E7] mt-2 space-y-1">
              <li>IRR includes both cash flow and appreciation over 5 years</li>
              <li>Cash-on-cash focuses only on annual cash flow relative to down payment</li>
              <li>Price forecast reflects short-term market conditions and trends</li>
            </ul>
          </div>
        </div>
      </div>
    );
  };

  // Update the LiquidityExitCard component to show better data
  const LiquidityExitCard: React.FC<{ domRatio: number; daysOnMarket: number; marketHealth: any; volatility: any }> = ({ domRatio, daysOnMarket, marketHealth, volatility }) => {
    // DOM ratio: property DOM / market median DOM
    // Lower is better (faster sale)
    const domRatioDisplay = domRatio > 0 ? `${domRatio.toFixed(2)}x` : 'N/A';
    
    const daysDisplay = daysOnMarket > 0 ? `${daysOnMarket} days` : 'N/A';

    const getDomColor5Category = (days: number) => {
      if (days < 20) return 'text-[#10B981]'; // Very Fast Sale
      if (days < 40) return 'text-[#34D399]'; // Fast Sale
      if (days < 70) return 'text-[#3B82F6]'; // Average Sale
      if (days < 100) return 'text-[#FBBF24]'; // Slow Sale
      return 'text-[#EF4444]'; // Very Slow Sale
    };

    const getDomRatioColor5Category = (ratio: number) => {
      if (ratio < 0.7) return 'text-[#10B981]'; // Very Fast (Much lower than market)
      if (ratio < 0.9) return 'text-[#34D399]'; // Fast (Lower than market)
      if (ratio < 1.1) return 'text-[#3B82F6]'; // Average (Near market)
      if (ratio < 1.3) return 'text-[#FBBF24]'; // Slow (Higher than market)
      return 'text-[#EF4444]'; // Very Slow (Much higher than market)
    };

    const getMarketLiquidityColor5Category = (turnover: number) => {
      if (turnover > 2.0) return 'text-[#10B981]'; // Very High Liquidity
      if (turnover > 1.5) return 'text-[#34D399]'; // High Liquidity
      if (turnover > 1.0) return 'text-[#3B82F6]'; // Moderate Liquidity
      if (turnover > 0.5) return 'text-[#FBBF24]'; // Low Liquidity
      return 'text-[#EF4444]'; // Very Low Liquidity
    };

    const getDomBadgeInfo = (ratio: number) => {
      if (ratio < 0.7) return { label: 'Very Fast Sale', color: 'bg-[#10B981] text-white' };
      if (ratio < 0.9) return { label: 'Fast Sale', color: 'bg-[#34D399] text-white' };
      if (ratio < 1.1) return { label: 'Average Sale', color: 'bg-[#3B82F6] text-white' };
      if (ratio < 1.3) return { label: 'Slow Sale', color: 'bg-[#FBBF24] text-white' };
      return { label: 'Very Slow Sale', color: 'bg-[#EF4444] text-white' };
    };

    const { label: domLabel, color: domBadge } = getDomBadgeInfo(domRatio);

    return (
      <InsightCard
        title="Liquidity & Exit Outlook"
        icon={<ClockIcon className="h-6 w-6 text-[#93C5FD]" />}
        insights={[
          `Properties sell in ${daysDisplay}, with a DOM ratio of ${domRatioDisplay} compared to market median`,
          `Market liquidity is ${marketHealth?.inventory_turnover || 'N/A'}x, indicating ${marketHealth?.market_balance || 'N/A'} conditions`
        ]}
        metrics={[
          { label: 'Days on Market', value: daysDisplay, color: getDomColor5Category(daysOnMarket) },
          { label: 'DOM Ratio', value: domRatioDisplay, color: getDomRatioColor5Category(domRatio) },
          { label: 'Market Liquidity', value: marketHealth?.inventory_turnover || 'N/A', color: getMarketLiquidityColor5Category(marketHealth?.inventory_turnover || 0) }
        ]}
        badges={[{ label: domLabel, color: domBadge }]}
      />
    );
  };

  // --- Unified Analysis Panel ---
  return (
    <div className="bg-[#121212] border border-[#2A2A2A] rounded-lg p-6 space-y-8 text-[#FFFFFF]">
      <h2 className="text-2xl font-semibold text-[#E4E4E7] mb-4">Investment Analysis</h2>
      {/* Only Full Analysis remains, no tab buttons */}
      {/* Unified AI/LLM analysis */}
      {renderAIAnalysis()}
      {/* All other insights remain below, always visible */}
      {renderEnhancedCashflowAnalysis()}
      {renderForecast()}
      {renderLongTermIRRProjection()}
      {renderHoldTimeRecommendation()}
      {renderFeatureImportance()}
      {renderRiskMetrics()}
      {renderMarketMetrics()}
      {renderBaseMetrics()}
      {renderInvestmentMetrics()}
      {/* Collapsible Investment Assumptions Panel */}
      <div className="my-6">
        <button
          className="w-full flex items-center justify-between px-6 py-3 bg-[#1A1A1A] border border-[#2A2A2A] shadow-sm hover:shadow-md transition-all focus:outline-none"
          onClick={() => setShowAssumptions(!showAssumptions)}
        >
          <span className="text-lg font-bold text-[#E4E4E7]">Investment Assumptions (Edit)</span>
          <svg className={`w-6 h-6 transform transition-transform ${showAssumptions ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showAssumptions && (
          <div className="mt-4">
            {renderInvestmentAssumptions()}
          </div>
        )}
      </div>
    </div>
  );
};

export default MLAnalysis; 
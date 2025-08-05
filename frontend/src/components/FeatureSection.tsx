import React from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon,
  LightBulbIcon,
  MagnifyingGlassIcon,
  SquaresPlusIcon,
  ScaleIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';
import PropertySearchDemo from '../components/features/PropertySearchDemo';
import InvestmentScoreDial from '../components/features/InvestmentScoreDial';

// Features based on actual implemented functionality from codebase
const features = [
  {
    title: 'Interactive Property Search',
    description: 'Filter and visualize properties on an interactive Mapbox map. Search by price, beds, baths, and location with real-time updates.',
    icon: MagnifyingGlassIcon,
    visual: <PropertySearchDemo />,
  },
  {
    title: 'Investment Score Analysis',
    description: 'Proprietary 0-1 scoring system combining cap rate, cash-on-cash return, ROI, DSCR, and rental yield metrics for instant property evaluation.',
    icon: ScaleIcon,
    visual: <InvestmentScoreDial score={85} />,
  },
  {
    title: 'Market Intelligence',
    description: 'Track price trends, inventory levels, and market momentum with live data from Realtor.com. View historical trends and price forecasts.',
    icon: ChartBarIcon,
    placeholder: 'Market trend charts and metrics',
  },
  {
    title: 'Risk Assessment',
    description: 'Comprehensive risk analysis covering market risk, property condition risk, and location risk factors with detailed scoring breakdowns.',
    icon: ArrowTrendingUpIcon,
    placeholder: 'Risk metrics visualization',
  },
  {
    title: 'AI Property Insights',
    description: 'GPT-powered analysis of property features, market conditions, and investment potential with actionable recommendations.',
    icon: LightBulbIcon,
    placeholder: 'AI analysis summary card',
  },
  {
    title: 'Comparable Analysis',
    description: 'Side-by-side comparison of similar properties with automatic adjustments for square footage, amenities, and location factors.',
    icon: SquaresPlusIcon,
    placeholder: 'Comparable properties grid',
  },
];

interface FeatureSectionProps {
  index: number;
  feature: typeof features[0];
}

const FeatureSection: React.FC<FeatureSectionProps> = ({ index, feature }) => {
  const isEven = index % 2 === 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="py-24 first:pt-0 last:pb-0"
    >
      <div className={`max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid lg:grid-cols-2 gap-12 items-center ${isEven ? '' : 'lg:grid-flow-dense'}`}>
        <div className={isEven ? 'lg:pr-8' : 'lg:pl-8 lg:col-start-2'}>
          <div className="flex items-center gap-4 mb-6">
            <div className="flex items-center justify-center h-12 w-12 rounded-xl bg-blue-500/10 text-blue-400">
              <feature.icon className="h-6 w-6" />
            </div>
            <h3 className="text-2xl font-semibold bg-gradient-to-r from-blue-400 to-indigo-400 text-transparent bg-clip-text">{feature.title}</h3>
          </div>
          <p className="text-lg text-gray-300">{feature.description}</p>
        </div>

        <div className={`${isEven ? '' : 'lg:col-start-1'} flex items-center justify-center`}>
          {feature.visual || (
            <div className="aspect-video w-full bg-[#18181B]/50 rounded-xl border border-[#27272A]/50 flex items-center justify-center text-gray-500">
              {feature.placeholder}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export { features };
export default FeatureSection; 
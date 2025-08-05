import React from 'react';
import { Link } from 'react-router-dom';
import {
  ChartBarIcon,
  LightBulbIcon,
  MagnifyingGlassIcon,
  SquaresPlusIcon,
  CheckCircleIcon,
  ScaleIcon,
} from '@heroicons/react/24/outline';
import CityBackground from '../components/CityBackground';
import AnimatedHeading from '../components/AnimatedHeading';
import FeatureSection, { features } from '../components/FeatureSection';

const Home: React.FC = () => {
  return (
    <>
      {/* Hero Section */}
      <div className="relative min-h-[90vh] flex items-center">
        <CityBackground />
        
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32">
          <div className="max-w-xl">
            <AnimatedHeading />
            
            <p className="mt-6 text-lg text-gray-300">
              Purpose-built tools for serious real estate analysis. Every feature is grounded in real data. No noise, just speed, clarity, and control.
            </p>
            
            <div className="mt-8 flex flex-col sm:flex-row gap-4">
              <Link
                to="/search"
                className="inline-flex items-center gap-2 justify-center px-6 py-3 rounded-lg text-white bg-blue-500 hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0A0A0B] focus:ring-blue-500 transition-all duration-200 hover:shadow-lg hover:shadow-blue-500/20"
              >
                <MagnifyingGlassIcon className="w-5 h-5" />
                Start Searching
              </Link>
              <Link
                to="/market-analysis"
                className="inline-flex items-center gap-2 justify-center px-6 py-3 rounded-lg bg-[#27272A] text-gray-300 hover:bg-[#3F3F46] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0A0A0B] focus:ring-blue-500 transition-all duration-200"
              >
                <ChartBarIcon className="w-5 h-5" />
                Market Dashboard
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="bg-[#0A0A0B]">
        <div className="py-24">
          <div className="text-center max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 mb-24">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 text-transparent bg-clip-text">
              Built for Investors Who Value Data Over Hype
            </h2>
            <p className="mt-4 text-lg text-gray-300">
              Every feature is rooted in real underwriting workflows. No fluff, just the tools you wish MLS websites shipped with.
            </p>
          </div>

          {features.map((feature, index) => (
            <FeatureSection 
              key={feature.title} 
              feature={feature} 
              index={index} 
            />
          ))}
        </div>
      </div>
    </>
  );
};

export default Home; 
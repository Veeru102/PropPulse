import React from 'react';
import { Link } from 'react-router-dom';
import { HomeIcon, MagnifyingGlassIcon, ChartBarIcon, ChartBarSquareIcon } from '@heroicons/react/24/outline';

const Navbar: React.FC = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-[#18181B]/80 backdrop-blur-sm border-b border-[#27272A]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 text-transparent bg-clip-text">
                PropPulse
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link
                to="/"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-white"
              >
                <HomeIcon className="h-5 w-5 mr-1" />
                Home
              </Link>
              <Link
                to="/search"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-400 hover:text-white transition-colors duration-200"
              >
                <MagnifyingGlassIcon className="h-5 w-5 mr-1" />
                Search
              </Link>
              <Link
                to="/market-analysis"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-400 hover:text-white transition-colors duration-200"
              >
                <ChartBarIcon className="h-5 w-5 mr-1" />
                Market Analysis
              </Link>
              <Link
                to="/comparative-analysis"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-400 hover:text-white transition-colors duration-200"
              >
                <ChartBarSquareIcon className="h-5 w-5 mr-1" />
                Compare Markets
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 
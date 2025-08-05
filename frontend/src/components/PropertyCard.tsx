import React from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowsRightLeftIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  MapPinIcon,
  HomeIcon,
} from '@heroicons/react/24/outline';
import { Tooltip } from '@mui/material';

interface Property {
  property_id: string;
  address: string;
  city: string;
  state: string;
  price: number;
  beds: number;
  baths: number;
  square_feet: number;
  property_type: string;
  investment_score: number;
  investment_score_explanation?: string;
  rental_yield?: number;
  flip_roi?: number;
  photos?: string[];
}

interface PropertyCardProps {
  property: Property;
}


const getInvestmentScoreColor = (score: number) => {
  if (score >= 0.8) return '#7C3AED'; // Excellent (Purple)
  if (score >= 0.6) return '#87CEEB'; // Good (Powder Blue)
  if (score >= 0.4) return '#3BA3DD'; // Average (Blue)
  if (score >= 0.2) return '#F59E0B'; // Below Average (Yellow)
  return '#EF4444'; // Poor (RED)
};




const getInvestmentScoreLabel = (score: number) => {
  if (score >= 0.8) return 'Excellent';
  if (score >= 0.6) return 'Good';
  if (score >= 0.4) return 'Average';
  if (score >= 0.2) return 'Below Avg';
  return 'Poor';
};

const PropertyCard: React.FC<PropertyCardProps> = ({ property }) => {
  const score = property.investment_score ?? 0;
  const scoreColor = getInvestmentScoreColor(score);
  const scoreLabel = getInvestmentScoreLabel(score);
  return (
    <Link
      to={`/property/${property.property_id}`}
      className="block rounded-xl shadow hover:shadow-lg transition-shadow duration-200 border border-[#232336]"
      style={{ background: 'linear-gradient(135deg, #232336 0%, #18181B 100%)' }}
    >
      <div className="relative">
        {property.photos && property.photos.length > 0 ? (
          <img
            src={property.photos[0]}
            alt={property.address}
            className="w-full h-48 object-cover rounded-t-xl"
          />
        ) : (
          <div className="w-full h-48 bg-[#232336] rounded-t-xl flex items-center justify-center">
            <HomeIcon className="h-12 w-12 text-gray-500" />
          </div>
        )}
        <div className="absolute top-2 right-2 bg-[#18181B] px-3 py-1 rounded-full text-sm font-semibold border border-[#232336] text-white shadow">
          ${property.price.toLocaleString()}
        </div>
      </div>

      <div className="p-4">
        <h3 className="text-lg font-bold text-white truncate">{property.address}</h3>
        <p className="text-sm text-gray-300">{property.city}, {property.state}</p>

        <div className="mt-2 flex items-center space-x-4 text-sm text-gray-400">
          <div className="flex items-center">
            <HomeIcon className="h-4 w-4 mr-1" />
            {property.beds} beds, {property.baths} baths
          </div>
          <div className="flex items-center">
            <CurrencyDollarIcon className="h-4 w-4 mr-1" />
            {property.square_feet.toLocaleString()} sqft
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-[#232336]">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-300">Investment Score:</span>
              <Tooltip 
                title={
                  <div style={{ whiteSpace: 'pre-line' }}>
                    {property.investment_score_explanation || 'Score explanation not available'}
                  </div>
                }
                arrow
                placement="top"
              >
                <span className="text-lg font-bold" style={{ color: scoreColor }}>
                  {property.investment_score?.toFixed(3) || 'N/A'}
                </span>
              </Tooltip>
              <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-semibold" style={{ background: scoreColor, color: '#18181B' }}>
                {scoreLabel}
              </span>
            </div>
            {property.rental_yield && (
              <div className="text-sm text-gray-400">
                Rental Yield: {property.rental_yield.toFixed(1)}%
              </div>
            )}
          </div>
          {property.flip_roi && (
            <div className="mt-1 text-sm text-gray-400">
              Flip ROI: {property.flip_roi.toFixed(1)}%
            </div>
          )}
        </div>
      </div>
    </Link>
  );
};

export default PropertyCard; 
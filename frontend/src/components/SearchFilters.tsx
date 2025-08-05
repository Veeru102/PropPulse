import React from 'react';

interface SearchFiltersProps {
  filters: {
    city: string;
    state_code: string;
    zipCode: string;
    minPrice: string;
    maxPrice: string;
    beds: string;
    baths: string;
    propertyType: string;
  };
  onFilterChange: (filters: any) => void;
  onSearch: () => void;
}

const SearchFilters: React.FC<SearchFiltersProps> = ({ filters, onFilterChange, onSearch }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    onFilterChange({ ...filters, [name]: value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch();
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 rounded-lg shadow" style={{ background: '#0D0D0D' }}>
      <h2 className="text-lg font-semibold mb-4 text-white">Search Filters</h2>
      
      <div className="space-y-4">
        {/* Location */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="city" className="block text-sm font-medium text-gray-300">
              City
            </label>
            <input
              type="text"
              name="city"
              id="city"
              value={filters.city}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white placeholder:text-gray-400" style={{ background: '#222222' }}
              placeholder="Enter city"
            />
          </div>
          <div>
            <label htmlFor="state_code" className="block text-sm font-medium text-gray-300">
              State
            </label>
            <input
              type="text"
              name="state_code"
              id="state_code"
              value={filters.state_code}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white placeholder:text-gray-400" style={{ background: '#222222' }}
              placeholder="Enter state code (e.g., CA)"
              maxLength={2}
            />
          </div>
        </div>

        <div>
          <label htmlFor="zipCode" className="block text-sm font-medium text-gray-300">
            ZIP Code
          </label>
          <input
            type="text"
            name="zipCode"
            id="zipCode"
            value={filters.zipCode}
            onChange={handleChange}
            className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white placeholder:text-gray-400" style={{ background: '#222222' }}
            placeholder="Enter ZIP code"
          />
        </div>

        {/* Price Range */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="minPrice" className="block text-sm font-medium text-gray-300">
              Min Price
            </label>
            <input
              type="number"
              name="minPrice"
              id="minPrice"
              value={filters.minPrice}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white placeholder:text-gray-400" style={{ background: '#222222' }}
              placeholder="Min price"
            />
          </div>
          <div>
            <label htmlFor="maxPrice" className="block text-sm font-medium text-gray-300">
              Max Price
            </label>
            <input
              type="number"
              name="maxPrice"
              id="maxPrice"
              value={filters.maxPrice}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white placeholder:text-gray-400" style={{ background: '#222222' }}
              placeholder="Max price"
            />
          </div>
        </div>

        {/* Property Details */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="beds" className="block text-sm font-medium text-gray-300">
              Beds
            </label>
            <select
              name="beds"
              id="beds"
              value={filters.beds}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white" style={{ background: '#222222' }}
            >
              <option value="" className="bg-[#222222]">Any</option>
              <option value="1" className="bg-[#222222]">1+</option>
              <option value="2" className="bg-[#222222]">2+</option>
              <option value="3" className="bg-[#222222]">3+</option>
              <option value="4" className="bg-[#222222]">4+</option>
              <option value="5" className="bg-[#222222]">5+</option>
            </select>
          </div>
          <div>
            <label htmlFor="baths" className="block text-sm font-medium text-gray-300">
              Baths
            </label>
            <select
              name="baths"
              id="baths"
              value={filters.baths}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white" style={{ background: '#222222' }}
            >
              <option value="" className="bg-[#222222]">Any</option>
              <option value="1" className="bg-[#222222]">1+</option>
              <option value="2" className="bg-[#222222]">2+</option>
              <option value="3" className="bg-[#222222]">3+</option>
              <option value="4" className="bg-[#222222]">4+</option>
            </select>
          </div>
        </div>

        {/* Property Type */}
        <div>
          <label htmlFor="propertyType" className="block text-sm font-medium text-gray-300">
            Property Type
          </label>
          <select
            name="propertyType"
            id="propertyType"
            value={filters.propertyType}
            onChange={handleChange}
            className="mt-1 block w-full rounded-md border-[#232336] shadow-sm focus:border-[#87CEEB] focus:ring-[#87CEEB] sm:text-sm text-white" style={{ background: '#222222' }}
          >
            <option value="" className="bg-[#222222]">Any</option>
            <option value="single_family" className="bg-[#222222]">Single Family</option>
            <option value="multi_family" className="bg-[#222222]">Multi Family</option>
            <option value="condo" className="bg-[#222222]">Condo</option>
            <option value="townhouse" className="bg-[#222222]">Townhouse</option>
          </select>
        </div>

        {/* Search Button */}
        <div className="pt-4">
          <button
            type="submit"
            className="w-full bg-[#87CEEB] text-white py-2 px-4 rounded-md hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-[#87CEEB] focus:ring-offset-2"
          >
            Search Properties
          </button>
        </div>
      </div>
    </form>
  );
};

export default SearchFilters; 
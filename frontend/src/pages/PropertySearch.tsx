import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import mapboxgl from 'mapbox-gl';
import MapboxGeocoder from '@mapbox/mapbox-gl-geocoder';
import 'mapbox-gl/dist/mapbox-gl.css';
import '@mapbox/mapbox-gl-geocoder/dist/mapbox-gl-geocoder.css';
import PropertyCard from '../components/PropertyCard';
import SearchFilters from '../components/SearchFilters';
import {
  searchProperties,
  getPropertyDetails,
  Property,
} from '../services/api';

// Initialize Mapbox
const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN || '';
if (!MAPBOX_TOKEN) {
  console.error('Mapbox token is not set in environment variables');
} else {
  mapboxgl.accessToken = MAPBOX_TOKEN;
}

const PropertySearch: React.FC = () => {
  const [filters, setFilters] = useState({
    city: '',
    state_code: '',
    zipCode: '',
    minPrice: '',
    maxPrice: '',
    beds: '',
    baths: '',
    propertyType: '',
  });

  const [searchParams, setSearchParams] = useState<typeof filters | null>(null);
  const [map, setMap] = useState<mapboxgl.Map | null>(null);
  const [markers, setMarkers] = useState<mapboxgl.Marker[]>([]);
  const [mapError, setMapError] = useState<string | null>(null);
  const mapContainerRef = React.useRef<HTMLDivElement>(null);

  // State for properties with details and loading state
  const [detailedProperties, setDetailedProperties] = useState<Property[]>([]);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);

  // Handle filter changes
  const handleFilterChange = (newFilters: typeof filters) => {
    setFilters(newFilters);
  };

  // Handle search submission
  const handleSearch = (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault();
    }
    // Set reasonable defaults for price range if not specified
    const searchParams = {
      ...filters,
      minPrice: filters.minPrice || '0',
      maxPrice: filters.maxPrice || '1000000', // Default to 1 million if not specified
      beds: filters.beds || '1',
      baths: filters.baths || '1'
    };
    setSearchParams(searchParams);
  };

  // Fetch properties based on search parameters
  const { data: properties = [], isLoading, error } = useQuery<Property[]>({
    queryKey: ['properties', searchParams],
    queryFn: () => {
      if (!searchParams) return Promise.resolve([]);
      setDetailedProperties([]); // Clear previous detailed properties
      return searchProperties(searchParams);
    },
    enabled: !!searchParams,
    retry: 1,
    retryDelay: 1000
  });

  // Remove the useEffect that fetches details for each property since we now get photos in the initial response
  useEffect(() => {
    if (!properties || properties.length === 0) {
      setDetailedProperties([]);
      setIsLoadingDetails(false);
      return;
    }

    // Just set the properties directly since they now include photos
    setDetailedProperties(properties.map(property => ({
      ...property,
      photos: property.photos || [] // Ensure photos is always an array
    })));
    setIsLoadingDetails(false);
  }, [properties]);

  // Initialize map
  React.useEffect(() => {
    if (!mapContainerRef.current || !MAPBOX_TOKEN) return;

    try {
      const newMap = new mapboxgl.Map({
        container: mapContainerRef.current,
        style: 'mapbox://styles/mapbox/light-v10',
        center: [-98.5795, 39.8283], // Center of US
        zoom: 3,
      });

      // Add navigation controls
      newMap.addControl(new mapboxgl.NavigationControl());

      // Add geocoder with error handling
      try {
        const geocoder = new MapboxGeocoder({
          accessToken: MAPBOX_TOKEN,
          mapboxgl: mapboxgl,
          placeholder: 'Search location...',
          marker: false, // Disable default marker
          countries: 'us', // Restrict to US addresses
          types: 'address,place,locality,neighborhood,postcode', // Restrict to these types
          limit: 5, // Limit results
        });

        geocoder.on('error', (e) => {
          console.error('Geocoder error:', e.error?.message || 'Unknown error');
          setMapError('Error with location search. Please try again.');
        });

        geocoder.on('result', (e) => {
          // Clear existing markers
          markers.forEach(marker => marker.remove());
          setMarkers([]);
          setMapError(null);
        });

        newMap.addControl(geocoder);
      } catch (error) {
        console.error('Error initializing geocoder:', error);
        setMapError('Error initializing location search. Please refresh the page.');
      }

      // Wait for map to load before setting it
      newMap.on('load', () => {
        setMap(newMap);
        setMapError(null);
      });

      newMap.on('error', (e) => {
        console.error('Mapbox error:', e.error?.message || 'Unknown error');
        setMapError('Error loading map. Please refresh the page.');
      });

      return () => {
        newMap.remove();
      };
    } catch (error) {
      console.error('Error initializing map:', error);
      setMapError('Error initializing map. Please refresh the page.');
    }
  }, []); // Only run once on mount

  // Update markers when detailedProperties change
  React.useEffect(() => {
    if (!map || !detailedProperties) return;

    // Clear existing markers
    markers.forEach(marker => marker.remove());
    const newMarkers: mapboxgl.Marker[] = [];

    // Add markers for each property
    detailedProperties.forEach(property => {
      if (!property.longitude || !property.latitude) return;

      try {
        const marker = new mapboxgl.Marker()
          .setLngLat([property.longitude, property.latitude])
          .setPopup(
            new mapboxgl.Popup({ offset: 25 }).setHTML(`
              <h3 class="font-bold">${property.address}</h3>
              <p>$${property.price.toLocaleString()}</p>
              <p>${property.beds} beds, ${property.baths} baths</p>
            `)
          )
          .addTo(map);

        newMarkers.push(marker);
      } catch (error) {
        console.error('Error adding marker for property:', property.property_id, error);
      }
    });

    setMarkers(newMarkers);

    // Fit map to markers if we have any
    if (newMarkers.length > 0) {
      try {
        const bounds = new mapboxgl.LngLatBounds();
        detailedProperties.forEach(property => {
          if (property.longitude && property.latitude) {
            bounds.extend([property.longitude, property.latitude]);
          }
        });
        map.fitBounds(bounds, { padding: 50 });
      } catch (error) {
        console.error('Error fitting map bounds:', error);
      }
    }
  }, [map, detailedProperties]); // Depend on detailedProperties

  return (
    <div className="flex h-[calc(100vh-4rem)]" style={{ background: '#0D0D0D' }}>
      {/* Search filters and results */}
      <div className="w-1/3 overflow-y-auto p-4">
        <SearchFilters 
          filters={filters} 
          onFilterChange={handleFilterChange} 
          onSearch={handleSearch}
        />
        
        {isLoading || isLoadingDetails ? (
          <div className="text-center py-4 text-gray-300">Loading properties...</div>
        ) : error ? (
          <div className="text-center py-4">
            <div className="rounded-md p-4" style={{ background: '#222222', border: '1px solid #4F627A' }}>
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-300">Error loading properties</h3>
                  <div className="mt-2 text-sm text-red-200">
                    {error instanceof Error ? error.message : 'An error occurred while fetching properties'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : !searchParams ? (
          <div className="text-center py-4">
            <div className="rounded-md p-4" style={{ background: '#222222', border: '1px solid #4F627A' }}>
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-blue-300">Enter search criteria</h3>
                  <div className="mt-2 text-sm text-blue-200">
                    Use the filters above to search for properties
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : detailedProperties?.length === 0 ? (
          <div className="text-center py-4">
            <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-yellow-800">No properties found</h3>
                  <div className="mt-2 text-sm text-yellow-700">
                    <p>Try adjusting your search criteria:</p>
                    <ul className="mt-2 list-disc list-inside">
                      <li>Make sure the city and state are correct</li>
                      <li>Try a broader price range</li>
                      <li>Remove some filters to see more results</li>
                      <li>Check if the ZIP code is valid</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {detailedProperties?.map(property => (
              <PropertyCard key={property.property_id} property={property} />
            ))}
          </div>
        )}
      </div>

      {/* Map */}
      <div className="w-2/3">
        <div ref={mapContainerRef} className="h-full" />
        {mapError && (
          <div className="absolute top-4 right-4 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Map Error</h3>
                <div className="mt-2 text-sm text-red-700">{mapError}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PropertySearch; 
import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import mapboxgl from 'mapbox-gl';
import { darkThemeStyles } from '../styles/darkTheme';
import {
  getPropertyDetails,
  analyzeProperty,
  getComparableProperties,
  getPropertyAnalysis,
  getPropertyPrediction,
  getPropertyRiskAssessment,
  Property,
} from '../services/api';
import MLAnalysis from '../components/MLAnalysis';

// Initialize Mapbox
const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN || '';
if (!MAPBOX_TOKEN) {
  console.error('Mapbox token is not set in environment variables');
} else {
  mapboxgl.accessToken = MAPBOX_TOKEN;
}

const PropertyDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [analysisType, setAnalysisType] = useState<'full' | 'rental' | 'flip'>('full');
  const [mapError, setMapError] = useState<string | null>(null);

  // Fetch property details
  const { data: property, isLoading: isLoadingProperty, error: propertyError } = useQuery<Property>({
    queryKey: ['property', id],
    queryFn: () => getPropertyDetails(id!),
    retry: 1,
    retryDelay: 1000
  });

  // Fetch property analysis
  const { data: analysis, isLoading: isLoadingAnalysis } = useQuery<Property>({
    queryKey: ['analysis', id, analysisType],
    queryFn: () => analyzeProperty(id!, analysisType),
    enabled: !!id,
  });

  // Fetch comparable properties
  const { data: comps, isLoading: isLoadingComps } = useQuery<Property[]>({
    queryKey: ['comps', id],
    queryFn: () => getComparableProperties(id!),
    enabled: !!id,
  });

  // Fetch ML/AI analysis
  const { data: mlAnalysis, isLoading: isLoadingMLAnalysis } = useQuery<Property>({
    queryKey: ['ml-analysis', id],
    queryFn: () => getPropertyAnalysis(id!),
    enabled: !!id,
  });

  // Fetch ML/AI prediction
  const { data: mlPrediction, isLoading: isLoadingMLPrediction } = useQuery<Property>({
    queryKey: ['ml-prediction', id],
    queryFn: () => getPropertyPrediction(id!),
    enabled: !!id,
  });

  // Fetch ML/AI risk assessment
  const { data: mlRisk, isLoading: isLoadingMLRisk } = useQuery<Property>({
    queryKey: ['ml-risk', id],
    queryFn: () => getPropertyRiskAssessment(id!),
    enabled: !!id,
  });

  // Add helper to compute label and percentile inside the component, just after hooks declarations
  const getInvestmentLabel = (score: number): string => {
    if (score >= 0.8) return 'Strong';
    if (score >= 0.6) return 'Moderate';
    return 'Low';
  };

  // Initialize map
  React.useEffect(() => {
    if (property && MAPBOX_TOKEN) {
      try {
        const map = new mapboxgl.Map({
          container: 'map',
          style: 'mapbox://styles/mapbox/light-v10',
          center: [property.longitude, property.latitude],
          zoom: 14,
        });

        // Add marker for the property
        new mapboxgl.Marker()
          .setLngLat([property.longitude, property.latitude])
          .setPopup(
            new mapboxgl.Popup({ offset: 25 }).setHTML(`
              <h3 class="font-bold">${property.address}</h3>
              <p>$${property.price.toLocaleString()}</p>
              <p>${property.beds} beds, ${property.baths} baths</p>
            `)
          )
          .addTo(map);

        map.on('load', () => {
          setMapError(null);
        });

        map.on('error', (e) => {
          console.error('Mapbox error:', e.error?.message || 'Unknown error');
          setMapError('Error loading map. Please refresh the page.');
        });

        return () => {
          map.remove();
        };
      } catch (error) {
        console.error('Error initializing map:', error);
        setMapError('Error initializing map. Please refresh the page.');
      }
    }
  }, [property]);

  if (isLoadingProperty) {
    return <div className="text-center py-4 text-[#A3A3A3]">Loading property details...</div>;
  }

  if (propertyError) {
    return (
      <div className="text-center py-4">
        <div className="rounded-md p-4" style={{ background: '#121212', border: '1px solid #2A2A2A' }}>
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-[#EF4444]" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-[#E4E4E7]">Error loading property details</h3>
              <div className="mt-2 text-sm text-[#A3A3A3]">
                {propertyError instanceof Error ? propertyError.message : 'An error occurred while fetching property details'}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!property) {
    return (
      <div className="text-center py-4">
        <div className="rounded-md p-4" style={{ background: '#121212', border: '1px solid #2A2A2A' }}>
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-[#60A5FA]" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-[#E4E4E7]">Property not found</h3>
              <div className="mt-2 text-sm text-[#A3A3A3]">
                The property you're looking for could not be found. Please try searching for another property.
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#000000] text-[#FFFFFF] p-6">
      {/* Property Header */}
      <div className="rounded-xl shadow-lg border border-[#000000]" style={{ background: '#000000' }}>
        <div className="relative h-96 sm:h-80 md:h-96 lg:h-[450px]">
          {property.photos && property.photos.length > 0 && property.photos[0] ? (
            <img
              src={property.photos[0]}
              alt={property.address}
              className="w-full h-full object-cover rounded-t-xl"
            />
          ) : comps && comps.length > 0 && comps[0].photos && comps[0].photos.length > 0 && comps[0].photos[0] ? (
            <img
              src={comps[0].photos[0]}
              alt={`Comparable property for ${property.address}`}
              className="w-full h-full object-cover rounded-t-xl"
            />
          ) : (
            <div className="w-full h-full bg-[#1A1A1A] rounded-t-xl flex items-center justify-center">
              <span className="text-[#A3A3A3]">No image available</span>
            </div>
          )}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#0D0D0D] to-transparent p-6">
            <h1 className="text-3xl font-bold text-[#E4E4E7]">{property.address}</h1>
            <p className="text-[#A3A3A3]">
              {property.city}, {property.state}
            </p>
          </div>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Property Details */}
            <div className="md:col-span-2 lg:col-span-3 p-6 rounded-xl border border-[#2A2A2A]" style={{ background: '#121212' }}>
              <h2 className="text-2xl font-bold text-[#E4E4E7] mb-4">Property Details</h2>
              <dl className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <dt className="text-sm font-medium text-[#FFFFFF]">Price</dt>
                  <dd className="text-lg font-semibold text-[#FFFFFF]">
                    ${property.price.toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[#A3A3A3]">Property Type</dt>
                  <dd className="text-lg font-semibold text-[#FFFFFF]">{property.property_type}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[#A3A3A3]">Beds</dt>
                  <dd className="text-lg font-semibold text-[#FFFFFF]">{property.beds}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[#A3A3A3]">Baths</dt>
                  <dd className="text-lg font-semibold text-[#FFFFFF]">{property.baths}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[#A3A3A3]">Square Feet</dt>
                  <dd className="text-lg font-semibold text-[#FFFFFF]">
                    {property.square_feet.toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[#A3A3A3]">Investment Score</dt>
                  <dd className="text-lg font-semibold text-[#FFFFFF]">
                    {property.investment_score.toFixed(3)}
                  </dd>
                </div>
              </dl>
            </div>
          </div>

          {/* ML/AI Analysis Section */}
          {(isLoadingMLAnalysis || isLoadingMLPrediction || isLoadingMLRisk) ? (
            <div className="mt-6 text-center py-4 text-[#FFFFFF]">Loading ML/AI analysis...</div>
          ) : (mlAnalysis || mlPrediction || mlRisk) ? (
            <div className="mt-6">
              <MLAnalysis
                property={{
                  ...property!,
                  ...mlAnalysis,
                  ...mlPrediction,
                  ...mlRisk,
                }}
              />
            </div>
          ) : null}
        </div>
      </div>

      {/* Map */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-[#E4E4E7] mb-4">Location</h2>
        <div id="map" className="rounded-xl shadow-lg border border-[#2A2A2A] h-96" />
      </div>

      {/* Comparable Properties */}
      {!isLoadingComps && comps && comps.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold text-[#E4E4E7] mb-4">Comparable Properties</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
            {comps.map(comp => (
              <div
                key={comp.property_id}
                className="rounded-xl shadow-lg border border-[#2A2A2A]"
                style={{ background: '#121212' }}
              >
                {comp.photos && comp.photos.length > 0 && (
                  <img
                    src={comp.photos[0]}
                    alt={comp.address}
                    className="w-full h-48 object-cover rounded-t-xl"
                  />
                )}
                <div className="p-4">
                  <h3 className="text-lg font-semibold text-[#E4E4E7]">{comp.address}</h3>
                  <p className="text-[#A3A3A3]">
                    {comp.city}, {comp.state}
                  </p>
                  <p className="text-lg font-semibold text-[#3B82F6] mt-2">
                    ${comp.price.toLocaleString()}
                  </p>
                  <p className="text-[#A3A3A3]">
                    {comp.beds} beds, {comp.baths} baths
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PropertyDetails; 
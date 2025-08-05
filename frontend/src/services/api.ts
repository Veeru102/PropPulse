import axios from 'axios';

// Use port 8000 for the API URL
const API_BASE_URL = 'http://localhost:8000/api/v1';

console.log('API Base URL:', API_BASE_URL);

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log('Making API request to:', config.url, 'with params:', config.params);
    return config;
  },
  (error) => {
    console.error('API request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => {
    console.log('API response:', response.data);
    return response;
  },
  (error) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API response error:', error.response.data);
      
      if (error.response.status === 429) {
        throw new Error('Rate limit exceeded. Please try again in a few moments.');
      } else if (error.response.status === 403) {
        throw new Error('Invalid API key or access denied.');
      } else if (error.response.status === 500) {
        throw new Error('Server error. Please try again later.');
      } else {
        throw new Error(error.response.data.detail || 'An error occurred while fetching data.');
      }
    } else if (error.request) {
      // The request was made but no response was received
      console.error('API request error:', error.request);
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('API error:', error.message);
      throw new Error('An unexpected error occurred.');
    }
  }
);

export interface SearchFilters {
  city?: string;
  state_code?: string;
  zipCode?: string;
  minPrice?: string;
  maxPrice?: string;
  beds?: string;
  baths?: string;
  propertyType?: string;
}

export interface Property {
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
  rental_yield?: number;
  flip_roi?: number;
  investment_score_explanation?: string;
  photos: string[];
  analysis?: string;
  recommendations?: string[];
  market_trends?: {
    market_analysis?: {
      price_trends?: {
        yoy_change: number;
        short_term_trend: number;
        medium_term_trend: number;
        long_term_trend: number;
        trend_strength: string;
      };
      market_health?: {
        price_momentum: number;
        inventory_turnover: number;
        market_balance: string;
        overall_health: string;
      };
    };
  };
  risk_factors?: string[];
  longitude: number;
  latitude: number;
  predicted_value?: number;
  value_confidence?: number;
  investment_confidence?: number;
  feature_importance?: {
    value_model: Record<string, number>;
    investment_model: Record<string, number>;
  };
  risk_metrics?: {
    market_risk: number;
    property_risk: number;
    location_risk: number;
    overall_risk: number;
  };
  market_metrics?: {
    market_health: number;
    market_momentum: number;
    market_stability: number;
  };
  base_metrics?: {
    price_per_sqft: number;
    price_to_median: number;
    sqft_per_bed: number;
    beds_baths_ratio: number;
    property_age: number;
    lot_size_per_sqft: number;
  };
  investment_metrics?: {
    cap_rate: number;
    cash_on_cash: number;
    price_to_rent: number;
    dom_ratio: number;
  };
}

export interface MarketTrends {
  error?: string;
  location?: string;
  market_data?: {
    current_metrics?: {
      median_price: number;
      avg_days_on_market: number;
      avg_price_per_sqft: number;
    };
    historical_data?: {
      median_list_price?: {
        dates: string[];
        values: number[];
      };
      median_dom?: {
        dates: string[];
        values: number[];
      };
      price_per_sqft?: {
        dates: string[];
        values: number[];
      };
    };
  };
  market_analysis?: {
    price_trends?: {
      short_term_trend: number;
      medium_term_trend: number;
      long_term_trend: number;
      yoy_change: number;
      trend_strength: string;
    };
    market_health?: {
      price_momentum: number;
      inventory_turnover: number;
      market_balance: string;
      overall_health: string;
    };
    seasonality?: {
      seasonal_pattern: string;
      strongest_month: {
        month: number;
        average_price: number;
      };
      weakest_month: {
        month: number;
        average_price: number;
      };
      seasonal_strength: number;
    };
    volatility?: {
      standard_deviation: number;
      coefficient_of_variation: number;
      price_range: number;
      volatility_level: string;
    };
    forecast?: {
      short_term_forecast: {
        value: number;
        confidence: number;
      };
      medium_term_forecast: {
        value: number;
        confidence: number;
      };
      long_term_forecast: {
        value: number;
        confidence: number;
      };
    };
    market_indicators?: {
      price_momentum: number;
      market_strength: number;
      price_stability: number;
    };
  };
  llm_insights?: {
    insights: string;
    recommendations: string[];
  };
}

export const searchProperties = async (filters: SearchFilters): Promise<Property[]> => {
  try {
    // Log the incoming filters
    console.log('Raw filters:', filters);

    // Format and validate the search parameters
    const params = formatSearchParams(filters);
    console.log('Formatted search params:', params);

    // Make the API request
    const response = await api.get('/properties', { params });
    console.log('API response:', response.data);

    // Validate the response
    if (!Array.isArray(response.data)) {
      console.error('Invalid response format:', response.data);
      throw new Error('Invalid response format from API');
    }

    return response.data;
  } catch (error) {
    console.error('Error searching properties:', error);
    throw error;
  }
};

const formatSearchParams = (filters: SearchFilters): Record<string, any> => {
  const params: Record<string, any> = {
    sort: "relevance",
    prop_type: "single_family"  // Add default property type
  };

  // Location parameters
  if (filters.city && filters.state_code) {
    params.city = filters.city;
    params.state_code = filters.state_code;
  } else if (filters.zipCode) {
    params.zip_code = filters.zipCode;  // Changed from postal_code to zip_code to match backend
  }

  // Price parameters - only include if they have reasonable values
  if (filters.minPrice && parseInt(filters.minPrice) > 0) {
    params.min_price = parseInt(filters.minPrice);  // Changed from price_min to min_price
  }
  if (filters.maxPrice && parseInt(filters.maxPrice) < 100000000) {
    params.max_price = parseInt(filters.maxPrice);  // Changed from price_max to max_price
  }

  // Property features - only include if they have reasonable values
  if (filters.beds && parseInt(filters.beds) > 0) {
    params.beds = parseInt(filters.beds);  // Changed from beds_min to beds
  }
  if (filters.baths && parseInt(filters.baths) > 0) {
    params.baths = parseInt(filters.baths);  // Changed from baths_min to baths
  }
  if (filters.propertyType) {
    params.property_type = filters.propertyType;  // Changed from prop_type to property_type
  }

  // Log the formatted parameters
  console.log('Formatted parameters:', params);
  return params;
};

export const getPropertyDetails = async (propertyId: string): Promise<Property> => {
  try {
    const response = await api.get(`/properties/${propertyId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching property details:', error);
    throw error;
  }
};

export const analyzeProperty = async (
  propertyId: string,
  analysisType: string = 'full'
): Promise<Property> => {
  const response = await api.post('/analyze', {
    property_id: propertyId,
    analysis_type: analysisType,
  });
  return response.data;
};

export const getMarketTrends = async (location: string): Promise<MarketTrends> => {
  const encodedLocation = encodeURIComponent(location);
  const response = await api.get(`/market-trends/${encodedLocation}`);
  return response.data;
};

export const getComparableProperties = async (propertyId: string): Promise<Property[]> => {
  const response = await api.get(`/comps/${propertyId}`);
  return response.data;
};

export const getPropertyAnalysis = async (propertyId: string): Promise<Property> => {
  const response = await api.get(`/properties/${propertyId}/analysis`);
  return response.data;
};

export const getPropertyPrediction = async (propertyId: string): Promise<Property> => {
  const response = await api.get(`/properties/${propertyId}/prediction`);
  return response.data;
};

export const getPropertyRiskAssessment = async (propertyId: string): Promise<Property> => {
  const response = await api.get(`/properties/${propertyId}/risk-assessment`);
  return response.data;
};

export const ragChat = async (question: string, context: string, history: string[] = []): Promise<{ answer: string }> => {
  const response = await api.post('/rag-chat', {
    question,
    context,
    history
  });
  return response.data;
};

export const getCrewAIAnalysis = async (propertyId: string, analysisType: 'full' | 'rental' | 'flip'): Promise<{ summary: string }> => {
  const response = await api.post('/crewai-analysis', {
    property_id: propertyId,
    analysis_type: analysisType,
  });
  return response.data;
}; 
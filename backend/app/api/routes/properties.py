from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from app.services.service_manager import ServiceManager
from app.services.market_data_service import MarketDataService
from app.services.property_analyzer import PropertyAnalyzer
from app.services.data_collector import DataCollector
from app.services.openai_service import OpenAIService
from app.core.logging import loggers
import traceback
from loguru import logger
import httpx
from app.core.config import settings
import logging
from datetime import datetime

router = APIRouter()

# Initialize services using ServiceManager
market_data_service = ServiceManager.get_market_data_service()
property_analyzer = ServiceManager.get_property_analyzer()
data_collector = ServiceManager.get_data_collector()
logger = loggers['api']

def transform_property_data(property_data: dict) -> dict:
    """Transform property data to match PropertyAnalyzer's expected field names."""
    # Convert zip_code to numeric, handling potential errors
    zip_code = property_data.get('zip_code', '')
    try:
        # Remove any non-numeric characters and convert to int
        numeric_zip = int(''.join(filter(str.isdigit, str(zip_code))))
    except (ValueError, TypeError):
        # If conversion fails, use a default value
        numeric_zip = 78701  # Default to a valid Austin ZIP code
    
    transformed = {
        'property_id': property_data.get('property_id', ''),
        'address': property_data.get('address', ''),
        'city': property_data.get('city', ''),
        'state': property_data.get('state', ''),
        'zip_code': numeric_zip,  # Use the numeric zip code
        'price': float(property_data.get('price', 0)),
        'bedrooms': int(property_data.get('beds', 0)),  # PropertyAnalyzer expects 'bedrooms'
        'bathrooms': float(property_data.get('baths', 0)),  # PropertyAnalyzer expects 'bathrooms'
        'square_feet': float(property_data.get('square_feet', property_data.get('area', 1000))),  # PropertyAnalyzer expects 'square_feet'
        'year_built': int(property_data.get('year_built', datetime.now().year - 20)),  # Default to 20 years old
        'lot_size': float(property_data.get('lot_size', property_data.get('square_feet', property_data.get('area', 1000)) * 2)),  # Default to 2x square feet
        'property_type': property_data.get('property_type', 'single_family'),
        'listing_date': property_data.get('listing_date'),
        'latitude': property_data.get('latitude'),
        'longitude': property_data.get('longitude'),
        'days_on_market': property_data.get('days_on_market', 30),  # Default to 30 days
        'estimated_rent': property_data.get('estimated_rent', property_data.get('price', 0) * 0.008),  # Default to 0.8% of price
        'estimated_mortgage': property_data.get('estimated_mortgage', property_data.get('price', 0) * 0.004),  # Default to 0.4% of price
        'estimated_expenses': property_data.get('estimated_expenses', property_data.get('price', 0) * 0.002),  # Default to 0.2% of price
    }
    return transformed

class PropertySearchParams(BaseModel):
    city: Optional[str] = None
    zip_code: Optional[str] = None
    state_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    beds: Optional[int] = None
    baths: Optional[int] = None
    property_type: Optional[str] = None

class PropertyResponse(BaseModel):
    property_id: str
    address: str
    city: str
    state: str
    zip_code: str
    price: float
    beds: int
    baths: float
    square_feet: Optional[float] = None
    property_type: Optional[str] = "single_family"
    listing_date: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    investment_score: Optional[float] = 0.0
    investment_score_explanation: Optional[str] = ""
    rental_yield: Optional[float] = None
    flip_roi: Optional[float] = None
    analysis: Optional[str] = ""
    market_trends: Optional[dict] = None
    investment_metrics: Optional[Dict[str, float]] = None
    base_metrics: Optional[Dict[str, float]] = None
    photos: Optional[List[str]] = None

@router.get(
    "/properties",
    response_model=list[PropertyResponse],
)
async def get_properties(
    city: str | None = None,
    zip_code: str | None = None,
    state_code: str | None = None,
    min_price: int | None = Query(default=None, ge=0),
    max_price: int | None = Query(default=None, ge=0),
    beds: int | None = Query(default=None, ge=0),
    baths: int | None = Query(default=None, ge=0),
    property_type: str | None = None,
    market_data_service: MarketDataService = Depends(lambda: ServiceManager.get_market_data_service())
):
    """
    Fetch property listings based on search criteria and analyze their investment potential.
    """
    try:
        logger.info(f"Searching properties with params: city={city}, state_code={state_code}, zip_code={zip_code}, min_price={min_price}, max_price={max_price}, beds={beds}, baths={baths}, property_type={property_type}")
        
        # Validate required parameters
        if not city and not zip_code:
            raise HTTPException(status_code=400, detail="Either city or zip_code is required")
        if city and not state_code:
            raise HTTPException(status_code=400, detail="state_code is required when city is provided")
        
        # Get properties from data collector
        properties = await data_collector.get_properties_by_location(
            city=city,
            zip_code=zip_code,
            state_code=state_code,
            min_price=min_price,
            max_price=max_price,
            beds=beds,
            baths=baths,
            property_type=property_type
        )
        
        if not properties:
            # Return empty list instead of raising an error
            return []
        
        # Get market data for the location
        location = f"{city}, {state_code}" if city and state_code else zip_code
        market_trends = await market_data_service.get_market_trends(location)
        
        # Initialize market data with default values
        market_data = {
            'median_listing_price': 0,
            'median_days_on_market': 30,
            'inventory_count': 100,
            'price_reduction_count': 10,
            'price_increase_count': 10,
            'price_change_1y': 0,
            'price_change_3y': 0,
            'price_change_5y': 0,
            'price_volatility': 0.1,
            'inventory_volatility': 0.1
        }
        
        if "error" not in market_trends:
            # Extract market data from market_trends
            current_metrics = market_trends.get('market_data', {}).get('current_metrics', {})
            market_analysis = market_trends.get('market_analysis', {})
            historical_data = market_trends.get('market_data', {}).get('historical_data', {})
            
            # Update market data with available values
            if current_metrics:
                market_data.update({
                    'median_listing_price': current_metrics.get('median_price') or current_metrics.get('median_listing_price') or market_data['median_listing_price'],
                    'median_days_on_market': current_metrics.get('avg_days_on_market') or market_data['median_days_on_market'],
                    'inventory_count': market_data.get('inventory_count', 100),
                    'price_reduction_count': market_data.get('price_reduction_count', 10),
                    'price_increase_count': market_data.get('price_increase_count', 10)
                })
            
            # Include historical data for price trend calculations
            if historical_data:
                market_data['historical_data'] = historical_data
                
            # Update with market analysis data if available
            if market_analysis:
                inventory_health = market_analysis.get('inventory_health', {})
                price_trends = market_analysis.get('price_trends', {})
                
                market_data.update({
                    'inventory_count': inventory_health.get('current_inventory', market_data['inventory_count']),
                    'price_reduction_count': price_trends.get('price_reductions', market_data['price_reduction_count']),
                    'price_increase_count': price_trends.get('price_increases', market_data['price_increase_count']),
                    'price_change_1y': price_trends.get('price_change_1y', market_data['price_change_1y']),
                    'price_change_3y': price_trends.get('price_change_3y', market_data['price_change_3y']),
                    'price_change_5y': price_trends.get('price_change_5y', market_data['price_change_5y']),
                    'price_volatility': price_trends.get('price_volatility', market_data['price_volatility']),
                    'inventory_volatility': inventory_health.get('inventory_volatility', market_data['inventory_volatility'])
                })
        
        # Ensure all market_data values are valid numbers
        for key, value in market_data.items():
            if value is None or (isinstance(value, float) and (value == float('inf') or value == float('-inf'))):
                market_data[key] = 0.0
            elif not isinstance(value, (int, float)):
                try:
                    market_data[key] = float(value)
                except (ValueError, TypeError):
                    market_data[key] = 0.0
        
        # Analyze each property
        analyzed_properties = []
        for property_data in properties:
            try:
                # Transform property data to match PropertyAnalyzer expectations
                transformed_property = transform_property_data(property_data)
                
                # Add market trends to property data if available
                if "error" not in market_trends:
                    transformed_property['market_trends'] = market_trends
                
                # Add historical data to market_data if available
                if "error" not in market_trends and 'market_data' in market_trends and 'historical_data' in market_trends['market_data']:
                    market_data['historical_data'] = market_trends['market_data']['historical_data']
                
                # Analyze the property with both property_data and market_data
                analysis = property_analyzer.analyze_property(transformed_property, market_data)
                
                # Generate score explanation
                score_explanation = generate_score_explanation(analysis)
                
                # Create PropertyResponse with both property data and analysis results
                property_response = PropertyResponse(
                    property_id=str(property_data.get('property_id', '')),
                    address=property_data.get('address', ''),
                    city=property_data.get('city', ''),
                    state=property_data.get('state', ''),
                    zip_code=property_data.get('zip_code', ''),
                    price=float(property_data.get('price', 0)),
                    beds=int(property_data.get('beds', 0)),
                    baths=float(property_data.get('baths', 0)),
                    square_feet=property_data.get('square_feet', property_data.get('area')),
                    property_type=property_data.get('property_type', 'single_family'),
                    listing_date=property_data.get('listing_date'),
                    latitude=property_data.get('latitude'),
                    longitude=property_data.get('longitude'),
                    investment_score=round(float(analysis.get('investment_score', 0.0)), 3),
                    investment_score_explanation=score_explanation,
                    rental_yield=analysis.get('investment_metrics', {}).get('cap_rate'),
                    flip_roi=analysis.get('investment_metrics', {}).get('cash_on_cash'),
                    analysis=f"Investment Score: {analysis.get('investment_score', 0.0):.3f}. " + ". ".join(analysis.get('recommendations', [])),
                    market_trends=transformed_property.get('market_trends'),
                    investment_metrics=analysis.get('investment_metrics', {}),
                    base_metrics=analysis.get('base_metrics', {}),
                    photos=property_data.get('photos', [])
                )
                analyzed_properties.append(property_response)
            except Exception as e:
                logger.error(f"Error analyzing property {property_data.get('property_id', 'unknown')}: {str(e)}")
                # Continue with other properties even if one fails
                continue
        
        return analyzed_properties
    
    except Exception as e:
        logger.error(f"Error during property search: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def generate_score_explanation(analysis: Dict[str, Any]) -> str:
    """Generate a detailed explanation of how the investment score was calculated."""
    try:
        base_metrics = analysis.get('base_metrics', {})
        investment_metrics = analysis.get('investment_metrics', {})
        risk_metrics = analysis.get('risk_metrics', {})
        market_metrics = analysis.get('market_metrics', {})
        
        explanation = "Investment Score Breakdown:\n"
        
        # Base Score (25%)
        explanation += "\nBase Score (25%):"
        explanation += f"\n• Price per sqft: {base_metrics.get('price_per_sqft', 0):.2f}"
        explanation += f"\n• Price to median: {base_metrics.get('price_to_median', 0):.2f}"
        explanation += f"\n• Sqft per bed: {base_metrics.get('sqft_per_bed', 0):.2f}"
        explanation += f"\n• Beds/baths ratio: {base_metrics.get('beds_baths_ratio', 0):.2f}"
        
        # Investment Score (35%)
        explanation += "\n\nInvestment Score (35%):"
        explanation += f"\n• Cap Rate: {investment_metrics.get('cap_rate', 0):.2f}%"
        explanation += f"\n• Cash on Cash: {investment_metrics.get('cash_on_cash', 0):.2f}%"
        explanation += f"\n• ROI: {investment_metrics.get('roi', 0):.2f}%"
        explanation += f"\n• DSCR: {investment_metrics.get('dscr', 0):.2f}"
        explanation += f"\n• Rental Yield: {investment_metrics.get('rental_yield', 0):.2f}%"
        
        # Risk Score (20%)
        explanation += "\n\nRisk Score (20%):"
        explanation += f"\n• Market Risk: {risk_metrics.get('market_risk', 0):.2f}"
        explanation += f"\n• Property Risk: {risk_metrics.get('property_risk', 0):.2f}"
        explanation += f"\n• Location Risk: {risk_metrics.get('location_risk', 0):.2f}"
        
        # Market Score (20%)
        explanation += "\n\nMarket Score (20%):"
        explanation += f"\n• Market Health: {market_metrics.get('market_health', 0):.2f}"
        explanation += f"\n• Market Momentum: {market_metrics.get('market_momentum', 0):.2f}"
        explanation += f"\n• Market Stability: {market_metrics.get('market_stability', 0):.2f}"
        
        # Final Score
        explanation += f"\n\nFinal Score: {analysis.get('investment_score', 0):.3f}"
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating score explanation: {str(e)}")
        return "Score explanation not available"

@router.get("/properties/{property_id}", response_model=PropertyResponse)
async def get_property_details(property_id: str):
    """
    Get detailed information about a specific property.
    """
    try:
        logger.info(f"Fetching property details for ID: {property_id}")
        
        # Get property details from data collector
        property_data = await data_collector.get_property_by_id(property_id)
        
        if not property_data:
            raise HTTPException(status_code=404, detail="Property not found")
        
        # Get market data for the location
        location = f"{property_data.get('city', '')}, {property_data.get('state', '')}"
        market_trends = await market_data_service.get_market_trends(location)
        
        # Initialize market data with default values
        market_data = {
            'median_listing_price': 0,
            'median_days_on_market': 30,
            'inventory_count': 100,
            'price_reduction_count': 10,
            'price_increase_count': 10,
            'price_change_1y': 0,
            'price_change_3y': 0,
            'price_change_5y': 0,
            'price_volatility': 0.1,
            'inventory_volatility': 0.1
        }
        
        if "error" not in market_trends:
            # Extract market data from market_trends
            current_metrics = market_trends.get('market_data', {}).get('current_metrics', {})
            market_analysis = market_trends.get('market_analysis', {})
            historical_data = market_trends.get('market_data', {}).get('historical_data', {})
            
            # Update market data with available values
            if current_metrics:
                market_data.update({
                    'median_listing_price': current_metrics.get('median_price') or current_metrics.get('median_listing_price') or market_data['median_listing_price'],
                    'median_days_on_market': current_metrics.get('avg_days_on_market') or market_data['median_days_on_market'],
                    'inventory_count': market_data.get('inventory_count', 100),
                    'price_reduction_count': market_data.get('price_reduction_count', 10),
                    'price_increase_count': market_data.get('price_increase_count', 10)
                })
            
            # Include historical data for price trend calculations
            if historical_data:
                market_data['historical_data'] = historical_data
                
            # Update with market analysis data if available
            if market_analysis:
                inventory_health = market_analysis.get('inventory_health', {})
                price_trends = market_analysis.get('price_trends', {})
                
                market_data.update({
                    'inventory_count': inventory_health.get('current_inventory', market_data['inventory_count']),
                    'price_reduction_count': price_trends.get('price_reductions', market_data['price_reduction_count']),
                    'price_increase_count': price_trends.get('price_increases', market_data['price_increase_count']),
                    'price_change_1y': price_trends.get('price_change_1y', market_data['price_change_1y']),
                    'price_change_3y': price_trends.get('price_change_3y', market_data['price_change_3y']),
                    'price_change_5y': price_trends.get('price_change_5y', market_data['price_change_5y']),
                    'price_volatility': price_trends.get('price_volatility', market_data['price_volatility']),
                    'inventory_volatility': inventory_health.get('inventory_volatility', market_data['inventory_volatility'])
                })
        
        # Ensure all market_data values are valid numbers
        for key, value in market_data.items():
            if value is None or (isinstance(value, float) and (value == float('inf') or value == float('-inf'))):
                market_data[key] = 0.0
            elif not isinstance(value, (int, float)):
                try:
                    market_data[key] = float(value)
                except (ValueError, TypeError):
                    market_data[key] = 0.0
        
        # Transform property data to match PropertyAnalyzer expectations
        transformed_property = transform_property_data(property_data)
        
        # Add market trends to property data if available
        if "error" not in market_trends:
            transformed_property['market_trends'] = market_trends
            
        # Add historical data to market_data if available
        if "error" not in market_trends and 'market_data' in market_trends and 'historical_data' in market_trends['market_data']:
            market_data['historical_data'] = market_trends['market_data']['historical_data']
            logger.debug(f"PRICE_TREND_FIX: Added historical_data to market_data for property {property_id}")
            logger.debug(f"PRICE_TREND_FIX: Historical data keys: {list(market_trends['market_data']['historical_data'].keys())}")
        else:
            logger.debug(f"PRICE_TREND_FIX: No historical data available for property {property_id}")
            logger.debug(f"PRICE_TREND_FIX: market_trends keys: {list(market_trends.keys()) if market_trends else 'None'}")
            if 'market_data' in market_trends:
                logger.debug(f"PRICE_TREND_FIX: market_data keys: {list(market_trends['market_data'].keys())}")
        
        # Analyze the property with both property_data and market_data
        analysis = property_analyzer.analyze_property(transformed_property, market_data)
        
        # Generate score explanation
        score_explanation = generate_score_explanation(analysis)
        
        # Create PropertyResponse with both property data and analysis results
        property_response = PropertyResponse(
            property_id=str(property_data.get('property_id', '')),
            address=property_data.get('address', ''),
            city=property_data.get('city', ''),
            state=property_data.get('state', ''),
            zip_code=property_data.get('zip_code', ''),
            price=float(property_data.get('price', 0)),
            beds=int(property_data.get('beds', 0)),
            baths=float(property_data.get('baths', 0)),
            square_feet=property_data.get('square_feet', property_data.get('area')),
            property_type=property_data.get('property_type', 'single_family'),
            listing_date=property_data.get('listing_date'),
            latitude=property_data.get('latitude'),
            longitude=property_data.get('longitude'),
            investment_score=round(float(analysis.get('investment_score', 0.0)), 3),
            investment_score_explanation=score_explanation,
            rental_yield=analysis.get('investment_metrics', {}).get('cap_rate'),
            flip_roi=analysis.get('investment_metrics', {}).get('cash_on_cash'),
            analysis=f"Investment Score: {analysis.get('investment_score', 0.0):.3f}. " + ". ".join(analysis.get('recommendations', [])),
            market_trends=transformed_property.get('market_trends'),
            investment_metrics=analysis.get('investment_metrics', {}),
            base_metrics=analysis.get('base_metrics', {}),
            photos=property_data.get('photos', [])
        )
        
        return property_response
        
    except Exception as e:
        logger.error(f"Error fetching property details: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/properties/{property_id}/analysis")
async def get_property_analysis(
    property_id: str,
    data_collector: DataCollector = Depends(lambda: ServiceManager.get_data_collector()),
    market_data_service: MarketDataService = Depends(lambda: ServiceManager.get_market_data_service()),
    property_analyzer: PropertyAnalyzer = Depends(lambda: ServiceManager.get_property_analyzer())
) -> Dict[str, Any]:
    """
    Get detailed investment analysis for a property.
    
    Args:
        property_id: The ID of the property to analyze
        data_collector: Service for collecting property data
        market_data_service: Service for collecting market data
        property_analyzer: Service for analyzing properties
        
    Returns:
        Analysis results including investment score and recommendations
    """
    try:
        # Get property data
        property_data = await data_collector.get_property_by_id(property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail=f"Property {property_id} not found")
            
        # Get market data
        location = f"{property_data.get('city', '')}, {property_data.get('state', '')}"
        market_trends = await market_data_service.get_market_trends(location)
        
        # Initialize market data with defaults
        market_data = {
            "median_price": 0,
            "price_trend": 0,
            "days_on_market": 0,
            "inventory_level": 0,
            "error": None
        }
        
        # Update market data if available
        if not market_trends.get("error"):
            market_data.update(market_trends)
            
            # Add historical data if available
            if 'market_data' in market_trends and 'historical_data' in market_trends['market_data']:
                market_data['historical_data'] = market_trends['market_data']['historical_data']
                logger.debug(f"PRICE_TREND_FIX: Added historical_data to market_data for analysis endpoint property {property_id}")
                logger.debug(f"PRICE_TREND_FIX: Historical data keys: {list(market_trends['market_data']['historical_data'].keys())}")
            else:
                logger.debug(f"PRICE_TREND_FIX: No historical data available for analysis endpoint property {property_id}")
                logger.debug(f"PRICE_TREND_FIX: market_trends keys: {list(market_trends.keys()) if market_trends else 'None'}")
                if 'market_data' in market_trends:
                    logger.debug(f"PRICE_TREND_FIX: market_data keys: {list(market_trends['market_data'].keys())}")
            
        # Transform property data to include all required fields (bedrooms, bathrooms, lot_size, zip_code, etc.)
        transformed_property = transform_property_data(property_data)
        # Add market trends for completeness (optional)
        transformed_property["market_trends"] = market_trends

        # Analyze property
        analysis_results = property_analyzer.analyze_property(transformed_property, market_data)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing property: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/properties/{property_id}/prediction")
async def get_property_prediction(
    property_id: str,
    data_collector: DataCollector = Depends(lambda: ServiceManager.get_data_collector()),
    market_data_service: MarketDataService = Depends(lambda: ServiceManager.get_market_data_service()),
    property_analyzer: PropertyAnalyzer = Depends(lambda: ServiceManager.get_property_analyzer())
) -> Dict[str, Any]:
    """
    Get price prediction and market forecast for a property.
    
    Args:
        property_id: The ID of the property to predict
        data_collector: Service for collecting property data
        market_data_service: Service for collecting market data
        property_analyzer: Service for analyzing properties
        
    Returns:
        Prediction results including price forecast and market trends
    """
    try:
        # Get property data
        property_data = await data_collector.get_property_by_id(property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail=f"Property {property_id} not found")
            
        # Get market data
        location = f"{property_data.get('city', '')}, {property_data.get('state', '')}"
        market_trends = await market_data_service.get_market_trends(location)
        
        # Use the property analyzer to generate predictions
        transformed_property = transform_property_data(property_data)
        prediction_results = property_analyzer.generate_predictions(transformed_property, market_trends)
        
        return prediction_results
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/properties/{property_id}/risk-assessment")
async def get_property_risk(
    property_id: str,
    data_collector: DataCollector = Depends(lambda: ServiceManager.get_data_collector()),
    market_data_service: MarketDataService = Depends(lambda: ServiceManager.get_market_data_service()),
    property_analyzer: PropertyAnalyzer = Depends(lambda: ServiceManager.get_property_analyzer())
) -> Dict[str, Any]:
    """
    Get risk assessment for a property.
    
    Args:
        property_id: The ID of the property to assess
        data_collector: Service for collecting property data
        market_data_service: Service for collecting market data
        property_analyzer: Service for analyzing properties
        
    Returns:
        Risk assessment results including market risk and property risk
    """
    try:
        # Get property data
        property_data = await data_collector.get_property_by_id(property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail=f"Property {property_id} not found")
            
        # Get market data
        location = f"{property_data.get('city', '')}, {property_data.get('state', '')}"
        market_trends = await market_data_service.get_market_trends(location)
        
        # Transform property data
        transformed_property = transform_property_data(property_data)
        risk_metrics = property_analyzer._calculate_risk_metrics(transformed_property, market_trends)
        
        return {
            "property_id": property_id,
            "risk_metrics": risk_metrics,
            "market_risk": risk_metrics.get("market_risk", 0),
            "property_risk": risk_metrics.get("property_risk", 0),
            "location_risk": risk_metrics.get("location_risk", 0),
            "overall_risk": risk_metrics.get("overall_risk", 0)
        }
        
    except Exception as e:
        logger.error(f"Error assessing property risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comps/{property_id}")
async def get_comparable_properties(
    property_id: str,
    data_collector: DataCollector = Depends(),
    property_analyzer: PropertyAnalyzer = Depends()
) -> List[Dict[str, Any]]:
    """
    Get comparable properties for a given property.
    
    Args:
        property_id: The ID of the property to find comparables for
        data_collector: Service for collecting property data
        property_analyzer: Service for analyzing properties
        
    Returns:
        List of comparable properties with similarity scores
    """
    try:
        # First attempt comparable properties search - this will check if property exists
        # in FAISS index first, and only require fresh data if not in index
        comparable_properties = await property_analyzer.get_comparable_properties(property_id)
        
        # If we got results, return them
        if comparable_properties:
            logger.info(f"Found {len(comparable_properties)} comparable properties for {property_id}")
            return comparable_properties
        
        # If no comparables found, check if the target property exists at all
        # This helps distinguish between "property not found" vs "no similar properties found"
        target_property = await data_collector.get_property_by_id(property_id)
        if not target_property:
            # Check if property analyzer has FAISS index and if property is in it
            if hasattr(property_analyzer, '_comparable_service') and property_analyzer._comparable_service:
                # Check if property exists in FAISS index mapping
                if hasattr(property_analyzer._comparable_service, '_id_to_idx') and property_id in property_analyzer._comparable_service._id_to_idx:
                    logger.info(f"Property {property_id} exists in FAISS index but no comparables found, returning empty list")
                    return []
                else:
                    raise HTTPException(status_code=404, detail=f"Target property {property_id} not found in current data or FAISS index")
            else:
                raise HTTPException(status_code=404, detail=f"Target property {property_id} not found")
        
        # Property exists but no comparables found
        logger.info(f"No comparable properties found for {property_id}, returning empty list")
        return []
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting comparable properties for {property_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 
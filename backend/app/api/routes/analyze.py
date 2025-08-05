from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.services.service_manager import ServiceManager
from app.services.market_data_service import MarketDataService
from app.services.property_analyzer import PropertyAnalyzer
from app.services.openai_service import OpenAIService
from loguru import logger
from app.api.endpoints import ml_analysis
from app.services.data_collector import DataCollector
from app.core.config import settings
from app.schemas.analysis import AnalysisRequest as SchemaAnalysisRequest, AnalysisResponse as SchemaAnalysisResponse
from app.api.routes.properties import transform_property_data

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    context: str
    history: List[str] = []

class ChatResponse(BaseModel):
    answer: str

@router.post("/analyze", response_model=SchemaAnalysisResponse)
async def analyze_property(
    request: SchemaAnalysisRequest,
    data_collector: DataCollector = Depends(lambda: ServiceManager.get_data_collector()),
    market_data_service: MarketDataService = Depends(lambda: ServiceManager.get_market_data_service()),
    property_analyzer: PropertyAnalyzer = Depends(lambda: ServiceManager.get_property_analyzer())
) -> Dict[str, Any]:
    """
    Analyze a property for investment potential.
    
    Args:
        request: Analysis request containing property ID
        data_collector: Service for collecting property data
        market_data_service: Service for collecting market data
        property_analyzer: Service for analyzing property data
        
    Returns:
        Analysis results including investment score and recommendations
    """
    try:
        # Get property data
        property_data = await data_collector.get_property_by_id(request.property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail=f"Property {request.property_id} not found")
            
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
            
        # Transform property data to match analyzer expectations
        transformed_property = transform_property_data(property_data)
        
        # Hybrid analysis (rules + LLM)
        analysis_results = await property_analyzer.generate_insights(transformed_property, request.analysis_type)
        comparable_properties = await property_analyzer.get_comparable_properties(request.property_id)
        return {
            "property_id": request.property_id,
            "investment_score": analysis_results.get("investment_score", 0),
            "base_metrics": analysis_results.get("base_metrics", {}),
            "investment_metrics": analysis_results.get("investment_metrics", {}),
            "risk_metrics": analysis_results.get("risk_metrics", {}),
            "market_metrics": analysis_results.get("market_metrics", {}),
            "recommendations": analysis_results.get("recommendations", []),
            "comparable_properties": comparable_properties,
            "insights": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Error analyzing property: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-trends/{location}")
async def get_market_trends(
    location: str,
    market_data_service: MarketDataService = Depends(lambda: ServiceManager.get_market_data_service())
):
    """
    Get market trends and analysis for a specific location.
    """
    try:
        logger.info(f"Getting market trends for location: {location}")
        
        # Get market data using the instance variable
        if ',' in location:
            city, state = location.split(',')
            market_data = await market_data_service.get_metro_level_metrics(city.strip(), state.strip())
        else:
            market_data = await market_data_service.get_zip_level_metrics(location.strip())
        
        if "error" in market_data:
            logger.error(f"Error getting market data: {market_data['error']}")
            return {"error": market_data["error"]}
        
        # Analyze market trends
        market_analysis = market_data_service.analyze_market_trends(market_data)
        
        if "error" in market_analysis:
            logger.error(f"Error analyzing market trends: {market_analysis['error']}")
            return {"error": market_analysis["error"]}
        
        logger.info("Successfully retrieved and analyzed market trends")
        return {
            "location": location,
            "market_data": market_data,
            "market_analysis": market_analysis
        }
        
    except Exception as e:
        logger.error(f"Error getting market trends for {location}: {str(e)}")
        return {"error": str(e)}

@router.get("/comps/{property_id}")
async def get_comparable_properties(
    property_id: str,
    property_analyzer: PropertyAnalyzer = Depends(lambda: ServiceManager.get_property_analyzer())
):
    """
    Get comparable properties for analysis.
    """
    try:
        comps = await property_analyzer.get_comparable_properties(property_id)
        return comps
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag-chat", response_model=ChatResponse)
async def rag_chat(
    request: ChatRequest,
    openai_service: OpenAIService = Depends(lambda: ServiceManager.get_openai_service())
):
    """
    Handle RAG-based chat requests with context and history.
    """
    try:
        # Generate response using OpenAI service
        response = await openai_service.generate_chat_response(
            question=request.question,
            context=request.context,
            history=request.history
        )
        
        return ChatResponse(answer=response)
    
    except Exception as e:
        logger.error(f"Error during RAG chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# At the bottom, ensure the router includes ml_analysis.router
# If this file defines a router, add:
# router.include_router(ml_analysis.router)

# If not, ensure main.py includes it (already done). 

# --------------------- CrewAI analysis endpoint ---------------------

# Hybrid LLM+Rules endpoint for frontend
class CrewAIRequest(BaseModel):
    property_id: str
    analysis_type: str = "full"  # full, rental, flip

from app.schemas.analysis import RiskMetricsResponse, MarketMetricsResponse

class CrewAIHybridResponse(BaseModel):
    """Hybrid response model combining ML metrics with LLM insights."""
    summary: str = Field(..., description="AI-generated analysis summary")
    investment_score: float = Field(..., description="Overall investment score")
    base_metrics: Dict[str, float] = Field(..., description="Basic property metrics")
    investment_metrics: Dict[str, float] = Field(..., description="Investment-related metrics")
    risk_metrics: RiskMetricsResponse = Field(..., description="Risk assessment metrics with sources")
    market_metrics: MarketMetricsResponse = Field(..., description="Market analysis metrics with sources")
    recommendations: List[str] = Field(..., description="AI-generated recommendations")
    analysis_date: str = Field(..., description="Timestamp of analysis")

@router.post("/crewai-analysis", response_model=CrewAIHybridResponse)
async def crewai_analysis(
    request: CrewAIRequest,
    data_collector: DataCollector = Depends(lambda: ServiceManager.get_data_collector()),
    property_analyzer: PropertyAnalyzer = Depends(lambda: ServiceManager.get_property_analyzer()),
):
    """Hybrid endpoint: returns both rules-based metrics and LLM expert insights for a property."""
    try:
        property_data = await data_collector.get_property_by_id(request.property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail="Property not found")
        transformed_property = transform_property_data(property_data)
        hybrid = await property_analyzer.generate_insights(
            transformed_property,
            analysis_type=request.analysis_type,
        )
        return CrewAIHybridResponse(
            summary=hybrid.get("expert_insights") or "AI insights are temporarily unavailable.",
            investment_score=hybrid.get("investment_score", 0),
            base_metrics=hybrid.get("base_metrics", {}),
            investment_metrics=hybrid.get("investment_metrics", {}),
            risk_metrics=hybrid.get("risk_metrics", {}),
            market_metrics=hybrid.get("market_metrics", {}),
            recommendations=hybrid.get("recommendations", []),
            analysis_date=hybrid.get("analysis_date", "")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating hybrid analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate hybrid analysis") 
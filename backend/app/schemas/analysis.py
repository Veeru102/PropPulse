from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal

class RiskMetricsResponse(BaseModel):
    """Structured response model for risk metrics."""
    market_risk: float = Field(..., description="Market-related risk score")
    property_risk: float = Field(..., description="Property-specific risk score")
    location_risk: float = Field(..., description="Location-based risk score")
    overall_risk: float = Field(..., description="Combined overall risk score")
    metrics_source: Dict[str, Literal["ml_model", "heuristic_fallback", "neutral_fallback", "weighted_average", "error_fallback"]] = Field(
        ..., 
        description="Source of each metric calculation"
    )

class MarketMetricsResponse(BaseModel):
    """Structured response model for market metrics."""
    market_health: float = Field(..., description="Overall market health score")
    market_momentum: float = Field(..., description="Market momentum/trend score")
    market_stability: float = Field(..., description="Market stability score")
    price_growth_rate: float = Field(..., description="Annualized price growth rate")
    metrics_source: Dict[str, Literal["ml_model", "heuristic_fallback", "neutral_fallback", "weighted_average", "error_fallback"]] = Field(
        ..., 
        description="Source of each metric calculation"
    )

class AnalysisRequest(BaseModel):
    """Request model for property analysis."""
    property_id: str
    analysis_type: str = Field(default="full", description="Analysis type: full, rental, or flip")

class AnalysisResponse(BaseModel):
    """Response model for property analysis."""
    property_id: str
    investment_score: float
    base_metrics: Dict[str, float]
    investment_metrics: Dict[str, float]
    risk_metrics: RiskMetricsResponse
    market_metrics: MarketMetricsResponse
    recommendations: List[str]
    comparable_properties: List[Dict[str, Any]]
    insights: Dict[str, Any] 
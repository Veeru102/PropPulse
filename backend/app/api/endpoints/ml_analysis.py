from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Dict, Any, List
from app.services.ml_predictor import MLPredictor
from app.services.market_data_service import MarketDataService
from app.services.realtor_api import RealtorAPIService
from app.services.data_collector import DataCollector
from app.services.model_trainer import ModelTrainer
from app.core.logging import loggers
import json
from app.services.openai_service import OpenAIService
from fastapi.responses import JSONResponse
from app.services.service_manager import ServiceManager
from pydantic import BaseModel

router = APIRouter()
logger = loggers['api']

# Initialize services
ml_predictor = MLPredictor()
market_data_service = MarketDataService()
realtor_service = RealtorAPIService()
data_collector = DataCollector()
model_trainer = ModelTrainer()
openai_service = OpenAIService()

class RAGChatRequest(BaseModel):
    question: str
    context: str
    history: List[str] = []

@router.post("/analyze-property/{property_id}")
async def analyze_property(
    property_id: str
) -> Dict[str, Any]:
    """
    Analyze a property using ML models to predict value and investment potential.
    """
    try:
        logger.info(f"Analyzing property {property_id}")
        
        # Get property data
        property_data = await realtor_service.get_property_details(property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail="Property not found")
        
        # Get market data
        location = f"{property_data['city']}, {property_data['state']}"
        market_data = await market_data_service.get_metro_level_metrics(
            property_data['city'],
            property_data['state']
        )
        
        # Get market analysis
        market_analysis_data = market_data_service.analyze_market_trends(market_data)
        
        # Get property value prediction
        value_prediction = ml_predictor.predict_property_value(property_data, market_data)
        
        # Get investment opportunity score
        investment_analysis = ml_predictor.score_investment_opportunity(property_data, market_data)

        # Generate LLM-powered market insights and recommendations
        llm_insights = await openai_service.get_market_insights_and_recommendations(
            current_metrics=market_data.get("current_metrics", {}),
            historical_data=market_data.get("historical_data", {}),
            market_analysis=market_analysis_data,
            location=location
        )
        
        logger.info(f"Successfully analyzed property {property_id}")
        
        return {
            "property_data": property_data,
            "market_analysis": market_analysis_data,
            "value_prediction": value_prediction,
            "investment_analysis": investment_analysis,
            "llm_insights": llm_insights
        }
    except Exception as e:
        logger.error(f"Error analyzing property {property_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-market/{location}")
async def analyze_market(
    location: str
) -> Dict[str, Any]:
    """
    Analyze market trends and provide detailed insights.
    """
    try:
        logger.info(f"Analyzing market for {location}")
        
        # Get market data
        if ',' in location:
            city, state = location.split(',')
            market_data = await market_data_service.get_metro_level_metrics(city.strip(), state.strip())
        else:
            market_data = await market_data_service.get_zip_level_metrics(location.strip())
        
        if "error" in market_data:
            raise HTTPException(status_code=404, detail=market_data["error"])
        
        # Get market analysis
        market_analysis = market_data_service.analyze_market_trends(market_data)

        # Generate LLM-powered market insights and recommendations
        llm_insights = await openai_service.get_market_insights_and_recommendations(
            current_metrics=market_data.get("current_metrics", {}),
            historical_data=market_data.get("historical_data", {}),
            market_analysis=market_analysis,
            location=location
        )
        
        logger.info(f"Successfully analyzed market for {location}")
        
        return {
            "location": location,
            "market_data": market_data,
            "market_analysis": market_analysis,
            "llm_insights": llm_insights
        }
    except Exception as e:
        logger.error(f"Error analyzing market for {location}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-models")
async def train_models(
    location: str,
    time_period: str = "1y",
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Train or retrain the ML models with new data.
    """
    try:
        logger.info(f"Starting model training for {location}")
        
        # Collect training data
        training_data = await data_collector.collect_training_data(location, time_period)
        
        # Train models
        training_results = model_trainer.train_models(training_data)
        
        logger.info(f"Successfully trained models for {location}")
        
        return {
            "status": "success",
            "location": location,
            "time_period": time_period,
            "training_results": training_results
        }
    except Exception as e:
        logger.error(f"Error training models for {location}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-metrics")
async def get_model_metrics() -> Dict[str, Any]:
    """
    Get current model performance metrics.
    """
    try:
        logger.info("Retrieving model metrics")
        
        # Load metrics from file
        metrics_path = model_trainer.model_dir / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Model metrics not found")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info("Successfully retrieved model metrics")
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/rag-chat')
async def rag_chat(request: RAGChatRequest):
    openai_service = ServiceManager.get_openai_service()
    prompt = f"""
You are a real estate market analysis assistant. Use the following context to answer the user's question. Be specific, cite numbers and trends from the context, and do not make up data.

Context:
{request.context}

"""
    if request.history:
        prompt += "Previous questions and answers (for context):\n"
        for h in request.history:
            prompt += f"- {h}\n"
    prompt += f"\nUser question: {request.question}\nAnswer:"
    try:
        answer = await openai_service.get_chat_completion(prompt)
        return {"answer": answer}
    except Exception as e:
        import traceback
        logger.error(f"RAG chat error: {str(e)}\n{traceback.format_exc()}")
        # Return the error message for debugging
        return JSONResponse(status_code=500, content={"answer": f"Sorry, error: {str(e)}"}) 
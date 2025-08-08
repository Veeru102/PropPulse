from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import properties, analyze
from app.api.endpoints import ml_analysis
from app.core.config import settings
from app.services.service_manager import ServiceManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PropPulse API",
    description="Real estate property analysis and prediction API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?:\/\/(www\.)?proppulse\.netlify\.app|https?:\/\/proppulse-hmgf\.onrender\.com|http:\/\/localhost:(3000|8000)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(properties.router, prefix="/api/v1", tags=["properties"])
app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
app.include_router(ml_analysis.router, prefix="/api/v1/ml", tags=["ml"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Initializing services...")
    ServiceManager.initialize_services()
    logger.info("Services initialized successfully")

@app.get("/")
async def root():
    return {"message": "Welcome to PropPulse API"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PropPulse API server")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True) 
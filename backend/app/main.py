from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import properties, analyze
from app.api.endpoints import ml_analysis
from app.core.config import settings
from app.services.service_manager import ServiceManager
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
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
    # Add debug logs for current working directory and directory contents
    current_working_dir = os.getcwd()
    logger.info(f"Current working directory: {current_working_dir}")

    backend_dir = Path(current_working_dir)
    models_dir = backend_dir / "models"
    data_dir = backend_dir / "data"

    logger.info(f"Contents of {models_dir}:")
    if models_dir.exists() and models_dir.is_dir():
        for item in os.listdir(models_dir):
            logger.info(f"  - {item}")
    else:
        logger.warning(f"Models directory not found or not a directory: {models_dir}")

    logger.info(f"Contents of {data_dir}:")
    if data_dir.exists() and data_dir.is_dir():
        for item in os.listdir(data_dir):
            logger.info(f"  - {item}")
    else:
        logger.warning(f"Data directory not found or not a directory: {data_dir}")
    
    # Initialize services after logging paths
    ServiceManager.initialize_services()
    logger.info("Services initialized successfully")

@app.get("/")
async def root():
    return {"message": "Welcome to PropPulse API"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PropPulse API server")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True) 
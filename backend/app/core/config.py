from pydantic_settings import BaseSettings
from typing import ClassVar, Dict, List, Any
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import Field

load_dotenv()

# Set BASE_DIR as a global variable
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Base directories
    BASE_DIR: Path = BASE_DIR
    MODEL_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # API Keys
    RAPIDAPI_KEY: str
    RAPIDAPI_HOST: str = "realty-in-us.p.rapidapi.com"
    OPENAI_API_KEY: str
    MAPBOX_API_KEY: str
    MAPBOX_TOKEN: str
    CENSUS_API_KEY: str
    REALTOR_API_KEY: str
    
    # Environment
    ENVIRONMENT: str = "development"
    
    # Database Configuration
    DATABASE_URL: str
    
    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DEBUG: bool = True
    
    # Model Parameters
    MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
        "property_value_model": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "enable_categorical": True,
            "tree_method": "hist"
        },
        "investment_model": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "enable_categorical": True,
            "tree_method": "hist"
        }
    }
    
    # Parameter validation ranges
    PARAM_RANGES: Dict[str, List[float]] = {
        "n_estimators": [100, 2000],
        "learning_rate": [0.001, 0.1],
        "max_depth": [3, 12],
        "min_child_weight": [1, 10],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "gamma": [0, 0.5],
        "reg_alpha": [0, 1.0],
        "reg_lambda": [0, 2.0]
    }
    
    # Required features for model training
    REQUIRED_FEATURES: List[str] = [
        "median_list_price",
        "median_list_price_per_sqft",
        "median_dom",
        "inventory",
        "sqft",
        "avg_list_price",
        "price_to_median_ratio",
        "price_to_avg_ratio"
    ]
    
    # Feature importance thresholds
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.01
    
    # Model performance thresholds
    MIN_R2_SCORE: float = 0.7
    MAX_RMSE_THRESHOLD: float = 0.3
    
    # Cross-validation settings
    N_SPLITS: int = 5
    TEST_SIZE: float = 0.2
    
    # Data validation settings
    MAX_MISSING_RATIO: float = 0.3
    OUTLIER_THRESHOLD: float = 3.0
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PropPulse"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]  # Frontend URL
    
    # ML Settings
    CACHE_DURATION: int = 24  
    
    HISTORICAL_INVENTORY_URL: str = "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_State_History.csv"
    HISTORICAL_METRICS_URL: str = "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv"
    
    # Model Settings
    MODEL_PATH: str = "app/models/property_roi_model.json"
    
    # ML Model Settings
    ML_MODEL_SETTINGS: ClassVar[Dict[str, Any]] = {
        'USE_NEUTRAL_FALLBACK': False,  # If True, use 0.5 as fallback; if False, use heuristic calculations
        'NEUTRAL_FALLBACK_VALUE': 0.5,  # Default neutral value when ML prediction fails
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create necessary directories
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True) 
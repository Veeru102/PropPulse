#!/bin/bash

# PropPulse FAISS-enabled startup script
# This script ensures FAISS is available by using the conda environment

echo "Starting PropPulse with FAISS support..."

# Activate conda base environment
source $HOME/miniconda/bin/activate

# Activate the faiss_env environment where FAISS is installed
conda activate faiss_env

# Install required dependencies in the conda environment that aren't already there
echo "Installing missing dependencies in conda environment..."
pip install fastapi uvicorn python-dotenv sqlalchemy psycopg2-binary pydantic pydantic-settings requests openai xgboost pandas scikit-learn python-jose[cryptography] passlib[bcrypt] python-multipart aiohttp geopy shapely geoalchemy2 pytest httpx loguru certifi torch

# Set PYTHONPATH to include the backend directory
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Start the FastAPI application
echo "Starting FastAPI server with FAISS support..."
uvicorn app.main:app --reload --port 8000

echo "Server stopped." 
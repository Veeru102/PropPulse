# PropPulse - AI-Powered Real Estate Intelligence Platform

PropPulse is an advanced real estate intelligence platform that leverages AI and machine learning to help investors, agents, and homebuyers make data-driven decisions. The platform analyzes property listings in real-time, providing investment insights and recommendations.

## Features

PropPulse provides a several features created to empower real estate investors, agents, and homebuyers:

- **Real-time Property Data Integration:** Fetches and processes real-time property listings from various sources, including Realtor.com API.
- **Property Comparison Tools:** Allows users to compare multiple properties side-by-side based on various metrics and analytical scores, leveraging FAISS for efficient similarity search.
- **Machine Learning for Predictive Analytics:** Employs ML models (XGBoost/LightGBM) to predict property values and potential return on investment (ROI).
- **AI-Powered Investment Analysis:** Utilizes AI (GPT-4o via OpenAI API) to provide deep investment insights, identify opportunities, and generate clear rationales.
- **Interactive Data Visualization:** Offers interactive map visualizations (using Mapbox GL JS) and dynamic charts to explore property data and market trends.
- **Comprehensive Market Analysis:** Provides detailed insights into real estate markets, including historical trends, supply-demand dynamics, and key performance indicators.
- **Investment Scoring & Risk Assessment:** Generates proprietary investment scores and assesses potential risks associated with properties and markets.
- **Data Pipeline:** Incorporates services for data collection, quality validation, feature extraction, and training/auditing of ML models to ensure data integrity and model accuracy.

## Tech Stack

### Frontend
- React
- TailwindCSS
- Mapbox GL JS
- TypeScript

### Backend
- FastAPI (Python)
- PostgreSQL with PostGIS
- XGBoost/LightGBM for ML predictions
- OpenAI GPT-4 API

### APIs
- Realtor.com API (via RapidAPI)
- Mapbox API
- OpenAI API

## Project Structure

```
proppulse/
├── backend/           # FastAPI backend application
│   ├── app/
│   │   ├── api/      # API endpoints and route definitions
│   │   │   ├── endpoints/ # Specific API endpoints (e.g., ml_analysis)
│   │   │   └── routes/    # Main API routes (e.g., properties, analyze)
│   │   ├── core/     # Core configurations, utilities, and settings
│   │   ├── data/     # Data storage and processing utilities
│   │   ├── ml_models/ # Machine learning model artifacts and utilities
│   │   ├── models/   # Database models and Pydantic schemas
│   │   ├── schemas/  # Pydantic schemas for request/response validation
│   │   ├── scripts/  # Backend-related scripts (e.g., data download, model training)
│   │   └── services/ # Core business logic, data processing, and external API integrations
│   │       ├── comparable_property_service.py # Logic for finding comparable properties
│   │       ├── data_collector.py       # Collects raw property data
│   │       ├── data_quality_auditor.py # Audits data for quality issues
│   │       ├── data_quality_validator.py # Validates incoming data
│   │       ├── enhanced_label_generator.py # Generates enhanced labels for ML training
│   │       ├── enhanced_model_trainer.py # Manages enhanced ML model training
│   │       ├── input_data_validator.py # Validates input data for analysis
│   │       ├── market_data_service.py  # Fetches and processes market trend data
│   │       ├── ml_predictor.py         # Handles ML model predictions
│   │       ├── model_trainer.py        # Core ML model training logic
│   │       ├── openai_service.py       # Integrates with OpenAI API for AI analysis
│   │       ├── property_analyzer.py    # Main logic for property investment analysis
│   │       ├── realtor_api.py          # Handles communication with Realtor.com API
│   │       ├── service_manager.py      # Manages initialization and access to services
│   │       └── training_data_generator.py # Generates data for model training
│   │       └── training_inference_auditor.py # Audits ML model training and inference
│   ├── tests/        # Backend tests
├── frontend/         # React frontend application
│   ├── src/
│   │   ├── components/  # Reusable UI components
│   │   │   ├── features/ # Feature-specific components (ex. InvestmentScoreDial)
│   │   ├── pages/       # Main application pages (ex. Home, PropertySearch, MarketAnalysis, ComparativeAnalysis, CompareMarkets, PropertyDetails)
│   │   ├── services/    # Frontend services for API interaction
│   │   └── styles/      # TailwindCSS configurations and other styles
__   └── public/      # Static assets
```



## Web Application Pages Overview

PropPulse features a intuitive web interface with the following main sections:

- **Home Page (`/`):** A welcoming landing page introducing PropPulse's capabilities and value proposition.
- **Property Search (`/search`):** Allows users to search for properties based on various criteria, view listings, and access detailed property information.
- **Property Details (`/property/:id`):** Displays comprehensive details for a selected property, including analytical insights, investment scores, and comparable properties.
- **Market Analysis (`/market-analysis`):** Provides in-depth analysis of real estate markets, showcasing trends, statistics, and key indicators to help users understand market dynamics.
- **Comparative Analysis (`/comparative-analysis`):** Enables users to compare multiple properties side-by-side, highlighting differences and similarities to aid in decision-making.
- **Compare Markets (`/compare-markets`):** Allows users to compare different real estate markets to identify investment opportunities or relocation insights.

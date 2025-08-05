# PropPulse - AI-Powered Real Estate Intelligence Platform

PropPulse is an advanced real estate intelligence platform that leverages AI and machine learning to help investors, agents, and homebuyers make data-driven decisions. The platform analyzes property listings in real-time, providing investment insights and recommendations.

## Features

- Real-time property listings from Realtor.com API
- AI-powered investment analysis using GPT-4
- Machine learning models for ROI prediction
- Interactive map visualization with Mapbox
- Property comparison and market trend analysis
- Investment scoring and rationale generation

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
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── api/      # API endpoints
│   │   ├── models/   # ML models
│   │   ├── services/ # Business logic
│   │   └── utils/    # Helper functions
│   └── tests/        # Backend tests
├── frontend/         # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   └── public/
└── docs/            # Documentation
```

## Setup Instructions

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for the interactive API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 
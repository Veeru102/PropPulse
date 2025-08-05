from typing import Optional
from app.services.market_data_service import MarketDataService
from app.services.property_analyzer import PropertyAnalyzer
from app.services.data_collector import DataCollector
from app.services.openai_service import OpenAIService

class ServiceManager:
    _instance: Optional['ServiceManager'] = None
    _market_data_service: Optional[MarketDataService] = None
    _property_analyzer: Optional[PropertyAnalyzer] = None
    _data_collector: Optional[DataCollector] = None
    _openai_service: Optional[OpenAIService] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_market_data_service(cls) -> MarketDataService:
        if cls._market_data_service is None:
            cls._market_data_service = MarketDataService()
        return cls._market_data_service

    @classmethod
    def get_property_analyzer(cls) -> PropertyAnalyzer:
        if cls._property_analyzer is None:
            cls._property_analyzer = PropertyAnalyzer()
        return cls._property_analyzer

    @classmethod
    def get_data_collector(cls) -> DataCollector:
        if cls._data_collector is None:
            cls._data_collector = DataCollector()
        return cls._data_collector

    @classmethod
    def get_openai_service(cls) -> OpenAIService:
        if cls._openai_service is None:
            cls._openai_service = OpenAIService()
        return cls._openai_service

    @classmethod
    def initialize_services(cls):
        """Pre-initialize all services to load data and models."""
        cls.get_market_data_service()
        cls.get_property_analyzer()
        cls.get_data_collector()
        cls.get_openai_service() 
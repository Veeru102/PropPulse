import logging
import sys
from pathlib import Path
from loguru import logger

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    logs_dir / "app.log",
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Create loggers dictionary
loggers = {
    'app': logger.bind(name='app'),
    'ml': logger.bind(name='ml'),
    'api': logger.bind(name='api'),
    'db': logger.bind(name='db'),
    'openai': logger.bind(name='openai')
}

def setup_logging():
    """Initialize and return the loggers dictionary"""
    return loggers 
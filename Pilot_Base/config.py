import os
from dotenv import load_dotenv, find_dotenv
from typing import Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Try to load .env file from multiple possible locations
env_locations = [
    PROJECT_ROOT / '.env',  # Root directory
    Path(__file__).parent / '.env',  # Pilot_Base directory
    Path.cwd() / '.env'  # Current working directory
]

# First, try to find .env file automatically
dotenv_path = find_dotenv()
if dotenv_path:
    logger.info(f"Found .env file automatically at: {dotenv_path}")
    load_dotenv(dotenv_path)
    env_loaded = True
else:
    # If automatic detection fails, try our specific locations
    env_loaded = False
    for env_path in env_locations:
        if env_path.exists():
            logger.info(f"Found .env file at: {env_path}")
            load_dotenv(env_path)
            env_loaded = True
            break

if not env_loaded:
    logger.warning("No .env file found in any of the expected locations")
    logger.warning("Expected locations:")
    for loc in env_locations:
        logger.warning(f"- {loc}")

class Config:
    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenRouter API key from environment variables"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        logger.debug(f"Loading API key from environment... Found: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            logger.error("No OPENROUTER_API_KEY found in environment variables")
            logger.debug("Available environment variables: %s", list(os.environ.keys()))
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Ensure the API key has the correct format
        if not api_key.startswith("sk-or-v1-"):
            logger.warning("API key format may be incorrect. Expected format: sk-or-v1-...")
        
        return api_key.strip()  # Remove any whitespace

    @staticmethod
    def get_openai_api_base_url() -> str:
        """Get OpenRouter API base URL"""
        url = os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1")
        logger.debug(f"Using API base URL: {url}")
        return url

    @staticmethod
    def get_openai_api_model() -> str:
        """Get OpenRouter API model"""
        model = os.getenv("OPENAI_API_MODEL", "openai/gpt-4o")
        logger.debug(f"Using model: {model}")
        return model

    @staticmethod
    def get_http_referer() -> str:
        """Get HTTP Referer for OpenRouter"""
        referer = os.getenv("HTTP_REFERER", "https://openrouter.ai")
        logger.debug(f"Using HTTP referer: {referer}")
        return referer

    @staticmethod
    def get_x_title() -> str:
        """Get X-Title for OpenRouter"""
        title = os.getenv("X_TITLE", "DataPilot")
        logger.debug(f"Using X-Title: {title}")
        return title.strip()  # Remove any whitespace 
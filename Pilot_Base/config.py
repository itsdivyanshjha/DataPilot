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

def load_env_files():
    """Load environment variables from all possible .env locations"""
    # First try to find .env file automatically
    env_path = find_dotenv()
    if env_path:
        logger.debug(f"Found .env file automatically at: {env_path}")
        load_dotenv(env_path)
        return True
        
    # If automatic detection fails, try specific locations
    env_locations = [
        PROJECT_ROOT / '.env',  # Root directory
        Path(__file__).parent / '.env',  # Pilot_Base directory
        Path.cwd() / '.env'  # Current working directory
    ]
    
    for env_path in env_locations:
        if env_path.exists():
            logger.debug(f"Loading .env file from: {env_path}")
            load_dotenv(env_path)
            return True
            
    logger.warning("No .env file found in any of the expected locations")
    return False

# Load environment variables
load_env_files()

class Config:
    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenRouter API key from environment variables"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        logger.debug(f"Loading API key... Found: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            logger.error("No OPENROUTER_API_KEY found in environment variables")
            logger.debug("Available environment variables: %s", list(os.environ.keys()))
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        api_key = api_key.strip()
        if not api_key.startswith("sk-or-v1-"):
            logger.warning(f"API key format may be incorrect. Expected format: sk-or-v1-... Got: {api_key[:10]}...")
        
        return api_key

    @staticmethod
    def get_openai_api_base_url() -> str:
        """Get OpenRouter API base URL"""
        url = os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1")
        logger.debug(f"Using API base URL: {url}")
        return url.strip()

    @staticmethod
    def get_openai_api_model() -> str:
        """Get OpenRouter API model"""
        model = os.getenv("OPENAI_API_MODEL", "openai/gpt-4")
        logger.debug(f"Using model: {model}")
        return model.strip()

    @staticmethod
    def get_http_referer() -> str:
        """Get HTTP Referer for OpenRouter"""
        referer = os.getenv("HTTP_REFERER", "https://openrouter.ai")
        logger.debug(f"Using HTTP referer: {referer}")
        return referer.strip()

    @staticmethod
    def get_x_title() -> str:
        """Get X-Title for OpenRouter"""
        title = os.getenv("X_TITLE", "DataPilot")
        logger.debug(f"Using X-Title: {title}")
        return title.strip() 
import os
from dotenv import load_dotenv
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_openrouter_connection():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("No API key found in environment variables")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "DataPilot",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        if response.status_code == 200:
            logger.info("Successfully connected to OpenRouter API")
            return True
        else:
            logger.error(f"Failed to connect to OpenRouter API. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to OpenRouter API: {str(e)}")
        return False

if __name__ == "__main__":
    test_openrouter_connection() 
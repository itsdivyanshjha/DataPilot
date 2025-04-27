import os
from dotenv import load_dotenv
import requests
import logging
from langchain_openai import ChatOpenAI
from config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_direct_api():
    """Test API directly with requests"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "DataPilot",
        "Content-Type": "application/json"
    }
    
    # Test models endpoint
    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
    print("\nDirect API Test (models endpoint):")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test chat completion endpoint
    chat_url = "https://openrouter.ai/api/v1/chat/completions"
    chat_data = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "Say hello"}]
    }
    
    response = requests.post(chat_url, headers=headers, json=chat_data)
    print("\nDirect API Test (chat endpoint):")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

def test_langchain_api():
    """Test API through LangChain"""
    try:
        llm = ChatOpenAI(
            model="openai/gpt-4",
            temperature=0,
            api_key=Config.get_openai_api_key(),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://openrouter.ai",
                "X-Title": "DataPilot"
            }
        )
        
        print("\nLangChain API Test:")
        result = llm.invoke("Say hello")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"LangChain Error: {str(e)}")

if __name__ == "__main__":
    print("Testing API connections...")
    print("API Key starts with:", os.getenv("OPENROUTER_API_KEY")[:10] if os.getenv("OPENROUTER_API_KEY") else "Not found")
    test_direct_api()
    test_langchain_api() 
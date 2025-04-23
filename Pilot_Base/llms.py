import streamlit as st
import logging
import os
import json

#------------------OPENAI------------
from langchain_openai import ChatOpenAI
from config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Get API key and verify it's not empty
    api_key = Config.get_openai_api_key()
    logger.debug(f"API Key starts with: {api_key[:10] if api_key else 'None'}")
    
    if not api_key or api_key.strip() == "":
        raise ValueError("API key is empty")
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": Config.get_http_referer(),
        "X-Title": Config.get_x_title(),
        "Content-Type": "application/json"
    }
    logger.debug(f"Using headers: {json.dumps(headers, indent=2)}")
    
    # Initialize OpenAI LLM through OpenRouter
    chatopenai_llm = ChatOpenAI(
        model=Config.get_openai_api_model(),
        temperature=0.0,
        api_key=api_key,
        base_url=Config.get_openai_api_base_url(),
        default_headers={
            "HTTP-Referer": Config.get_http_referer(),
            "X-Title": Config.get_x_title()
        }
    )
    logger.info("Successfully initialized ChatOpenAI with OpenRouter")
    logger.debug(f"Using base URL: {Config.get_openai_api_base_url()}")
    logger.debug(f"Using model: {Config.get_openai_api_model()}")
except Exception as e:
    logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Environment variables: {dict(os.environ)}")
    raise

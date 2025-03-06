import streamlit as st

#------------------OPENAI------------
from langchain_openai import ChatOpenAI
import os

# Initialize OpenAI LLM through OpenRouter
chatopenai_llm = ChatOpenAI(
    model="openai/gpt-4-turbo",  # Using GPT-4 Turbo through OpenRouter (with tool call support)
    temperature=0.0,
    openai_api_key="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",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://openrouter.ai",  # Required by OpenRouter
        "X-Title": "DataPilot"
    },
    tool_choice="auto"  # Added to support the new tools parameter
)

# -------------------GROQ--------------
from langchain_groq import ChatGroq

# Initialize the LLM
# groq_api_key = os.environ['GROQ_API_KEY'] # Setup your API Key
groq_api_key = st.secrets['GROQ_API_KEY'] # Setup your API Key
# groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768', temperature=0.0)
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192', temperature=0.0)

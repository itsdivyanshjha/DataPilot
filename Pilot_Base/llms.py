import streamlit as st

#------------------OPENAI------------
from langchain_openai import ChatOpenAI
import os

# Initialize OpenAI LLM through OpenRouter
chatopenai_llm = ChatOpenAI(
    model="openai/gpt-4-turbo",
    temperature=0.0,
    openai_api_key="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",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "DataPilot",
        "Authorization": "Bearer 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"
    }
)

# -------------------GROQ--------------
from langchain_groq import ChatGroq

# Initialize the LLM
groq_api_key = st.secrets['GROQ_API_KEY']
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192', temperature=0.0)

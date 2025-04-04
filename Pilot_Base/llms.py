import streamlit as st

#------------------OPENAI------------
from langchain_openai import ChatOpenAI
import os

# Initialize OpenAI LLM through OpenRouter
chatopenai_llm = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.0,
    openai_api_key="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",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "DataPilot",
        "Authorization": "Bearer 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"
    }
)

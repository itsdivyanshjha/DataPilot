from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # Change to OpenAI
import os

# Initialize OpenRouter LLM for chains
llm = ChatOpenAI(
    model="anthropic/claude-3-haiku",  # Using Claude 3 Haiku as it's more stable for this use case
    temperature=0.0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    default_headers={
        "HTTP-Referer": "localhost:8501",  # Replace with your site URL
        "X-Title": "DataPilot",  # Your application name
    }
)

#-------------------Additonal Info Summarization Chain____________________

summarization_chain_template = """
You are a Data Analysis Assistant. Your task is to provide a comprehensive summary of the dataset based on the information provided by the user.

1. Provide basic statistics about the dataset, including:
   - Number of entries
   - Number of columns
   - Data types of each column
   - Any missing values

2. Highlight key observations, such as:
   - Most common values in categorical columns
   - Range of numerical values
   - Any notable trends or patterns

3. Use bullet points for clarity and structure your response with headings.

*Information provided by the User:*
{additional_info_dataset}
"""

summarization_chain_prompt = PromptTemplate(
    template=summarization_chain_template,
    input_variables=['additional_info_dataset']
)

# Updated chain creation using the new recommended approach
summarization_chain = summarization_chain_prompt | llm

# The run method will still work the same way
def run(self, input_text):
    return summarization_chain.invoke({"additional_info_dataset": input_text}).content

#-------------------Query Enhancer Chain____________________

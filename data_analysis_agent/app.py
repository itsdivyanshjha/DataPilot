import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import os
import re

# Set Groq API key first, before any other imports
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Now import our local modules after setting the API key
from chains import summarization_chain
from utils import check_image_file_exists, read_image_file

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",  # Use the correct model name
    temperature=0.0,
)

st.set_page_config(
    page_title="DataPilot",
    page_icon="🧞",
)

# Session Variables
if 'additional_info_dataset' not in st.session_state:
    st.session_state.additional_info_dataset = ""

if 'summarized_dataset_info' not in st.session_state:
    st.session_state.summarized_dataset_info = ""

if 'img_list' not in st.session_state:
    st.session_state.img_list = []

if 'img_flag' not in st.session_state:
    st.session_state.img_flag = 0

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 2 Columns, one with the sidebar+chat area and other with features
main_col, right_col = st.columns([4, 1])

with main_col:

    st.title("Chat Area")

    with st.sidebar:
        st.title("DataPilot : Your Data Analysis Agent🧞")

        st.write("Upload your CSV File and ask your questions!")
        file = st.file_uploader("Select File", type=["csv"])
        st.divider()


    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Display the message content
            st.markdown(message["content"])
            
            # Check for images in the message
            images = check_image_file_exists(message["content"])
            if isinstance(images, list):
                for img_path in images:
                    st.image(img_path)
                    image_data = read_image_file(img_path)
                    st.download_button(
                        label=f"Download {os.path.basename(img_path)}",
                        data=image_data,
                        file_name=os.path.basename(img_path),
                        mime="image/png"
                    )

    # If the user uploads a CSV file
    if file is not None:

        # Read the CSV file in a pandas dataframe 'df'
        df = pd.read_csv(file)

        from prompts import PREFIX

        # Initialize the agent executor with the pandas agent
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="tool-calling",
            prefix=PREFIX.format(
                # Only keep last 3 messages to reduce context size
                chat_history="\n".join([f"{entry['role'].capitalize()}: {entry['content']}" 
                                      for entry in st.session_state.chat_history[-3:]]), 
                additional_info_dataset=st.session_state.summarized_dataset_info[:500]  # Limit additional info
            ),
            verbose=True,
            allow_dangerous_code=True
        )

        # import uvicorn
        # import subprocess
        # add_routes(fastapiapp, agent_executor, path="/pandas_agent")
        # subprocess.Popen(["uvicorn", "app:fastapiapp", "--host", "localhost", "--port", "8003"])

        # Get user input
        user_input = st.chat_input("Ask your question here...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Invoke the agent executor with the user's query
            res = agent_executor.invoke(user_input)
            
            # Process the response and check for multiple images
            response_text = res['output']
            image_tokens = re.findall(r'<image : r"(charts/[^"]+)">', response_text)
            
            # Format the response text
            formatted_response = res['output']
            
            # Add images to the session state if they exist
            if image_tokens:
                for img_path in image_tokens:
                    if os.path.isfile(img_path):
                        st.session_state.img_list.append(img_path)
            
            st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
            
            # Display chat messages after new input
            st.rerun()

with right_col:
    # st.write("Features")
    st.subheader("Features", divider="blue")

    page = st.radio("Select", ["None", "View Charts", "Add Dataset Info"])
    st.divider()

    if page == "View Charts":
        # st.write("Generated Charts")
        st.subheader('Generated Charts', divider='rainbow')
        if st.session_state.img_list:
            for img_path in st.session_state.img_list:
                st.image(img_path)
                image_data = read_image_file(img_path)
                st.download_button(label="Download",
                                   data=image_data,
                                   file_name=os.path.basename(img_path),
                                   mime="image/png")
        else:
            st.write("No images to display.")


    if page == "Add Dataset Info":
        additional_info_input = st.text_area("Additional Information About Dataset (Optional)",
                                             key='additional_info_input')

        if additional_info_input:
            if st.button("Update Info"):
                whitespace = " "
                st.session_state.additional_info_dataset += (whitespace + additional_info_input)
                summarization_chain_res = summarization_chain.run(additional_info_input)
                st.session_state.summarized_dataset_info = st.session_state.summarized_dataset_info + " " + summarization_chain_res

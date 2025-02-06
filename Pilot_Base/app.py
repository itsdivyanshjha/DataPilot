import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from data_ingestion import DataIngestionManager
from utils import (
    check_image_file_exists, 
    read_image_file, 
    generate_dataset_overview, 
    preprocess_dataframe,
    calculate_data_quality_score
)

# Set OpenRouter API key and configuration
OPENROUTER_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY  # LangChain uses this env variable

from chains import summarization_chain

def handle_groq_error(error):
    """Handle Groq API errors gracefully"""
    if "413" in str(error):
        st.error("The query is too complex. Please try a simpler question or break it down into smaller parts.")
    elif "rate_limit_exceeded" in str(error):
        st.error("Rate limit exceeded. Please wait a moment before trying again.")
    else:
        st.error(f"An error occurred: {str(error)}")

# Initialize OpenRouter LLM
llm = ChatOpenAI(
    model="anthropic/claude-3-haiku",  # Using Claude 3 Haiku as it's more stable for this use case
    temperature=0.0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "localhost:8501",
        "X-Title": "DataPilot",
    }
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
    st.title("Data Analysis Dashboard")

    with st.sidebar:
        st.title("DataPilot : Your Data Analysis Agent🧞")
        
        # File upload section with multiple file support
        st.write("Upload your data file")
        supported_types = ['csv', 'json', 'xlsx', 'xls', 'xml', 'db', 'sqlite', 'yaml', 'yml']
        file = st.file_uploader("Select File", type=supported_types)
        
        if file is not None:
            file_type = file.name.split('.')[-1].lower()
            
            try:
                with st.spinner('Reading file...'):
                    # Use the DataIngestionManager to read the file
                    data = DataIngestionManager.read_file(file, file_type)
                    
                    # Handle multiple tables from SQLite
                    if isinstance(data, dict):
                        selected_table = st.selectbox("Select Table", list(data.keys()))
                        df = data[selected_table]
                    else:
                        df = data
                    
                    # Display basic file info
                    file_info = DataIngestionManager.get_file_info(df)
                    st.success(f"File loaded successfully: {file.name}")
                    with st.expander("File Information"):
                        st.write(f"Rows: {file_info['rows']}")
                        st.write(f"Columns: {file_info['columns']}")
                        st.write(f"Memory Usage: {file_info['memory_usage']}")
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.stop()
        
        st.divider()

    # If the user uploads a file
    if file is not None:
        # Data preprocessing and overview generation
        with st.spinner('Analyzing dataset...'):
            df = preprocess_dataframe(df)
            dataset_overview = generate_dataset_overview(df)
        
        # Dataset Insights Section
        st.header("📊 Dataset Insights")
        
        # Quick Stats in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{dataset_overview['basic_info']['rows']:,}")
        with col2:
            st.metric("Total Features", dataset_overview['basic_info']['columns'])
        with col3:
            st.metric("Memory Usage", dataset_overview['basic_info']['memory_usage'])
        
        # Trend Analysis Section
        st.subheader("📈 Key Trends and Patterns")
        
        # Numeric Columns Analysis
        if dataset_overview['numeric_columns']:
            numeric_df = df[dataset_overview['numeric_columns']]
            
            # Calculate correlations
            corr = numeric_df.corr()
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i,j]) > 0.5:  # Threshold for strong correlation
                        strong_corr.append(f"Strong {corr.iloc[i,j]:.2f} correlation between {corr.columns[i]} and {corr.columns[j]}")
            
            if strong_corr:
                st.write("🔍 Notable Correlations:")
                for corr in strong_corr[:3]:  # Show top 3 correlations
                    st.write(f"- {corr}")
        
        # Categorical Analysis
        if dataset_overview['categorical_columns']:
            st.write("📊 Category Distribution Highlights:")
            for col in dataset_overview['categorical_columns'][:3]:  # Show top 3 categorical columns
                value_counts = df[col].value_counts()
                st.write(f"- {col}: Dominated by '{value_counts.index[0]}' ({(value_counts.iloc[0]/len(df)*100):.1f}% of data)")
        
        # Data Quality Insights
        st.subheader("🔍 Data Quality Overview")
        quality_metrics, overall_score = calculate_data_quality_score(df)
        st.progress(overall_score/100, text=f"Overall Data Quality Score: {overall_score:.1f}%")
        
        # Technical Details in Expandable Section
        st.subheader("📊 Detailed Analysis")
        tab1, tab2, tab3 = st.tabs(["Column Details", "Missing Values", "Statistical Summary"])

        with tab1:
            # Create a clean dataframe for column overview
            col_overview = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique(),
                'Missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_overview, use_container_width=True)

        with tab2:
            # Create missing values visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            
            if not missing_data.empty:
                sns.barplot(data=missing_data, x='Column', y='Percentage')
                plt.xticks(rotation=45, ha='right')
                plt.title('Missing Values by Column (%)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No missing values found in the dataset!")

        with tab3:
            # Numeric Columns
            if len(df.select_dtypes(include=['int64', 'float64']).columns) > 0:
                st.markdown("#### Numeric Columns")
                st.dataframe(df.describe(), use_container_width=True)
            
            # Categorical Columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()  # Convert to list
            if categorical_cols:  # Simple check if list is not empty
                st.markdown("#### Categorical Columns")
                
                # Create tabs for each categorical column
                cat_tabs = st.tabs(categorical_cols)
                for col, tab in zip(categorical_cols, cat_tabs):
                    with tab:
                        try:
                            value_counts = df[col].value_counts().head(10)
                            
                            # Create visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
                            plt.xticks(rotation=45, ha='right')
                            plt.title(f'Top 10 Values in {col}')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show value counts table
                            st.markdown("##### Distribution Table")
                            dist_df = pd.DataFrame({
                                'Value': value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': (value_counts.values / len(df) * 100).round(2)
                            })
                            st.dataframe(dist_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error processing column {col}: {str(e)}")

        # Quick Actions Section
        st.markdown("### 🔍 Quick Actions")
        quick_action = st.selectbox(
            "Choose an analysis type:",
            ["Select an action", "Correlation Matrix", "Distribution Plots", "Summary Statistics", "Data Quality Report"]
        )
        
        if quick_action == "Correlation Matrix":
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
                st.pyplot(fig)
                plt.close()
        
        elif quick_action == "Distribution Plots":
            col = st.selectbox("Select column for distribution:", df.columns)
            
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if df[col].dtype in ['int64', 'float64']:
                    # For numeric columns
                    sns.histplot(data=df, x=col, kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    
                    # Add summary statistics
                    summary_stats = df[col].describe()
                    stats_text = (f"Mean: {summary_stats['mean']:.2f}\n"
                                 f"Median: {summary_stats['50%']:.2f}\n"
                                 f"Std Dev: {summary_stats['std']:.2f}")
                    
                    # Add text box with statistics
                    plt.text(0.95, 0.95, stats_text,
                            transform=ax.transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                else:
                    # For categorical columns
                    value_counts = df[col].value_counts()
                    total_count = len(df)
                    
                    # Calculate percentages
                    percentages = (value_counts / total_count * 100).round(1)
                    
                    # Plot top 15 categories if there are more
                    if len(value_counts) > 15:
                        value_counts = value_counts.head(15)
                        percentages = percentages.head(15)
                        ax.set_title(f'Top 15 Categories in {col}')
                    else:
                        ax.set_title(f'Distribution of {col}')
                    
                    # Create the bar plot
                    bars = ax.bar(range(len(value_counts)), value_counts.values)
                    
                    # Customize x-axis labels
                    plt.xticks(range(len(value_counts)), 
                              [str(x)[:20] + '...' if len(str(x)) > 20 else str(x) for x in value_counts.index],
                              rotation=45, ha='right')
                    
                    # Add value labels on top of bars
                    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:,.0f}\n({percentage}%)',
                               ha='center', va='bottom')
                
                # Adjust layout
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
                plt.close()
                
                # Display summary table
                st.markdown("#### Summary Statistics")
                if df[col].dtype in ['int64', 'float64']:
                    st.dataframe(df[col].describe().round(2))
                else:
                    summary_df = pd.DataFrame({
                        'Category': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': percentages
                    })
                    st.dataframe(summary_df)
                    
            except Exception as e:
                st.error(f"Error generating distribution plot: {str(e)}")
                st.info("Displaying basic value counts instead:")
                st.dataframe(df[col].value_counts().head(15))
        
        elif quick_action == "Summary Statistics":
            st.subheader("Summary Statistics", divider="rainbow")
            st.dataframe(df.describe(), use_container_width=True)
        
        elif quick_action == "Data Quality Report":
            st.subheader("Data Quality Report", divider="rainbow")
            st.write("This feature is not implemented in the current version.")

        from prompts import PREFIX

        # Initialize the agent executor with the pandas agent
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="tool-calling",
            prefix=PREFIX.format(
                chat_history="\n".join([
                    f"{entry['role'].capitalize()}: {entry['content'][:200]}..." 
                    for entry in st.session_state.chat_history[-2:]
                ]), 
                additional_info_dataset=st.session_state.summarized_dataset_info[:300]
            ),
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=30,
            allow_dangerous_code=True
        )

        # import uvicorn
        # import subprocess
        # add_routes(fastapiapp, agent_executor, path="/pandas_agent")
        # subprocess.Popen(["uvicorn", "app:fastapiapp", "--host", "localhost", "--port", "8003"])

        # Chat Interface
        st.header("💬 Ask Questions About Your Data")
        
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
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
        
        # Chat input
        user_input = st.chat_input("Ask about your data...")
        
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

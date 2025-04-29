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
    calculate_data_quality_score,
    format_large_number,
    analyze_categorical_counts,
    analyze_data,
    create_generalized_visualization,
    analyze_distribution
)
from chains import summarization_chain
from prompts import get_prefix
from langchain.tools import Tool
from langchain.agents import AgentType
from rag_manager import RAGManager
from config import Config
import tempfile
import logging
import warnings
import matplotlib.font_manager as fm
import time
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from st_chat_message import message

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_experimental')

# Configure matplotlib to use a font that supports emojis
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def initialize_llm():
    """Initialize the LLM with proper configuration"""
    try:
        api_key = Config.get_openai_api_key()
        logger.debug(f"Initializing LLM with API key starting with: {api_key[:10] if api_key else 'None'}")
        
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is empty")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": Config.get_http_referer(),
            "X-Title": Config.get_x_title(),
            "Content-Type": "application/json"
        }
        
        logger.debug(f"Using headers: {headers}")
        logger.debug(f"Using base URL: {Config.get_openai_api_base_url()}")
        logger.debug(f"Using model: {Config.get_openai_api_model()}")
        
        return ChatOpenAI(
            model=Config.get_openai_api_model(),
            temperature=0.0,
            openai_api_key=api_key,
            base_url=Config.get_openai_api_base_url(),
            default_headers=headers
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Environment variables: {dict(os.environ)}")
        raise

# Initialize RAG Manager with OpenAI embeddings
rag_manager = RAGManager()

st.set_page_config(
    page_title="DataPilot",
    page_icon="📊",
    layout="wide"
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

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

if 'df' not in st.session_state:
    st.session_state.df = None

# 2 Columns, one with the sidebar+chat area and other with features
main_col, right_col = st.columns([4, 1])

with main_col:
    st.title("DataPilot Analysis Dashboard: ")

    with st.sidebar:
        st.title("DataPilot 📊👨🏻‍✈️")
        st.markdown("""
        ### Welcome to Your Data Analysis Journey! 📊
        
        DataPilot helps you explore and understand your data with just a few clicks.
        No coding required - just upload your data and start exploring!
        """)
        
        # File upload section with enhanced UI
        st.markdown("### 📁 Upload Your Dataset:")
        st.markdown("""
        Supported formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - JSON files (.json)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'json', 'xlsx', 'xls', 'xml', 'db'],
            help="Upload your dataset to start analyzing!"
        )
        
        if uploaded_file is not None:
            try:
                # Get file extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Read the file with a loading spinner and progress message
                with st.spinner('Reading your data file... Please wait a moment.'):
                    df = DataIngestionManager.read_file(uploaded_file, file_extension)
                
                # Store the DataFrame in session state
                st.session_state.df = df
                
                # Initialize RAG system with progress updates
                with st.spinner('Setting up advanced analysis capabilities...(RAG)'):
                    try:
                        rag_manager.clear()
                        rag_manager.create_knowledge_base(df)
                        st.session_state.rag_initialized = True
                        st.success("✨ Advanced analysis (RAG)system ready!")
                    except Exception as rag_error:
                        st.warning("⚠️ Basic analysis mode active (some advanced features may be limited)")
                        st.session_state.rag_initialized = False
                
                # Display file information in a more organized way
                st.markdown("### 📊 Dataset Overview")
                file_info = DataIngestionManager.get_file_info(df)
                
                # Use columns for better organization
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{file_info['rows']:,}")
                    st.metric("Memory Usage", file_info['memory_usage'])
                with col2:
                    st.metric("Columns", file_info['columns'])
                
                # Data preview in an expandable section
                with st.expander("👀 Preview Your Data", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                    
                # Quick tips in an expandable section
                with st.expander("💡 Quick Tips", expanded=False):
                    st.markdown("""
                    - Use the **Quick Actions** section for instant insights
                    - Try the **Chat** feature to ask questions about your data
                    - Check the **Data Quality Report** to understand your data better
                    - Use **Visualizations** to explore patterns and trends
                    """)
                
            except Exception as e:
                st.error("⚠️ Oops! Something went wrong while reading your file.")
                st.info("🔍 Common issues to check:")
                st.markdown("""
                - Is the file empty?
                - Are the column headers correct?
                - Is the file using a standard format?
                - Is the file encoding standard (UTF-8, Latin-1)?
                
                **Error details:** {}
                """.format(str(e)))
                st.session_state.df = None
                st.session_state.rag_initialized = False
        
        st.divider()
        
        # Add helpful resources section
        with st.expander("📚 Help & Resources", expanded=False):
            st.markdown("""
            ### Need Help?
            
            - 📖 **Data Formats**: Learn about [supported data formats](https://pandas.pydata.org/docs/user_guide/io.html)
            - 🎯 **Analysis Tips**: Check out our [data analysis guide](https://www.analyticsvidhya.com/blog/2021/08/complete-guide-to-data-analysis/)
            - 🔍 **Visualization**: Explore [data visualization best practices](https://www.tableau.com/learn/articles/data-visualization-tips)
            
            ### Common Questions
            
            1. **What file formats are supported?**
               - Most common data file formats including CSV, Excel, JSON, and more
            
            2. **How big can my file be?**
               - For optimal performance, files under 100MB are recommended
            
            3. **What can I do with my data?**
               - Explore trends and patterns
               - Create visualizations
               - Get statistical insights
               - Ask questions in natural language
            """)

    # If the user uploads a file
    if uploaded_file is not None:
        # Data preprocessing and overview generation
        with st.spinner('Analyzing your dataset...'):
            df = preprocess_dataframe(df)
            dataset_overview = generate_dataset_overview(df)
        
        # Main Dashboard Header
        st.markdown("""
        # 📊 Data Analysis Dashboard
        Your interactive data analysis companion
        """)
        
        # Dataset Insights Section with enhanced visuals
        st.markdown("## 📈 Dataset Insights")
        
        # Quick Stats in columns with icons and better formatting
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📝 Total Records",
                f"{dataset_overview['basic_info']['rows']:,}",
                help="Total number of rows in your dataset"
            )
        with col2:
            st.metric(
                "🎯 Features",
                dataset_overview['basic_info']['columns'],
                help="Number of columns in your dataset"
            )
        with col3:
            st.metric(
                "💾 Memory Usage",
                dataset_overview['basic_info']['memory_usage'],
                help="Amount of memory used by your dataset"
            )

        # Data Overview Section
        with st.expander("📊 Data Overview", expanded=False):
            # Data Types Overview
            st.markdown("### Column Types Distribution")
            col_types = df.dtypes.value_counts()
            
            # Create two columns for the data types summary
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Create a smaller pie chart
                fig, ax = plt.subplots(figsize=(4, 3))
                colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
                plt.pie(
                    col_types.values,
                    labels=col_types.index,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                plt.title("Column Types", pad=10, fontsize=10)
                st.pyplot(fig, use_container_width=False)
                plt.close()
            
            with col2:
                # Display column type counts in a clean format
                st.markdown("#### Column Counts")
                for dtype, count in col_types.items():
                    st.markdown(f"- **{dtype}**: {count} columns")
            
            # Sample Data Preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

        # Data Quality Section
        with st.expander("✨ Data Quality Assessment", expanded=False):
            quality_metrics, overall_score = calculate_data_quality_score(df)
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Completeness", f"{quality_metrics['completeness']:.1f}%",
                         help="Percentage of non-null values")
                st.metric("Consistency", f"{quality_metrics['consistency']:.1f}%",
                         help="Percentage of columns with expected data types")
            with col2:
                st.metric("Uniqueness", f"{quality_metrics['uniqueness']:.1f}%",
                         help="Average ratio of unique values")
                st.metric("Data Quality Score", f"{overall_score:.1f}%",
                         help="Overall quality score based on all metrics")

        # Quick Actions Section
        st.markdown("## 🔍 Quick Actions")
        st.markdown("Choose an analysis type to explore your data:")
        
        # Analysis Type Selection
        quick_action = st.selectbox(
            "Choose Analysis Type:",
            ["Distribution Plots", "Box Plots", "Bar Charts", "Correlation Matrix", "Scatter Plots", "Time Series Plots", "Summary Statistics"],
            help="Select the type of analysis you want to perform"
        )
        
        # Show description for selected analysis
        analysis_descriptions = {
            "Distribution Plots": "Visualize how values are distributed in a column. This helps you understand the spread, shape, and potential outliers in your numerical data. It shows if your data is normally distributed, skewed, or has multiple peaks.",
            "Box Plots": "Identify outliers and understand data spread using quartiles. Box plots show the median, quartiles, and outliers, making it easy to spot unusual values and compare distributions across groups.",
            "Bar Charts": "Compare frequencies or counts across different categories. Perfect for showing the distribution of categorical data, market share analysis, or comparing values across different groups.",
            "Correlation Matrix": "Discover relationships between numeric columns using a color-coded matrix. Red indicates positive correlations, blue indicates negative correlations, with darker colors showing stronger relationships.",
            "Scatter Plots": "Explore relationships between two numeric columns. This helps identify patterns, trends, and potential correlations between variables. Useful for finding linear relationships or clusters in your data.",
            "Time Series Plots": "Analyze trends and patterns over time. Visualize how values change across different time periods, identify seasonality, trends, and potential anomalies in temporal data.",
            "Summary Statistics": "Get detailed statistical insights including mean, median, standard deviation, and quartiles for numeric data, plus frequency distributions for categorical variables."
        }
        
        if quick_action in analysis_descriptions:
            st.info(analysis_descriptions[quick_action])

        try:
            if quick_action == "Distribution Plots":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column for distribution plot:", numeric_cols)
                    if selected_col:
                        plt.figure(figsize=(10, 6))
                        sns.histplot(data=df, x=selected_col, kde=True)
                        plt.title(f"Distribution of {selected_col}")
                        plt.xlabel(selected_col)
                        plt.ylabel("Count")
                        
                        # Add summary statistics
                        mean = df[selected_col].mean()
                        median = df[selected_col].median()
                        std = df[selected_col].std()
                        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
                        plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
                        plt.legend()
                        
                        chart_path = "charts/distribution_plot.png"
                        plt.savefig(chart_path, bbox_inches='tight')
                        plt.close()
                        
                        st.image(chart_path)
                        st.write(f"Standard Deviation: {std:.2f}")
                else:
                    st.warning("No numeric columns found for distribution plot.")

            elif quick_action == "Box Plots":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column for box plot:", numeric_cols)
                    if selected_col:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(data=df, y=selected_col)
                        plt.title(f"Box Plot of {selected_col}")
                        
                        chart_path = "charts/box_plot.png"
                        plt.savefig(chart_path, bbox_inches='tight')
                        plt.close()
                        
                        st.image(chart_path)
                        
                        # Add summary statistics
                        q1 = df[selected_col].quantile(0.25)
                        q3 = df[selected_col].quantile(0.75)
                        iqr = q3 - q1
                        st.write(f"Summary Statistics for {selected_col}:")
                        st.write(f"- Q1 (25th percentile): {q1:.2f}")
                        st.write(f"- Median: {df[selected_col].median():.2f}")
                        st.write(f"- Q3 (75th percentile): {q3:.2f}")
                        st.write(f"- IQR: {iqr:.2f}")
                else:
                    st.warning("No numeric columns found for box plot.")

            elif quick_action == "Bar Charts":
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    selected_col = st.selectbox("Select column for bar chart:", categorical_cols)
                    if selected_col:
                        value_counts = df[selected_col].value_counts()
                        total = len(df)
                        
                        # Limit to top 20 categories if there are too many
                        if len(value_counts) > 20:
                            value_counts = value_counts.head(20)
                        
                        plt.figure(figsize=(12, 6))
                        bars = plt.bar(range(len(value_counts)), value_counts.values,
                                     color=sns.color_palette("husl", len(value_counts)))
                        
                        plt.title(f"Distribution of {selected_col}")
                        plt.xlabel(selected_col)
                        plt.ylabel("Count")
                        
                        # Rotate x-axis labels for better readability
                        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
                        
                        # Add value labels on top of bars
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentage = (height / total) * 100
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height)}\n({percentage:.1f}%)',
                                    ha='center', va='bottom')
                        
                        plt.tight_layout()
                        chart_path = "charts/bar_chart.png"
                        plt.savefig(chart_path, bbox_inches='tight')
                        plt.close()
                        
                        st.image(chart_path)
                        st.write(f"Most common category: {value_counts.index[0]} ({(value_counts.values[0]/total*100):.1f}%)")
                        if len(value_counts) > 20:
                            st.info("Note: Showing top 20 categories only.")
                else:
                    st.warning("No categorical columns found for bar chart.")

            elif quick_action == "Correlation Matrix":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) >= 2:
                    plt.figure(figsize=(10, 8))
                    correlation_matrix = df[numeric_cols].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                    plt.title("Correlation Matrix")
                    
                    chart_path = "charts/correlation_matrix.png"
                    plt.savefig(chart_path, bbox_inches='tight')
                    plt.close()
                    
                    st.image(chart_path)
                    
                    # Add correlation interpretation
                    st.write("### Correlation Interpretation:")
                    st.write("- Values close to 1 (dark red) indicate strong positive correlation")
                    st.write("- Values close to -1 (dark blue) indicate strong negative correlation")
                    st.write("- Values close to 0 indicate weak or no correlation")
                else:
                    st.warning("Need at least 2 numeric columns for correlation matrix.")

            elif quick_action == "Scatter Plots":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select X-axis column:", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Select Y-axis column:", numeric_cols)
                    
                    if x_col and y_col:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(df[x_col], df[y_col],
                                  alpha=0.5,
                                  c=df[y_col],  # Color based on y-values
                                  cmap='viridis',  # Use a colormap
                                  edgecolors='white',
                                  s=50)
                        
                        plt.colorbar(label=y_col)
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        correlation = df[x_col].corr(df[y_col])
                        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}',
                                transform=plt.gca().transAxes,
                                bbox=dict(facecolor='white', alpha=0.8))
                        
                        chart_path = "charts/scatter_plot.png"
                        plt.savefig(chart_path, bbox_inches='tight')
                        plt.close()
                        
                        st.image(chart_path)
                        
                        st.write(f"### Relationship between {x_col} and {y_col}:")
                        if abs(correlation) > 0.7:
                            st.write(f"Strong {'positive' if correlation > 0 else 'negative'} correlation detected.")
                        elif abs(correlation) > 0.3:
                            st.write(f"Moderate {'positive' if correlation > 0 else 'negative'} correlation detected.")
                        else:
                            st.write("Weak or no correlation detected.")
                else:
                    st.warning("Need at least 2 numeric columns for scatter plot.")

            elif quick_action == "Time Series Plots":
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    date_col = st.selectbox("Select date column:", datetime_cols)
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numeric_cols) > 0:
                        value_col = st.selectbox("Select value column:", numeric_cols)
                        if date_col and value_col:
                            plt.figure(figsize=(12, 6))
                            plt.plot(df[date_col], df[value_col],
                                   color='#2ecc71',
                                   linewidth=2)
                            
                            plt.xlabel(date_col)
                            plt.ylabel(value_col)
                            plt.title(f"Time Series: {value_col} over {date_col}")
                            plt.xticks(rotation=45)
                            plt.grid(True, linestyle='--', alpha=0.7)
                            
                            # Add trend line
                            z = np.polyfit(range(len(df[date_col])), df[value_col], 1)
                            p = np.poly1d(z)
                            plt.plot(df[date_col], p(range(len(df[date_col]))),
                                    "r--", alpha=0.8, label="Trend Line")
                            plt.legend()
                            
                            chart_path = "charts/time_series.png"
                            plt.savefig(chart_path, bbox_inches='tight')
                            plt.close()
                            
                            st.image(chart_path)
                            
                            # Add summary statistics
                            st.write("### Time Series Summary:")
                            st.write(f"- Start date: {df[date_col].min()}")
                            st.write(f"- End date: {df[date_col].max()}")
                            st.write(f"- Average {value_col}: {df[value_col].mean():.2f}")
                            st.write(f"- Trend: {'Increasing' if z[0] > 0 else 'Decreasing'}")
                    else:
                        st.warning("No numeric columns found for time series plot.")
                else:
                    st.warning("No datetime columns found for time series plot.")

            elif quick_action == "Summary Statistics":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    st.write("### Numeric Columns Summary")
                    st.write(df[numeric_cols].describe())
                    
                    st.write("### Categorical Columns Summary")
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        for col in categorical_cols:
                            st.write(f"\n**{col}**")
                            value_counts = df[col].value_counts()
                            total = len(df)
                            
                            # Display top 5 categories with percentages
                            summary_df = pd.DataFrame({
                                'Count': value_counts.head(),
                                'Percentage': (value_counts.head() / total * 100).round(2)
                            })
                            summary_df['Percentage'] = summary_df['Percentage'].astype(str) + '%'
                            st.write(summary_df)
                else:
                    st.warning("No numeric columns found for summary statistics.")

        except Exception as e:
            st.error(f"An error occurred while creating the visualization: {str(e)}")
            st.warning("Please try a different selection or check your data format.")

        # AI Assistant Section
        st.markdown("## 🤖 AI Data Assistant")
        st.markdown("""
        Ask questions about your data in natural language. The AI will help you:
        - Analyze trends and patterns
        - Create visualizations
        - Generate insights
        - Answer specific questions
        """)
        
        # Chat Interface
        st.markdown("### 💬 Chat with Your Data")

        # Display chat messages from history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
            # Display any images generated in the response
            if message["role"] == "assistant":
                images = check_image_file_exists(message["content"])
                if isinstance(images, list):
                    for img_path in images:
                        with st.chat_message("assistant"):
                            st.image(img_path)
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown("*Click to download this visualization:*")
                            with col2:
                                image_data = read_image_file(img_path)
                                st.download_button(
                                    label="Download",
                                    data=image_data,
                                    file_name=os.path.basename(img_path),
                                    mime="image/png"
                                )

        # Chat input at the bottom
        user_input = st.chat_input("Ask anything about your data...", key="user_chat_input")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            try:
                # Initialize LLM if not already done
                if 'llm' not in st.session_state:
                    st.session_state.llm = initialize_llm()

                # Get dataset context using RAG
                rag_context = []
                if st.session_state.rag_initialized:
                    try:
                        dataset_info = rag_manager.get_dataset_info()
                        if dataset_info:
                            rag_context.append({
                                'content': f"Dataset Info: {json.dumps(dataset_info, indent=2)}",
                                'metadata': {'type': 'dataset_info'}
                            })
                        
                        query_context = rag_manager.get_relevant_context(user_input)
                        if query_context:
                            rag_context.extend(query_context)
                    except Exception as rag_error:
                        pass
                
                # Prepare context-enhanced input
                context_text = "\n\n".join([
                    ctx['content'] for ctx in rag_context
                ]) if rag_context else ""
                
                # Create the agent with tools
                agent_executor = create_pandas_dataframe_agent(
                    st.session_state.llm,
                    df,
                    agent_type="openai-tools",
                    prefix=get_prefix(
                        chat_history="\n".join([
                            f"{entry['role'].capitalize()}: {entry['content'][:200]}..." 
                            for entry in st.session_state.chat_history[-2:]
                        ]), 
                        additional_info_dataset=context_text
                    ),
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=20,
                    max_execution_time=180,
                    allow_dangerous_code=True,
                    extra_tools=[]
                )

                # Execute the agent
                with st.spinner("Thinking..."):
                    result = agent_executor.invoke(user_input)
                    
                    # Process and format the response
                    if isinstance(result, dict):
                        formatted_response = "\n".join([f"{v}" for v in result.values() if v != user_input])
                    elif isinstance(result, str):
                        formatted_response = result
                        # Clean up the response
                        formatted_response = re.sub(r'^.*?\?', '', formatted_response, flags=re.DOTALL).strip()
                        formatted_response = re.sub(r'\*\*input:\*\*.*?\n', '', formatted_response, flags=re.DOTALL)
                        formatted_response = re.sub(r'\*\*output:\*\*', '', formatted_response)
                        formatted_response = formatted_response.replace('input:', '').replace('output:', '')
                        formatted_response = re.sub(r'{.*?}', '', formatted_response)
                        formatted_response = formatted_response.strip()
                        # Remove any leading phrases and question repetition
                        formatted_response = re.sub(r'^(The |Based on |According to |Here\'s |)*(answer is|analysis shows|data shows|data indicates|we can see that|I can tell you that|)', '', formatted_response, flags=re.IGNORECASE).strip()
                    else:
                        formatted_response = str(result)

                    # Add assistant's response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": formatted_response
                    })

            except Exception as e:
                # Add error message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I encountered an error. Please try rephrasing your question."
                })
            
            # Rerun the app to update the chat display
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

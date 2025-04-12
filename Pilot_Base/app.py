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
    create_generalized_visualization
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_experimental')

# Configure matplotlib to use a font that supports emojis
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Initialize LangChain's ChatOpenAI
llm = ChatOpenAI(
    model=Config.get_openai_api_model(),
    temperature=0.0,
    openai_api_key=Config.get_openai_api_key(),
    base_url=Config.get_openai_api_base_url(),
    default_headers={
        "Authorization": f"Bearer {Config.get_openai_api_key()}",
        "HTTP-Referer": Config.get_http_referer(),
        "X-Title": Config.get_x_title(),
        "Content-Type": "application/json"
    }
)

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
    st.title("Data Analysis Dashboard")

    with st.sidebar:
        st.title("DataPilot : Your Dataset Analysis Agent🧞")
        
        # File upload section with multiple file support
        st.write("Upload your data file")
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'json', 'xlsx', 'xls', 'xml', 'db', 'sqlite', 'yaml', 'yml']
        )
        
        if uploaded_file is not None:
            try:
                # Get file extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Read the file
                with st.spinner('Reading file...'):
                    df = DataIngestionManager.read_file(uploaded_file, file_extension)
                
                # Store the DataFrame in session state
                st.session_state.df = df
                
                # Clear previous RAG knowledge base and create new one
                with st.spinner('Initializing RAG system...'):
                    try:
                        rag_manager.clear()
                        rag_manager.create_knowledge_base(df)
                        st.session_state.rag_initialized = True
                        st.success("RAG system initialized successfully!")
                    except Exception as rag_error:
                        st.warning(f"RAG system initialization failed: {str(rag_error)}")
                        st.info("You can still use basic data analysis features.")
                        st.session_state.rag_initialized = False
                
                # Display file information
                file_info = DataIngestionManager.get_file_info(df)
                st.success("File uploaded successfully!")
                st.write(f"Rows: {file_info['rows']:,}")
                st.write(f"Columns: {file_info['columns']:,}")
                st.write(f"Memory Usage: {file_info['memory_usage']}")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please check that:")
                st.markdown("""
                - The file is not empty
                - The file has proper column headers
                - The file uses a standard delimiter (comma, semicolon, tab, or pipe)
                - The file encoding is standard (UTF-8, Latin-1, etc.)
                """)
                st.session_state.df = None
                st.session_state.rag_initialized = False
        
        st.divider()

    # If the user uploads a file
    if uploaded_file is not None:
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
        st.markdown("""
The Data Quality Score is calculated based on three key metrics:

1. **Completeness (40% weight)**: Measures how much data is not null/missing
2. **Uniqueness (30% weight)**: Evaluates the diversity of values in each column
3. **Consistency (30% weight)**: Checks if data types match expected formats

Additional metrics include:
- Null value percentage
- Duplicate row percentage
- Data type distribution
""")
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
            try:
                # Get numeric columns only for summary statistics
                numeric_df = df.select_dtypes(include=['int64', 'float64'])
                
                if not numeric_df.empty:
                    # Calculate and display basic statistics for each numeric column
                    st.markdown("### Numeric Columns Analysis")
                    
                    for column in numeric_df.columns:
                        st.markdown(f"#### {column}")
                        col_data = numeric_df[column]
                        
                        # Create a clean statistics dictionary
                        stats = {
                            "Count": len(col_data),
                            "Mean": col_data.mean(),
                            "Std": col_data.std(),
                            "Min": col_data.min(),
                            "25%": col_data.quantile(0.25),
                            "50% (Median)": col_data.median(),
                            "75%": col_data.quantile(0.75),
                            "Max": col_data.max()
                        }
                        
                        # Convert to DataFrame for better display
                        stats_df = pd.DataFrame({
                            "Statistic": list(stats.keys()),
                            "Value": [f"{v:.2f}" if isinstance(v, (float, int)) else str(v) for v in stats.values()]
                        })
                        
                        # Display the statistics
                        st.table(stats_df)
                        
                        # Add a visual separator
                        st.markdown("---")
                    
                    # Display correlation matrix if there are multiple numeric columns
                    if len(numeric_df.columns) > 1:
                        st.markdown("### Correlation Matrix")
                        corr_matrix = numeric_df.corr().round(2)
                        st.table(corr_matrix)
                else:
                    st.info("No numeric columns found in the dataset for summary statistics.")
                    
                # Display categorical columns summary if any exist
                categorical_df = df.select_dtypes(include=['object', 'category'])
                if not categorical_df.empty:
                    st.markdown("### Categorical Columns Summary")
                    for col in categorical_df.columns:
                        st.markdown(f"#### {col}")
                        value_counts = categorical_df[col].value_counts()
                        total = len(categorical_df[col])
                        
                        # Create summary DataFrame
                        cat_summary = pd.DataFrame({
                            "Value": value_counts.index[:5],  # Show top 5 categories
                            "Count": value_counts.values[:5],
                            "Percentage": (value_counts.values[:5] / total * 100).round(2)
                        })
                        
                        st.table(cat_summary)
                        if len(value_counts) > 5:
                            st.info(f"Showing top 5 out of {len(value_counts)} unique values")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"Error displaying summary statistics: {str(e)}")
                st.info("Please try a different analysis option or check your data format.")
        
        elif quick_action == "Data Quality Report":
            st.subheader("Data Quality Report", divider="rainbow")
            st.write("This feature is not implemented in the current version.")

        # Initialize the agent executor with the pandas agent and RAG context
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="openai-tools",
            prefix=get_prefix(
                chat_history="\n".join([
                    f"{entry['role'].capitalize()}: {entry['content'][:200]}..." 
                    for entry in st.session_state.chat_history[-2:]
                ]), 
                additional_info_dataset=st.session_state.summarized_dataset_info[:300] if st.session_state.summarized_dataset_info else ""
            ),
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            max_execution_time=180,
            allow_dangerous_code=True,
            extra_tools=[
                Tool(
                    name="format_large_number",
                    func=format_large_number,
                    description="Format large numbers with commas for better readability. Example: 1000000 -> 1,000,000"
                ),
                Tool(
                    name="analyze_categorical_counts",
                    func=lambda df_json, category_column, count_column=None, value_to_count=None, title=None, top_n=None, sort_by='count', ascending=False, include_percentages=True, chart_type='bar': analyze_categorical_counts(
                        df_json=df_json,
                        category_column=category_column,
                        count_column=count_column,
                        value_to_count=value_to_count,
                        title=title,
                        top_n=top_n,
                        sort_by=sort_by,
                        ascending=ascending,
                        include_percentages=include_percentages,
                        chart_type=chart_type
                    ),
                    description="""Analyze and visualize the distribution of values in any categorical column. 
                    This tool can:
                    1. Show the distribution of any categorical column
                    2. Filter the distribution based on another column's value
                    3. Sort results by count or category name
                    4. Show top N categories
                    5. Display percentages
                    6. Create bar or pie charts
                    
                    Parameters:
                    - category_column: The column to analyze (required)
                    - count_column: Optional column to filter by
                    - value_to_count: Optional value to filter for
                    - title: Optional custom chart title
                    - top_n: Optional number of top categories to show
                    - sort_by: 'count' or 'category' (default: 'count')
                    - ascending: Sort order (default: False)
                    - include_percentages: Show percentages (default: True)
                    - chart_type: 'bar' or 'pie' (default: 'bar')
                    
                    Returns a dictionary with counts, percentages, and chart path."""
                ),
                Tool(
                    name="analyze_data",
                    func=analyze_data,
                    description="""Perform comprehensive data analysis on any dataset. This tool can:
                    1. Filter data by any column and value
                    2. Group data by any column
                    3. Calculate statistics (max, min, mean, sum, count)
                    4. Sort results by any column
                    5. Create visualizations
                    6. Generate summary statistics
                    
                    Parameters:
                    - filter_column: Column to filter by
                    - filter_value: Value to filter for
                    - group_by: Column to group by
                    - agg_function: 'max', 'min', 'mean', 'sum', 'count'
                    - sort_column: Column to sort by
                    - ascending: Sort order
                    - top_n: Number of results to return
                    
                    Returns comprehensive analysis results with visualizations."""
                ),
                Tool(
                    name="get_dataset_context",
                    func=lambda x: rag_manager.get_relevant_context(x) if st.session_state.rag_initialized else [],
                    description="""Get relevant context from the dataset for a given query. This tool:
                    1. Uses semantic search to find relevant information
                    2. Returns context about column meanings, data relationships, and patterns
                    3. Helps understand the dataset structure and content
                    4. Provides insights about data quality and characteristics
                    
                    Always use this tool first to understand the dataset context before performing analysis."""
                ),
                Tool(
                    name="get_dataset_info",
                    func=lambda x: rag_manager.get_dataset_info() if st.session_state.rag_initialized else {},
                    description="""Get comprehensive information about the current dataset. This tool:
                    1. Returns column names and data types
                    2. Provides basic statistics for each column
                    3. Shows data quality metrics
                    4. Identifies relationships between columns
                    5. Highlights important patterns and trends
                    
                    Use this tool to understand the dataset structure before analysis."""
                ),
                Tool(
                    name="get_column_semantics",
                    func=lambda x: rag_manager.get_column_semantics(x) if st.session_state.rag_initialized else {},
                    description="""Get semantic understanding of specific columns. This tool:
                    1. Explains the meaning and purpose of columns
                    2. Identifies relationships between columns
                    3. Provides context about data values
                    4. Suggests relevant analysis approaches
                    
                    Use this tool to understand specific columns before analyzing them."""
                ),
                Tool(
                    name="create_visualization",
                    func=lambda df_json, x_column, y_column=None, filter_column=None, filter_value=None, chart_type='bar', title=None, x_label=None, y_label=None, top_n=None, sort_by='count', ascending=False, include_percentages=True: create_generalized_visualization(
                        df=df_json,
                        x_column=x_column,
                        y_column=y_column,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        chart_type=chart_type,
                        title=title,
                        x_label=x_label,
                        y_label=y_label,
                        top_n=top_n,
                        sort_by=sort_by,
                        ascending=ascending,
                        include_percentages=include_percentages
                    ),
                    description="""Create a visualization for any dataset structure. This tool can:
                    1. Create bar, pie, scatter, line, and histogram charts
                    2. Filter data by any column and value
                    3. Show top N values
                    4. Sort by count or value
                    5. Display percentages
                    6. Customize labels and titles
                    
                    Parameters:
                    - x_column: The column to use for x-axis (required)
                    - y_column: Optional column for y-axis (required for scatter/line plots)
                    - filter_column: Optional column to filter by
                    - filter_value: Optional value to filter for
                    - chart_type: 'bar', 'pie', 'scatter', 'line', or 'hist' (default: 'bar')
                    - title: Optional custom chart title
                    - x_label: Optional custom x-axis label
                    - y_label: Optional custom y-axis label
                    - top_n: Optional number of top values to show
                    - sort_by: 'count' or 'value' (default: 'count')
                    - ascending: Sort order (default: False)
                    - include_percentages: Show percentages (default: True)
                    
                    Returns a dictionary with the chart path and analysis results."""
                )
            ]
        )

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

            # Add a loading indicator
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Analyzing your data... ⏳")
            
            try:
                # Invoke the agent executor with the user's query
                res = agent_executor.invoke({"input": user_input})
                
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
                        else:
                            # If image doesn't exist, add a note to the response
                            formatted_response += f"\n\nNote: Could not find image at {img_path}."
                
                st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
                
                # Display chat messages after new input
                st.rerun()
            except Exception as e:
                error_message = f"Error processing your query: {str(e)}\n\nPlease try rephrasing your question or ask something else."
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
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

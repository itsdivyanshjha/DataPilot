import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from chains import summarization_chain
from utils import check_image_file_exists, read_image_file, generate_dataset_overview

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",  
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
        
        # Generate dataset overview
        dataset_overview = generate_dataset_overview(df)
        
        # Display Dataset Overview
        st.subheader("Dataset Overview")
        
        # Basic Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", dataset_overview['basic_info']['rows'])
        with col2:
            st.metric("Total Columns", dataset_overview['basic_info']['columns'])
        with col3:
            st.metric("Memory Usage", dataset_overview['basic_info']['memory_usage'])
        
        # Column Information Tabs
        st.markdown("### Column Analysis")
        tab1, tab2, tab3 = st.tabs(["Column Overview", "Missing Values", "Statistical Summary"])
        
        with tab1:
            # Create a clean dataframe for column overview
            col_overview = pd.DataFrame({
                'Data Type': [info['dtype'] for info in dataset_overview['column_info'].values()],
                'Unique Values': [info['unique_values'] for info in dataset_overview['column_info'].values()],
                'Missing (%)': [info['missing_percentage'] for info in dataset_overview['column_info'].values()]
            }, index=dataset_overview['column_info'].keys())
            st.dataframe(col_overview, use_container_width=True)
        
        with tab2:
            # Create missing values visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data = pd.DataFrame(dataset_overview['missing_values'], index=['Missing Count']).T
            missing_data['Percentage'] = missing_data['Missing Count'] / len(df) * 100
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            
            if not missing_data.empty:
                sns.barplot(data=missing_data.reset_index(), x='index', y='Percentage')
                plt.xticks(rotation=45)
                plt.title('Missing Values by Column (%)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No missing values found in the dataset!")
        
        with tab3:
            # Display statistical summary for numeric columns
            if dataset_overview['numeric_columns']:
                st.markdown("#### 📈 Numeric Columns")
                st.dataframe(df[dataset_overview['numeric_columns']].describe(), use_container_width=True)
            
            if dataset_overview['categorical_columns']:
                st.markdown("#### 📊 Categorical Columns")
                for col in dataset_overview['categorical_columns']:
                    with st.expander(f"Distribution of {col}"):
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            value_counts = df[col].value_counts().head(10)
                            
                            # Clean text for display
                            clean_labels = [str(x).replace('$', '\$').replace('_', '\_') 
                                          if isinstance(x, str) else str(x) 
                                          for x in value_counts.index]
                            
                            # Create bar plot
                            bars = ax.bar(range(len(value_counts)), value_counts.values)
                            
                            # Customize the plot
                            ax.set_xticks(range(len(value_counts)))
                            ax.set_xticklabels(clean_labels, rotation=45, ha='right')
                            ax.set_title(f'Top 10 Values in {col}')
                            
                            # Add value labels on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{int(height)}',
                                       ha='center', va='bottom')
                            
                            # Adjust layout with specific margins
                            plt.subplots_adjust(bottom=0.2, top=0.9)
                            st.pyplot(fig)
                            plt.close()
                            
                            # Also display as a table for better readability
                            st.markdown("##### Top 10 Values (Table View)")
                            value_table = pd.DataFrame({
                                'Value': value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': (value_counts.values / len(df) * 100).round(2)
                            })
                            st.dataframe(value_table, use_container_width=True)
                            
                        except Exception as e:
                            st.warning(f"Could not generate visualization for {col}. Error: {str(e)}")
                            # Display as table instead
                            st.markdown(f"##### Top 10 Values in {col} (Table View)")
                            value_counts = df[col].value_counts().head(10)
                            st.dataframe(pd.DataFrame({
                                'Value': value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': (value_counts.values / len(df) * 100).round(2)
                            }), use_container_width=True)
        
        # Quick Actions Section
        st.markdown("### Quick Actions")
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

import re
import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create charts directory if it doesn't exist
CHARTS_DIR = "charts"
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

def check_image_file_exists(text: str) -> Union[list, int]:
    # Function to extract all image locations
    def extract_image_locations(text: str) -> Union[list, None]:
        pattern = r'<image : r"(charts/[^"]+)">'
        matches = re.findall(pattern, text)
        return matches if matches else None

    # Extract the image locations
    image_locations = extract_image_locations(text)

    if image_locations:
        valid_images = [loc for loc in image_locations if os.path.isfile(loc)]
        
        return valid_images if valid_images else 0
    return -1

def display_images(image_paths: list) -> None:
    """Display multiple images with their download buttons"""
    if isinstance(image_paths, list):
        for img_path in image_paths:
            st.image(img_path)
            image_data = read_image_file(img_path)
            st.download_button(
                label=f"Download {os.path.basename(img_path)}",
                data=image_data,
                file_name=os.path.basename(img_path),
                mime="image/png"
            )

# Example usage:
# text = """The correlation matrix plot has been saved. <image : r"charts/correlation_matrix.png">"""
# text = """The average Air Temperature is 56K"""
text = """The chart has been saved at <image : r"charts/air_temp_machine_failure_relation.png">"""

# image_loc = check_image_file_exists(text)
# print(image_loc)

#---------------------

# Function to read image file and prepare it for download
def read_image_file(image_path: str) -> Union[bytes, None]:
    """Read image file and prepare it for download"""
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return None

def generate_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive overview of the dataset with trend analysis"""
    overview = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        },
        'column_info': {},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'trends': {
            'numeric_trends': {},
            'categorical_trends': {},
            'temporal_trends': {}
        }
    }
    
    # Analyze numeric columns for trends
    for col in overview['numeric_columns']:
        overview['trends']['numeric_trends'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': df[col].skew(),
            'is_normal': abs(df[col].skew()) < 0.5  # Check if distribution is roughly normal
        }
    
    # Analyze categorical columns for trends
    for col in overview['categorical_columns']:
        value_counts = df[col].value_counts()
        if len(value_counts) > 0:
            dominant_category = value_counts.index[0]
            dominance_ratio = value_counts.iloc[0] / len(df)
            overview['trends']['categorical_trends'][col] = {
                'dominant_category': dominant_category,
                'dominance_ratio': dominance_ratio,
                'category_count': len(value_counts),
                'is_imbalanced': dominance_ratio > 0.7  # Check if heavily imbalanced
            }
    
    return overview

def calculate_data_quality_score(df: pd.DataFrame) -> tuple[dict, float]:
    """Calculate a data quality score based on various metrics"""
    # Calculate completeness (how much data is not null)
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    # Calculate uniqueness (average ratio of unique values)
    uniqueness = np.mean([len(df[col].unique()) / len(df) for col in df.columns]) * 100
    
    # Calculate consistency (ratio of columns with expected data types)
    valid_types = ['int64', 'float64', 'object', 'datetime64[ns]', 'bool']
    consistency = sum([1 for col in df.columns if str(df[col].dtype) in valid_types]) / len(df.columns) * 100
    
    # Calculate data type distribution
    dtype_counts = df.dtypes.value_counts()
    type_distribution = {str(k): v / len(df.columns) * 100 for k, v in dtype_counts.items()}
    
    # Compile quality metrics
    quality_metrics = {
        'completeness': completeness,
        'uniqueness': uniqueness,
        'consistency': consistency,
        'type_distribution': type_distribution,
        'null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows_percentage': (df.duplicated().sum() / len(df)) * 100
    }
    
    # Calculate overall score (weighted average)
    weights = {
        'completeness': 0.4,  # Highest weight as missing data is critical
        'uniqueness': 0.3,    # Important for data diversity
        'consistency': 0.3    # Important for data reliability
    }
    
    overall_score = (
        quality_metrics['completeness'] * weights['completeness'] +
        quality_metrics['uniqueness'] * weights['uniqueness'] +
        quality_metrics['consistency'] * weights['consistency']
    )
    
    return quality_metrics, overall_score

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe for analysis"""
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Convert date columns with consistent format
    date_patterns = [
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}'
    ]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains dates
            sample = df[col].dropna().head(100)
            if not sample.empty:
                is_date = any(sample.astype(str).str.match(pattern).any() for pattern in date_patterns)
                if is_date:
                    try:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    except:
                        try:
                            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                        except:
                            try:
                                df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                            except:
                                try:
                                    df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
                                except:
                                    # If all format attempts fail, keep as string
                                    pass
    
    return df

def process_hashtags(df_json, hashtag_column='hashtags'):
    """
    Process hashtags from a column that may contain comma-separated values.
    Returns a DataFrame with each hashtag as a separate row.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the hashtag column, or a JSON string representation
    hashtag_column : str
        The name of the column containing hashtags
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with each hashtag as a separate row
    """
    import pandas as pd
    
    # Convert JSON to DataFrame if needed
    if isinstance(df_json, str):
        import json
        df_dict = json.loads(df_json)
        df = pd.DataFrame(df_dict)
    else:
        df = df_json
    
    # Check if the hashtag column exists
    if hashtag_column not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure the hashtag column is string type
    result_df[hashtag_column] = result_df[hashtag_column].astype(str)
    
    # Split the hashtags and explode the dataframe
    # This creates a new row for each hashtag in a post
    result_df[hashtag_column] = result_df[hashtag_column].str.split(',')
    exploded_df = result_df.explode(hashtag_column)
    
    # Clean up the hashtags (remove leading/trailing spaces)
    exploded_df[hashtag_column] = exploded_df[hashtag_column].str.strip()
    
    return exploded_df

def format_large_number(number: Union[int, float]) -> str:
    """
    Format large numbers with commas for better readability.
    
    Parameters:
    -----------
    number : int or float
        The number to format
        
    Returns:
    --------
    str
        Formatted number as string with commas
    """
    try:
        if isinstance(number, (int, float)):
            if float(number).is_integer():
                return f"{int(number):,d}"
            return f"{float(number):,.2f}"
        return str(number)
    except (ValueError, TypeError):
        return str(number)

def analyze_hashtags(df_json, hashtag_column='hashtags', metric_column='likes'):
    """
    Analyze hashtags to find the most popular ones based on a metric.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the hashtag and metric columns, or a JSON string representation
    hashtag_column : str
        The name of the column containing hashtags
    metric_column : str
        The name of the column containing the metric to analyze (e.g., likes)
        
    Returns:
    --------
    str
        A string with the analysis results and path to the generated chart
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert JSON to DataFrame if needed
    if isinstance(df_json, str):
        import json
        df_dict = json.loads(df_json)
        df = pd.DataFrame(df_dict)
    else:
        df = df_json
    
    # Process hashtags to get one hashtag per row
    exploded_df = process_hashtags(df, hashtag_column)
    
    # Group by hashtag and sum the metric
    hashtag_stats = exploded_df.groupby(hashtag_column)[metric_column].sum().reset_index()
    
    # Sort by the metric in descending order
    hashtag_stats = hashtag_stats.sort_values(by=metric_column, ascending=False)
    
    # Create a visualization of the top 10 hashtags
    plt.figure(figsize=(12, 8))
    
    # Use only top 10 for the chart
    top_hashtags = hashtag_stats.head(10)
    
    # Create the bar plot with proper formatting
    ax = sns.barplot(x=hashtag_column, y=metric_column, data=top_hashtags)
    
    # Add value labels on top of bars
    for i, v in enumerate(top_hashtags[metric_column]):
        ax.text(i, v + (v * 0.01), format_large_number(v), ha='center')
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_large_number(x)))
    
    plt.title(f'Top 10 Hashtags by {metric_column.capitalize()}', fontsize=16)
    plt.xlabel('Hashtag', fontsize=14)
    plt.ylabel(f'Total {metric_column.capitalize()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the chart
    chart_path = f'charts/top_hashtags_by_{metric_column}.png'
    plt.savefig(chart_path)
    plt.close()
    
    # Get the top hashtag and its metric value
    top_hashtag = hashtag_stats.iloc[0][hashtag_column]
    top_metric = hashtag_stats.iloc[0][metric_column]
    
    # Return a formatted string with the results
    result = f"The hashtag with the most {metric_column} is {top_hashtag} with {format_large_number(top_metric)} {metric_column}.\n"
    result += f"<image : r\"{chart_path}\">"
    
    return result

def find_platform_with_most_engagement(df_json, hashtag, engagement_column='likes'):
    """
    Find which platform has the most engagement for a specific hashtag.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the data, or a JSON string representation
    hashtag : str
        The hashtag to analyze
    engagement_column : str
        The column to use for measuring engagement (e.g., likes)
        
    Returns:
    --------
    str
        A string with the analysis results and path to the generated chart
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert JSON to DataFrame if needed
    if isinstance(df_json, str):
        import json
        df_dict = json.loads(df_json)
        df = pd.DataFrame(df_dict)
    else:
        df = df_json
    
    # Process hashtags to get one hashtag per row
    exploded_df = process_hashtags(df, 'hashtags')
    
    # Filter for the specific hashtag
    hashtag_df = exploded_df[exploded_df['hashtags'] == hashtag]
    
    # Group by platform and sum the engagement
    platform_stats = hashtag_df.groupby('platform')[engagement_column].sum().reset_index()
    
    # Sort by engagement in descending order
    platform_stats = platform_stats.sort_values(by=engagement_column, ascending=False)
    
    # Create a visualization
    plt.figure(figsize=(10, 6))
    
    # Create the bar plot with proper formatting
    ax = sns.barplot(x='platform', y=engagement_column, data=platform_stats)
    
    # Add value labels on top of bars
    for i, v in enumerate(platform_stats[engagement_column]):
        ax.text(i, v + (v * 0.01), format_large_number(v), ha='center')
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_large_number(x)))
    
    plt.title(f'Engagement ({engagement_column}) for #{hashtag} by Platform', fontsize=16)
    plt.xlabel('Platform', fontsize=14)
    plt.ylabel(f'Total {engagement_column.capitalize()}', fontsize=14)
    plt.tight_layout()
    
    # Save the chart
    chart_path = f'charts/hashtag_{hashtag}_by_platform.png'
    plt.savefig(chart_path)
    plt.close()
    
    # Get the platform with the most engagement
    if not platform_stats.empty:
        top_platform = platform_stats.iloc[0]['platform']
        top_engagement = platform_stats.iloc[0][engagement_column]
        
        # Return a formatted string with the results
        result = f"The platform with the most engagement for #{hashtag} is {top_platform} with {format_large_number(top_engagement)} {engagement_column}.\n"
        result += f"<image : r\"{chart_path}\">"
        
        return result
    else:
        return f"No data found for hashtag #{hashtag}."

def find_hashtag_with_most_likes_and_platform(df_json):
    """
    Find which hashtag has the most likes and on which platform it appears.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the data, or a JSON string representation
        
    Returns:
    --------
    str
        A string with the analysis results and paths to the generated charts
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert JSON to DataFrame if needed
    if isinstance(df_json, str):
        import json
        df_dict = json.loads(df_json)
        df = pd.DataFrame(df_dict)
    else:
        df = df_json
    
    # Process hashtags to get one hashtag per row
    exploded_df = process_hashtags(df, 'hashtags')
    
    # Group by hashtag and sum the likes
    hashtag_stats = exploded_df.groupby('hashtags')['likes'].sum().reset_index()
    
    # Sort by likes in descending order
    hashtag_stats = hashtag_stats.sort_values(by='likes', ascending=False)
    
    # Get the top hashtag and its likes
    top_hashtag = hashtag_stats.iloc[0]['hashtags']
    top_likes = hashtag_stats.iloc[0]['likes']
    
    # Create a visualization of the top 10 hashtags by likes
    plt.figure(figsize=(12, 8))
    
    # Use only top 10 for the chart
    top_hashtags = hashtag_stats.head(10)
    
    # Create the bar plot with proper formatting
    ax = sns.barplot(x='hashtags', y='likes', data=top_hashtags)
    
    # Add value labels on top of bars
    for i, v in enumerate(top_hashtags['likes']):
        ax.text(i, v + (v * 0.01), format_large_number(v), ha='center')
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_large_number(x)))
    
    plt.title('Top 10 Hashtags by Likes', fontsize=16)
    plt.xlabel('Hashtag', fontsize=14)
    plt.ylabel('Total Likes', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the chart
    hashtags_chart_path = 'charts/top_hashtags_by_likes.png'
    plt.savefig(hashtags_chart_path)
    plt.close()
    
    # Filter for posts with the top hashtag
    top_hashtag_df = exploded_df[exploded_df['hashtags'] == top_hashtag]
    
    # Group by platform and sum the likes
    platform_stats = top_hashtag_df.groupby('platform')['likes'].sum().reset_index()
    
    # Sort by likes in descending order
    platform_stats = platform_stats.sort_values(by='likes', ascending=False)
    
    # Get the top platform and its likes for this hashtag
    top_platform = platform_stats.iloc[0]['platform']
    platform_likes = platform_stats.iloc[0]['likes']
    
    # Create a visualization of the platforms for this hashtag
    plt.figure(figsize=(10, 6))
    
    # Create the bar plot with proper formatting
    ax = sns.barplot(x='platform', y='likes', data=platform_stats)
    
    # Add value labels on top of bars
    for i, v in enumerate(platform_stats['likes']):
        ax.text(i, v + (v * 0.01), format_large_number(v), ha='center')
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_large_number(x)))
    
    plt.title(f'Likes for #{top_hashtag} by Platform', fontsize=16)
    plt.xlabel('Platform', fontsize=14)
    plt.ylabel('Total Likes', fontsize=14)
    plt.tight_layout()
    
    # Save the chart
    platform_chart_path = f'charts/hashtag_{top_hashtag}_by_platform.png'
    plt.savefig(platform_chart_path)
    plt.close()
    
    # Return a formatted string with the results
    result = f"The hashtag with the most likes is #{top_hashtag} with {format_large_number(top_likes)} likes.\n"
    result += f"This hashtag appears most on {top_platform} with {format_large_number(platform_likes)} likes on that platform.\n\n"
    result += f"<image : r\"{hashtags_chart_path}\">\n"
    result += f"<image : r\"{platform_chart_path}\">"
    
    return result

def analyze_negative_posts_by_platform(df_json):
    """
    Analyze the number of negative posts by platform and create a visualization.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the data, or a JSON string representation
        
    Returns:
    --------
    str
        A string with the analysis results and path to the generated chart
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert JSON to DataFrame if needed
    if isinstance(df_json, str):
        import json
        df_dict = json.loads(df_json)
        df = pd.DataFrame(df_dict)
    else:
        df = df_json
    
    # Group by platform and count negative posts
    platform_stats = df[df['sentiment'] == 'negative'].groupby('platform').size().reset_index(name='negative_posts')
    
    # Sort by number of negative posts in descending order
    platform_stats = platform_stats.sort_values(by='negative_posts', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create the bar plot with proper formatting
    ax = sns.barplot(x='platform', y='negative_posts', data=platform_stats)
    
    # Add value labels on top of bars
    for i, v in enumerate(platform_stats['negative_posts']):
        ax.text(i, v + 1, format_large_number(v), ha='center')
    
    plt.title('Negative Posts by Platform', fontsize=16)
    plt.xlabel('Platform', fontsize=14)
    plt.ylabel('Number of Negative Posts', fontsize=14)
    plt.tight_layout()
    
    # Save the chart
    chart_path = 'charts/negative_posts_by_platform.png'
    plt.savefig(chart_path)
    plt.close()
    
    # Get the platform with the most negative posts
    top_platform = platform_stats.iloc[0]['platform']
    top_count = platform_stats.iloc[0]['negative_posts']
    
    # Get counts for other platforms for comparison
    platform_counts = {row['platform']: row['negative_posts'] for _, row in platform_stats.iterrows()}
    
    # Create a detailed response
    result = f"{top_platform} has the highest number of negative posts with {format_large_number(top_count)} posts.\n"
    result += "Here's the breakdown by platform:\n"
    for platform, count in platform_counts.items():
        result += f"- {platform}: {format_large_number(count)} negative posts\n"
    result += f"\n<image : r\"{chart_path}\">"
    
    return result

def analyze_categorical_counts(
    df_json: Union[pd.DataFrame, str],
    category_column: str,
    count_column: str = None,
    value_to_count: Any = None,
    title: str = None,
    top_n: int = None,
    sort_by: str = 'count',
    ascending: bool = False,
    include_percentages: bool = True,
    chart_type: str = 'bar'
) -> Dict[str, Any]:
    """
    Analyze and visualize categorical data in a dataset.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the data, or a JSON string representation
    category_column : str
        The column containing categories to analyze
    count_column : str, optional
        If provided, the column to filter by before counting
    value_to_count : any, optional
        If provided, the specific value in count_column to count
    title : str, optional
        Custom title for the visualization
    top_n : int, optional
        Number of top categories to show (if None, shows all)
    sort_by : str, optional
        How to sort the results ('count' or 'category')
    ascending : bool, optional
        Whether to sort in ascending order
    include_percentages : bool, optional
        Whether to include percentages in the visualization
    chart_type : str, optional
        Type of chart to create ('bar' or 'pie')
        
    Returns:
    --------
    dict
        A dictionary containing analysis results and path to the generated chart
    """
    try:
        if isinstance(df_json, str):
            df = pd.DataFrame(json.loads(df_json))
        else:
            df = df_json.copy()
        
        # Validate required column exists
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found in dataset")
        
        # Apply filtering if specified
        if count_column and value_to_count:
            if count_column not in df.columns:
                raise ValueError(f"Filter column '{count_column}' not found in dataset")
            df = df[df[count_column] == value_to_count]
        
        # Calculate counts and percentages
        counts = df[category_column].value_counts()
        total = len(df)
        percentages = (counts / total * 100).round(2)
        
        # Sort results
        if sort_by == 'category':
            counts = counts.sort_index(ascending=ascending)
            percentages = percentages[counts.index]
        else:  # sort by count
            counts = counts.sort_values(ascending=ascending)
            percentages = percentages[counts.index]
        
        # Limit to top N if specified
        if top_n is not None:
            counts = counts.head(top_n)
            percentages = percentages.head(top_n)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        if chart_type == 'bar':
            ax = sns.barplot(x=counts.index, y=counts.values)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            if include_percentages:
                for i, v in enumerate(counts.values):
                    ax.text(i, v, f'{v}\n({percentages[i]}%)', ha='center', va='bottom')
            else:
                for i, v in enumerate(counts.values):
                    ax.text(i, v, str(v), ha='center', va='bottom')
                    
        elif chart_type == 'pie':
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%' if include_percentages else None)
            plt.axis('equal')
        
        # Set title
        if title:
            plt.title(title)
        else:
            if count_column and value_to_count:
                plt.title(f'Distribution of {category_column} for {count_column} = {value_to_count}')
            else:
                plt.title(f'Distribution of {category_column}')
        
        plt.tight_layout()
        
        # Save the chart
        timestamp = int(time.time())
        chart_path = f"charts/categorical_counts_{timestamp}.png"
        plt.savefig(chart_path)
        plt.close()
        
        # Prepare results
        result = {
            'highest_category': counts.index[0],
            'highest_count': counts.values[0],
            'highest_percentage': percentages[0],
            'chart_path': chart_path,
            'counts': counts.to_dict(),
            'percentages': percentages.to_dict(),
            'total_count': total,
            'summary': {
                'total_categories': len(counts),
                'unique_values': df[category_column].nunique(),
                'missing_values': df[category_column].isnull().sum(),
                'missing_percentage': (df[category_column].isnull().sum() / total * 100).round(2)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_categorical_counts: {str(e)}")
        raise ValueError(f"Error analyzing categorical counts: {str(e)}")

def analyze_data(
    df_json: Union[pd.DataFrame, str], 
    filter_column: str = None,
    filter_value: Any = None,
    sort_column: str = None,
    ascending: bool = True,
    group_by: str = None,
    agg_function: str = None,
    top_n: int = 5,
    title: str = None
) -> Dict[str, Any]:
    """
    Enhanced generic function to analyze data with comprehensive analysis capabilities.
    
    Args:
        df_json: DataFrame or JSON string of the data
        filter_column: Column to filter on (optional)
        filter_value: Value to filter by (optional)
        sort_column: Column to sort by (optional)
        ascending: Sort order (default: True)
        group_by: Column to group by (optional)
        agg_function: Aggregation function to apply ('max', 'min', 'mean', 'sum', 'count')
        top_n: Number of results to return (default: 5)
        title: Custom title for the visualization (optional)
    
    Returns:
        Dictionary containing analysis results and visualization path
    """
    try:
        # Convert JSON to DataFrame if needed
        df = pd.DataFrame(json.loads(df_json)) if isinstance(df_json, str) else df_json.copy()
        
        # Initial data validation
        if df.empty:
            return {
                "success": False,
                "error": "Empty dataset provided",
                "results": None
            }
            
        # Apply filtering if specified
        if filter_column and filter_value:
            if filter_column not in df.columns:
                return {
                    "success": False,
                    "error": f"Filter column '{filter_column}' not found in dataset",
                    "results": None
                }
            df = df[df[filter_column] == filter_value]
            
        # Handle grouping and aggregation
        if group_by:
            if group_by not in df.columns:
                return {
                    "success": False,
                    "error": f"Group by column '{group_by}' not found in dataset",
                    "results": None
                }
                
            if not sort_column and not agg_function:
                # If no sort column or agg function specified, use count as default
                grouped_df = df.groupby(group_by).size().reset_index(name='count')
                sort_column = 'count'
            else:
                if sort_column not in df.columns:
                    return {
                        "success": False,
                        "error": f"Sort column '{sort_column}' not found in dataset",
                        "results": None
                    }
                    
                agg_func = {
                    'max': 'max',
                    'min': 'min',
                    'mean': 'mean',
                    'sum': 'sum',
                    'count': 'count'
                }.get(agg_function, 'max')
                
                grouped_df = df.groupby(group_by).agg({sort_column: agg_func}).reset_index()
        else:
            grouped_df = df
            
        # Apply sorting
        if sort_column:
            if sort_column not in grouped_df.columns:
                return {
                    "success": False,
                    "error": f"Sort column '{sort_column}' not found in processed dataset",
                    "results": None
                }
                
            # Convert to numeric if possible
            if grouped_df[sort_column].dtype == 'object':
                grouped_df[sort_column] = pd.to_numeric(grouped_df[sort_column], errors='coerce')
            
            grouped_df = grouped_df.sort_values(sort_column, ascending=ascending)
        
        # Get top N results
        results = grouped_df.head(top_n)
        
        # Create visualization
        viz_path = None
        if len(results) > 0:
            plt.figure(figsize=(12, 6))
            
            if pd.api.types.is_numeric_dtype(results[sort_column if sort_column else results.columns[-1]]):
                # For numeric data, create a bar plot
                plot_col = sort_column if sort_column else results.columns[-1]
                x_col = group_by if group_by else results.index
                
                bars = plt.bar(range(len(results)), results[plot_col])
                plt.xticks(range(len(results)), 
                          [str(x)[:30] + '...' if len(str(x)) > 30 else str(x) for x in results[x_col]],
                          rotation=45, ha='right')
                
                # Add value labels on top of bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:,.0f}',
                            ha='center', va='bottom')
            else:
                # For categorical data, create a count plot
                sns.countplot(data=results, x=group_by if group_by else results.columns[0])
                plt.xticks(rotation=45, ha='right')
            
            # Customize plot
            plt.title(title or f"{'Top' if not ascending else 'Bottom'} {top_n} {group_by or ''} by {sort_column or 'Count'}")
            plt.tight_layout()
            
            # Save plot
            viz_path = f"charts/analysis_{int(time.time())}.png"
            os.makedirs("charts", exist_ok=True)
            plt.savefig(viz_path)
            plt.close()
        
        # Prepare comprehensive results
        analysis_results = {
            "success": True,
            "results": results.to_dict('records'),
            "visualization_path": viz_path,
            "summary": {
                "total_records": len(df),
                "filtered_records": len(grouped_df),
                "metric_stats": {
                    "min": grouped_df[sort_column].min() if sort_column else None,
                    "max": grouped_df[sort_column].max() if sort_column else None,
                    "mean": grouped_df[sort_column].mean() if sort_column else None,
                    "median": grouped_df[sort_column].median() if sort_column else None
                } if sort_column and pd.api.types.is_numeric_dtype(grouped_df[sort_column]) else None,
                "query_details": {
                    "filter_applied": bool(filter_column and filter_value),
                    "grouping_applied": bool(group_by),
                    "sorting_applied": bool(sort_column),
                    "aggregation_used": agg_function if agg_function else None
                }
            }
        }
        
        return analysis_results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": None,
            "visualization_path": None
        }

def create_generalized_visualization(
    df: pd.DataFrame,
    x_column: str,
    y_column: str = None,
    filter_column: str = None,
    filter_value: Any = None,
    chart_type: str = 'bar',
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    top_n: int = None,
    sort_by: str = 'count',
    ascending: bool = False,
    include_percentages: bool = True
) -> Dict[str, Any]:
    """
    Create a generalized visualization that can handle any dataset structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
    x_column : str
        The column to use for x-axis
    y_column : str, optional
        The column to use for y-axis (for scatter plots or line charts)
    filter_column : str, optional
        Column to filter by
    filter_value : any, optional
        Value to filter for
    chart_type : str, optional
        Type of chart ('bar', 'pie', 'scatter', 'line', 'hist')
    title : str, optional
        Custom title for the visualization
    x_label : str, optional
        Custom x-axis label
    y_label : str, optional
        Custom y-axis label
    top_n : int, optional
        Number of top values to show
    sort_by : str, optional
        How to sort the results ('count' or 'value')
    ascending : bool, optional
        Whether to sort in ascending order
    include_percentages : bool, optional
        Whether to include percentages in the visualization
        
    Returns:
    --------
    dict
        A dictionary containing the visualization path and analysis results
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Validate columns exist
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset")
        if y_column and y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in dataset")
        if filter_column and filter_column not in df.columns:
            raise ValueError(f"Filter column '{filter_column}' not found in dataset")
        
        # Apply filtering if specified
        if filter_column and filter_value is not None:
            df = df[df[filter_column] == filter_value]
        
        # Handle different chart types
        plt.figure(figsize=(12, 6))
        
        if chart_type in ['bar', 'pie']:
            # For categorical data
            if y_column:
                # If y_column is specified, use it for values
                counts = df.groupby(x_column)[y_column].sum()
            else:
                # Otherwise, count occurrences
                counts = df[x_column].value_counts()
            
            # Calculate percentages
            total = counts.sum()
            percentages = (counts / total * 100).round(2)
            
            # Sort results
            if sort_by == 'value':
                counts = counts.sort_values(ascending=ascending)
                percentages = percentages[counts.index]
            else:  # sort by count
                counts = counts.sort_values(ascending=ascending)
                percentages = percentages[counts.index]
            
            # Limit to top N if specified
            if top_n is not None:
                counts = counts.head(top_n)
                percentages = percentages.head(top_n)
            
            if chart_type == 'bar':
                ax = sns.barplot(x=counts.index, y=counts.values)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels
                if include_percentages:
                    for i, v in enumerate(counts.values):
                        ax.text(i, v, f'{v}\n({percentages[i]}%)', ha='center', va='bottom')
                else:
                    for i, v in enumerate(counts.values):
                        ax.text(i, v, str(v), ha='center', va='bottom')
            
            elif chart_type == 'pie':
                plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%' if include_percentages else None)
                plt.axis('equal')
        
        elif chart_type == 'scatter':
            if not y_column:
                raise ValueError("y_column is required for scatter plots")
            sns.scatterplot(data=df, x=x_column, y=y_column)
        
        elif chart_type == 'line':
            if not y_column:
                raise ValueError("y_column is required for line plots")
            sns.lineplot(data=df, x=x_column, y=y_column)
        
        elif chart_type == 'hist':
            sns.histplot(data=df, x=x_column, bins=30)
        
        # Set labels and title
        plt.xlabel(x_label or x_column)
        if y_label:
            plt.ylabel(y_label)
        elif y_column:
            plt.ylabel(y_column)
        elif chart_type in ['bar', 'pie']:
            plt.ylabel('Count')
        
        if title:
            plt.title(title)
        else:
            title_parts = []
            if filter_column and filter_value is not None:
                title_parts.append(f"for {filter_column} = {filter_value}")
            if chart_type in ['bar', 'pie']:
                title_parts.append(f"Distribution of {x_column}")
            plt.title(' '.join(title_parts))
        
        plt.tight_layout()
        
        # Save the chart
        timestamp = int(time.time())
        chart_path = f"charts/visualization_{timestamp}.png"
        os.makedirs("charts", exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        
        # Prepare results
        result = {
            'chart_path': chart_path,
            'analysis': {
                'total_records': len(df),
                'unique_values': df[x_column].nunique(),
                'missing_values': df[x_column].isnull().sum(),
                'missing_percentage': (df[x_column].isnull().sum() / len(df) * 100).round(2)
            }
        }
        
        if chart_type in ['bar', 'pie']:
            result['analysis'].update({
                'counts': counts.to_dict(),
                'percentages': percentages.to_dict(),
                'highest_value': counts.index[0],
                'highest_count': counts.values[0],
                'highest_percentage': percentages[0]
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in create_generalized_visualization: {str(e)}")
        raise ValueError(f"Error creating visualization: {str(e)}")

def analyze_distribution(
    df: Union[pd.DataFrame, str],
    column: str,
    group_by: str = None,
    bins: int = 30,
    top_n: int = None,
    include_stats: bool = True,
    chart_type: str = 'auto',
    title: str = None,
    x_label: str = None,
    y_label: str = None
) -> Dict[str, Any]:
    """
    Analyze the distribution of any column in the dataset with advanced options.
    """
    try:
        # Convert JSON to DataFrame if needed
        if isinstance(df, str):
            df = pd.DataFrame(json.loads(df))
        
        # Validate column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Initialize results dictionary
        results = {
            'column': column,
            'total_records': len(df),
            'null_count': df[column].isnull().sum(),
            'null_percentage': (df[column].isnull().sum() / len(df) * 100).round(2)
        }
        
        # Handle numerical data
        if pd.api.types.is_numeric_dtype(df[column]):
            results['data_type'] = 'numerical'
            
            # Calculate statistics
            stats = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q1': df[column].quantile(0.25),
                'q3': df[column].quantile(0.75)
            }
            results['statistics'] = stats
            
            # Format statistics for display
            stats_text = (
                f"Summary Statistics:\n"
                f"Mean: {format_large_number(stats['mean'])}\n"
                f"Median: {format_large_number(stats['median'])}\n"
                f"Std Dev: {format_large_number(stats['std'])}\n"
                f"Min: {format_large_number(stats['min'])}\n"
                f"Max: {format_large_number(stats['max'])}\n"
                f"25th percentile: {format_large_number(stats['q1'])}\n"
                f"75th percentile: {format_large_number(stats['q3'])}"
            )
            results['formatted_stats'] = stats_text
            
            if chart_type == 'auto' or chart_type == 'hist':
                # Create histogram with KDE
                sns.histplot(data=df, x=column, bins=bins, kde=True)
                if include_stats:
                    # Add statistical annotations
                    plt.text(0.95, 0.95, stats_text,
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif chart_type == 'box':
                sns.boxplot(data=df, y=column)
            elif chart_type == 'violin':
                sns.violinplot(data=df, y=column)
            
            if group_by and group_by in df.columns:
                plt.figure(figsize=(12, 6))
                if chart_type == 'box':
                    sns.boxplot(data=df, x=group_by, y=column)
                else:
                    sns.violinplot(data=df, x=group_by, y=column)
                plt.xticks(rotation=45, ha='right')
                
                # Calculate and format group statistics
                group_stats = df.groupby(group_by)[column].agg([
                    'mean', 'median', 'std', 'min', 'max'
                ]).round(2)
                
                # Format group statistics for display
                group_stats_text = "Group Statistics:\n"
                for group in group_stats.index:
                    group_stats_text += f"\n{group}:\n"
                    group_stats_text += f"  Mean: {format_large_number(group_stats.loc[group, 'mean'])}\n"
                    group_stats_text += f"  Median: {format_large_number(group_stats.loc[group, 'median'])}\n"
                    group_stats_text += f"  Min: {format_large_number(group_stats.loc[group, 'min'])}\n"
                    group_stats_text += f"  Max: {format_large_number(group_stats.loc[group, 'max'])}\n"
                
                results['group_statistics'] = group_stats.to_dict()
                results['formatted_group_stats'] = group_stats_text
        
        # Handle categorical data
        else:
            results['data_type'] = 'categorical'
            value_counts = df[column].value_counts()
            total_count = len(df)
            
            if top_n:
                value_counts = value_counts.head(top_n)
            
            # Calculate and format statistics
            stats = {
                'unique_values': df[column].nunique(),
                'most_common': value_counts.index[0],
                'most_common_count': int(value_counts.iloc[0]),
                'most_common_percentage': (value_counts.iloc[0] / total_count * 100).round(2)
            }
            results['statistics'] = stats
            
            # Format statistics for display
            stats_text = (
                f"Category Statistics:\n"
                f"Total unique values: {stats['unique_values']}\n"
                f"Most common: {stats['most_common']}\n"
                f"Occurrences: {format_large_number(stats['most_common_count'])} "
                f"({stats['most_common_percentage']}%)"
            )
            results['formatted_stats'] = stats_text
            
            if chart_type == 'auto' or chart_type == 'bar':
                # Create bar plot
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for i, v in enumerate(value_counts.values):
                    percentage = (v / total_count * 100).round(1)
                    plt.text(i, v, f'{format_large_number(v)}\n({percentage}%)', 
                            ha='center', va='bottom')
            
            if group_by and group_by in df.columns:
                # Calculate cross-tabulation with percentages
                crosstab = pd.crosstab(
                    df[group_by], df[column], normalize='index'
                ).round(2)
                results['group_statistics'] = crosstab.to_dict()
                
                # Format cross-tabulation for display
                group_stats_text = "Distribution by Group:\n"
                for group in crosstab.index:
                    group_stats_text += f"\n{group}:\n"
                    for category in crosstab.columns:
                        percentage = crosstab.loc[group, category] * 100
                        if percentage > 0:
                            group_stats_text += f"  {category}: {percentage:.1f}%\n"
                
                results['formatted_group_stats'] = group_stats_text
        
        # Set labels and title
        plt.xlabel(x_label or column)
        plt.ylabel(y_label or ('Count' if chart_type != 'box' else 'Value'))
        plt.title(title or f'Distribution of {column}')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = int(time.time())
        chart_path = f"charts/distribution_{column}_{timestamp}.png"
        plt.savefig(chart_path)
        plt.close()
        
        results['chart_path'] = chart_path
        
        # Add a natural language summary
        if results['data_type'] == 'numerical':
            summary = (
                f"The {column} distribution shows:\n"
                f"- Average (mean) of {format_large_number(stats['mean'])}\n"
                f"- Median of {format_large_number(stats['median'])}\n"
                f"- Range from {format_large_number(stats['min'])} to {format_large_number(stats['max'])}\n"
                f"- Standard deviation of {format_large_number(stats['std'])}"
            )
        else:
            summary = (
                f"The {column} distribution shows:\n"
                f"- {stats['unique_values']} unique values\n"
                f"- Most common: {stats['most_common']} "
                f"({stats['most_common_percentage']}% of total)\n"
                f"- Top {len(value_counts)} categories shown in the chart"
            )
        
        if group_by:
            summary += f"\n\nThe distribution is broken down by {group_by} in the visualization."
        
        results['summary'] = summary
        
        return results
    
    except Exception as e:
        logger.error(f"Error in analyze_distribution: {str(e)}")
        raise ValueError(f"Error analyzing distribution: {str(e)}")

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

# Create charts directory if it doesn't exist
CHARTS_DIR = "charts"
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

def check_image_file_exists(text):
    # Function to extract all image locations
    def extract_image_locations(text):
        pattern = r'<image : r"(charts/[^"]+)">'
        matches = re.findall(pattern, text)
        return matches if matches else None

    # Extract the image locations
    image_locations = extract_image_locations(text)

    if image_locations:
        valid_images = []
        for loc in image_locations:
            if os.path.isfile(loc):
                valid_images.append(loc)
        
        return valid_images if valid_images else 0
    return -1

def display_images(image_paths):
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
def read_image_file(image_path):
    """Read image file and prepare it for download"""
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except Exception as e:
        print(f"Error reading image file: {e}")
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
    
    # Convert date columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
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

def format_large_number(number):
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
    return "{:,}".format(number)

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

def analyze_categorical_counts(df_json, category_column, count_column=None, value_to_count=None, title=None):
    """
    Analyze and visualize counts by category in a dataset.
    
    Parameters:
    -----------
    df_json : str or pandas.DataFrame
        The dataframe containing the data, or a JSON string representation
    category_column : str
        The column containing categories to group by
    count_column : str, optional
        If provided, the column to filter by before counting
    value_to_count : any, optional
        If provided, the specific value in count_column to count
    title : str, optional
        Custom title for the visualization
        
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
    
    # Filter data if count_column and value_to_count are provided
    if count_column and value_to_count is not None:
        filtered_df = df[df[count_column] == value_to_count]
    else:
        filtered_df = df
    
    # Group by category and count
    if count_column:
        category_stats = filtered_df.groupby(category_column).size().reset_index(name='count')
    else:
        category_stats = df.groupby(category_column).size().reset_index(name='count')
    
    # Sort by count in descending order
    category_stats = category_stats.sort_values(by='count', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create the bar plot with proper formatting
    ax = sns.barplot(x=category_column, y='count', data=category_stats)
    
    # Add value labels on top of bars
    for i, v in enumerate(category_stats['count']):
        ax.text(i, v + 1, format_large_number(v), ha='center')
    
    # Set title and labels
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f'Count by {category_column}', fontsize=16)
    plt.xlabel(category_column, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the chart
    safe_column_name = re.sub(r'[^a-zA-Z0-9]', '_', category_column)
    chart_path = f'charts/count_by_{safe_column_name}.png'
    plt.savefig(chart_path)
    plt.close()
    
    # Get the category with the highest count
    top_category = category_stats.iloc[0][category_column]
    top_count = category_stats.iloc[0]['count']
    
    # Get counts for all categories
    category_counts = {row[category_column]: row['count'] for _, row in category_stats.iterrows()}
    
    # Create a detailed response
    if count_column and value_to_count is not None:
        result = f"For {value_to_count} {count_column}, {top_category} has the highest count with {format_large_number(top_count)}.\n"
    else:
        result = f"{top_category} has the highest count with {format_large_number(top_count)}.\n"
    
    result += f"\nBreakdown by {category_column}:\n"
    for category, count in category_counts.items():
        result += f"- {category}: {format_large_number(count)}\n"
    result += f"\n<image : r\"{chart_path}\">"
    
    return result

def analyze_data(df_json: Union[pd.DataFrame, str], 
                filter_column: str = None,
                filter_value: Any = None,
                sort_column: str = None,
                ascending: bool = True,
                group_by: str = None,
                agg_function: str = None,
                top_n: int = 5,
                title: str = None) -> Dict[str, Any]:
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

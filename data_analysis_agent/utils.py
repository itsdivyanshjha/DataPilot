import re
import os
import streamlit as st
import numpy as np

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

def generate_dataset_overview(df):
    """Generate comprehensive overview of the dataset"""
    overview = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        },
        'column_info': {},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    # Generate column-specific information
    for column in df.columns:
        col_info = {
            'dtype': str(df[column].dtype),
            'unique_values': len(df[column].unique()),
            'missing_values': df[column].isnull().sum(),
            'missing_percentage': f"{(df[column].isnull().sum() / len(df)) * 100:.2f}%"
        }
        
        # Add specific stats for numeric columns
        if df[column].dtype in ['int64', 'float64']:
            col_info.update({
                'mean': f"{df[column].mean():.2f}",
                'median': f"{df[column].median():.2f}",
                'std': f"{df[column].std():.2f}",
                'min': df[column].min(),
                'max': df[column].max()
            })
        # Add specific stats for categorical columns
        elif df[column].dtype == 'object':
            value_counts = df[column].value_counts()
            col_info.update({
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0
            })
            
        overview['column_info'][column] = col_info
    
    return overview

def calculate_data_quality_score(df):
    """Calculate a data quality score based on various metrics"""
    quality_metrics = {
        'completeness': (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'uniqueness': np.mean([len(df[col].unique()) / len(df) for col in df.columns]) * 100,
        'consistency': sum([1 for col in df.columns if df[col].dtype in ['int64', 'float64', 'object']]) / len(df.columns) * 100
    }
    
    # Calculate overall score (weighted average)
    weights = {'completeness': 0.4, 'uniqueness': 0.3, 'consistency': 0.3}
    overall_score = sum(score * weights[metric] for metric, score in quality_metrics.items())
    
    return quality_metrics, overall_score

import re
import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

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

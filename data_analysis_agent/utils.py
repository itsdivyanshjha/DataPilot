import re
import os
import streamlit as st

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

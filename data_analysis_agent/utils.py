import re
import os

# Create charts directory if it doesn't exist
CHARTS_DIR = "charts"
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

def check_image_file_exists(text):
    # Function to extract the image location
    def extract_image_location(text):
        pattern = r'<image : (r".*?")>'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip('r"')
        return None

    # Extract the image location
    image_location = extract_image_location(text)

    if image_location:
        # Ensure the path is relative to the charts directory
        if not image_location.startswith(CHARTS_DIR):
            image_location = os.path.join(CHARTS_DIR, os.path.basename(image_location))
            
        if os.path.isfile(image_location):
            return fr"{image_location}"  # Return the image location if detected in agent response and actually exists
        else:
            # return "0"  # Image location detected in agent response but doesn't exist
            return 0  # Image location detected in agent response but doesn't exist
            # return fr"{image_location}"
    else:
        return -1  # Image location not detected in agent response


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

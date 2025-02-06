import os
import gdown
import zipfile
from gdown.exceptions import FileURLRetrievalError

# Google Drive folder URL
folder_url = "https://drive.google.com/drive/folders/1Tquahp2HWBP_R2tNi5cxsVJ3oEJ8F0Xx"

# Output directory where the folder will be downloaded
output_dir = "datasets/Motion-X++"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download the folder using gdown, skipping files with permission issues
try:
    gdown.download_folder(folder_url, output=output_dir, quiet=False)
except FileURLRetrievalError as e:
    print(f"Skipping file due to permission error: {e}")

# Function to unzip specific files
def unzip_specific_files(directory, target_zip_name):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == target_zip_name:
                zip_path = os.path.join(root, file)
                extract_path = os.path.splitext(zip_path)[0]  # Extract to a folder with the same name as the zip file
                os.makedirs(extract_path, exist_ok=True)
                
                print(f"Unzipping {zip_path} to {extract_path}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"Unzipped {zip_path} successfully.")
                except zipfile.BadZipFile:
                    print(f"Skipping {zip_path} as it is not a valid zip file.")
                except PermissionError:
                    print(f"Skipping {zip_path} due to permission error.")

# Unzip all files named 'music.zip'
unzip_specific_files(output_dir, "music.zip")
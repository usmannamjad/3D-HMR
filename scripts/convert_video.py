# convert video from mp4 to mov required for tram
import os
import subprocess
from tqdm import tqdm
import time

def convert_mp4_to_mov(input_file, output_file):
    command = ["ffmpeg", "-i", input_file, "-c:v", "copy", "-c:a", "copy", output_file]
    subprocess.run(command, check=True)


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            file_paths.append(path)
    return file_paths


all_files = get_all_file_paths("/home/usmannamjad/vh/datasets/Motion-X++/video/music")

for file in tqdm(all_files):
    output_file = f"/home/usmannamjad/vh/datasets/Motion-X++/video-mov/music/{file.split('/')[-1].replace('.mp4', '.mov')}"
    try:
        convert_mp4_to_mov(file, output_file)
    except Exception as e:
        print(f"Error converting {file} to mov: {e}")
    
    time.sleep(0.2)
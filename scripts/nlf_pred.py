import os
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
import pickle
import gc
import time
import warnings
warnings.filterwarnings("ignore")


MODEL_PATH = '/home/usmannamjad/vh/models/nlf_l_multi.torchscript'
VIDEOS_PATH = Path('/home/usmannamjad/vh/datasets/Motion-X++/video/')
OUTPUT_PATH = Path('/home/usmannamjad/vh/datasets/outputs/nlf')
DATASET_SUBSETS = ['music']

# Load the TorchScript model
model = torch.jit.load(MODEL_PATH).cuda().eval()
batch_size = 128
key_names = ['boxes', 'pose', 'betas', 'trans', 'vertices3d', 
             'joints3d', 'vertices2d', 'joints2d', 'vertices3d_nonparam', 
             'joints3d_nonparam', 'vertices2d_nonparam', 'joints2d_nonparam', 
             'vertex_uncertainties', 'joint_uncertainties']

def get_total_paths():
    paths = []
    for video_path in get_video_paths():
        paths.append(video_path)
    total_paths = len(paths)
    del paths
    return total_paths


def get_video_paths():
    for subset in DATASET_SUBSETS:
        subset_path = VIDEOS_PATH / subset
        if subset_path.exists() and subset_path.is_dir():
            for video_file in subset_path.rglob("*.mp4"):  # Recursively finds .mp4 files
                yield video_file  # Yields full path as a generator


def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            path = path.split('/')[-1]
            path = path.replace('.pkl', '')
            file_paths.append(path)
    return file_paths

all_files = get_all_files(OUTPUT_PATH / 'music')
print("all files: ", len(all_files))


total_paths = get_total_paths()
videos_not_processed = []

paths = ["/home/usmannamjad/vh/datasets/Motion-X++/video/music/Play_Cello_7_clip1.mp4"]

for video_path in paths:
# for video_path in tqdm(get_video_paths(), total=total_paths):
    video_name = str(video_path).split('/')[-1]
    video_name = video_name.replace('.mp4', '')
    if video_name not in all_files:
        try:
            frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")

            # Convert frames to tensor and move to GPU
            frames = frames.permute(0, 3, 1, 2).cuda()  # Shape: (num_frames, C, H, W)

            # Process video frames in batches
            num_frames = frames.shape[0]
            print("frames shape: ", frames.shape, num_frames)

            results = {key: [] for key in key_names}
            # with torch.inference_mode():
            with torch.no_grad():
                for i in range(0, num_frames, batch_size):
                    frame_batch = frames[i:i+batch_size]
                    preds = model.detect_smpl_batched(frame_batch, model_name='smplx')

                    for key in preds:
                        results[key].extend([p.cpu() for p in preds[key]])
        
            del frames
            torch.cuda.empty_cache()
            gc.collect()

            print("output: ", len(results['betas']))
            video_path = str(video_path).replace('.mp4', '')
            video_path = video_path.split('/')
            video_name = video_path[-1]
            subset = video_path[-2]
            save_path = OUTPUT_PATH / subset / f"{video_name}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            del results
            time.sleep(1)

        except Exception as e:
            videos_not_processed.append(str(video_path))
            print(f"Error processing video: {video_path}")
            print(e)
    else:
        print(f"Video {video_name} already processed")
        

print(f"Videos not processed:", len(videos_not_processed))
output_file = OUTPUT_PATH / "videos_not_processed.txt"
with open(output_file, 'w') as f:
    f.write("\n".join(videos_not_processed))
        
        
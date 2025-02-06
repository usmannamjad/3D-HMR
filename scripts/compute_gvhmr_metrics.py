import torch
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (N, J, 3).
        fps (float).
    Returns:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0

def compute_acceleration(joints, fps=30):
    acceleration = torch.norm(
        (joints[2:] - 2 * joints[1:-1] + joints[:-2]) * (fps**2),
        dim=2,
    ).mean(dim=-1)
    return acceleration.cpu().numpy()



def align_trajectories(gt_joints, pred_joints):
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    return gt_joints, pred_joints

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            path = path.split('/')[-1]
            path = path.replace('.pkl', '')
            file_paths.append(path)
    return file_paths

def get_predicted_motion(file):
    motion = np.load(file)
    motion = torch.tensor(motion).float()
    return motion

dir_path = Path("/home/usmannamjad/vh/datasets/outputs/gvhmr/music")
predicted_files = get_all_file_paths(dir_path)

accel_error = []
mpjpe_error = []
pampjpe_error = []
pred_jitter = []

gt_jitter_list = []
gt_accel_list = []
pred_jitter_list = []
pred_accel_list = []

files_with_error = []
for predicted_file in tqdm(predicted_files, total=len(predicted_files)):
# predicted_files = ['Play_Cello_7_clip1', 'Play_Cello_7_clip2']
# for predicted_file in predicted_files:
    try: 
        # gt_motion = get_ground_truth_motion(predicted_file)
        pred_motion = get_predicted_motion(dir_path / predicted_file)
        # gt_motion, pred_motion = align_trajectories(gt_motion, pred_motion)

        pred_jitter = compute_jitter(pred_motion)
        pred_accel = compute_acceleration(pred_motion)
        # gt_jitter = compute_jitter(gt_motion)
        # gt_accel = compute_acceleration(gt_motion)

        # gt_jitter_list.append(gt_jitter.mean())
        # gt_accel_list.append(gt_accel.mean())
        pred_jitter_list.append(pred_jitter.mean())
        pred_accel_list.append(pred_accel.mean())
    except Exception as e:
        print(e)
        files_with_error.append(predicted_file)


# print("GT Jitter: ", sum(gt_jitter_list)/len(gt_jitter_list))
# print("GT Acceleration: ", sum(gt_accel_list)/len(gt_accel_list))
print("Pred Jitter: ", sum(pred_jitter_list)/len(pred_jitter_list))
print("Pred Acceleration: ", sum(pred_accel_list)/len(pred_accel_list))
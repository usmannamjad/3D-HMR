import os
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm
import time
from typing import Tuple

PREDICTIONS_PATH = Path("/home/usmannamjad/vh/datasets/outputs/tram/")
GROUND_TRUTH_PATH = Path("/home/usmannamjad/vh/datasets/Motion-X++/motion/motion_generation/smplx322")

SUBSETS = ['music']


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            path = path.split('/')[-1]
            path = path.replace('.pkl', '')
            file_paths.append(path)
    return file_paths

# def compute_error_accel(joints_gt, joints_pred, valid_mask=None, fps=None):
#     """
#     Use [i-1, i, i+1] to compute acc at frame_i. The acceleration error:
#         1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
#     Note that for each frame that is not visible, three entries(-1, 0, +1) in the
#     acceleration error will be zero'd out.
#     Args:
#         joints_gt : (F, J, 3)
#         joints_pred : (F, J, 3)
#         valid_mask : (F)
#     Returns:
#         error_accel (F-2) when valid_mask is None, else (F'), F' <= F-2
#     """
#     # (F, J, 3) -> (F-2) per-joint
#     accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
#     accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
#     normed = np.linalg.norm(accel_pred - accel_gt, axis=-1).mean(axis=-1)
#     if fps is not None:
#         normed = normed * fps**2

#     if valid_mask is None:
#         new_vis = np.ones(len(normed), dtype=bool)
#     else:
#         invis = np.logical_not(valid_mask)
#         invis1 = np.roll(invis, -1)
#         invis2 = np.roll(invis, -2)
#         new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
#         new_vis = np.logical_not(new_invis)
#         if new_vis.sum() == 0:
#             print("Warning!!! no valid acceleration error to compute.")

#     return normed[new_vis]

def compute_error_accel(joints_gt, joints_pred, vis=None, fps=None):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)
    if fps is not None:
        normed = normed * fps**2

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re.cpu().numpy()

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstuction_error
    r_error = reconstruction_error(pred_joints.cpu(), gt_joints.cpu())
    return 1000 * mpjpe, 1000 * r_error


def get_ground_truth_motion(file):
    gt_path = GROUND_TRUTH_PATH / SUBSETS[0] / f'{file}.npy'

    motion = np.load(gt_path)
    motion = torch.tensor(motion).float()
    gt_motion = motion[:, :3+63]     # 22 body joints
    gt_motion = gt_motion.reshape(-1, 22, 3)
    return gt_motion

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


def get_predicted_motion(file):
    pred_path = PREDICTIONS_PATH / SUBSETS[0] / f'{file}' / 'hps/joints.npy'
    pred = np.load(pred_path)
    pred_motion = torch.tensor(pred[:-1, :])    
    return pred_motion


from pathlib import Path

def get_subdirectories(path):
    """Returns a list of subdirectory names in the given directory (depth 1)."""
    return [p.name for p in Path(path).iterdir() if p.is_dir()]

def align_trajectories(gt_joints, pred_joints):
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    return gt_joints, pred_joints


predicted_files = get_subdirectories(PREDICTIONS_PATH / SUBSETS[0])
print(predicted_files)
accel_error = []
mpjpe_error = []
pampjpe_error = []
pred_jitter = []

files_with_error = []

gt_jitter_list = []
gt_accel_list = []
pred_jitter_list = []
pred_accel_list = []

for predicted_file in tqdm(predicted_files, total=len(predicted_files)):
# predicted_files = ['Play_Cello_7_clip1', 'Play_Cello_7_clip2']
# for predicted_file in predicted_files:
    try: 
        gt_motion = get_ground_truth_motion(predicted_file)
        pred_motion = get_predicted_motion(predicted_file)
        gt_motion, pred_motion = align_trajectories(gt_motion, pred_motion)
        # error = compute_error_accel(gt_motion, pred_motion)
        # accel_error.append(error.mean())
        # mpjpe, pampjpe = eval_pose(pred_motion, gt_motion)
        # mpjpe_error.append(mpjpe.mean())
        # pampjpe_error.append(pampjpe.mean())
        pred_jitter = compute_jitter(pred_motion)
        pred_accel = compute_acceleration(pred_motion)
        gt_jitter = compute_jitter(gt_motion)
        gt_accel = compute_acceleration(gt_motion)

        gt_jitter_list.append(gt_jitter.mean())
        gt_accel_list.append(gt_accel.mean())
        pred_jitter_list.append(pred_jitter.mean())
        pred_accel_list.append(pred_accel.mean())
        # pred_jitter.append(jitter.mean())
    except Exception as e:
        print(e)
        files_with_error.append(predicted_file)

# print("accel: ", sum(accel_error)/len(accel_error) * 1000)
# print("mpjpe :", sum(mpjpe_error)/len(mpjpe_error))
# print("pampjpe :", sum(pampjpe_error)/len(pampjpe_error))
# print("num of files not processed: ", len(files_with_error))
# print("pred_jitter: ", sum(pred_jitter)/len(pred_jitter))

print("GT Jitter: ", sum(gt_jitter_list)/len(gt_jitter_list))
print("GT Acceleration: ", sum(gt_accel_list)/len(gt_accel_list))
print("Pred Jitter: ", sum(pred_jitter_list)/len(pred_jitter_list))
print("Pred Acceleration: ", sum(pred_accel_list)/len(pred_accel_list))


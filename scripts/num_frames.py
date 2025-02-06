import torchvision
import pickle
import numpy as np
import torch

gt = "/home/usmannamjad/vh/datasets/Motion-X++/motion/motion_generation/smplx322/music/Play_Cello_7_clip1.npy"
pred = "/home/usmannamjad/vh/datasets/outputs/nlf/music/Play_Cello_7_clip1.pkl"
video = "/home/usmannamjad/vh/datasets/Motion-X++/video/music/Play_Cello_7_clip1.mp4"

gt = np.load(gt, allow_pickle=True)
print("gt: ", gt.shape)
gt = gt[:, :3+63]
gt = gt.reshape(gt.shape[0], 22, 3)
print("gt: ", gt.shape)


with open(pred, 'rb') as f:
    pred = pickle.load(f)

pred = pred['pose']
pred = torch.cat(pred, dim=0)
print("pred: ", pred.shape)
pred = pred[:, :3+63].reshape(-1, 22, 3)
pred = pred[:-1]
print("pred: ", pred.shape)


frames, _, info = torchvision.io.read_video(video, pts_unit='sec')

print("frames: ", len(frames))

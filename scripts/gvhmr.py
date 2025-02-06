import torch

# Load the file
file_path = "/home/usmannamjad/vh/GVHMR/outputs/demo/tennis/hmr4d_results.pt"
data = torch.load(file_path)

# Check the type and keys
print(type(data))
print(data.keys())  # To see available keys
print(data["smpl_params_global"].keys())
print(data["smpl_params_global"]["body_pose"].shape)
print(data["smpl_params_global"]["betas"].shape)
print(data["smpl_params_global"]["global_orient"].shape)
print(data["smpl_params_global"]["transl"].shape)


import numpy as np

print("::::::::::::::::::::::::::::::::::::::::::::::::::::")
# Provide the path to your .npy file
file_path = '/home/usmannamjad/vh/datasets/Motion-X++/motion/motion_generation/smplx322/music/Ancient_Drum_1_clip1.npy'

# Load the .npy file
data = np.load(file_path)

# Print the content of the file
print(data.shape)


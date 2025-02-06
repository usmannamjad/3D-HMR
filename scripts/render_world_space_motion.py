# render ground truth motion from the dataset
import numpy as np
import torch
import torch
import numpy as np
import cv2
import pyrender
from smplx import SMPLX
from tqdm import tqdm
import trimesh
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"



name = "Ancient_Drum_1_clip3"
video_path = f"datasets/Motion-X++/video/music/{name}.mp4"
gt_path = f"/home/usmannamjad/vh/datasets/Motion-X++/motion/motion_generation/smplx322/music/{name}.npy"


motion = np.load(gt_path)
motion = torch.tensor(motion).float()



# Load SMPL-X model
model_path = '/home/usmannamjad/vh/smplx/SMPLX_NEUTRAL.npz'
model = SMPLX(model_path, gender='neutral', use_pca=False)

num_frames = motion.shape[0]

# # Renderer setup
# scene = pyrender.Scene()
# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
# scene.add(camera, pose=np.eye(4))
# scene.add(light, pose=np.eye(4))
# renderer = pyrender.OffscreenRenderer(800, 800)

frames = []
for i in tqdm(range(num_frames)):
    # Extract parameters
    root_orient = torch.tensor(motion[i, :3]).unsqueeze(0)
    pose_body = torch.tensor(motion[i, 3:66]).unsqueeze(0)
    trans = torch.tensor(motion[i, 309:312]).unsqueeze(0)
    betas = torch.tensor(motion[i, 312:]).unsqueeze(0)

    # Generate SMPL-X model output
    output = model(global_orient=root_orient, body_pose=pose_body,
                   betas=betas, transl=trans)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Create a pyrender scene
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # Set up the camera and light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 3])  # Move camera further away

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Render the scene
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    color, depth = renderer.render(scene)



    frames.append(color)
print("here")
# Save as video
video_path = 'output.mp4'
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 800))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print(f'Video saved at {video_path}')

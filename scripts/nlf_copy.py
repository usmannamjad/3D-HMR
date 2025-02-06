# file used for rendering

import torch
import torchvision
import smplx
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex, MeshRenderer, look_at_view_transform
)
from PIL import Image

# Load the TorchScript model
model = torch.jit.load('/home/usmannamjad/vh/models/nlf_l_multi.torchscript').cuda().eval()

# Load input image
image = torchvision.io.read_image('/home/usmannamjad/vh/nlf/ex.jpeg').cuda()
frame_batch = image.unsqueeze(0)  # Add batch dimension

# Get predictions from the model
with torch.inference_mode():
    pred = model.detect_smpl_batched(frame_batch, model_name='smplx')

# Define the path to the SMPL-X model file
smplx_model_path = "/home/usmannamjad/vh/smplx/SMPLX_NEUTRAL.npz"

# Load the SMPL-X model and move it to the GPU
smplx_model = smplx.create(
    model_path=smplx_model_path,
    model_type="smplx",
    gender="neutral",  # or "male", "female"
    num_betas=10,      # number of shape parameters
    use_pca=False,     # whether to use PCA for pose parameters
    ext="npz"          # file extension of the model
).cuda()  # Move the model to GPU

# Extract pose and betas from the prediction dictionary
pose = pred['pose'][0].clone().detach().to(torch.float32).cuda()  # Shape: [1, 165]
global_orientation = pose[:, :3]  # Global orientation (first 3 elements)
body_pose = pose[:, 3:66]  # Body pose (next 63 elements)
body_pose = body_pose.view(1, 21, 3)  # Reshape to [1, 21, 3]
betas = pred['betas'][0].clone().detach().to(torch.float32).cuda()  # Shape: [1, 10]

# If translation is available, extract it as well
trans = pred['trans'][0].clone().detach().to(torch.float32).cuda()  # Shape: [1, 3]

# Generate the SMPL-X mesh
output = smplx_model(
    betas=betas,
    body_pose=body_pose,  # Body pose in axis-angle representation
    global_orient=global_orientation,  # Global orientation
    transl=trans,  # Translation
    return_verts=True  # Return vertices
)

# Access the vertices of the generated mesh
vertices = output.vertices  # Shape: [1, 10475, 3]

# Get the faces of the SMPL-X model
faces = smplx_model.faces  # Shape: [20908, 3]

# Add vertex colors (e.g., a uniform color for the entire mesh)
vertex_colors = torch.ones_like(vertices)  # Shape: [1, 10475, 3]
vertex_colors = vertex_colors * torch.tensor([0.7, 0.7, 1.0], device="cuda")  # Light blue color

# Create a TexturesVertex object
textures = TexturesVertex(verts_features=vertex_colors)

# Create a Meshes object for rendering
meshes = Meshes(
    verts=vertices,  # Vertices of the mesh
    faces=torch.tensor(faces.astype(np.int64)).cuda().unsqueeze(0),  # Faces of the mesh
    textures=textures  # Vertex colors
)

# Set up a camera
R, T = look_at_view_transform(dist=2.7, elev=0, azim=180)  # Camera position
cameras = PerspectiveCameras(device="cuda", R=R, T=T)

# Set up rasterization settings
raster_settings = RasterizationSettings(
    image_size=512,  # Size of the rendered image
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Set up lights
lights = PointLights(device="cuda", location=[[0.0, 0.0, -3.0]])

# Set up the renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device="cuda",
        cameras=cameras,
        lights=lights
    )
)

# Render the mesh
rendered_image = renderer(meshes)

# Convert the rendered image to a numpy array
rendered_image = rendered_image[0, ..., :3].detach().cpu().numpy()  # Shape: [512, 512, 3]

# Convert to a PIL image and save
rendered_image_pil = Image.fromarray((rendered_image * 255).astype(np.uint8))
rendered_image_pil.save("rendered_smplx.png")

print("Rendered image saved as rendered_smplx.png")
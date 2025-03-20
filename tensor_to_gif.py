import torch
import numpy as np
import imageio
import os


tensor = torch.load('vorticity.pt')['vorticity']
mask = torch.load('vorticity.pt')['mask']


# Convert to NumPy and remove the last dimension
tensor = tensor.numpy().squeeze(-1) 
mask = mask.numpy().squeeze(-1) 

print(f"Tensor Shape is : {tensor.shape}")
print(f"Mask Shape is : {mask.shape}")

# Parameters
num_frames = 20  
stride = 1
save_dir = "flow_dataset"
mask_dir = "mask_dataset"

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Generate GIFs with a sliding window
for i in range(0, tensor.shape[0] - num_frames + 1, stride):
    frames = tensor[i : i + num_frames]  
    masks = mask[i : i + num_frames]
    gif_path = os.path.join(save_dir, f"vorticity_{i}.gif")
    mask_path = os.path.join(mask_dir, f"mask_{i}.gif")
    imageio.mimsave(gif_path, (frames * 255).astype(np.uint8), duration=0.1) 
    imageio.mimsave(mask_path, (masks * 255).astype(np.uint8), duration=0.1)

print(f"GIFs saved in {save_dir}")

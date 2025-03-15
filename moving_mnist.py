import numpy as np
import os
import imageio.v2 as imageio

data = np.load('mnist_test_seq.npy')

output_dir = "moving_mnist"
os.makedirs(output_dir, exist_ok=True)

# Iterate over the datapoints (second axis)
num_datapoints = data.shape[1]

for i in range(num_datapoints//10):
    # Extract sequence of 20 images
    sequence = data[:, i, :, :]
    
    # Define output file name
    output_path = os.path.join(output_dir, f"moving_mnist_{i+1}.gif")

    # Save as GIF
    imageio.mimsave(output_path, sequence, duration=0.1)  # 0.1s per frame

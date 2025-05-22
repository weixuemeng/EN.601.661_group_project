import pickle
import numpy as np
import os

# === INPUT FILE ===
input_pkl = "border_extrinsics.pkl"  # replace with your actual .pkl file
output_folder = "extrinsics_txt"
os.makedirs(output_folder, exist_ok=True)

# === LOAD POSES ===
with open(input_pkl, 'rb') as f:
    poses = pickle.load(f)

# === CONVERT AND SAVE EACH MATRIX ===
for pose in poses:
    image_name = os.path.splitext(pose["image"])[0]
    output_path = os.path.join(output_folder, f"{image_name}.txt")

    matrix_3x4 = np.array(pose["extrinsic"])
    if matrix_3x4.shape != (3, 4):
        raise ValueError(f"{image_name}: Expected 3x4 matrix, got {matrix_3x4.shape}")

    # Convert to 4x4 by adding a [0 0 0 1] row
    matrix_4x4 = np.vstack([matrix_3x4, [0, 0, 0, 1]])

    with open(output_path, 'w') as f:
        for row in matrix_4x4:
            line = " ".join(f"{val:.17e}" for val in row)
            f.write(line + "\n")


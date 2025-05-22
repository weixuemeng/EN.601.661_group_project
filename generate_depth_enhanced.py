
import os
import argparse

import cv2
import numpy as np

def mask_depth_images(depth_dir, mask_dir, output_dir):
    """
    For each file in depth_dir, look for a mask with the same name in mask_dir,
    then zero out all depth pixels where the mask is 0 (background).
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(depth_dir):
        depth_path = os.path.join(depth_dir, filename)
        mask_path  = os.path.join(mask_dir,  filename)

        if not os.path.isfile(mask_path):
            print(f"[WARN] no mask for {filename}, skipping")
            continue

        # Read depth (keep original bit‐depth) and mask (grayscale)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask  = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)

        if depth is None:
            print(f"[ERROR] failed to read depth image {depth_path}")
            continue
        if mask is None:
            print(f"[ERROR] failed to read mask image  {mask_path}")
            continue

        # Build a boolean mask (True where ROI)
        roi = (mask == 255)

        # Allocate output and copy only ROI pixels
        masked_depth = np.zeros_like(depth)
        masked_depth[roi] = depth[roi]

        # Write out
        out_path = os.path.join(output_dir, filename)
        if cv2.imwrite(out_path, masked_depth):
            print(f"[OK] saved masked depth → {out_path}")
        else:
            print(f"[ERROR] could not write {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply ROI masks (0/255 PNGs) to depth images"
    )
    parser.add_argument(
        "--depth_folder",  "-d", required=True,
        help="Path to folder containing depth images"
    )
    parser.add_argument(
        "--mask_folder",   "-m", required=True,
        help="Path to folder containing 0/255 ROI masks (same filenames)"
    )
    parser.add_argument(
        "--output_folder", "-o", required=True,
        help="Where to save the masked depth images"
    )
    args = parser.parse_args()
    mask_depth_images(args.depth_folder, args.mask_folder, args.output_folder)

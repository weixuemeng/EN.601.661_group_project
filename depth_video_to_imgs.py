#!/usr/bin/env python3
import argparse
import os

import imageio
import numpy as np
from PIL import Image

def main():
    p = argparse.ArgumentParser(
        description="Read a Z16 depth AVI (via ffmpeg) and dump each to a 16-bit PNG"
    )
    p.add_argument("input_file", help="Path to your depth_gorilla.avi")
    p.add_argument("output_dir", help="Where to save the PNGs")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    reader = imageio.get_reader(args.input_file, "ffmpeg")

    for idx, frame in enumerate(reader):
        # frame might be H×W×3 or H×W; pick one channel if RGB
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        depth = frame
        # up‐cast if needed
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        img = Image.fromarray(depth)
        if img.mode != "I;16":
            img = img.convert("I;16")

        outpath = os.path.join(args.output_dir, f"depth_{idx:05d}.png")
        img.save(outpath)
        print(f"Saved {outpath}")

    reader.close()

if __name__ == "__main__":
    main()

import cv2
import os

# Path to your PNG image
image_path = "/home/kuka/6dpose/FoundationPose/datasets/linemod/ref_views/ob_0000001/model/material_0.png"  # change this to your image path
output_path = "grayscale_output.png"

# Read the image in color
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite(output_path, gray)

print(f"[INFO] Grayscale image saved to {output_path}")

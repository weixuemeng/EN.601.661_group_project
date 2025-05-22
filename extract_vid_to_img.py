import cv2
import os

# ========== CONFIGURATION ==========
video_path = "/home/kuka/6dpose/EN.601.661_group_project/recordings/depth_gorilla.avi"        # Replace with your video file path
output_folder = "depth"       # Folder to save extracted images
image_prefix = "frame"                # Prefix for image filenames
image_format = "png"                  # Can be 'jpg', 'png', etc.

# ========== CREATE OUTPUT FOLDER ==========
os.makedirs(output_folder, exist_ok=True)

# ========== OPEN VIDEO ==========
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Cannot open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    filename = f"{image_prefix}_{frame_count:03d}.{image_format}"
    filepath = os.path.join(output_folder, filename)

    cv2.imwrite(filepath, frame)
    frame_count += 1

print(f"[INFO] Saved {frame_count} frames to '{output_folder}'")

cap.release()

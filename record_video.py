import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create output folder
os.makedirs("recordings", exist_ok=True)

# Setup pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Define video writers
color_writer = cv2.VideoWriter('recordings/color_gorilla.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
depth_writer = cv2.VideoWriter('recordings/depth_gorilla.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), isColor=False)

print("Recording... Press 'q' to stop.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth for visualization (optional)
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # Write to files
        color_writer.write(color_image)
        depth_writer.write(depth_vis)

        # Display
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Stopping...")
    pipeline.stop()
    color_writer.release()
    depth_writer.release()
    cv2.destroyAllWindows()


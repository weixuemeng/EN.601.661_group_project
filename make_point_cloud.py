import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()
# Generate point cloud
pc = rs.pointcloud()
pc.map_to(color_frame)
points = pc.calculate(depth_frame)  # compute the point cloud&#8203;:contentReference[oaicite:5]{index=5}
points.export_to_ply("object_scan.ply", color_frame)  # save to PLY file


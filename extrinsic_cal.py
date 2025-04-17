import cv2
import numpy as np
import glob
import os
import argparse
import pickle

def create_border_board_objectPoints(
    markersX, markersY, marker_length, marker_separation, aruco_dict
):
    """
    Create objectPoints and ids arrays for a border-only ArUco board.
    """
    objectPoints = []
    ids = []
    marker_id = 0

    for row in range(markersY):
        for col in range(markersX):
            # Only include border markers
            if row in (0, markersY - 1) or col in (0, markersX - 1):
                x = col * (marker_length + marker_separation)
                y = row * (marker_length + marker_separation)
                corners = np.array([
                    [x, y, 0],
                    [x + marker_length, y, 0],
                    [x + marker_length, y + marker_length, 0],
                    [x, y + marker_length, 0]
                ], dtype=np.float32)
                objectPoints.append(corners)
                ids.append(marker_id)
                marker_id += 1

    return objectPoints, np.array(ids, dtype=np.int32)

def estimate_extrinsics_border(
    image_folder,
    intrinsics_file,
    dict_id,
    markersX,
    markersY,
    marker_length,
    marker_separation,
    show=False
):
    """
    Estimate extrinsic parameters for a border-only ArUco board using the new ArucoDetector API.
    """
    # Load intrinsics
    data = np.load(intrinsics_file)
    K, dist = data['K'], data['dist']

    # Prepare dictionary and detector (new API)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    parameters = cv2.aruco.DetectorParameters()  # new API class
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Create custom Board
    objectPoints, ids = create_border_board_objectPoints(
        markersX, markersY, marker_length, marker_separation, aruco_dict
    )
    board = cv2.aruco.Board_create(objectPoints, aruco_dict, ids)

    poses = []
    img_paths = sorted(glob.glob(os.path.join(image_folder, "*.[pj][pn]g")))
    for path in img_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detectMarkers via new ArucoDetector
        corners, detected_ids, _ = detector.detectMarkers(gray)
        if detected_ids is None or len(detected_ids) == 0:
            continue

        # estimatePoseBoard unchanged
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, detected_ids, board, K, dist, None, None
        )
        if not retval:
            continue

        R, _ = cv2.Rodrigues(rvec)
        extrinsic = np.hstack((R, tvec.reshape(3,1)))

        poses.append({
            "image": os.path.basename(path),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "extrinsic": extrinsic.tolist()
        })

        print(poses[-1])

        if show:
            cv2.aruco.drawDetectedMarkers(img, corners, detected_ids)
            cv2.aruco.drawAxis(img, K, dist, rvec, tvec, marker_length)
            cv2.imshow("Pose", img)
            cv2.waitKey(200)

    if show:
        cv2.destroyAllWindows()

    return poses

def main():
    parser = argparse.ArgumentParser(
        description="Extrinsic calibration for border-only ArUco board"
    )
    parser.add_argument("--images", "-i", required=True, help="Folder with calibration images")
    parser.add_argument("--intrinsics", "-n", required=True, help=".npz file with 'K' and 'dist'")
    parser.add_argument("--markersX", type=int, default=5, help="Markers in X direction")
    parser.add_argument("--markersY", type=int, default=7, help="Markers in Y direction")
    parser.add_argument("--marker-length", "-l", type=float, required=True, help="Marker side length (meters)")
    parser.add_argument("--marker-sep", "-s", type=float, required=True, help="Marker separation (meters)")
    parser.add_argument("--dict", "-d", default="DICT_6X6_250", help="ArUco dictionary name")
    parser.add_argument("--output", "-o", default="extrinsics_border.pkl", help="Output pickle filename")
    parser.add_argument("--show", action="store_true", help="Show detected markers and axes")
    args = parser.parse_args()

    dict_id = getattr(cv2.aruco, args.dict)
    poses = estimate_extrinsics_border(
        args.images,
        args.intrinsics,
        dict_id,
        args.markersX,
        args.markersY,
        args.marker_length,
        args.marker_sep,
        show=args.show
    )

    with open(args.output, "wb") as f:
        pickle.dump(poses, f)
    print(f"Saved {len(poses)} extrinsic poses to {args.output}")

if __name__ == "__main__":
    main()
'''
python calibrate_extrinsics_border.py \
  --images ./calib_border \
  --intrinsics realsense_intrinsics_border.npz \
  --marker-length 0.04 \
  --marker-sep 0.01 \
  --dict DICT_6X6_250 \
  --output border_extrinsics.pkl'''
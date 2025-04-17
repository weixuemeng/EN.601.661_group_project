#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import argparse

def calibrate_intrinsics(image_folder, dict_id, marker_length, marker_separation, show=False):
    """
    Detects ArUco markers in all images of a folder and performs intrinsic calibration.
    Returns: (rms_error, camera_matrix, dist_coeffs)
    """
    # 1. Prepare dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    parameters = cv2.aruco.DetectorParameters_create()

    all_corners, all_ids = [], []
    image_size = None

    # 2. Load and detect in each image
    img_paths = sorted(glob.glob(f"{image_folder}/*.[pj][pn]g"))
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > 0:
            all_corners.append(corners)
            all_ids.append(ids)

            if show:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                cv2.imshow("Detection", img)
                cv2.waitKey(200)

    if show:
        cv2.destroyAllWindows()

    if not all_ids:
        raise RuntimeError("No markers detected in any image.")

    # 3. Prepare count of detected markers per image
    marker_counts = np.array([len(ids) for ids in all_ids])

    # 4. Calibrate
    rms, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraAruco(
        all_corners,
        all_ids,
        marker_counts,
        marker_length,
        marker_separation,
        image_size,
        None,
        None
    )

    return rms, camera_matrix, dist_coeffs

def main():
    p = argparse.ArgumentParser(description="Intrinsic calibration from ArUco images")
    p.add_argument("--images", "-i", required=True,
                   help="Folder with calibration images (.png/.jpg)")
    p.add_argument("--marker-length", "-l", type=float, required=True,
                   help="Marker side length (in meters)")
    p.add_argument("--marker-sep", "-s", type=float, required=True,
                   help="Separation between markers (in meters)")
    p.add_argument("--dict", "-d", default="DICT_6X6_250",
                   help="ArUco dictionary (e.g. DICT_6X6_250)")
    p.add_argument("--show", action="store_true",
                   help="Show detected markers during processing")
    p.add_argument("--output", "-o", default="intrinsics.npz",
                   help="Output .npz filename for K and dist")
    args = p.parse_args()

    dict_id = getattr(cv2.aruco, args.dict)
    rms, K, dist = calibrate_intrinsics(
        args.images, dict_id, args.marker_length, args.marker_sep, args.show
    )

    print(f"RMS re-projection error: {rms:.4f}")
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients:\n", dist.ravel())

    np.savez(args.output, K=K, dist=dist)
    print(f"Saved intrinsics to: {args.output}")

if __name__ == "__main__":
    main()

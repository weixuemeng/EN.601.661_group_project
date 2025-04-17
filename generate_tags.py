#!/usr/bin/env python3
import cv2
import numpy as np
import cv2.aruco as aruco

def create_border_board(
    markersX=5,
    markersY=7,
    markerSizePx=200,
    sepPx=50,
    dictionary=aruco.DICT_6X6_250
):
    """
    Generate a border-only ArUco board image.

    Args:
        markersX (int): number of markers along top/bottom edges.
        markersY (int): number of markers along left/right edges.
        markerSizePx (int): size of each marker in pixels.
        sepPx (int): separation (in pixels) between markers and from edges.
        dictionary (int): OpenCV ArUco dictionary identifier.

    Returns:
        board_img (ndarray): grayscale image of the board.
    """
    # 1. Prepare dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary)

    # 2. Compute full canvas size
    img_w = markersX * markerSizePx + (markersX + 1) * sepPx
    img_h = markersY * markerSizePx + (markersY + 1) * sepPx
    board_img = 255 * np.ones((img_h, img_w), dtype=np.uint8)

    # 3. Draw only border markers
    marker_id = 0
    for row in range(markersY):
        for col in range(markersX):
            # If on any border side, place a marker
            if row in (0, markersY - 1) or col in (0, markersX - 1):
                # Low‐level marker generator
                marker = aruco.generateImageMarker(
                    aruco_dict, marker_id, markerSizePx
                )
                # Position on canvas
                x0 = sepPx + col * (markerSizePx + sepPx)
                y0 = sepPx + row * (markerSizePx + sepPx)
                board_img[y0:y0 + markerSizePx, x0:x0 + markerSizePx] = marker
                marker_id += 1

    return board_img

if __name__ == "__main__":
    # Customize parameters here if desired
    board = create_border_board(
        markersX=5,
        markersY=7,
        markerSizePx=200,
        sepPx=50,
        dictionary=aruco.DICT_6X6_250
    )

    # Save and (optionally) preview
    out_path = "border_aruco_board.png"
    cv2.imwrite(out_path, board)
    print(f"Saved border ArUco board to: {out_path}")

    # Uncomment below to preview using OpenCV's GUI
    # cv2.imshow("Border ArUco Board", board)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

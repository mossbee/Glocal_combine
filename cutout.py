import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceLandmarker object.
landmark_base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
landmark_options = vision.FaceLandmarkerOptions(base_options=landmark_base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
landmark_detector = vision.FaceLandmarker.create_from_options(landmark_options)

def get_position(image_path):
    try:
        image = mp.Image.create_from_file(image_path)
        landmarks_detected = landmark_detector.detect(image)

        normalized_face_landmarks = landmarks_detected.face_landmarks[0]
        face_landmarks = [[int(normalized_face_landmark.x * image.width), int(normalized_face_landmark.y * image.height)] for normalized_face_landmark in normalized_face_landmarks]
        return face_landmarks
    except Exception as e:
        print(e)
        return None, None, None
    
def visualize_landmarks(image_path, face_landmarks):
    img = cv2.imread(image_path)
    for idx, landmark in enumerate(face_landmarks):
        cv2.circle(img, (landmark[0], landmark[1]), radius=1, color=(0,255,0), thickness=-1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(idx)
        font_scale = 0.2
        font_thickness = 1
        text_color = (255, 0, 0)
        text_pos = (landmark[0], landmark[1])
        cv2.putText(img, text, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # save image
    cv2.imwrite("output_image.jpg", img)

def visualize_one_landmarks(image_path, face_landmarks, indexes):
    img = cv2.imread(image_path)
    for index in indexes:
        landmark = face_landmarks[index]
        cv2.circle(img, (landmark[0], landmark[1]), radius=1, color=(0,255,0), thickness=-1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(index)
        font_scale = 0.2
        font_thickness = 1
        text_color = (255, 0, 0)
        text_pos = (landmark[0], landmark[1])
        cv2.putText(img, text, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # save image
    cv2.imwrite("output_image.jpg", img)

def square_from_side_midpoints(A, B):
    """
    Given two 2D points A and B (each a tuple/list of (x, y)), which are midpoints of two sides of a square,
    return a list of the 4 vertices of the square as (x, y) tuples, in any order.
    """
    x1, y1 = A
    x2, y2 = B

    # Vector between midpoints
    dx = x2 - x1
    dy = y2 - y1

    # Perpendicular vector for corner directions (normalize to half-length)
    n_half = (-dy/2, dx/2)

    # Four vertices
    corners = [
        (x1 + n_half[0], y1 + n_half[1]),
        (x1 - n_half[0], y1 - n_half[1]),
        (x2 + n_half[0], y2 + n_half[1]),
        (x2 - n_half[0], y2 - n_half[1]),
    ]
    return corners

def order_points(pts):
    """
    Given a list/array of four (x, y) points, returns them ordered as:
    [top-left, top-right, bottom-right, bottom-left]
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]         # Top-left has smallest sum
    ordered[2] = pts[np.argmax(s)]         # Bottom-right has largest sum
    ordered[1] = pts[np.argmin(diff)]      # Top-right has smallest diff
    ordered[3] = pts[np.argmax(diff)]      # Bottom-left has largest diff
    return ordered

def warp_perspective_cutout(img_path, corners, output_path, output_size=None):
    """
    Args:
        img_path: str, path to input image
        corners: list of 4 (x, y) tuples, order: [top-left, top-right, bottom-right, bottom-left]
        output_path: str, path to save the result
        output_size: tuple of (width, height), optional; if None, inferred from distances
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not open: {img_path}")
    if len(corners) != 4:
        raise ValueError("corners must be a list of four (x, y) tuples.")

    # Convert to np.array
    pts = order_points(corners)
    

    # If output size not specified, compute width and height from points
    (tl, tr, br, bl) = pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if output_size is not None:
        maxWidth, maxHeight = output_size

    # Destination points for "unwarped" rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight -1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Save image
    cv2.imwrite(output_path, warped)
    print(f"Warped image saved to {output_path}")

def visualize_square(image_path, face_landmarks, indexes, output):
    img = cv2.imread(image_path)
    corners = square_from_side_midpoints(face_landmarks[indexes[0]], face_landmarks[indexes[1]])
    warp_perspective_cutout(image_path, corners, output)

face_part = {
    "left_eye" : [35, 168],
    "right_eye" : [168, 265],
    "mouth" : [61, 291],
    "nose" : [36, 266],
    "chin" : [32, 262]
}

    
if __name__ == "__main__":
    image_path = "/home/mossbee/Work/AdaFace/ffhq/00001.png"
    face_landmarks = get_position(image_path)
    for key in face_part.keys():
        visualize_square(image_path, face_landmarks, face_part[key], f"{key}.jpg")
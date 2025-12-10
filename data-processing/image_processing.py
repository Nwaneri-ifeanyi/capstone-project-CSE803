import cv2
import numpy as np

def align_and_process(image, parsed_objects, face_class_id=0, eye_class_id=1):
    """
    Executes the alignment pipeline:
    1. Angle Calculation (from eyes)
    2. Rotation
    3. Face Cropping (adjusted for rotation)
    4. Filtering (HistEq + Bilateral)
    """
    # --- 1. Filter Data ---
    faces = [obj for obj in parsed_objects if obj['class_id'] == face_class_id]
    eyes = [obj for obj in parsed_objects if obj['class_id'] == eye_class_id]
    
    if not faces:
        raise ValueError("No face detected for cropping.")
    if len(eyes) < 2:
        raise ValueError("Need at least 2 eyes for alignment calculation.")

    # Select largest face and sort eyes left-to-right
    main_face = max(faces, key=lambda o: o['bbox'][2] * o['bbox'][3])
    eyes_sorted = sorted(eyes, key=lambda o: o['center'][0])
    left_eye = eyes_sorted[0]
    right_eye = eyes_sorted[-1]

    # --- 2. Calculate Angle & Rotate ---
    # "Calculating the center of each eye and measuring angle" [cite: 120]
    dY = right_eye['center'][1] - left_eye['center'][1]
    dX = right_eye['center'][0] - left_eye['center'][0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Center of rotation (midpoint between eyes)
    eyes_midpoint = (
        int((left_eye['center'][0] + right_eye['center'][0]) // 2),
        int((left_eye['center'][1] + right_eye['center'][1]) // 2)
    )
    
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # --- 3. Transform Face Bounding Box ---
    # We  rotate the center of the face box to find where to crop on the new image
    fx, fy, fw, fh = main_face['bbox']
    face_center_original = np.array([fx + fw / 2, fy + fh / 2, 1.0])
    
    new_face_center = M.dot(face_center_original)
    new_cx, new_cy = int(new_face_center[0]), int(new_face_center[1])
    
    # "Crop it into fixed regions" 
    start_x = max(0, int(new_cx - fw / 2))
    start_y = max(0, int(new_cy - fh / 2))
    end_x = min(w, int(new_cx + fw / 2))
    end_y = min(h, int(new_cy + fh / 2))
    
    face_crop = rotated_img[start_y:end_y, start_x:end_x]
    
    if face_crop.size == 0:
        raise ValueError("Crop failed - computed region is outside image bounds.")

    # --- 4. Final Processing ---

    resized = cv2.resize(face_crop, (64, 64))
    
    # Convert to Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # "Histogram equalization is performed to increase the contrast" 
    gray_eq = cv2.equalizeHist(gray)
    
    # "Bilateral filter used for noise reduction" 
    # Parameters (9, 75, 75) are standard for preserving edges while smoothing
    processed_final = cv2.bilateralFilter(gray_eq, 9, 75, 75)
    
    return rotated_img, processed_final
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import image_processing as processor

# --- CONFIGURATION ---
image_path = r'/CSE Assignment/HW5/CSE_final_project/upload/18.jpg'
txt_path = r'/CSE_final_project/upload/18.txt'
CLASS_MAPPING = ["face", "eye", "mouth", "ear"] 

def draw_alignment_geometry(image, parsed_data):
    """
    Draws the visual guides (Eye-line, Horizontal-line, Angle) on the image.
    """
    vis_img = image.copy()
    
    # 1. Get Eyes
    eyes = [obj for obj in parsed_data if obj['class_id'] == 1] # Class 1 = Eye
    if len(eyes) < 2: return vis_img, None, None
    
    # Sort Left to Right
    eyes_sorted = sorted(eyes, key=lambda o: o['center'][0])
    left_eye = eyes_sorted[0]['center']
    right_eye = eyes_sorted[-1]['center']
    
    # 2. Calculate Geometry
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 3. Draw Lines
    cv2.line(vis_img, left_eye, right_eye, (0, 255, 0), 3) # Green Angle Line
    
    horizontal_end = (int(left_eye[0] + dX), left_eye[1])
    cv2.line(vis_img, left_eye, horizontal_end, (255, 0, 0), 2, cv2.LINE_AA) # Blue Horizontal
    
    # 4. Draw Text
    text_pos = (left_eye[0] + 20, left_eye[1] + 20)
    cv2.putText(vis_img, f"Angle: {angle:.1f} deg", text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Calculate Matrix for return
    eyes_midpoint = (
        int((left_eye[0] + right_eye[0]) // 2),
        int((left_eye[1] + right_eye[1]) // 2)
    )
    M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
    
    return vis_img, M, (left_eye, right_eye)

def draw_aligned_visuals(aligned_img, M, original_eye_points):
    """
    Draws the horizontal line on the ALIGNED image.
    """
    if M is None or original_eye_points is None: return aligned_img
    vis_aligned = aligned_img.copy()
    
    pts = np.array([original_eye_points[0], original_eye_points[1]], dtype="float32")
    pts = np.array([pts]) 
    new_pts = cv2.transform(pts, M)[0]
    
    new_left = (int(new_pts[0][0]), int(new_pts[0][1]))
    new_right = (int(new_pts[1][0]), int(new_pts[1][1]))
    
    cv2.line(vis_aligned, new_left, new_right, (0, 255, 0), 3)
    return vis_aligned

def get_high_res_crop(image, parsed_data):
    """
    Extracts the aligned face at ORIGINAL resolution for visual verification.
    """
    faces = [obj for obj in parsed_data if obj['class_id'] == 0]
    eyes = [obj for obj in parsed_data if obj['class_id'] == 1]
    
    if not faces or len(eyes) < 2: return None

    # Get Geometry
    main_face = max(faces, key=lambda o: o['bbox'][2] * o['bbox'][3])
    eyes_sorted = sorted(eyes, key=lambda o: o['center'][0])
    left_center = eyes_sorted[0]['center']
    right_center = eyes_sorted[-1]['center']
    
    # Calculate Rotation
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    eyes_midpoint = (
        int((left_center[0] + right_center[0]) // 2),
        int((left_center[1] + right_center[1]) // 2)
    )
    
    # Rotate Full Image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # Transform Face Box Center
    fx, fy, fw, fh = main_face['bbox']
    face_center_orig = np.array([fx + fw / 2, fy + fh / 2, 1.0])
    new_face_center = M.dot(face_center_orig)
    new_cx, new_cy = int(new_face_center[0]), int(new_face_center[1])
    
    # Crop at High Res
    start_x = max(0, int(new_cx - fw / 2))
    start_y = max(0, int(new_cy - fh / 2))
    end_x = min(w, int(new_cx + fw / 2))
    end_y = min(h, int(new_cy + fh / 2))
    
    return rotated_img[start_y:end_y, start_x:end_x]

def main():
    # 1. Load Data
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image at {image_path}")
        return
    H_img, W_img, _ = image.shape

    parsed_data = utils.parse_yolo_annotations(txt_path, W_img, H_img)
    if not parsed_data:
        print("No valid data found.")
        return

    # 2. Create Visualization of Intermediate Steps (Original Image)
    geo_viz_img, M, eye_pts = draw_alignment_geometry(image, parsed_data)

    # 3. Get High Resolution Crop (New Step)
    high_res_crop = get_high_res_crop(image, parsed_data)

    # 4. Run the Actual Processing (Low Res Output)
    try:
        aligned_img, final_img = processor.align_and_process(
            image, parsed_data, face_class_id=0, eye_class_id=1
        )
        
        # 5. Create Visualization for the ALIGNED full image
        aligned_viz_img = draw_aligned_visuals(aligned_img, M, eye_pts)

        # --- PLOT EVERYTHING (4 Columns) ---
        plt.figure(figsize=(20, 6))

        # View 1: Geometry
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(geo_viz_img, cv2.COLOR_BGR2RGB))
        plt.title("1. Angle Calculation")
        plt.axis('off')

        # View 2: Aligned Body
        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(aligned_viz_img, cv2.COLOR_BGR2RGB))
        plt.title("2. Aligned (Horizontal Eyes)")
        plt.axis('off')
        
        # View 3: High Res Crop
        plt.subplot(1, 4, 3)
        if high_res_crop is not None and high_res_crop.size > 0:
            plt.imshow(cv2.cvtColor(high_res_crop, cv2.COLOR_BGR2RGB))
            shape_str = f"{high_res_crop.shape[1]}x{high_res_crop.shape[0]}"
            plt.title(f"3. High-Res Crop\n({shape_str})")
        else:
            plt.text(0.5, 0.5, "Crop Failed", ha='center')
        plt.axis('off')

        # View 4: Final Output
        plt.subplot(1, 4, 4)
        plt.imshow(final_img, cmap='gray')
        plt.title("4. Final CNN Input\n(64x64)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        print("Processing complete.")

    except ValueError as e:
        print(f"Processing Error: {e}")

if __name__ == "__main__":
    main()


















# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import utils as utils
# import image_processing as processor

# # --- CONFIGURATION ---
# image_path = r'/Users/ifeanyinwaneri/Library/CloudStorage/OneDrive-MichiganStateUniversity/MSU/Fall 2025/Computer Vision/CSE Assignment/HW5/CSE_final_project/1selected/47/IMG_4713.jpeg'
# txt_path = r'/Users/ifeanyinwaneri/Library/CloudStorage/OneDrive-MichiganStateUniversity/MSU/Fall 2025/Computer Vision/CSE Assignment/HW5/CSE_final_project/capstone-project-CSE803/main/18_detected.txt'
# # Adjust these to match your dataset's class IDs
# CLASS_MAPPING = ["face", "eye", "mouth", "ear"] 

# def draw_alignment_geometry(image, parsed_data):
#     """
#     Draws the visual guides (Eye-line, Horizontal-line, Angle) on the image.
#     Returns:
#         vis_img: The image with lines drawn.
#         M: The rotation matrix used (so we can map points to the next stage).
#         eyes_center: The point we rotated around.
#     """
#     vis_img = image.copy()
    
#     # 1. Get Eyes
#     eyes = [obj for obj in parsed_data if obj['class_id'] == 1] # Class 1 = Eye
#     if len(eyes) < 2: return vis_img, None, None
    
#     # Sort Left to Right
#     eyes_sorted = sorted(eyes, key=lambda o: o['center'][0])
#     left_eye = eyes_sorted[0]['center']  # (x, y)
#     right_eye = eyes_sorted[-1]['center'] # (x, y)
    
#     # 2. Calculate Geometry
#     dY = right_eye[1] - left_eye[1]
#     dX = right_eye[0] - left_eye[0]
#     angle = np.degrees(np.arctan2(dY, dX))
#     dist = np.sqrt(dX**2 + dY**2)
    
#     # 3. Draw "Angled" Line (Connecting Eyes)
#     # Green Line
#     cv2.line(vis_img, left_eye, right_eye, (0, 255, 0), 3)
    
#     # 4. Draw "Horizontal" Reference Line
#     # Blue Dashed Line (from left eye extending right)
#     horizontal_end = (int(left_eye[0] + dX), left_eye[1])
#     cv2.line(vis_img, left_eye, horizontal_end, (255, 0, 0), 2, cv2.LINE_AA)
    
#     # 5. Draw Angle Text
#     text_pos = (left_eye[0] + 20, left_eye[1] + 20)
#     cv2.putText(vis_img, f"Angle: {angle:.1f} deg", text_pos, 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     # --- Calculate Rotation Matrix for the "After" view ---
#     eyes_midpoint = (
#         int((left_eye[0] + right_eye[0]) // 2),
#         int((left_eye[1] + right_eye[1]) // 2)
#     )
#     M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
    
#     return vis_img, M, (left_eye, right_eye)

# def draw_aligned_visuals(aligned_img, M, original_eye_points):
#     """
#     Draws the horizontal line on the ALIGNED image to prove it worked.
#     """
#     if M is None or original_eye_points is None: return aligned_img
    
#     vis_aligned = aligned_img.copy()
    
#     # Transform the original eye points using the Rotation Matrix M
#     # We need to reshape points to (N, 1, 2) for cv2.transform
#     pts = np.array([original_eye_points[0], original_eye_points[1]], dtype="float32")
#     pts = np.array([pts]) 
    
#     # Apply rotation to points
#     new_pts = cv2.transform(pts, M)[0] # Returns list of new coordinates
    
#     new_left = (int(new_pts[0][0]), int(new_pts[0][1]))
#     new_right = (int(new_pts[1][0]), int(new_pts[1][1]))
    
#     # Draw the new line (Should be perfectly horizontal now)
#     cv2.line(vis_aligned, new_left, new_right, (0, 255, 0), 3)
    
#     return vis_aligned

# def main():
#     # 1. Load Data
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not open image at {image_path}")
#         return
#     H_img, W_img, _ = image.shape

#     parsed_data = utils.parse_yolo_annotations(txt_path, W_img, H_img)
#     if not parsed_data:
#         print("No valid data found.")
#         return

#     # 2. Create Visualization of Intermediate Steps
#     # This draws the angle and lines on the ORIGINAL image
#     geo_viz_img, M, eye_pts = draw_alignment_geometry(image, parsed_data)

#     # 3. Run the Actual Processing (Alignment)
#     try:
#         aligned_img, final_img = processor.align_and_process(
#             image, parsed_data, face_class_id=0, eye_class_id=1
#         )
        
#         # 4. Create Visualization for the ALIGNED image
#         # Use the matrix M to figure out where the eyes moved to
#         aligned_viz_img = draw_aligned_visuals(aligned_img, M, eye_pts)

#         # --- PLOT EVERYTHING ---
#         plt.figure(figsize=(15, 6))

#         # View 1: Geometry (Original with Angle)
#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(geo_viz_img, cv2.COLOR_BGR2RGB))
#         plt.title("1. Intermediate: Angle Calculation")
#         plt.axis('off')

#         # View 2: Aligned (With Horizontal Line)
#         plt.subplot(1, 3, 2)
#         plt.imshow(cv2.cvtColor(aligned_viz_img, cv2.COLOR_BGR2RGB))
#         plt.title("2. Result: Aligned (Horizontal Eyes)")
#         plt.axis('off')

#         # View 3: Final Output
#         plt.subplot(1, 3, 3)
#         plt.imshow(final_img, cmap='gray')
#         plt.title("3. Final CNN Input (64x64)")
#         plt.axis('off')

#         plt.tight_layout()
#         plt.show()
#         print("Processing complete.")

#     except ValueError as e:
#         print(f"Processing Error: {e}")

# # if __name__ == "__main__":
#     main()




# ###------------------- New Alihgnment Pipeline Code Ends Here -------------------###
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import utils as utils
# import image_processing as processor

# # --- CONFIGURATION ---
# image_path = r'../../upload/18.jpg'
# txt_path = r'../../upload/18.txt'
# CLASS_MAPPING = ["face", "eye", "mouth", "ear"] 

# def get_high_res_crop(image, parsed_data):
#     """
#     Extracts the aligned face at ORIGINAL resolution for visual verification.
#     """
#     # Reuse logic to get rotation matrix
#     faces = [obj for obj in parsed_data if obj['class_id'] == 0]
#     eyes = [obj for obj in parsed_data if obj['class_id'] == 1]
    
#     if not faces or len(eyes) < 2: return None

#     # Get Geometry
#     main_face = max(faces, key=lambda o: o['bbox'][2] * o['bbox'][3])
#     eyes_sorted = sorted(eyes, key=lambda o: o['center'][0])
#     left_center = eyes_sorted[0]['center']
#     right_center = eyes_sorted[-1]['center']
    
#     # Calculate Rotation
#     dY = right_center[1] - left_center[1]
#     dX = right_center[0] - left_center[0]
#     angle = np.degrees(np.arctan2(dY, dX))
    
#     eyes_midpoint = (
#         int((left_center[0] + right_center[0]) // 2),
#         int((left_center[1] + right_center[1]) // 2)
#     )
    
#     # Rotate Full Image
#     h, w = image.shape[:2]
#     M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
#     rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
#     # Crop Face (High Res)
#     fx, fy, fw, fh = main_face['bbox']
#     face_center_orig = np.array([fx + fw / 2, fy + fh / 2, 1.0])
#     new_face_center = M.dot(face_center_orig)
#     new_cx, new_cy = int(new_face_center[0]), int(new_face_center[1])
    
#     start_x = max(0, int(new_cx - fw / 2))
#     start_y = max(0, int(new_cy - fh / 2))
#     end_x = min(w, int(new_cx + fw / 2))
#     end_y = min(h, int(new_cy + fh / 2))
    
#     return rotated_img[start_y:end_y, start_x:end_x]

# def main():
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error loading image.")
#         return
#     H, W = image.shape[:2]

#     parsed_data = utils.parse_yolo_annotations(txt_path, W, H)
#     if not parsed_data: return

#     # 1. Get Visuals
#     annotated_img = utils.draw_annotations(image, parsed_data, CLASS_MAPPING)
#     high_res_crop = get_high_res_crop(image, parsed_data)

#     # 2. Run Pipeline
#     try:
#         aligned_img, final_img = processor.align_and_process(
#             image, parsed_data, face_class_id=0, eye_class_id=1
#         )

#         # --- PLOT (Now with 4 Columns) ---
#         plt.figure(figsize=(16, 5))

#         # 1. Original
#         plt.subplot(1, 4, 1)
#         plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
#         plt.title("1. Original Detection")
#         plt.axis('off')

#         # 2. High Res Crop (New!)
#         plt.subplot(1, 4, 2)
#         if high_res_crop is not None and high_res_crop.size > 0:
#             plt.imshow(cv2.cvtColor(high_res_crop, cv2.COLOR_BGR2RGB))
#             plt.title(f"2. High-Res Crop\n({high_res_crop.shape[1]}x{high_res_crop.shape[0]})")
#         else:
#             plt.text(0.5, 0.5, "Crop Failed", ha='center')
#         plt.axis('off')

#         # 3. Final Input (Low Res)
#         plt.subplot(1, 4, 3)
#         plt.imshow(final_img, cmap='gray')
#         plt.title("3. Final CNN Input\n(64x64 + Filter)")
#         plt.axis('off')
        
#         # 4. Final Histogram (Check Data Distribution)
#         plt.subplot(1, 4, 4)
#         plt.hist(final_img.ravel(), 256, [0, 256])
#         plt.title("4. Pixel Intensity Dist.")
        
#         plt.tight_layout()
#         plt.show()

#     except ValueError as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION ---
INPUT_ROOT = r'/Computer Vision/CSE Assignment/HW5/CSE_final_project/1selected'
OUTPUT_ROOT = r'/HW5/CSE_final_project/processed_dataset_highres'

# YOLO Files
WEIGHTS_PATH = "../main/yolov4-custom_best.weights"
CONFIG_PATH = "../main/yolov4-custom.cfg"
CLASSES_PATH = "../main/classes.txt"

# --- HELPER 1: OPTIMIZED DETECTION ---
def detect_single_image(net, image, output_layers):
    """
    Runs detection on a pre-loaded network.
    Returns list of parsed objects dictionary.
    """
    h, w = image.shape[:2]
    
    # Preprocess (608x608 as per paper)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Parse Outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # YOLO returns center coordinates
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    parsed_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            cid = class_ids[i]
            
            parsed_objects.append({
                'class_id': cid,
                'bbox': (x, y, bw, bh),
                'center': (x + bw // 2, y + bh // 2)
            })
            
    return parsed_objects

# --- HELPER 2: HIGH-RES ALIGNMENT & CROP ---
def crop_aligned_face_high_res(image, parsed_objects, face_id=0, eye_id=1):
    """
    Performs geometric alignment based on eyes, then crops the face 
    WITHOUT resizing or converting to grayscale.
    """
    faces = [obj for obj in parsed_objects if obj['class_id'] == face_id]
    eyes = [obj for obj in parsed_objects if obj['class_id'] == eye_id]
    
    # Validation: Need 1 Face and 2 Eyes
    if not faces or len(eyes) < 2:
        return None

    # 1. Select Main Face (Largest)
    main_face = max(faces, key=lambda o: o['bbox'][2] * o['bbox'][3])
    
    # 2. Sort Eyes (Left to Right)
    eyes_sorted = sorted(eyes, key=lambda o: o['center'][0])
    left_center = eyes_sorted[0]['center']
    right_center = eyes_sorted[-1]['center']
    
    # 3. Calculate Rotation Angle
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 4. Rotation Matrix (Rotate around midpoint between eyes)
    eyes_midpoint = (
        int((left_center[0] + right_center[0]) // 2),
        int((left_center[1] + right_center[1]) // 2)
    )
    
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
    
    # 5. Rotate Full Image
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # 6. Transform Face Box to Rotated Coordinates
    fx, fy, fw, fh = main_face['bbox']
    face_center_orig = np.array([fx + fw / 2, fy + fh / 2, 1.0])
    new_face_center = M.dot(face_center_orig)
    new_cx, new_cy = int(new_face_center[0]), int(new_face_center[1])
    
    # 7. Crop Original Resolution
    start_x = max(0, int(new_cx - fw / 2))
    start_y = max(0, int(new_cy - fh / 2))
    end_x = min(w, int(new_cx + fw / 2))
    end_y = min(h, int(new_cy + fh / 2))
    
    face_crop = rotated_img[start_y:end_y, start_x:end_x]
    
    # Return None if crop is invalid/empty
    if face_crop.size == 0:
        return None
        
    return face_crop

# --- MAIN BATCH LOOP ---
def main():
    # 1. Load Model ONCE
    print(f"Loading YOLO model from {WEIGHTS_PATH}...")
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 2. Get Class Folders
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input directory not found: {INPUT_ROOT}")
        return

    class_folders = sorted([f for f in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, f))])
    print(f"Found classes: {class_folders}")

    total_processed = 0
    total_failed = 0

    # 3. Process Folders
    for class_name in class_folders:
        input_dir = os.path.join(INPUT_ROOT, class_name)
        output_dir = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = glob.glob(os.path.join(input_dir, "*.[jJ][pP]*[gG]"))
        print(f"\nProcessing Class {class_name}: {len(image_files)} images...")

        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            try:
                # A. Read Image
                image = cv2.imread(img_path)
                if image is None: continue

                # B. Detect
                objects = detect_single_image(net, image, output_layers)
                if not objects:
                    total_failed += 1
                    continue

                # C. Align & Crop High-Res (No resizing/grayscale)
                high_res_face = crop_aligned_face_high_res(image, objects)
                
                if high_res_face is not None:
                    # D. Save
                    save_path = os.path.join(output_dir, filename)
                    cv2.imwrite(save_path, high_res_face)
                    total_processed += 1
                else:
                    total_failed += 1
                
            except Exception as e:
                print(f"  [Error] {filename}: {e}")
                total_failed += 1

    print("\n" + "="*30)
    print(f"High-Res Processing Complete.")
    print(f"Saved:   {total_processed}")
    print(f"Skipped: {total_failed}")
    print(f"Location: {OUTPUT_ROOT}")
    print("="*30)

if __name__ == "__main__":
    main()
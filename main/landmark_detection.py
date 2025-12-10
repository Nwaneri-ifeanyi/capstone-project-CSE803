import cv2
import numpy as np

def run_yolo_detection(image_path, config_path, weights_path, names_path, output_txt_path):
    """
    Runs YOLOv4 detection and saves the bounding boxes to a .txt file.
    
    Args:
        image_path (str): Path to input image.
        config_path (str): Path to .cfg file.
        weights_path (str): Path to .weights file.
        names_path (str): Path to .names file (class labels).
        output_txt_path (str): Path to save the resulting .txt annotation.
    """
    
    # 1. Load Class Names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 2. Load YOLO Model
    print(f"Loading YOLO model from {weights_path}...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Use GPU if available (optional, defaults to CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 3. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    h, w = img.shape[:2]

    # 4. Preprocess Image (Blob)
    # The paper mentions 608x608 input resolution
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)

    # 5. Forward Pass
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # 6. Process Detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter weak detections (Threshold: 0.5)
            if confidence > 0.5:
                # YOLO returns center_x, center_y, width, height (Normalized)
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Calculate top-left corner
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 7. Non-Maximum Suppression (NMS)
    # Removes overlapping boxes for the same object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # 8. Save Output to .txt File
    # Format matches your previous parser: "class_id normalized_cx normalized_cy normalized_w normalized_h"
    if len(indices) > 0:
        with open(output_txt_path, 'w') as f:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                cls_id = class_ids[i]
                
                # Convert back to normalized coordinates for consistency with your pipeline
                n_cx = (x + bw / 2) / w
                n_cy = (y + bh / 2) / h
                n_w = bw / w
                n_h = bh / h
                
                line = f"{cls_id} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f}\n"
                f.write(line)
        print(f"Success! Detection saved to {output_txt_path}")
    else:
        print("No objects detected.")

# --- Example Usage ---
# Adjust paths to match your folder structure
if __name__ == "__main__":
    # Your downloaded files
    weights = "yolov4-custom_best.weights"
    cfg = "yolov4-custom.cfg"  # You need to find this file in the authors' dataset
    names = "classes.txt"        # You need this file too (list of classes)
    
    # Input/Output
    img_in = "/IMG_4713.jpeg"
    txt_out = "18_detected.txt"
    
    try:
        run_yolo_detection(img_in, cfg, weights, names, txt_out)
    except Exception as e:
        print(e)
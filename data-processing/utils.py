import cv2

def parse_yolo_annotations(txt_path, img_width, img_height):
    """
    Parses a YOLO format text file into a list of dictionaries.
    Returns: [{'class_id': int, 'bbox': (x, y, w, h), 'center': (cx, cy)}, ...]
    """
    parsed_objects = []
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {txt_path}")
        return []

    for line in lines:
        data = line.strip().split()
        if len(data) < 5: continue
        
        class_id = int(data[0])
        # Normalized coordinates: center_x, center_y, width, height
        cx_n, cy_n, w_n, h_n = map(float, data[1:5])
        
        # Convert to absolute pixel coordinates
        cx_abs = cx_n * img_width
        cy_abs = cy_n * img_height
        w_abs = w_n * img_width
        h_abs = h_n * img_height
        
        # Calculate top-left corner (x, y) for OpenCV
        x = int(cx_abs - w_abs / 2)
        y = int(cy_abs - h_abs / 2)
        w = int(w_abs)
        h = int(h_abs)
        
        parsed_objects.append({
            'class_id': class_id,
            'bbox': (x, y, w, h),
            'center': (int(cx_abs), int(cy_abs))
        })
        
    return parsed_objects

def draw_annotations(image, parsed_objects, class_names):
    """
    Draws bounding boxes and labels on a copy of the image.
    """
    img_viz = image.copy()
    H_img, W_img, _ = img_viz.shape
    
    # Define color (Green) and thickness
    color = (0, 255, 0) 
    thickness = 2
    
    for obj in parsed_objects:
        class_id = obj['class_id']
        x, y, w, h = obj['bbox']
        
        # Ensure coordinates are within bounds
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(W_img - 1, x + w)
        y_max = min(H_img - 1, y + h)

        # Draw the rectangle
        cv2.rectangle(img_viz, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw Label
        if class_id < len(class_names):
            label = class_names[class_id]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # Text background
            cv2.rectangle(img_viz, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
            # Text
            cv2.putText(img_viz, label, (x_min, y_min - baseline), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
            
    return img_viz
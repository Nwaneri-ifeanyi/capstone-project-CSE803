import cv2
import numpy as np


image_path = r'/1selected/47/IMG_4713.jpeg' # image path 
txt_path = r'18_detected.txt'   # run landmark detection to get this file if you don't have it already 
CLASS_NAMES = ["face", "eye", "mouth", "ear"] 
# Read image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not open or find the image at {image_path}")
    exit()

# Get image dimensions
H_img, W_img, _ = image.shape

# Read bounding box data
with open(txt_path, 'r') as f:
    lines = f.readlines()

# process and Draw Each Bounding Box ---
for line in lines:
    data = line.strip().split()
    
    # Check if the line has the expected format (ClassID, Cx, Cy, W, H)
    if len(data) < 5:
        continue
    
    # Parse data: Class ID and normalized coordinates
    class_id = int(data[0])
    # Convert remaining strings to float for calculations
    cx_n, cy_n, w_n, h_n = map(float, data[1:5])
    
    # Convert normalized coordinates to absolute pixel coordinates (XYXY format)
    
    # Absolute center and dimensions
    cx_abs = cx_n * W_img
    cy_abs = cy_n * H_img
    w_abs = w_n * W_img
    h_abs = h_n * H_img
    
    # Calculate top-left (x_min, y_min) and bottom-right (x_max, y_max)
    x_min = int(cx_abs - w_abs / 2)
    y_min = int(cy_abs - h_abs / 2)
    x_max = int(cx_abs + w_abs / 2)
    y_max = int(cy_abs + h_abs / 2)
    
    # Ensure coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W_img - 1, x_max)
    y_max = min(H_img - 1, y_max)
    
    # Draw on the Image ---
    
    # Define color (BGR format for OpenCV) and thickness
    color = (0, 255, 0) # Green box
    thickness = 2
    
    # Draw the rectangle (bounding box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Add class label text
    if class_id < len(CLASS_NAMES):
        label = CLASS_NAMES[class_id]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw background for text
        cv2.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x_min, y_min - baseline), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

# Display or Save the Image ---

output_path = "1676_annotated.jpg"
cv2.imwrite(output_path, image)
print(f"Annotated image saved to {output_path}")

cv2.imshow("Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
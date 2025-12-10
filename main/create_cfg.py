import urllib.request
import re

def generate_goat_config():
    # 1. Download the standard YOLOv4 template from the official Darknet repo
    url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-custom.cfg"
    print(f"Downloading template from {url}...")
    
    try:
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    # 2. Define the changes based on the paper
    #  "input resolution was set to 608x608"
    new_width = 608
    new_height = 608
    
    # [cite: 108] "face, eye, nose and ear" -> 4 classes
    num_classes = 4
    
    # Formula: (classes + 5) * 3 -> (4 + 5) * 3 = 27
    num_filters = (num_classes + 5) * 3

    print("Applying configuration changes...")
    
    # Change 1: Update Network Resolution (Lines 8-9 usually)
    content = re.sub(r'width=\d+', f'width={new_width}', content)
    content = re.sub(r'height=\d+', f'height={new_height}', content)

    # Change 2: Update Classes (in every [yolo] layer)
    content = re.sub(r'classes=80', f'classes={num_classes}', content)

    # Change 3: Update Filters (ONLY in the [convolutional] layer before [yolo])
    # The pattern finds 'filters=255' followed closely by '[yolo]' and replaces it.
    # We use a specific regex to ensure we only change the correct conv layers.
    pattern = r'(filters=)255(\s+activation=linear\s+\[yolo\])'
    replacement = f'\\g<1>{num_filters}\\g<2>'
    content = re.sub(pattern, replacement, content)

    # 3. Save the new file
    output_filename = "yolov4-custom.cfg"
    with open(output_filename, 'w') as f:
        f.write(content)
        
    print(f"Success! Generated '{output_filename}' with:")
    print(f"- Width/Height: {new_width}x{new_height}")
    print(f"- Classes: {num_classes}")
    print(f"- Filters: {num_filters}")

if __name__ == "__main__":
    generate_goat_config()
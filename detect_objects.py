import torch
import torchvision
from torchvision.io import read_image
import requests
import sqlite3
from datetime import datetime  # Add this import statement
import json

# Define the cameras dictionary
cameras = {
    "5th Ave @ 34 St": "https://webcams.nyctmc.org/api/cameras/3a3d7bc0-7f35-46ba-9cca-75fe83aac34d/image",
    "2 Ave @ 74 St": "https://webcams.nyctmc.org/api/cameras/6316453d-6161-4b98-a8e7-0e36c69d267c/image",
    "E 14 St @ Irving Pl": "https://webcams.nyctmc.org/api/cameras/f9cb9d4c-10ad-42e4-8997-dbc9e12bd55a/image"
}

# Initialize SQLite database connection
conn = sqlite3.connect('detections.db')
c = conn.cursor()

# Create table to store detection information
c.execute('''CREATE TABLE IF NOT EXISTS detections
             (utc_timestamp TEXT, camera_location TEXT, detected_object TEXT, score REAL, bounding_box TEXT)''')

# Function to perform object detection and insert data into the database
def process_camera(camera_name, camera_url):
    # Download the image from the URL
    response = requests.get(camera_url)
    with open("downloaded_image.jpg", "wb") as f:
        f.write(response.content)

    # Load the image using read_image
    image = read_image("downloaded_image.jpg")

    # Convert the image to floating-point format
    image = image.float() / 255.0
        
    # Load COCO dataset annotations
    # coco = torchvision.datasets.CocoDetection('.', 'instances_train2017.json', transform=None)

    # Get category information from COCO dataset annotations
    # category_info = coco.coco.cats

    category_info = {1: {'supercategory': 'person', 'id': 1, 'name': 'person'},
                     2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                     3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                     4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                     5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
                     6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                     7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
                     8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
                     9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
                     10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
                     11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
                     13: {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
                     14: {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
                     15: {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
                     16: {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
                     17: {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
                     18: {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
                     19: {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
                     20: {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
                     21: {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
                     22: {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
                     23: {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
                     24: {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
                     25: {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
                     27: {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
                     28: {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
                     31: {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
                     32: {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
                     33: {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
                     34: {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
                     35: {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
                     36: {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
                     37: {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
                     38: {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
                     39: {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
                     40: {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
                     41: {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
                     42: {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
                     43: {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
                     44: {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
                     46: {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
                     47: {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
                     48: {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
                     49: {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
                     50: {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
                     51: {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
                     52: {'supercategory': 'food', 'id': 52, 'name': 'banana'},
                     53: {'supercategory': 'food', 'id': 53, 'name': 'apple'},
                     54: {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
                     55: {'supercategory': 'food', 'id': 55, 'name': 'orange'},
                     56: {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
                     57: {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
                     58: {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
                     59: {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
                     60: {'supercategory': 'food', 'id': 60, 'name': 'donut'},
                     61: {'supercategory': 'food', 'id': 61, 'name': 'cake'},
                     62: {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
                     63: {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
                     64: {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
                     65: {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
                     67: {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
                     70: {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
                     72: {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
                     73: {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
                     74: {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
                     75: {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
                     76: {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
                     77: {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
                     78: {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
                     79: {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
                     80: {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
                     81: {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
                     82: {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
                     84: {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
                     85: {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
                     86: {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
                     87: {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
                     88: {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
                     89: {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
                     90: {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}}

    # Load COCO dataset annotations
    # with open('./instances_train2017.json', 'r') as f:
    #     coco_annotations = json.load(f)
    
    # Get category information from COCO dataset annotations
    # category_info = {cat['id']: cat for cat in coco_annotations['categories']}

    # Use the image for object detection
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    # pth_path = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    # Load the model with the pretrained weights from the .pth file
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    # model.load_state_dict(torch.load(pth_path))
    model.eval()
    with torch.no_grad():
        output = model([image])[0]

    # Filter the output to only include detections with score > 0.5
    scores = output["scores"]
    boxes = output["boxes"]
    labels = output["labels"]
    # keep = scores > 0.5
    # scores = scores[keep]
    # boxes = boxes[keep]
    # labels = labels[keep]

    # Apply non-maximum suppression to remove redundant bounding boxes
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    scores = scores[keep]
    boxes = boxes[keep]
    labels = labels[keep]

    # Get current UTC timestamp
    utc_timestamp = str(datetime.utcnow())

    # Insert detection information into the database
    for i in range(len(scores)):
        detected_object = category_info[int(labels[i].item())]["name"]
        score = scores[i].item()
        bounding_box = ','.join([str(coord) for coord in boxes[i].tolist()])
        c.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?)", (utc_timestamp, camera_name, detected_object, score, bounding_box))

# Process each camera and insert data into the database
for location, url in cameras.items():
    process_camera(location, url)

# Commit changes and close connection
conn.commit()
conn.close()

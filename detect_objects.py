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
    coco = torchvision.datasets.CocoDetection('.', 'instances_train2017.json', transform=None)

    # Get category information from COCO dataset annotations
    category_info = coco.coco.cats

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

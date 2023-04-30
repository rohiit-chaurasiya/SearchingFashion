import cv2
import torch
import os
import hashlib
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the image transformation pipeline
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open the video file
video_path = "static/searchvideo/vid.mp4"
video = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = video.get(cv2.CAP_PROP_FPS)

# Set the frame interval (in seconds) to extract frames from
frame_interval = 1  # extract one frame per second

# Get the total number of frames in the video
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a folder to store the extracted clothing items
output_folder = "clothing_items"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Keep track of clothing items that have already been extracted
extracted_items = set()

# Loop through the video frames and extract the detected clothing items
for i in range(0, num_frames, int(fps * frame_interval)):
    # Set the current frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, i)

    # Read the frame from the video
    ret, frame = video.read()

    if ret:
        # Apply the transformation pipeline to the frame
        input_image = transform(frame)

        # Run the model on the input image
        with torch.no_grad():
            predictions = model([input_image])

        # Extract the predicted bounding boxes and labels from the model output
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Loop through the predicted bounding boxes and extract the clothing items
        for j, (box, label) in enumerate(zip(boxes, labels)):
            if label == 1:  # class 1 corresponds to 'person'
                x1, y1, x2, y2 = box.astype(int)
                clothing_item = frame[y1:y2, x1:x2]

                # Compute the hash of the clothing item to check if it has already been extracted
                clothing_item_hash = hashlib.sha256(clothing_item.copy()).hexdigest()
                if clothing_item_hash in extracted_items:
                    continue  # skip clothing item if it has already been extracted

                # Add the hash of the clothing item to the set of extracted items
                extracted_items.add(clothing_item_hash)

                clothing_item_filename = os.path.join(output_folder, "clothing_item_{}_{}_{}.jpg".format(i, j, label))
                cv2.imwrite(clothing_item_filename, clothing_item)
    else:
        break

# Release the video object
video.release()

import numpy as np
from PIL import Image
from DeepFeatures import DeepFeatures
from datetime import datetime
from flask import Flask, request, render_template, url_for
from pathlib import Path
import subprocess
import os
from flask import Flask, render_template

import cv2
import torch
import os
import hashlib
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import os
df = DeepFeatures()
# # set the path of the image folder
# image_folder = "clothing_items"
#
# # loop through all the images in the folder
# for image_file in os.listdir(image_folder):
#     # check if the file is an image file
#     if image_file.endswith('.jpg') or image_file.endswith('.png'):
#         # get the path of the image file
#         image_path = os.path.join(image_folder, image_file)
#
#         # convert the image to a numpy array
#         img = Image.open(image_path)
#         query = df.extract(img)
#
#         # get the features of all images in the feature folder
#         features = []
#         img_paths = []
#         for feature_path in Path("./static/feature").glob("*.npy"):
#             features.append(np.load(feature_path))
#             img_paths.append(Path("./static/image") / (feature_path.stem + ".jpg"))
#         features = np.array(features)
#
#         # compute the distance between the query image and all images in the feature folder
#         dists = np.linalg.norm(features - query, axis=1)
#         ids = np.argsort(dists)[:2]
#         scores = [(img_paths[id].stem, img_paths[id]) for id in ids]
#
#         # render the index.html template with the query image and the top 5 similar images
#         return render_template('index.html', query_path=image_path, scores=scores, header="header.html")


import os
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime

# Define the path to the image and feature folders
image_path = Path("static/image")
feature_path = Path("static/feature")

# Load the pre-extracted image features
features = []
img_paths = []
for feature_file in feature_path.glob("*.npy"):
    features.append(np.load(feature_file))
    img_paths.append(image_path / (feature_file.stem + ".jpg"))
features = np.array(features)

# Define the query image save path
query_img_save_path = Path("static/top_matches")

# Get all image files from the specified folder
for query_img_file in Path("clothing_items").glob("*"):
    # Load the query image
    query_img = Image.open(query_img_file)

    # Extract the features of the query image
    query = df.extract(query_img)

    # Compute the distances between the query and all the images
    dists = np.linalg.norm(features-query, axis=1)

    # Get the top 5 matching images
    ids = np.argsort(dists)[:1]

    # Save the matching images to the specified folder
    for id in ids:
        img_filename = img_paths[id].name
        img = Image.open(img_paths[id])
        matching_img_save_path = query_img_save_path / (query_img_file.stem + "_" + img_filename)
        img.save(matching_img_save_path)


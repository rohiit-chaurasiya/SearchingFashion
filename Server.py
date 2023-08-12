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

app = Flask(__name__)

df = DeepFeatures()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['query_img']
        filename = file.filename
        if filename.endswith('.jpg') or filename.endswith('.png'):
            features = []
            img_paths = []
            for feature_path in Path("./static/feature").glob("*.npy"):
                features.append(np.load(feature_path))
                img_paths.append(Path("./static/image") / (feature_path.stem + ".jpg"))
            features = np.array(features)
            # file = request.files['query_img']
            img = Image.open(file.stream)
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)
            query = df.extract(img)
            dists = np.linalg.norm(features-query, axis=1)
            ids = np.argsort(dists)[:5]
            scores = [(img_paths[id].stem, img_paths[id]) for id in ids]

            return render_template('index.html', query_path=uploaded_img_path, scores=scores,header="header.html")
        elif filename.endswith('.mp4'):
            # Define the folder containing the files
            folder1 = "clothing_items"
            folder2 = "static/top_matches"

            # Define the file extensions to delete
            extensions = (".jpg", ".png", ".mp4")

            # Delete all files with the specified extensions in the folder
            for folder in (folder1, folder2):
                for filename in os.listdir(folder):
                    if filename.endswith(extensions):
                        os.remove(os.path.join(folder, filename))

            # file = request.files['query_video']
            name = 'vid'
            uploaded_img_path = "static/searchvideo/" + name + ".mp4"
            file.save(uploaded_img_path)

            features = []
            img_paths = []

            # Replace the file path with the path to your Python file
            file_path_1 = 'ExtractImageFromVideo.py'
            file_path_2 = 'topmatchingcloths.py'
            file_path_3 = 'imagematching.py'


            # set the directory path where your images are stored
            folder_path = "top_matches"

            # Run the Python file using the subprocess module
            process_1 = subprocess.Popen(['python', file_path_1], stdout=subprocess.PIPE)
            process_1.wait()
            # Run the Python file using the subprocess module
            process_2 = subprocess.Popen(['python', file_path_2], stdout=subprocess.PIPE)
            process_2.wait()

            process_3 = subprocess.Popen(['python', file_path_3], stdout=subprocess.PIPE)
            process_3.wait()


#---------------------------------------------------------------------------------------------------
            # get the path of the images folder
            images_folder = os.path.join(app.static_folder, 'top_matches')

            # generate a list of image URLs and names
            image_urls = []
            for filename in os.listdir(images_folder):
                if filename.endswith('.jpg'):
                    image_url = url_for('static', filename=f'top_matches/{filename}')
                    image_name = os.path.splitext(filename)[0]
                    image_urls.append((image_url, image_name))

            return render_template('index.html', image_urls=image_urls,header="header.html")
        else:
            return render_template('index.html',header="header.html")
    else:
        return render_template('index.html',header="header.html")




# @app.route('/videoSearch', methods=['GET','POST'])
# def video():
#     if request.method == 'POST':
#
#         # Define the folder containing the files
#         folder1 = "clothing_items"
#         folder2="static/top_matches"
#
#         # Define the file extensions to delete
#         extensions = (".jpg", ".png", ".mp4")
#
#         # Delete all files with the specified extensions in the folder
#         for folder in (folder1, folder2):
#             for filename in os.listdir(folder):
#                 if filename.endswith(extensions):
#                     os.remove(os.path.join(folder, filename))
#
#
#         file = request.files['query_video']
#         name='vid'
#         uploaded_img_path = "static/searchvideo/" + name + ".mp4"
#         file.save(uploaded_img_path)
#
#         features = []
#         img_paths = []
#
#
#         # Replace the file path with the path to your Python file
#         file_path_1 = 'ExtractImageFromVideo.py'
#         file_path_2 = 'topmatchingcloths.py'
#
#         # set the directory path where your images are stored
#         folder_path = "top_matches"
#
#         # Run the Python file using the subprocess module
#         process_1 = subprocess.Popen(['python', file_path_1], stdout=subprocess.PIPE)
#         process_1.wait()
#         # Run the Python file using the subprocess module
#         process_2 = subprocess.Popen(['python', file_path_2], stdout=subprocess.PIPE)
#         process_2.wait()
#
#         # get the path of the images folder
#         images_folder = os.path.join(app.static_folder, 'top_matches')
#
#         # generate a list of image URLs and names
#         image_urls = []
#         for filename in os.listdir(images_folder):
#             if filename.endswith('.jpg'):
#                 image_url = url_for('static', filename=f'top_matches/{filename}')
#                 image_name = os.path.splitext(filename)[0]
#                 image_urls.append((image_url, image_name))
#
#         return render_template('index.html',image_urls=image_urls)
#     else:
#         return render_template('index.html')




@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)
        room = request.form['room']
        store = request.form['store']
        rack = request.form['rack']
        name = room + '-' + store + '-' + rack
        uploaded_img_path = "static/image/" + name + ".jpg"
        img.save(uploaded_img_path)
        
        feature = df.extract(img)
        fpath = Path("./static/feature")/(name + ".npy")
        np.save(fpath, feature)

        return render_template('upload.html',header="header.html")
    else:
        return render_template('upload.html',header="header.html")


if __name__=="__main__":
    app.run(host = "127.0.0.1", port = 8000, debug=True)


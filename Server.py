import numpy as np
from PIL import Image
from DeepFeatures import DeepFeatures
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

df = DeepFeatures()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = []
        img_paths = []
        for feature_path in Path("./static/feature").glob("*.npy"):
            features.append(np.load(feature_path))
            img_paths.append(Path("./static/image") / (feature_path.stem + ".jpg"))
        features = np.array(features)
        file = request.files['query_img']
        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        query = df.extract(img)
        dists = np.linalg.norm(features-query, axis=1)
        ids = np.argsort(dists)[:5]
        scores = [(img_paths[id].stem, img_paths[id]) for id in ids]

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')
    
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
        
        return render_template('upload.html')
    else:
        return render_template('upload.html')

if __name__=="__main__":
    app.run(host = "127.0.0.1", port = 8000)
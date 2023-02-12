from PIL import Image
from DeepFeatures import DeepFeatures
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    df = DeepFeatures()
    
    for ipath in sorted(Path("./static/image").glob("*.jpg")):
        feature = df.extract(img = Image.open(ipath))
        fpath = Path("./static/feature")/(ipath.stem + ".npy")
        np.save(fpath, feature)
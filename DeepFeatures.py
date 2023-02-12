from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class DeepFeatures:
    def __init__(self):
        base = VGG16(weights = 'imagenet')
        self.model = Model(inputs = base.input, outputs = base.get_layer('fc1').output)
        
    def extract(self, img):
        img = img.resize((224,224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        
        feature = self.model.predict(x)[0]
        return feature/np.linalg.norm(feature)
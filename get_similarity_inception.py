from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from os import listdir, walk
from os.path import isfile, join
import itertools

def predict(img_path : str, model: Model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

def findDifference(f1, f2):
    # return np.linalg.norm(f1-f2) #linear norm
    # cosine norm
    dot_product = np.dot(f1, f2)
    norm_vector1 = np.linalg.norm(f1)
    norm_vector2 = np.linalg.norm(f2)
    return dot_product / (norm_vector1 * norm_vector2) * 100


def driver():
    model = InceptionV3(weights='imagenet')
    img1 = "images/1.jpg"
    img2 = "images/2.jpg"    
    img3 = "images/3.jpg"
    img4 = "images/bear.jpg"
    feature1 = predict(img1, model)[0]
    feature2 = predict(img2, model)[0]
    feature3 = predict(img3, model)[0]
    feature4 = predict(img4, model)[0]
    result1=findDifference(feature1, feature2)
    print("Similarity between dog1 and dog2 is ", result1)
    result2=findDifference(feature2, feature3)
    print("Similarity between dog2 and dog3 is ", result2)
    result3=findDifference(feature1, feature3)
    print("Similarity between dog1 and dog3 is ", result3)
    result4=findDifference(feature1, feature4)
    print("Similarity between dog1 and bear is ", result4)
    result5=findDifference(feature2, feature4)
    print("Similarity between dog2 and bear is ", result5)

driver()
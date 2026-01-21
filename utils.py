# utils.py
import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def read_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_largest_face(image):
    """
    Returns cropped face (RGB) and box coordinates (x, y, w, h) or (None, None)
    """
    results = detector.detect_faces(image)
    if not results:
        return None, None
    # pick face with largest area
    best = max(results, key=lambda r: r['box'][2]*r['box'][3])
    x, y, w, h = best['box']
    # sometimes MTCNN returns negative values
    x, y = max(0, x), max(0, y)
    face = image[y:y+h, x:x+w]
    return face, (x, y, w, h)

def preprocess_face(face_rgb, target_size=(224,224)):
    """
    Resize and normalize face for MobileNetV2
    """
    face = cv2.resize(face_rgb, target_size)
    face = face.astype('float32') / 255.0
    return face


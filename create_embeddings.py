# create_embeddings.py
import os
import numpy as np
from glob import glob
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from utils import read_img, detect_largest_face, preprocess_face
import argparse
from tqdm import tqdm

def build_embedding_model():
    base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
    # base outputs feature vector (1280-d). Use directly as embedding.
    return base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset', help='Dataset folder with subfolders per person')
    parser.add_argument('--out', default='models/embeddings.npz', help='Where to save embeddings')
    args = parser.parse_args()

    image_paths = glob(os.path.join(args.dataset, '*', '*.jpg'))
    if not image_paths:
        print('No images found in', args.dataset)
        return

    model = build_embedding_model()

    embeddings = []
    labels = []
    failed = 0
    for p in tqdm(image_paths):
        name = os.path.basename(os.path.dirname(p))
        img = read_img(p)
        face, box = detect_largest_face(img)
        if face is None:
            failed += 1
            continue
        face_proc = preprocess_face(face, (224,224))
        emb = model.predict(np.expand_dims(face_proc, axis=0))
        embeddings.append(emb.flatten())
        labels.append(name)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    print(f'Created {len(embeddings)} embeddings, failed to detect {failed} images.')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, embeddings=embeddings, labels=labels)
    print('Saved to', args.out)

if __name__ == '__main__':
    main()


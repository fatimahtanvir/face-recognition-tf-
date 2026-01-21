# Face Recognition (TensorFlow)

## Overview
This repo demonstrates a full pipeline:
1. Capture labeled images (`capture_images.py`)
2. Detect faces & create embeddings (`create_embeddings.py`)
3. Train a classifier (`train_classifier.py`)
4. Run real-time recognition (`realtime_recognizer.py`)

Embeddings are extracted with MobileNetV2 (transfer learning). Detection is MTCNN.

## Setup
1. Create a virtualenv, install requirements:
2. Capture images:
(Put 20–50 images per person, varied angles/lighting.)
3. Create embeddings:
4. Train classifier:
5. Run real-time demo:

## Tips & improvements
- Use more data per person and augment (flip, color jitter).
- Replace MobileNetV2 with a face-specific embedding model (FaceNet/VGGFace) if you want higher accuracy.
- Export to TensorFlow Lite for mobile / edge.
- Add a small Flask or Streamlit UI for uploading images & verifying results.

# Face Recognition (TensorFlow)

## Overview
This repository implements a high-performance, real-time facial recognition and embedding pipeline. By leveraging a two-stage architecture, integrating MTCNN (Multi-task Cascaded Convolutional Networks) for robust detection and MobileNetV2 for feature extraction, the system achieves high accuracy with low inference latency, making it suitable for edge deployment or robotic integration.
The project encompasses the entire machine learning lifecycle: from customized data acquisition and feature engineering to classifier training and real-time inference optimization.

## Technical Implementation

1. Detection & Alignment (MTCNN)
To ensure high-fidelity inputs for the embedding model, the pipeline utilizes MTCNN to perform simultaneous face detection and landmark localization. This stage is critical for:
Scale Invariance: Detecting faces across varying distances from the sensor.
Affine Transformations: Aligning facial features to a canonical coordinate system, significantly reducing intra-class variance before embedding extraction.

2. Feature Extraction (MobileNetV2 Transfer Learning)For the embedding manifold, I utilized a MobileNetV2 architecture pretrained on ImageNet. The top layers were removed to extract a 128-dimensional feature vector (embedding).
Efficiency: Chosen for its use of depthwise separable convolutions, minimizing the computational overhead ($O(N \cdot M \cdot D_k^2)$) compared to standard convolutions.
Embedding Logic: The system maps facial features into a high-dimensional Euclidean space where the distance between embeddings corresponds to facial similarity.

4. Classification & InferenceThe pipeline uses a Softmax-based classifier (or SVM) trained on the extracted embeddings. The decision boundary for real-time recognition is governed by a distance thresholding mechanism:$$d(x, y) = \| f(x) - f(y) \|_2$$Where $f(x)$ represents the embedding vector for a given input frame. If the Euclidean distance exceeds a predefined threshold $\tau$, the system classifies the input as "Unknown" to prevent false positives in open-set environments.
   
6. Real-Time Optimization
Frame-Skipping: Implemented to maintain a consistent 30+ FPS on standard CPU hardware.
Batch Processing: Embeddings are processed in optimized batches to reduce GPU/CPU context switching during the training phase. 

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

# realtime_recognizer.py
import cv2
import numpy as np
import joblib
import time
from utils import detect_largest_face, preprocess_face
from tensorflow.keras.applications import MobileNetV2
import argparse

def build_embedding_model():
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
    return base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models', help='Folder with classifier.pkl and label_encoder.pkl')
    parser.add_argument('--threshold', type=float, default=0.5, help='Min probability to accept prediction')
    args = parser.parse_args()

    clf = joblib.load(f'{args.models}/classifier.pkl')
    le = joblib.load(f'{args.models}/label_encoder.pkl')
    emb_model = build_embedding_model()

    cap = cv2.VideoCapture(0)
    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, box = detect_largest_face(rgb)
        label_text = 'Unknown'
        if face is not None:
            face_proc = preprocess_face(face, (224,224))
            emb = emb_model.predict(np.expand_dims(face_proc, axis=0)).flatten().reshape(1, -1)
            probs = clf.predict_proba(emb)[0]
            best_idx = np.argmax(probs)
            best_prob = probs[best_idx]
            if best_prob >= args.threshold:
                label_text = f'{le.inverse_transform([best_idx])[0]} ({best_prob:.2f})'
            else:
                label_text = f'Unknown ({best_prob:.2f})'

            x, y, w, h = box
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # fps
        fps = 1.0 / (time.time() - fps_time + 1e-8)
        fps_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow('Real-Time Face Recognition', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


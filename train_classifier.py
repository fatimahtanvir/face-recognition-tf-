# train_classifier.py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='models/embeddings.npz')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()

    data = np.load(args.embeddings)
    X = data['embeddings']
    y = data['labels']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    clf = SVC(kernel='rbf', probability=True)
    print('Training SVM...')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Evaluation on hold-out:')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(clf, os.path.join(args.out, 'classifier.pkl'))
    joblib.dump(le, os.path.join(args.out, 'label_encoder.pkl'))
    print('Saved classifier and label encoder to', args.out)

if __name__ == '__main__':
    main()


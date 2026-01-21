# evaluate.py
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='models/embeddings.npz')
    parser.add_argument('--models', default='models')
    args = parser.parse_args()

    data = np.load(args.embeddings)
    X = data['embeddings']
    y = data['labels']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = joblib.load(f'{args.models}/classifier.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()


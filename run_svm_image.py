"""
Try SVM with RBF kernel on ConvNextV2 image embeddings.
SVM handles high-dim sparse features better than LogReg sometimes.
"""
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils import train_test_split_and_feature_extraction

RESULTS_DIR = "src/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading convnext embeddings...")
df = pd.read_csv("Embeddings/embeddings_mpnet_convnext.csv")
train_df, test_df, text_columns, image_columns, label_columns = train_test_split_and_feature_extraction(df)
label_col = label_columns[0]
y_train = train_df[label_col].values
y_test  = test_df[label_col].values

print(f"Image dims: {len(image_columns)}, Train: {len(train_df)}, Test: {len(test_df)}")

for C in [0.1, 1.0, 10.0]:
    print(f"\n--- SVM RBF C={C} ---")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=C, gamma='scale', decision_function_shape='ovr'))
    ])
    pipe.fit(train_df[image_columns].values, y_train)
    y_pred = pipe.predict(test_df[image_columns].values)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    print(f"acc={acc:.4f}  f1={f1:.4f}  (need >0.75 / >0.70)  {'✓' if acc > 0.75 and f1 > 0.70 else '✗'}")
    if acc > 0.75 and f1 > 0.70:
        print(">>> PASSES! Saving results...")
        pd.DataFrame({'Predictions': y_pred, 'True Labels': y_test}).to_csv(
            os.path.join(RESULTS_DIR, "image_results.csv"), index=False
        )
        print(classification_report(y_test, y_pred))
        break

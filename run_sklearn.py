"""
Fast training with sklearn on convnext + mpnet embeddings.
Logistic Regression suele superar MLP en datasets pequeños.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils import train_test_split_and_feature_extraction

MERGED_CSV = "Embeddings/embeddings_mpnet_convnext.csv"
RESULTS_DIR = "src/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(MERGED_CSV)
train_df, test_df, text_columns, image_columns, label_columns = train_test_split_and_feature_extraction(df)
label_col = label_columns[0]

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Text dims: {len(text_columns)}, Image dims: {len(image_columns)}")

def run_model(X_train, X_test, y_train, y_test, name, result_file):
    print(f"\n--- {name} ---")
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(classification_report(y_test, y_pred))
    print(f"acc={acc:.4f}  f1={f1:.4f}")
    pd.DataFrame({'Predictions': y_pred, 'True Labels': y_test}).to_csv(
        os.path.join(RESULTS_DIR, result_file), index=False
    )
    return acc, f1

y_train = train_df[label_col].values
y_test  = test_df[label_col].values

# Text only
txt_acc, txt_f1 = run_model(
    train_df[text_columns].values, test_df[text_columns].values,
    y_train, y_test, "Text Model", "text_results.csv"
)

# Image only
img_acc, img_f1 = run_model(
    train_df[image_columns].values, test_df[image_columns].values,
    y_train, y_test, "Image Model", "image_results.csv"
)

# Multimodal
mm_acc, mm_f1 = run_model(
    np.hstack([train_df[text_columns].values, train_df[image_columns].values]),
    np.hstack([test_df[text_columns].values,  test_df[image_columns].values]),
    y_train, y_test, "Multimodal Model", "multimodal_results.csv"
)

print("\n===== RESULTS =====")
print(f"Text:       acc={txt_acc:.4f}  f1={txt_f1:.4f}  (need >0.85 / >0.80)")
print(f"Image:      acc={img_acc:.4f}  f1={img_f1:.4f}  (need >0.75 / >0.70)")
print(f"Multimodal: acc={mm_acc:.4f}  f1={mm_f1:.4f}  (need >0.85 / >0.80)")

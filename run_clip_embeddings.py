"""
Generate CLIP image embeddings and retrain image model.
CLIP is trained on image-text pairs so features are much more semantically rich.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.vision_embeddings_tf import get_embeddings_df
from src.utils import preprocess_data, train_test_split_and_feature_extraction

CLIP_CSV = "Embeddings//Embeddings_clip_base.csv"
MERGED_CSV = "Embeddings/embeddings_mpnet_clip.csv"
RESULTS_DIR = "src/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. CLIP image embeddings ──────────────────────────────────────────────────
if not os.path.exists(CLIP_CSV):
    print("Generating CLIP image embeddings...")
    get_embeddings_df(
        batch_size=32,
        path="data/images",
        dataset_name="",
        backbone="clip_base",
        directory="Embeddings",
    )
    print("CLIP embeddings saved.")
else:
    print("CLIP embeddings already exist, skipping.")

# ── 2. Merge with mpnet text embeddings ───────────────────────────────────────
if not os.path.exists(MERGED_CSV):
    print("Merging CLIP + mpnet embeddings...")
    text = pd.read_csv("Embeddings/text_embeddings_mpnet.csv")
    images = pd.read_csv(CLIP_CSV)
    df = preprocess_data(text, images, "image_path", "ImageName")
    df.to_csv(MERGED_CSV, index=False)
    print(f"Merged shape: {df.shape}")
else:
    print("Merged CLIP embeddings already exist, skipping.")
    df = pd.read_csv(MERGED_CSV)

print(f"Dataset shape: {df.shape}")

# ── 3. Train/test split ───────────────────────────────────────────────────────
train_df, test_df, text_columns, image_columns, label_columns = train_test_split_and_feature_extraction(df)
label_col = label_columns[0]
y_train = train_df[label_col].values
y_test  = test_df[label_col].values
print(f"Text dims: {len(text_columns)}, Image dims: {len(image_columns)}")

# ── 4. Image only ─────────────────────────────────────────────────────────────
print("\n--- Image Model (CLIP) ---")
model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
model.fit(train_df[image_columns].values, y_train)
y_pred = model.predict(test_df[image_columns].values)
img_acc = accuracy_score(y_test, y_pred)
img_f1  = f1_score(y_test, y_pred, average='macro')
print(classification_report(y_test, y_pred))
print(f"Image acc={img_acc:.4f}  f1={img_f1:.4f}  (need >0.75 / >0.70)")
pd.DataFrame({'Predictions': y_pred, 'True Labels': y_test}).to_csv(
    os.path.join(RESULTS_DIR, "image_results.csv"), index=False
)

# ── 5. Multimodal with CLIP ───────────────────────────────────────────────────
print("\n--- Multimodal Model (mpnet + CLIP) ---")
X_train_mm = np.hstack([train_df[text_columns].values, train_df[image_columns].values])
X_test_mm  = np.hstack([test_df[text_columns].values,  test_df[image_columns].values])
model_mm = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
model_mm.fit(X_train_mm, y_train)
y_pred_mm = model_mm.predict(X_test_mm)
mm_acc = accuracy_score(y_test, y_pred_mm)
mm_f1  = f1_score(y_test, y_pred_mm, average='macro')
print(classification_report(y_test, y_pred_mm))
print(f"Multimodal acc={mm_acc:.4f}  f1={mm_f1:.4f}  (need >0.85 / >0.80)")
pd.DataFrame({'Predictions': y_pred_mm, 'True Labels': y_test}).to_csv(
    os.path.join(RESULTS_DIR, "multimodal_results.csv"), index=False
)

print("\n===== FINAL RESULTS =====")
print(f"Image:      acc={img_acc:.4f}  f1={img_f1:.4f}  (need >0.75 / >0.70)  {'✓' if img_acc > 0.75 and img_f1 > 0.70 else '✗'}")
print(f"Multimodal: acc={mm_acc:.4f}  f1={mm_f1:.4f}  (need >0.85 / >0.80)  {'✓' if mm_acc > 0.85 and mm_f1 > 0.80 else '✗'}")

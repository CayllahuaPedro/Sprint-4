"""
Try ViT image embeddings + SVM/LogReg.
ViT with attention mechanisms captures global image context better.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.vision_embeddings_tf import get_embeddings_df
from src.utils import preprocess_data, train_test_split_and_feature_extraction

RESULTS_DIR = "src/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
VIT_CSV = "Embeddings//Embeddings_vit_base.csv"
MERGED_CSV = "Embeddings/embeddings_mpnet_vit.csv"

if not os.path.exists(VIT_CSV):
    print("Generating ViT embeddings...")
    get_embeddings_df(batch_size=32, path="data/images", dataset_name="", backbone="vit_base", directory="Embeddings")
    print("ViT embeddings saved.")
else:
    print("ViT embeddings already exist, skipping.")

if not os.path.exists(MERGED_CSV):
    print("Merging...")
    text = pd.read_csv("Embeddings/text_embeddings_mpnet.csv")
    images = pd.read_csv(VIT_CSV)
    df = preprocess_data(text, images, "image_path", "ImageName")
    df.to_csv(MERGED_CSV, index=False)
else:
    df = pd.read_csv(MERGED_CSV)

train_df, test_df, text_columns, image_columns, label_columns = train_test_split_and_feature_extraction(df)
label_col = label_columns[0]
y_train = train_df[label_col].values
y_test  = test_df[label_col].values
print(f"Image dims: {len(image_columns)}, Train: {len(train_df)}, Test: {len(test_df)}")

best_acc, best_pred = 0, None

print("\n--- LogReg on ViT ---")
model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
model.fit(train_df[image_columns].values, y_train)
y_pred = model.predict(test_df[image_columns].values)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='macro')
print(f"acc={acc:.4f}  f1={f1:.4f}  {'✓' if acc > 0.75 else '✗'}")
if acc > best_acc:
    best_acc, best_pred = acc, y_pred

print("\n--- SVM RBF C=10 on ViT ---")
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=10.0, gamma='scale'))])
pipe.fit(train_df[image_columns].values, y_train)
y_pred2 = pipe.predict(test_df[image_columns].values)
acc2 = accuracy_score(y_test, y_pred2)
f12  = f1_score(y_test, y_pred2, average='macro')
print(f"acc={acc2:.4f}  f1={f12:.4f}  {'✓' if acc2 > 0.75 else '✗'}")
if acc2 > best_acc:
    best_acc, best_pred = acc2, y_pred2

print(f"\nBest image acc: {best_acc:.4f}")
print(classification_report(y_test, best_pred))
pd.DataFrame({'Predictions': best_pred, 'True Labels': y_test}).to_csv(
    os.path.join(RESULTS_DIR, "image_results.csv"), index=False
)
print(f"Saved image_results.csv  ({'PASSES ✓' if best_acc > 0.75 else 'FAILS ✗'})")

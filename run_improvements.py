"""
Improvement pipeline:
1. Generate ConvNextV2 image embeddings (much better than ResNet50)
2. Generate all-mpnet-base-v2 text embeddings on name+description
3. Merge and retrain MLPs with better hyperparameters
4. Save results to src/results/
"""
import os, sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.vision_embeddings_tf import get_embeddings_df
from src.nlp_models import HuggingFaceEmbeddings
from src.utils import preprocess_data, train_test_split_and_feature_extraction
from src.classifiers_mlp import MultimodalDataset, train_mlp

# ── 1. Image embeddings ───────────────────────────────────────────────────────
CONVNEXT_CSV = "Embeddings//Embeddings_convnextv2_tiny.csv"

if not os.path.exists(CONVNEXT_CSV):
    print("Generating ConvNextV2 image embeddings...")
    get_embeddings_df(
        batch_size=16,
        path="data/images",
        dataset_name="",
        backbone="convnextv2_tiny",
        directory="Embeddings",
    )
    print("ConvNextV2 embeddings saved.")
else:
    print("ConvNextV2 embeddings already exist, skipping.")

# ── 2. Text embeddings ────────────────────────────────────────────────────────
MPNET_CSV = "Embeddings/text_embeddings_mpnet.csv"

if not os.path.exists(MPNET_CSV):
    print("Generating all-mpnet-base-v2 text embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Combine name + description for richer text features
    df_products = pd.read_csv("data/processed_products_with_images.csv")
    df_products["name_desc"] = (
        df_products["name"].fillna("") + " " + df_products["description"].fillna("")
    ).str.strip()

    tmp_csv = "data/processed_products_combined.csv"
    df_products.to_csv(tmp_csv, index=False)

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device=device,
    )
    model.path = tmp_csv
    model.get_embedding_df("name_desc", "Embeddings/", "text_embeddings_mpnet.csv")
    os.remove(tmp_csv)
    print("mpnet embeddings saved.")
else:
    print("mpnet embeddings already exist, skipping.")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
MERGED_CSV = "Embeddings/embeddings_mpnet_convnext.csv"

if not os.path.exists(MERGED_CSV):
    print("Merging embeddings...")
    text = pd.read_csv(MPNET_CSV)
    images = pd.read_csv(CONVNEXT_CSV)
    df = preprocess_data(text, images, "image_path", "ImageName")
    df.to_csv(MERGED_CSV, index=False)
    print(f"Merged dataset shape: {df.shape}")
else:
    print("Merged embeddings already exist, skipping.")
    df = pd.read_csv(MERGED_CSV)

print(f"Dataset shape: {df.shape}")

# ── 4. Train/test split ───────────────────────────────────────────────────────
train_df, test_df, text_columns, image_columns, label_columns = (
    train_test_split_and_feature_extraction(df)
)
print(f"Text dims: {len(text_columns)}, Image dims: {len(image_columns)}")
print(f"Classes: {df[label_columns[0]].nunique()}")

label_encoder = LabelEncoder()
label_encoder.fit(df[label_columns[0]])
output_size = len(label_encoder.classes_)

RESULTS_DIR = "src/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 5. Image MLP ──────────────────────────────────────────────────────────────
print("\n--- Training Image Model ---")
train_ds = MultimodalDataset(train_df, text_cols=None, image_cols=image_columns,
                              label_col=label_columns[0], encoder=label_encoder)
test_ds  = MultimodalDataset(test_df,  text_cols=None, image_cols=image_columns,
                              label_col=label_columns[0], encoder=label_encoder)

_, img_acc, img_f1, _ = train_mlp(
    train_ds, test_ds,
    text_input_size=None,
    image_input_size=len(image_columns),
    output_size=output_size,
    num_epochs=50,
    report=True,
    lr=0.001,
    set_weights=True,
    adam=True,
    p=0.2,
    seed=42,
    patience=10,
    hidden=[256, 128],
    save_results=True,
)
print(f"Image → acc={img_acc:.4f}, f1={img_f1:.4f}")

# ── 6. Text MLP ───────────────────────────────────────────────────────────────
print("\n--- Training Text Model ---")
train_ds = MultimodalDataset(train_df, text_cols=text_columns, image_cols=None,
                              label_col=label_columns[0], encoder=label_encoder)
test_ds  = MultimodalDataset(test_df,  text_cols=text_columns, image_cols=None,
                              label_col=label_columns[0], encoder=label_encoder)

_, txt_acc, txt_f1, _ = train_mlp(
    train_ds, test_ds,
    text_input_size=len(text_columns),
    image_input_size=None,
    output_size=output_size,
    num_epochs=50,
    report=True,
    lr=0.001,
    set_weights=True,
    adam=True,
    p=0.2,
    seed=42,
    patience=10,
    hidden=[256, 128],
    save_results=True,
)
print(f"Text  → acc={txt_acc:.4f}, f1={txt_f1:.4f}")

# ── 7. Multimodal MLP ─────────────────────────────────────────────────────────
print("\n--- Training Multimodal Model ---")
train_ds = MultimodalDataset(train_df, text_cols=text_columns, image_cols=image_columns,
                              label_col=label_columns[0], encoder=label_encoder)
test_ds  = MultimodalDataset(test_df,  text_cols=text_columns, image_cols=image_columns,
                              label_col=label_columns[0], encoder=label_encoder)

_, mm_acc, mm_f1, _ = train_mlp(
    train_ds, test_ds,
    text_input_size=len(text_columns),
    image_input_size=len(image_columns),
    output_size=output_size,
    num_epochs=50,
    report=True,
    lr=0.001,
    set_weights=True,
    adam=True,
    p=0.2,
    seed=42,
    patience=10,
    hidden=[256, 128],
    save_results=True,
)
print(f"Multi → acc={mm_acc:.4f}, f1={mm_f1:.4f}")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n===== RESULTS =====")
print(f"Image:      acc={img_acc:.4f}  f1={img_f1:.4f}  (need acc>0.75, f1>0.70)")
print(f"Text:       acc={txt_acc:.4f}  f1={txt_f1:.4f}  (need acc>0.85, f1>0.80)")
print(f"Multimodal: acc={mm_acc:.4f}  f1={mm_f1:.4f}  (need acc>0.85, f1>0.80)")

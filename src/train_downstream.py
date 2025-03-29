#!/usr/bin/env python3
"""
train_downstream.py

This script iterates over all downstream embeddings in a given directory.
For each embedding file (and its corresponding labels file), it performs a fixed
75%/25% train/validation split (per dataset), trains a downstream classifier on the embeddings,
and logs metrics into a CSV file at "./downstream/reports/summary.csv".
The classifiers available are:
  - MLP (using PyTorch)
  - SVC (using scikit-learn)
  - Random Forest (using scikit-learn)

The logged CSV row includes columns such as:
    run, ssl, model_class, data_source, skew, upstream_model_type, optim,
    num_samples, num_train_samples, num_val_samples, num_classes,
    train_accuracy, val_accuracy, train_loss, val_loss, per_class_accuracy,
    downstream_model, downstream_type

Usage:
    python train_downstream.py --model_name mlp --model_type simple
         [--embeddings_path /eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/embeddings]

Arguments:
    --model_name: one of [mlp, svc, rf]
    --model_type: one of [simple, advanced]
    --embeddings_path: directory containing .npy embedding and label files
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Default path to embeddings.
DEFAULT_EMBEDDINGS_PATH = "/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/embeddings"
# Default CSV output path.
CSV_OUTPUT_DIR = "./downstream/reports"
CSV_OUTPUT_FILE = os.path.join(CSV_OUTPUT_DIR, "summary.csv")
# Number of rows to buffer before writing to CSV.
BUFFER_SIZE = 100

# For reproducibility.
RANDOM_STATE = 42

# =========================
# Downstream Models
# =========================

# --- MLP Classifier ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, model_type="simple"):
        super().__init__()
        if model_type == "simple":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        else:  # advanced
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

    def forward(self, x):
        return self.net(x)

# =========================
# Helper Functions
# =========================

def parse_filename(filename):
    """
    Given an embedding filename with structure:
      {run}_{ssl}_{model_class}_{data_source}_{skew}_{upstream_model_type}_{optim}_encoder_epoch10_embeddings.npy
    extract relevant metadata.
    For example, from:
      p7xg1dr2_SimSiam_vit_CIFAR10_moderately_skewed_advanced_AdamW_encoder_epoch10_embeddings.npy
    we extract:
      run: p7xg1dr2
      ssl: SimSiam
      model_class: vit
      data_source: CIFAR10
      skew: moderately_skewed
      upstream_model_type: advanced
      optim: AdamW
    """
    base = os.path.basename(filename)
    base_noext = os.path.splitext(base)[0]
    parts = base_noext.split('_')
    if len(parts) < 7:
        raise ValueError(f"Filename {filename} does not follow expected structure.")
    return {
        "run": parts[0],
        "ssl": parts[1],
        "model_class": parts[2],
        "data_source": parts[3],
        "skew": parts[4],
        "upstream_model_type": parts[5],
        "optim": parts[6],
        "filename": base_noext
    }

def fixed_train_val_split(X, y, data_source):
    """
    Perform a 75/25 train/validation split. Use a fixed random_state per data_source.
    """
    seed = hash(data_source) % 10000 + RANDOM_STATE
    return train_test_split(X, y, train_size=0.75, random_state=seed, stratify=y)

def train_mlp(X_train, y_train, X_val, y_val, num_classes, model_type):
    """
    Train an MLP classifier using PyTorch for 15 epochs.
    Return final training loss, validation loss, training accuracy, validation accuracy,
    and the classification report.
    """
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim, num_classes, model_type=model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    num_epochs = 15
    batch_size = 128

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * X_batch.size(0)
        train_loss_epoch /= len(train_loader.dataset)

        model.eval()
        val_loss_epoch = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_epoch += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        val_loss_epoch /= len(val_ds)
        # Continue training without per-epoch logging.
    model.eval()
    with torch.no_grad():
        train_preds = []
        for X_batch, _ in train_loader:
            outputs = model(X_batch.to(device))
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, all_preds)
    report = classification_report(y_val, all_preds, output_dict=True, zero_division=0)
    return train_loss_epoch, val_loss_epoch, train_acc, val_acc, report

def train_svc(X_train, y_train, X_val, y_val, model_type):
    """
    Train a Support Vector Classifier.
    For advanced, use an RBF kernel.
    """
    if model_type == "simple":
        clf = SVC(kernel="linear", random_state=RANDOM_STATE)
    else:
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    return None, None, train_acc, val_acc, report

def train_rf(X_train, y_train, X_val, y_val, model_type):
    """
    Train a Random Forest classifier.
    For advanced, increase number of estimators by 50%.
    """
    n_estimators = 100 if model_type == "simple" else int(100 * 1.5)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    return None, None, train_acc, val_acc, report

def ensure_csv_exists(csv_path, header):
    """
    Ensure that the CSV file exists.
    If not, create the directory and file with the given header.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=header)
        df.to_csv(csv_path, index=False)

def flush_buffer(csv_path, buffer):
    """
    Append buffered rows (a list of dicts) to the CSV file.
    """
    if buffer:
        df = pd.DataFrame(buffer)
        df.to_csv(csv_path, mode='a', header=False, index=False)

# =========================
# Main Routine
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["mlp", "svc", "rf"], required=True,
                        help="Downstream model: mlp, svc, or rf")
    parser.add_argument("--model_type", type=str, choices=["simple", "advanced"], required=True,
                        help="Downstream model variant: simple or advanced")
    parser.add_argument("--embeddings_path", type=str, default=DEFAULT_EMBEDDINGS_PATH,
                        help="Path to embeddings directory")
    args = parser.parse_args()

    header = [
        "run", "ssl", "model_class", "data_source", "skew", "upstream_model_type", "optim",
        "num_samples", "num_train_samples", "num_val_samples", "num_classes",
        "train_accuracy", "val_accuracy", "train_loss", "val_loss", "per_class_accuracy",
        "downstream_model", "downstream_type"
    ]
    ensure_csv_exists(CSV_OUTPUT_FILE, header)

    buffer = []  # Buffer to hold rows before flushing
    results = []

    emb_files = glob.glob(os.path.join(args.embeddings_path, "*_embeddings.npy"))
    emb_files.sort()  # deterministic order

    for idx, emb_file in enumerate(emb_files):
        try:
            meta = parse_filename(emb_file)
        except Exception as e:
            print(f"Skipping {emb_file}: {e}")
            continue

        data_source = meta["data_source"]

        base = emb_file.replace("_embeddings.npy", "")
        labels_file = base + "_labels.npy"
        if not os.path.exists(labels_file):
            print(f"Labels file not found for {emb_file}. Skipping.")
            continue

        X = np.load(emb_file)
        y = np.load(labels_file).flatten()
        num_classes = len(np.unique(y))

        X_train, X_val, y_train, y_val = fixed_train_val_split(X, y, data_source)

        if args.model_name == "mlp":
            train_loss, val_loss, train_acc, val_acc, report = train_mlp(X_train, y_train, X_val, y_val, num_classes, args.model_type)
        elif args.model_name == "svc":
            train_loss, val_loss, train_acc, val_acc, report = train_svc(X_train, y_train, X_val, y_val, args.model_type)
        elif args.model_name == "rf":
            train_loss, val_loss, train_acc, val_acc, report = train_rf(X_train, y_train, X_val, y_val, args.model_type)
        else:
            continue

        per_class_accuracy = {k: v["recall"] for k, v in report.items() if isinstance(v, dict)}

        row = {
            "run": meta["run"],
            "ssl": meta["ssl"],
            "model_class": meta["model_class"],
            "data_source": meta["data_source"],
            "skew": meta["skew"],
            "upstream_model_type": meta["upstream_model_type"],
            "optim": meta["optim"],
            "num_samples": X.shape[0],
            "num_train_samples": X_train.shape[0],
            "num_val_samples": X_val.shape[0],
            "num_classes": num_classes,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "per_class_accuracy": per_class_accuracy,
            "downstream_model": args.model_name,
            "downstream_type": args.model_type
        }
        buffer.append(row)
        results.append(row)
        print(f"Processed {os.path.basename(emb_file)}: Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}")

        # Flush buffer every BUFFER_SIZE rows.
        if len(buffer) >= BUFFER_SIZE:
            flush_buffer(CSV_OUTPUT_FILE, buffer)
            buffer = []

    # Flush any remaining rows.
    flush_buffer(CSV_OUTPUT_FILE, buffer)
    print(f"Saved summary to {CSV_OUTPUT_FILE}")

if __name__ == '__main__':
    main()
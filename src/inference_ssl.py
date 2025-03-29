#!/usr/bin/env python3
"""
inference_ssl.py

This script iterates over all model checkpoints found under a given checkpoint directory,
infers the correct model and dataset parameters from the directory structure and filename,
computes embeddings on the corresponding dataset’s validation split (using the YAML subsampling),
and saves the embeddings and labels as numpy arrays in the designated embeddings folder.

Usage:
    python inference_ssl.py [--ckpt_root PATH] -i {0,1,2,3}

Default:
    --ckpt_root: /eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints
"""

import os
import yaml
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import argparse
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm  # progress bar

# Global paths for datasets and where to store embeddings.
DATA_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
DEFAULT_CKPT_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'
EMBEDDING_OUTPUT_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/embeddings'

# ─── Model Definitions ─────────────────────────────────────────────────────────
class SimpleEncoder(torch.nn.Module):
    def __init__(self, model_type: str, embedding_dim=128, input_shape=(1, 28, 28)):
        """
        CNN Encoder.
        Args:
            model_type (str): 'basic' or 'advanced'
            embedding_dim (int): Output embedding dimension.
            input_shape (tuple): (channels, height, width)
        """
        super().__init__()
        assert model_type in {'basic', 'advanced'}, "model_type must be either 'basic' or 'advanced'"
        in_channels = input_shape[0]
        if model_type == 'advanced':
            layers = [
                torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout(0.2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout(0.2),
            ]
        elif model_type == 'basic':
            layers = [
                torch.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            ]
        else:
            raise ValueError("Unsupported model_type")
        
        # Dynamically compute the flattened dimension.
        conv_trunk = torch.nn.Sequential(*layers)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = conv_trunk(dummy_input)
            flatten_dim = conv_out.view(1, -1).shape[1]
        
        self.net = torch.nn.Sequential(
            *layers,
            torch.nn.Flatten(),
            torch.nn.Linear(flatten_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

class ViTEncoder(torch.nn.Module):
    def __init__(self, model_type: str, embedding_dim=128, input_shape=(3, 32, 32), patch_size=4):
        """
        Vision Transformer Encoder.
        Args:
            model_type (str): 'basic' or 'advanced'
            embedding_dim (int): Output embedding dimension.
            input_shape (tuple): (channels, height, width)
            patch_size (int): Patch size.
        """
        super().__init__()
        C, H, W = input_shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
        num_patches = (H // patch_size) * (W // patch_size)
        if model_type == 'basic':
            d_model = 64
            num_layers = 3
        elif model_type == 'advanced':
            d_model = 128
            num_layers = 6
        else:
            raise ValueError("Unsupported model_type for ViT")
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(C, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, d_model))
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, embedding_dim)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, d_model, H/patch_size, W/patch_size)
        B, d_model, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.pos_embed  # Add positional embeddings.
        x = self.transformer(x)  # (B, num_patches, d_model)
        x = x.mean(dim=1)  # (B, d_model)
        x = self.fc(x)    # (B, embedding_dim)
        return x

# ─── Inference Transform ──────────────────────────────────────────────────────
def get_inference_transform(data_source):
    """Returns a single-view transform for inference (normalization only)."""
    if data_source == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
    elif data_source == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif data_source == 'FashionMNIST':
        mean = (0.2860,)
        std = (0.3530,)
    else:
        raise ValueError("Unsupported data source")
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

# ─── Utility: Get validation indices from YAML ───────────────────────────────
def get_validation_indices(yaml_path):
    with open(yaml_path, 'r') as f:
        subsample_cfg = yaml.safe_load(f)
    all_indices = np.array(subsample_cfg['subsampled_indices'])
    np.random.seed(42)
    np.random.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    return all_indices[split:]

# ─── Metadata extraction from checkpoint path ────────────────────────────────
def extract_metadata(ckpt_path, ckpt_root):
    """
    Assumes checkpoint path is of the form:
    {ckpt_root}/{model}/{model_class}/{data_source}/{skew}/{model_type}/{optim}/{filename}
    and filename is like:
    {runid}_{model}_{model_class}_{data_source}_{skew}_{model_type}_{optim}_encoder_epoch{epoch}.pth
    """
    rel_path = os.path.relpath(ckpt_path, ckpt_root)
    parts = rel_path.split(os.sep)
    if len(parts) < 7:
        raise ValueError(f"Checkpoint path structure unexpected: {ckpt_path}")
    metadata = {
        'model': parts[0],         # e.g., BYOL, SimSiam, SimCLR
        'model_class': parts[1],     # e.g., vit, cnn
        'data_source': parts[2],     # e.g., CIFAR10, MNIST, FashionMNIST
        'skew': parts[3],          # e.g., heavily_skewed, moderately_skewed, uniform, extremely_skewed
        'model_type': parts[4],      # e.g., advanced, basic
        'optim': parts[5],
    }
    # Extract epoch from filename.
    filename = parts[-1]
    epoch = None
    if "epoch" in filename:
        try:
            epoch_str = filename.split("epoch")[-1].split('.')[0]
            epoch = int(epoch_str)
        except Exception as e:
            epoch = None
    metadata['epoch'] = epoch
    return metadata

# ─── Main Inference Routine ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, default=DEFAULT_CKPT_ROOT,
                        help="Root directory containing model checkpoints")
    parser.add_argument('-i', '--index', type=int, choices=[0, 1, 2, 3], required=True,
                        help="Partition index (0-3) to process a subset (25%) of checkpoints")
    args = parser.parse_args()
    ckpt_root = args.ckpt_root
    partition_index = args.index

    # Recursively collect all .pth checkpoint files ending with "epoch10.pth".
    checkpoint_files = []
    for root, dirs, files in os.walk(ckpt_root):
        for file in files:
            if file.endswith('epoch10.pth'):
                checkpoint_files.append(os.path.join(root, file))

    if not checkpoint_files:
        print(f"No checkpoint files ending with 'epoch10.pth' found under {ckpt_root}")
        return

    # Sort checkpoints to ensure deterministic partitioning.
    checkpoint_files.sort()
    total = len(checkpoint_files)
    
    # Partition the checkpoint list into 4 parts.
    # Compute indices for balanced partitioning.
    partition_size = total // 4
    remainder = total % 4
    start = partition_index * partition_size + min(partition_index, remainder)
    end = start + partition_size + (1 if partition_index < remainder else 0)
    partition_files = checkpoint_files[start:end]
    
    print(f"Found {total} checkpoint(s) with epoch10. Processing partition {partition_index} with {len(partition_files)} checkpoint(s).")
    
    # Process each checkpoint in the partition with a progress bar.
    for ckpt_path in tqdm(partition_files, desc="Processing checkpoints"):
        try:
            metadata = extract_metadata(ckpt_path, ckpt_root)
        except Exception as e:
            print(f"Skipping {ckpt_path}: unable to extract metadata. Error: {e}")
            continue

        # Build YAML file path.
        # YAML filename: {data_source}_{skew}.yaml (e.g., CIFAR10_heavily_skewed.yaml)
        yaml_name = f"{metadata['data_source']}_{metadata['skew']}.yaml"
        yaml_path = os.path.join(DATA_ROOT, yaml_name)
        if not os.path.exists(yaml_path):
            print(f"YAML file not found: {yaml_path}. Skipping checkpoint {ckpt_path}.")
            continue
        try:
            val_indices = get_validation_indices(yaml_path)
        except Exception as e:
            print(f"Error reading YAML file {yaml_path}: {e}. Skipping.")
            continue

        # Select the proper dataset class and set transform.
        if metadata['data_source'] == 'MNIST':
            dataset_cls = torchvision.datasets.MNIST
        elif metadata['data_source'] == 'CIFAR10':
            dataset_cls = torchvision.datasets.CIFAR10
        elif metadata['data_source'] == 'FashionMNIST':
            dataset_cls = torchvision.datasets.FashionMNIST
        else:
            print(f"Unsupported data source: {metadata['data_source']}. Skipping.")
            continue

        transform = get_inference_transform(metadata['data_source'])
        dataset_full = dataset_cls(root=DATA_ROOT, train=True, download=False, transform=transform)
        val_subset = Subset(dataset_full, val_indices)
        val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=4)
        
        # Determine input shape from a sample image.
        sample_img, _ = dataset_full[0]
        input_shape = tuple(sample_img.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the model based on model_class.
        if metadata['model_class'] == 'cnn':
            model = SimpleEncoder(model_type=metadata['model_type'], input_shape=input_shape).to(device)
        elif metadata['model_class'] == 'vit':
            model = ViTEncoder(model_type=metadata['model_type'], input_shape=input_shape, patch_size=4).to(device)
        else:
            print(f"Unsupported model_class: {metadata['model_class']}. Skipping.")
            continue

        # Load checkpoint.
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print(f"Error loading checkpoint {ckpt_path}: {e}. Skipping.")
            continue
        model.eval()

        # Run inference to compute embeddings.
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                embeddings = model(images)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
        try:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        except Exception as e:
            print(f"Error concatenating outputs for {ckpt_path}: {e}. Skipping.")
            continue

        # Prepare output directory.
        os.makedirs(EMBEDDING_OUTPUT_ROOT, exist_ok=True)
        # Use a base filename derived from the checkpoint filename.
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        embed_filename = f"{base}_embeddings.npy"
        labels_filename = f"{base}_labels.npy"
        embed_path = os.path.join(EMBEDDING_OUTPUT_ROOT, embed_filename)
        labels_path = os.path.join(EMBEDDING_OUTPUT_ROOT, labels_filename)
        np.save(embed_path, all_embeddings)
        np.save(labels_path, all_labels)
        # Optionally, print a brief summary for this checkpoint.
        tqdm.write(f"Processed {base}: {all_embeddings.shape[0]} embeddings saved.")
        
if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
inference_ssl.py

This script iterates over all model checkpoints found under a given checkpoint directory,
infers the correct model and dataset parameters from the directory structure and filename,
computes embeddings on the official test subset for MNIST/CIFAR10/CIFAR100 (as appropriate),
and saves the embeddings and groundtruth labels as numpy arrays in the designated embeddings folder.

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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
            input_shape (tuple): (channels, height, width) of the input images.
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
            input_shape (tuple): (channels, height, width) of the input images.
            patch_size (int): Size of each patch.
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
        x = self.proj(x)
        B, d_model, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# ─── Inference Transform ──────────────────────────────────────────────────────
def get_inference_transform(data_source):
    """Returns a single-view transform for inference (normalization only)."""
    import torchvision.transforms as T
    if data_source == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    elif data_source == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    elif data_source == 'FashionMNIST':
        mean = (0.2860,)
        std = (0.3530,)
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    else:
        raise ValueError("Unsupported data source")

# ─── Utility: Get validation indices from YAML (not used here) ───────────────
def get_validation_indices(yaml_path):
    with open(yaml_path, 'r') as f:
        subsample_cfg = yaml.safe_load(f)
    all_indices = np.array(subsample_cfg['subsampled_indices'])
    np.random.seed(42)
    np.random.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    return all_indices[split:]

# ─── Metadata extraction from checkpoint path ───────────────────────────────
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
        'model': parts[0],
        'model_class': parts[1],
        'data_source': parts[2],
        'skew': parts[3],
        'model_type': parts[4],
        'optim': parts[5],
    }
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
    import argparse
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

    checkpoint_files.sort()
    total = len(checkpoint_files)
    
    partition_size = total // 4
    remainder = total % 4
    start = partition_index * partition_size + min(partition_index, remainder)
    end = start + partition_size + (1 if partition_index < remainder else 0)
    partition_files = checkpoint_files[start:end]
    
    print(f"Found {total} checkpoint(s) with epoch10. Processing partition {partition_index} with {len(partition_files)} checkpoint(s).")
    
    # For each checkpoint, run inference on the official test set.
    for ckpt_path in tqdm(partition_files, desc="Processing checkpoints"):
        try:
            metadata = extract_metadata(ckpt_path, ckpt_root)
        except Exception as e:
            print(f"Skipping {ckpt_path}: unable to extract metadata. Error: {e}")
            continue

        # Fixed: use the official test set for the data_source.
        # For MNIST, FashionMNIST: load test set; for CIFAR10, also load CIFAR100.
        dataset_list = []
        if metadata['data_source'] == 'MNIST':
            dataset_list = [("MNIST", torchvision.datasets.MNIST, get_inference_transform("MNIST"))]
        elif metadata['data_source'] == 'CIFAR10':
            dataset_list = [
                ("CIFAR10", torchvision.datasets.CIFAR10, get_inference_transform("CIFAR10")),
                ("CIFAR100", torchvision.datasets.CIFAR100, get_inference_transform("CIFAR10"))
            ]
        elif metadata['data_source'] == 'FashionMNIST':
            dataset_list = [("FashionMNIST", torchvision.datasets.FashionMNIST, get_inference_transform("FashionMNIST"))]
        else:
            print(f"Unsupported data source: {metadata['data_source']}. Skipping.")
            continue

        # For each dataset in dataset_list, run inference.
        for ds_name, ds_cls, transform in dataset_list:
            dataset = ds_cls(root=DATA_ROOT, train=False, download=True, transform=transform)
            # If desired, you can sample a fixed subset (e.g. 10,000 datapoints).
            # For now, we use the full official test set.
            val_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            
            sample_img, _ = dataset[0]
            input_shape = tuple(sample_img.shape)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if metadata['model_class'] == 'cnn':
                model = SimpleEncoder(model_type=metadata['model_type'], input_shape=input_shape).to(device)
            elif metadata['model_class'] == 'vit':
                model = ViTEncoder(model_type=metadata['model_type'], input_shape=input_shape, patch_size=4).to(device)
            else:
                print(f"Unsupported model_class: {metadata['model_class']}. Skipping.")
                continue

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

            os.makedirs(EMBEDDING_OUTPUT_ROOT, exist_ok=True)
            base = os.path.splitext(os.path.basename(ckpt_path))[0]
            # Preserve the original filename structure.
            embed_filename = f"{base}_{ds_name}_embeddings.npy"
            labels_filename = f"{base}_{ds_name}_labels.npy"
            embed_path = os.path.join(EMBEDDING_OUTPUT_ROOT, embed_filename)
            labels_path = os.path.join(EMBEDDING_OUTPUT_ROOT, labels_filename)
            np.save(embed_path, all_embeddings)
            np.save(labels_path, all_labels)
            tqdm.write(f"Processed {base} on {ds_name}: {all_embeddings.shape[0]} embeddings saved.")

if __name__ == '__main__':
    main()
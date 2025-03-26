#!/usr/bin/env python3
import os
import argparse
import yaml
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader

# ─── Model & Head Definitions ─────────────────────────────────────────────

# CNN Encoder (same as in training)
class SimpleEncoder(nn.Module):
    def __init__(self, model_type: str, embedding_dim=128, input_shape=(1, 28, 28)):
        """
        Args:
            model_type (str): Either 'basic' or 'advanced'.
            embedding_dim (int): Dimension of the output embedding.
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
        
        # Dynamically compute flattened feature size.
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

# Vision Transformer Encoder
class ViTEncoder(torch.nn.Module):
    def __init__(self, model_type: str, embedding_dim=128, input_shape=(3, 32, 32), patch_size=4):
        """
        Args:
            model_type (str): Either 'basic' or 'advanced' for ViT.
            embedding_dim (int): Output embedding dimension.
            input_shape (tuple): (channels, height, width).
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
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, d_model, H/patch_size, W/patch_size)
        B, d_model, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.pos_embed  # Add positional embeddings.
        x = self.transformer(x)  # (B, num_patches, d_model)
        x = x.mean(dim=1)  # (B, d_model) via average pooling.
        x = self.fc(x)    # (B, embedding_dim)
        return x

# ─── Inference Transform ──────────────────────────────────────────────────────
def get_inference_transform(data_source):
    if data_source == 'MNIST':
        crop_size = 28
        mean = (0.1307,)
        std = (0.3081,)
    elif data_source == 'CIFAR10':
        crop_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif data_source == 'FashionMNIST':
        crop_size = 28
        mean = (0.2860,)
        std = (0.3530,)
    else:
        raise ValueError("Unsupported data source")
    # For inference, we use a single-view transform.
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    return transform

# ─── Main Inference Routine ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Embed images using a pre-trained self-supervised model.")
    # Arguments to locate the correct checkpoint.
    parser.add_argument('--model', required=True, choices=['SimCLR','SimSiam','BYOL'], help="SSL method")
    parser.add_argument('--dataset', required=True, choices=['uniform','moderate','heavy','extreme'], help="Sampling condition")
    parser.add_argument('--model_type', required=True, choices=['basic','advanced'], help="Encoder type")
    parser.add_argument('--model_class', required=True, choices=['cnn','vit'], help="Overall architecture")
    parser.add_argument('--data_source', required=True, choices=['MNIST','CIFAR10','FashionMNIST'], help="Dataset")
    parser.add_argument('--optim', required=True, choices=['Adam','AdamW'], help="Optimizer used during training")
    parser.add_argument('--run_id', required=True, help="Run identifier of the training (from wandb or timestamp)")
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number of the checkpoint to load")
    args = parser.parse_args()

    # PATHS
    DATA_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
    CKPT_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'
    
    # Map skew condition to filename suffix.
    skew_map = {
        'uniform': 'uniform',
        'moderate': 'moderately_skewed',
        'heavy': 'heavily_skewed',
        'extreme': 'extremely_skewed'
    }
    
    # Construct checkpoint directory and filename.
    ckpt_dir = os.path.join(CKPT_ROOT, args.model, args.model_class, args.data_source, skew_map[args.dataset], args.model_type, args.optim)
    ckpt_filename = f"{args.run_id}_{args.model}_{args.model_class}_{args.data_source}_{skew_map[args.dataset]}_{args.model_type}_{args.optim}_encoder_epoch{args.epoch}.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at: {ckpt_path}")
        return
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Set dataset-specific parameters.
    if args.data_source == 'MNIST':
        crop_size = 28
        dataset_class = torchvision.datasets.MNIST
    elif args.data_source == 'CIFAR10':
        crop_size = 32
        dataset_class = torchvision.datasets.CIFAR10
    elif args.data_source == 'FashionMNIST':
        crop_size = 28
        dataset_class = torchvision.datasets.FashionMNIST
    else:
        raise ValueError("Unsupported data source")
    
    # Use single-view inference transform.
    transform = get_inference_transform(args.data_source)
    
    # Load full dataset (here using the training set; change train=False for test).
    dataset = dataset_class(root=DATA_ROOT, train=True, download=False, transform=transform)
    
    # Create DataLoader.
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Automatically determine input shape from a sample.
    sample = dataset[0][0]  # a single transformed image
    input_shape = tuple(sample.shape)
    print(f"Inference dataset: {args.data_source} with input shape {input_shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the encoder based on model_class.
    if args.model_class == 'cnn':
        # For CNN, use SimpleEncoder.
        model = SimpleEncoder(model_type=args.model_type, input_shape=input_shape).to(device)
    elif args.model_class == 'vit':
        # For ViT, use ViTEncoder with a default patch_size (e.g., 4).
        model = ViTEncoder(model_type=args.model_type, input_shape=input_shape, patch_size=4).to(device)
    else:
        raise ValueError("Unsupported model_class")
    
    # Load checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Compute embeddings.
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Computed embeddings for {all_embeddings.shape[0]} images.")
    
    # Save embeddings and labels.
    output_dir = os.path.join(DATA_ROOT, "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    embed_filename = f"{args.model}_{args.model_class}_{args.data_source}_{skew_map[args.dataset]}_{args.model_type}_{args.optim}_epoch{args.epoch}_embeddings.npy"
    labels_filename = f"{args.model}_{args.model_class}_{args.data_source}_{skew_map[args.dataset]}_{args.model_type}_{args.optim}_epoch{args.epoch}_labels.npy"
    embed_path = os.path.join(output_dir, embed_filename)
    labels_path = os.path.join(output_dir, labels_filename)
    
    np.save(embed_path, all_embeddings)
    np.save(labels_path, all_labels)
    
    print(f"Embeddings saved to: {embed_path}")
    print(f"Labels saved to: {labels_path}")

if __name__ == '__main__':
    main()

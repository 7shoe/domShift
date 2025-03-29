#!/usr/bin/env python3
import os
import argparse
import yaml
import wandb
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy
from torch.utils.data import Subset, DataLoader

# ─── Model & Head Definitions ─────────────────────────────────────────────

# CNN Encoder (same as before)
class SimpleEncoder(nn.Module):
    def __init__(self, model_type: str, embedding_dim=128, input_shape=(1, 28, 28)):
        """
        Args:
            model_type (str): Either 'basic' or 'advanced' for the CNN.
            embedding_dim (int): Dimension of the output embedding.
            input_shape (tuple): (channels, height, width) of the input images.
        """
        super().__init__()
        assert model_type in {'basic', 'advanced'}, "model_type must be either 'basic' or 'advanced'"
        in_channels = input_shape[0]
        if model_type == 'advanced':
            layers = [
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),
            ]
        elif model_type == 'basic':
            layers = [
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        else:
            raise ValueError("Unsupported model_type")
        
        # Determine the flattened dimension dynamically.
        conv_trunk = nn.Sequential(*layers)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = conv_trunk(dummy_input)
            flatten_dim = conv_out.view(1, -1).shape[1]
        
        self.net = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(flatten_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

# Vision Transformer Encoder
class ViTEncoder(nn.Module):
    def __init__(self, model_type: str, embedding_dim=128, input_shape=(3, 32, 32), patch_size=4):
        """
        Args:
            model_type (str): Either 'basic' or 'advanced' for the ViT.
            embedding_dim (int): Dimension of the output embedding.
            input_shape (tuple): (channels, height, width) of the input images.
            patch_size (int): Size of each patch.
        """
        super().__init__()
        # Assume input_shape = (C, H, W)
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
        # Project patches to d_model using a Conv2d with kernel=patch_size and stride=patch_size.
        self.proj = nn.Conv2d(C, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final projection to the desired embedding dimension.
        self.fc = nn.Linear(d_model, embedding_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, d_model, H/patch_size, W/patch_size)
        B, d_model, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.pos_embed  # Add positional embeddings.
        x = self.transformer(x)  # (B, num_patches, d_model)
        # Aggregate tokens (average pooling).
        x = x.mean(dim=1)  # (B, d_model)
        x = self.fc(x)    # (B, embedding_dim)
        return x

# ProjectionHead and Predictor remain the same.
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ─── Transform ───────────────────────────────────────────────────────────────
# A transform class that accepts crop size and normalization parameters.
class SimCLRTransform:
    def __init__(self, crop_size, mean, std):
        self.transform = T.Compose([
            T.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    def __call__(self, x):
        return self.transform(x), self.transform(x)

# ─── Loss Functions ────────────────────────────────────────────────────────────
def nt_xent(z1, z2, temperature):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, float('-inf'))
    positives = torch.cat([torch.diag(sim_matrix, batch_size),
                           torch.diag(sim_matrix, -batch_size)], dim=0).unsqueeze(1)
    logits = torch.cat([positives, sim_matrix], dim=1)
    labels = torch.zeros(2 * batch_size, device=z.device, dtype=torch.long)
    return nn.CrossEntropyLoss()(logits, labels)

def byol_loss(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()

@torch.no_grad()
def update_target(online, target, momentum):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(momentum).add_(p_o.data, alpha=1 - momentum)

# ─── Training Epoch ─────────────────────────────────────────────────────────────
def train_one_epoch(loader, encoder, proj, predictor, target_enc, target_proj, optimizer, config, device):
    encoder.train(); proj.train()
    if target_enc:
        target_enc.eval(); target_proj.eval()
    total_loss = 0.0

    for (x1, x2), _ in loader:
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()

        h1, h2 = encoder(x1), encoder(x2)
        z1, z2 = proj(h1), proj(h2)

        if config.model == 'SimCLR':
            loss = nt_xent(z1, z2, config.temperature)
        elif config.model == 'SimSiam':
            p1, p2 = predictor(z1), predictor(z2)
            loss = byol_loss(p1, z2.detach()) / 2 + byol_loss(p2, z1.detach()) / 2
        else:  # BYOL
            p1, p2 = predictor(z1), predictor(z2)
            with torch.no_grad():
                tz1, tz2 = target_proj(target_enc(x1)), target_proj(target_enc(x2))
            loss = byol_loss(p1, tz2) + byol_loss(p2, tz1)

        loss.backward()
        optimizer.step()

        if config.model == 'BYOL':
            update_target(encoder, target_enc, 0.99)
            update_target(proj, target_proj, 0.99)

        total_loss += loss.item()

    return total_loss / len(loader)

# ─── Validation Epoch ───────────────────────────────────────────────────────────
def validate_epoch(loader, encoder, proj, predictor, target_enc, target_proj, config, device):
    encoder.eval(); proj.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            h1, h2 = encoder(x1), encoder(x2)
            z1, z2 = proj(h1), proj(h2)
            if config.model == 'SimCLR':
                loss = nt_xent(z1, z2, config.temperature)
            elif config.model == 'SimSiam':
                p1, p2 = predictor(z1), predictor(z2)
                loss = byol_loss(p1, z2) / 2 + byol_loss(p2, z1) / 2
            else:  # BYOL
                p1, p2 = predictor(z1), predictor(z2)
                tz1, tz2 = target_proj(target_enc(x1)), target_proj(target_enc(x2))
                loss = byol_loss(p1, tz2) + byol_loss(p2, tz1)
            total_loss += loss.item()
    return total_loss / len(loader)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['SimCLR','SimSiam','BYOL'])
    # This argument distinguishes the subsample selection configuration.
    parser.add_argument('--dataset', required=True, choices=['uniform','moderate','heavy','extreme'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.5)
    # New argument for encoder model type (for CNNs and ViTs).
    parser.add_argument('--model_type', required=True, choices=['basic','advanced'])
    # New argument for choosing the overall model architecture: CNN or ViT.
    parser.add_argument('--model_class', required=True, choices=['cnn','vit'])
    # New argument for training dataset.
    parser.add_argument('--data_source', default='MNIST', choices=['MNIST','CIFAR10','FashionMNIST'])
    # New argument to limit the number of training datapoints.
    parser.add_argument('--n_train', type=int, default=None,
                        help='Limit the number of training datapoints (default: use all available)')
    # New argument for optimizer: Adam or AdamW.
    parser.add_argument('--optim', default='Adam', choices=['Adam','AdamW'])
    args = parser.parse_args()

    # PATHS
    DATA_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
    CKPT_ROOT = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'

    # Map the dataset skew choice to a filename suffix.
    skew_map = {
        'uniform': 'uniform',
        'moderate': 'moderately_skewed',
        'heavy': 'heavily_skewed',
        'extreme': 'extremely_skewed'
    }
    yaml_name = f"{args.data_source}_{skew_map[args.dataset]}.yaml"
    yaml_path = os.path.join(DATA_ROOT, yaml_name)
    subsample_cfg = yaml.safe_load(open(yaml_path))
    all_indices = subsample_cfg['subsampled_indices']

    # Fixed train/val split (90% training, 10% validation) with a fixed seed.
    np.random.seed(42)
    all_indices = np.array(all_indices)
    np.random.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    train_indices = all_indices[:split].tolist()
    val_indices = all_indices[split:].tolist()

    # Limit the training datapoints to n_train if specified.
    if args.n_train is not None:
        train_indices = train_indices[:args.n_train]

    # Initialize wandb (if --use_wandb OR this is a sweep run)
    config = vars(args)
    if args.use_wandb or os.getenv("WANDB_RUN_ID") is not None:
        wandb.init(project='domShift-extensive', config=config)
        # Ensure all CLI defaults end up in config
        wandb.config.update(config)
        config = wandb.config

    # Set dataset-specific transform parameters.
    if args.data_source == 'MNIST':
        crop_size = 28
        mean = (0.1307,)
        std = (0.3081,)
        dataset_class = torchvision.datasets.MNIST
    elif args.data_source == 'CIFAR10':
        crop_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
    elif args.data_source == 'FashionMNIST':
        crop_size = 28
        mean = (0.2860,)
        std = (0.3530,)
        dataset_class = torchvision.datasets.FashionMNIST
    else:
        raise ValueError("Unsupported data source")

    transform = SimCLRTransform(crop_size, mean, std)
    ds = dataset_class(root=DATA_ROOT, train=True, download=False, transform=transform)

    train_loader = DataLoader(Subset(ds, train_indices), batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=256, shuffle=False, num_workers=4)

    # Automatically determine input shape from the dataset.
    # ds[0] returns ( (view1, view2), label ), so we take the shape of the first view.
    input_shape = tuple(ds[0][0][0].shape)
    print(f"Training dataset: {args.data_source} with input shape {input_shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the encoder based on the model_class argument.
    if args.model_class == 'cnn':
        encoder = SimpleEncoder(model_type=args.model_type, input_shape=input_shape).to(device)
    elif args.model_class == 'vit':
        # For ViT, we can optionally set a patch_size. Here we use 4 by default.
        encoder = ViTEncoder(model_type=args.model_type, input_shape=input_shape, patch_size=4).to(device)
    else:
        raise ValueError("Unsupported model_class")
    
    proj = ProjectionHead().to(device)

    # Choose optimizer based on argument.
    if args.optim == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=config.get('learning_rate', 3e-4))
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=config.get('learning_rate', 3e-4))
    else:
        raise ValueError("Unsupported optimizer")

    predictor = None
    target_enc = target_proj = None
    if args.model in ['BYOL', 'SimSiam']:
        predictor = Predictor().to(device)
    if args.model == 'BYOL':
        target_enc = deepcopy(encoder)
        target_proj = deepcopy(proj)

    # Create checkpoint directory including model, model_class, data_source, skew attribute, model_type, and optimizer.
    ckpt_dir = os.path.join(CKPT_ROOT, args.model, args.model_class, args.data_source, skew_map[args.dataset], args.model_type, args.optim)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create a run identifier (using wandb run id if available, else a timestamp)
    if True and wandb.run is not None:
        run_id = wandb.run.id
    else:
        run_id = str(int(time.time()))
    print(f"Run ID: {run_id}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(train_loader, encoder, proj, predictor, target_enc, target_proj, optimizer, args, device)
        val_loss = validate_epoch(val_loader, encoder, proj, predictor, target_enc, target_proj, args, device)
        print(f"[{args.model}-{args.model_class}-{args.data_source}-{skew_map[args.dataset]}-{args.model_type}-{args.optim}] Epoch {epoch}/{args.epochs} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if True:
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            wandb.watch(encoder, log='all')

        # Save checkpoint with an identifiable filename.
        ckpt_path = os.path.join(ckpt_dir, f'{run_id}_{args.model}_{args.model_class}_{args.data_source}_{skew_map[args.dataset]}_{args.model_type}_{args.optim}_encoder_epoch{epoch}.pth')
        torch.save(encoder.state_dict(), ckpt_path)
        if True:
            wandb.log({'checkpoint': ckpt_path})
            if epoch==args.epochs:
                wandb.save(ckpt_path)

    # finish
    if True:
        wandb.finish()

if __name__ == '__main__':
    main()
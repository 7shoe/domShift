import os
import json
import random
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Subset

# Updated SimCLRTransform accepts parameters.
class SimCLRTransform:
    def __init__(self, crop_size, mean, std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    def __call__(self, x):
        # Return two different augmented views.
        return self.transform(x), self.transform(x)

def main(args):
    # ---------------------
    # Parameters and Setup
    # ---------------------
    DATA_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
    CKPT_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'
    
    ALPHA = args.alpha                  # Pareto parameter (or -1 for extreme sampling).
    DATASET_NAME = args.dataset_name      # Name for the resulting subsampled dataset.
    SEED = 42
    # For MNIST, CIFAR10, and FashionMNIST we assume 10 classes labeled 0-9.
    CLASS_ORDER = list(range(10))

    # Set random seeds for reproducibility.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------------------
    # Define Transform and Dataset based on Data Source.
    # ---------------------
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
    full_dataset = dataset_class(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )

    # ---------------------
    # Compute Subsampled Indices with Fixed Overall Sample Size
    # ---------------------
    # Build a dictionary mapping each class label to its list of indices.
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_dataset):
        label_to_indices[label].append(idx)
    
    # Use the minimum number of samples available across classes.
    n_total = min(len(indices) for indices in label_to_indices.values())
    # Compute the harmonic sum over 10 classes (as in the alpha=1.0 setting).
    harmonic_sum = sum(1.0 / (i + 1) for i in range(10))
    target_total = n_total * harmonic_sum  # Overall target sample size.

    subsampled_indices = []
    not_subsampled_indices = []

    if ALPHA == -1:
        # Extreme sampling: sample uniformly from only the top few classes.
        # Here, we choose k = 3 classes (classes 0, 1, and 2).
        k = 3
        selected_classes = CLASS_ORDER[:k]
        per_class_target = int(round(target_total / k))
        for label in selected_classes:
            indices = label_to_indices[label].copy()
            random.shuffle(indices)
            n_to_sample = min(per_class_target, len(indices))
            selected = indices[:n_to_sample]
            not_selected = indices[n_to_sample:]
            subsampled_indices.extend(selected)
            not_subsampled_indices.extend(not_selected)
    else:
        # Standard Pareto sampling using the given alpha.
        raw_fractions = [1.0 / ((r + 1) ** ALPHA) for r in range(10)]
        sum_raw = sum(raw_fractions)
        scale = target_total / (n_total * sum_raw)
        for rank, label in enumerate(CLASS_ORDER):
            indices = label_to_indices[label].copy()
            desired_frac = (1.0 / ((rank + 1) ** ALPHA)) * scale
            n_to_sample = int(round(n_total * desired_frac))
            n_to_sample = min(n_to_sample, len(indices))
            random.shuffle(indices)
            selected = indices[:n_to_sample]
            not_selected = indices[n_to_sample:]
            subsampled_indices.extend(selected)
            not_subsampled_indices.extend(not_selected)
    
    subsampled_indices.sort()
    not_subsampled_indices.sort()

    # ---------------------
    # Save the Subsampling Information to YAML
    # ---------------------
    reconstruction_info = {
        'DATA_PATH': DATA_PATH,
        'CKPT_PATH': CKPT_PATH,
        'alpha': ALPHA,
        'dataset_name': DATASET_NAME,
        'seed': SEED,
        'data_source': args.data_source,
        'class_order': CLASS_ORDER,
        'n_total_per_class': n_total,
        'target_total_sample_size': target_total,
        'subsampled_indices': subsampled_indices,
        'num_subsampled': len(subsampled_indices),
        'not_subsampled_indices': not_subsampled_indices,
        'num_not_subsampled': len(not_subsampled_indices)
    }

    yaml_save_path = os.path.join(DATA_PATH, f'{DATASET_NAME}.yaml')
    with open(yaml_save_path, 'w') as f:
        yaml.dump(reconstruction_info, f, default_flow_style=False)
    print("Saved subsampling information to", yaml_save_path)
    print("Total subsampled images:", len(subsampled_indices))
    print("Total remaining images:", len(not_subsampled_indices))

    # ---------------------
    # Create DataLoader(s)
    # ---------------------
    subsampled_dataset = Subset(full_dataset, subsampled_indices)
    data_loader = DataLoader(subsampled_dataset, batch_size=256, shuffle=True, num_workers=2)

    remaining_dataset = Subset(full_dataset, not_subsampled_indices)
    remaining_loader = DataLoader(remaining_dataset, batch_size=256, shuffle=False, num_workers=2)

    # Now, you can continue with training using data_loader or perform inference on remaining_loader.
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a subsampled dataset with controlled class imbalance.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Pareto parameter (e.g., 1.0 for standard skew; use -1 for extreme sampling from only a few classes).')
    parser.add_argument('--dataset_name', type=str, default='MNIST_heavily_skewed',
                        help='Name for the resulting subsampled dataset (e.g., MNIST_uniform, MNIST_heavily_skewed, CIFAR10_extremely_skewed)')
    parser.add_argument('--data_source', type=str, default='MNIST', choices=['MNIST','CIFAR10','FashionMNIST'],
                        help='Data source to subsample from (MNIST, CIFAR10, or FashionMNIST).')
    args = parser.parse_args()
    main(args)
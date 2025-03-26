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

def main(args):
    # ---------------------
    # Parameters and Setup
    # ---------------------
    DATA_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
    CKPT_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'
    
    # User-provided parameters via argparse
    ALPHA = args.alpha            # Pareto parameter controlling imbalance degree.
    DATASET_NAME = args.dataset_name  # Name for the resulting subsampled dataset.
    SEED = 42
    # Define the order of class labels (first in the list is considered “most frequent”)
    # MNIST specific class labels
    CLASS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------------------
    # Define Transform(s)
    # ---------------------
    # This example transform returns two augmented views (e.g. for SimCLR).
    class SimCLRTransform:
        def __init__(self):
            # MNIST-specific transform
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # TODO: other transforms (CIFAR10, FashionMNIST)

        def __call__(self, x):
            # Return two different augmented views.
            return self.transform(x), self.transform(x)

    transform = SimCLRTransform()

    # ---------------------
    # Load Full MNIST Dataset
    # ---------------------
    full_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )

    # ---------------------
    # Compute Subsampled Indices with Fixed Overall Sample Size
    # ---------------------
    # MNIST training: each class has (roughly) 6000 examples.
    # We assume n_total is the same for every class.
    # (If not, the code uses each class’s own count.)
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_dataset):
        label_to_indices[label].append(idx)

    # n_total: number of samples available for each class (should be 6000 for MNIST train)
    n_total = min(len(indices) for indices in label_to_indices.values())

    # We'll fix the overall sample size to that of the heavy-skew setting (alpha = 1.0).
    # Compute the harmonic sum over 10 classes:
    harmonic_sum = sum(1.0 / (i + 1) for i in range(10))
    target_total = n_total * harmonic_sum  # target overall number of images

    # For the current alpha, compute raw fractions per class.
    raw_fractions = [1.0 / ((r + 1) ** ALPHA) for r in range(10)]
    sum_raw = sum(raw_fractions)
    # Compute scaling factor so that overall sample size becomes target_total.
    scale = target_total / (n_total * sum_raw)

    subsampled_indices = []
    not_subsampled_indices = []

    # For each class (using the predefined order), randomly select the desired number of samples.
    for rank, label in enumerate(CLASS_ORDER):
        indices = label_to_indices[label].copy()  # make a copy
        # Compute desired fraction for this class, scaled to achieve target_total.
        desired_frac = (1.0 / ((rank + 1) ** ALPHA)) * scale
        n_to_sample = int(round(n_total * desired_frac))
        # To ensure we don't request more samples than available:
        n_to_sample = min(n_to_sample, len(indices))
        # Shuffle indices for reproducibility.
        random.shuffle(indices)
        selected = indices[:n_to_sample]
        not_selected = indices[n_to_sample:]
        subsampled_indices.extend(selected)
        not_subsampled_indices.extend(not_selected)

    # (Optional) Sort indices for convenience.
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
        'class_order': CLASS_ORDER,
        'n_total_per_class': n_total,
        'target_total_sample_size': target_total,
        'raw_fractions': raw_fractions,
        'scale': scale,
        'num_subsampled': len(subsampled_indices),
        'num_not_subsampled': len(not_subsampled_indices),
        'subsampled_indices': subsampled_indices,
        'not_subsampled_indices': not_subsampled_indices
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

    # (Optional) DataLoader for remaining images (for inference or evaluation)
    remaining_dataset = Subset(full_dataset, not_subsampled_indices)
    remaining_loader = DataLoader(remaining_dataset, batch_size=256, shuffle=False, num_workers=2)

    # Now, you can continue with training using data_loader and perform inference on full_dataset
    # or on remaining_dataset as needed.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate subsampled MNIST with controlled class imbalance.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Pareto distribution parameter (e.g. 0.0 for uniform, 1.0 for heavy skew).')
    parser.add_argument('--dataset_name', type=str, default='MNIST_heavily_skewed',
                        help='Name for the resulting subsampled dataset (e.g. MNIST_uniform, MNIST_heavily_skewed, MNIST_moderately_skewed)')
    args = parser.parse_args()
    main(args)


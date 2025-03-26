#!/usr/bin/env python3
import os
import random
import yaml
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

def main():
    # Paths
    DATA_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
    CKPT_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'
    DATASET_NAME = 'MNIST_extreme'
    SEED = 42

    # Fix randomness
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Transform (same as your SimCLRTransform)
    class SimCLRTransform:
        def __init__(self):
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        def __call__(self, x):
            return self.transform(x), self.transform(x)

    transform = SimCLRTransform()

    # Load full MNIST train
    full = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)

    # Collect indices by label
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, lbl) in enumerate(full):
        label_to_indices[lbl].append(idx)

    # Sample exactly 1000 of class 0 and 1000 of class 8
    subsampled = []
    for lbl in (0, 8):
        indices = label_to_indices[lbl].copy()
        random.shuffle(indices)
        subsampled.extend(indices[:1000])
    subsampled.sort()

    # Everything else is “not subsampled”
    all_indices = set(range(len(full)))
    not_subsampled = sorted(all_indices - set(subsampled))

    # Write YAML
    info = {
        'DATA_PATH': DATA_PATH,
        'CKPT_PATH': CKPT_PATH,
        'dataset_name': DATASET_NAME,
        'seed': SEED,
        'class_order': list(range(10)),
        'n_total_per_class': min(len(label_to_indices[0]), len(label_to_indices[8])),
        'target_total_sample_size': len(subsampled),
        'num_subsampled': len(subsampled),
        'num_not_subsampled': len(not_subsampled),
        'subsampled_indices': subsampled,
        'not_subsampled_indices': not_subsampled
    }
    yaml_path = os.path.join(DATA_PATH, f'{DATASET_NAME}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)
    print(f"Saved YAML → {yaml_path}")

    # DataLoaders
    subsampled_loader = DataLoader(Subset(full, subsampled), batch_size=256, shuffle=True, num_workers=2)
    remaining_loader  = DataLoader(Subset(full, not_subsampled), batch_size=256, shuffle=False, num_workers=2)
    print(f"Subsampled dataset size: {len(subsampled)}")
    print(f"Remaining dataset size: {len(not_subsampled)}")

if __name__ == '__main__':
    main()

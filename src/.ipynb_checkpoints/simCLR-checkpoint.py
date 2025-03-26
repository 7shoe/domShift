import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# paths
DATA_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/datasets'
CKPT_PATH = '/eagle/projects/argonne_tpc/siebenschuh/domain_shift_data/checkpoints'

# Define a transform that produces two random augmented views for SimCLR
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def __call__(self, x):
        return self.transform(x), self.transform(x)

# Simple CNN encoder (same as above)
class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Projection head to map encoder outputs to a space where contrastive loss is applied
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss implementation
def nt_xent_loss(z1, z2, temperature=0.5):
    # Normalize embeddings
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # shape: [2*B, D]
    similarity_matrix = torch.matmul(z, z.T)  # cosine similarity between all pairs

    # Create labels for contrastive loss
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)

    # Mask to remove self-comparisons
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    # For each sample, the positive example is the other view in the batch.
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), 
                           torch.diag(similarity_matrix, -batch_size)], dim=0).unsqueeze(1)
    
    negatives = similarity_matrix
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    # Positive examples are at index 0 for each sample.
    loss = nn.CrossEntropyLoss()(logits, torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device))
    return loss

# Prepare the MNIST dataset with our SimCLR transforms
transform = SimCLRTransform()
mnist_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=256, shuffle=True, num_workers=2)

# Set up the model, projection head, optimizer, and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SimpleEncoder().to(device)
projection_head = ProjectionHead().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=1e-3)

# Pre-training loop
num_epochs = 10
encoder.train()
projection_head.train()
for epoch in range(num_epochs):
    total_loss = 0
    for (x1, x2), _ in data_loader:
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()
        
        # Get encoder representations
        h1 = encoder(x1)
        h2 = encoder(x2)
        # Get projections
        z1 = projection_head(h1)
        z2 = projection_head(h2)
        
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(data_loader):.4f}")

# Save the pre-trained encoder for future inference
torch.save(encoder.state_dict(), f'{CKPT_PATH}/pretrained_simclr_mnist.pth')

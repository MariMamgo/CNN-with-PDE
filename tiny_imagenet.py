# PDE Diffusion Neural Network for TinyImageNet
# Time-dependent alpha and beta matrices for RGB images (64x64x3, 200 classes)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import requests
import tarfile

# --- TinyImageNet Dataset Class ---
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, download=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if download:
            self.download_dataset()
        
        self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200')
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        self.load_data()
    
    def download_dataset(self):
        """Download TinyImageNet dataset if not present"""
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.root_dir, "tiny-imagenet-200.zip")
        extract_path = os.path.join(self.root_dir, "tiny-imagenet-200")
        
        if not os.path.exists(extract_path):
            print("Downloading TinyImageNet dataset...")
            os.makedirs(self.root_dir, exist_ok=True)
            
            # Note: In practice, you would download this manually due to size
            # This is a placeholder for the download logic
            print(f"Please download TinyImageNet from {url}")
            print(f"Extract to {extract_path}")
            print("Creating dummy data for demonstration...")
            
            # Create dummy structure for demo
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy data structure for demonstration"""
        dummy_dir = os.path.join(self.root_dir, 'tiny-imagenet-200')
        os.makedirs(dummy_dir, exist_ok=True)
        
        # Create some dummy classes
        for i in range(10):  # Just 10 classes for demo
            class_id = f"n{i:08d}"
            class_dir = os.path.join(dummy_dir, 'train', class_id, 'images')
            os.makedirs(class_dir, exist_ok=True)
            
            # Create dummy images
            for j in range(50):  # 50 images per class
                dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                dummy_img.save(os.path.join(class_dir, f"{class_id}_{j}.JPEG"))
        
        # Create val structure
        val_dir = os.path.join(dummy_dir, 'val', 'images')
        os.makedirs(val_dir, exist_ok=True)
        for i in range(100):  # 100 val images
            dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            dummy_img.save(os.path.join(val_dir, f"val_{i}.JPEG"))
        
        # Create val annotations
        with open(os.path.join(dummy_dir, 'val', 'val_annotations.txt'), 'w') as f:
            for i in range(100):
                class_idx = i % 10
                f.write(f"val_{i}.JPEG\tn{class_idx:08d}\t0\t0\t64\t64\n")
    
    def load_data(self):
        """Load dataset paths and labels"""
        if self.split == 'train':
            train_dir = os.path.join(self.data_dir, 'train')
            if not os.path.exists(train_dir):
                print("Train directory not found, using dummy data")
                return
                
            class_dirs = sorted(os.listdir(train_dir))
            for idx, class_dir in enumerate(class_dirs):
                self.class_to_idx[class_dir] = idx
                images_dir = os.path.join(train_dir, class_dir, 'images')
                if os.path.exists(images_dir):
                    for img_name in os.listdir(images_dir):
                        if img_name.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(images_dir, img_name))
                            self.labels.append(idx)
        
        elif self.split == 'val':
            val_dir = os.path.join(self.data_dir, 'val')
            if not os.path.exists(val_dir):
                print("Val directory not found, using dummy data")
                return
                
            # Load class mapping from train
            train_dir = os.path.join(self.data_dir, 'train')
            if os.path.exists(train_dir):
                class_dirs = sorted(os.listdir(train_dir))
                for idx, class_dir in enumerate(class_dirs):
                    self.class_to_idx[class_dir] = idx
            
            # Load validation annotations
            annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            if os.path.exists(annotations_file):
                with open(annotations_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        img_name, class_id = parts[0], parts[1]
                        img_path = os.path.join(val_dir, 'images', img_name)
                        if os.path.exists(img_path) and class_id in self.class_to_idx:
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_id])
    
    def __len__(self):
        return len(self.image_paths) if self.image_paths else 1000  # dummy length
    
    def __getitem__(self, idx):
        if not self.image_paths:  # dummy data
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            label = idx % 10
        else:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# --- Multi-Channel PDE Diffusion Layer ---
class MultiChannelDiffusionLayer(nn.Module):
    def __init__(self, size=64, channels=3, dt=0.2, dx=1.0, num_steps=3):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.num_steps = num_steps

        # Per-channel diffusion coefficients as matrices (learnable)
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 1.5)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 1.5)

        # Time-dependent modulation parameters as matrices (learnable)
        self.alpha_time_coeff = nn.Parameter(torch.zeros(channels, size, size))
        self.beta_time_coeff = nn.Parameter(torch.zeros(channels, size, size))

        # Cross-channel coupling (optional)
        self.channel_coupling = nn.Parameter(torch.eye(channels) * 0.1)

        # Stability parameters
        self.stability_eps = 1e-6

    def get_alpha_beta_at_time(self, t, channel=None):
        """Get alpha and beta coefficient matrices at time t for specific channel"""
        if channel is None:
            alpha_t = self.alpha_base + self.alpha_time_coeff * t
            beta_t = self.beta_base + self.beta_time_coeff * t
        else:
            alpha_t = self.alpha_base[channel] + self.alpha_time_coeff[channel] * t
            beta_t = self.beta_base[channel] + self.beta_time_coeff[channel] * t

        # Ensure positive coefficients for stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps)
        beta_t = torch.clamp(beta_t, min=self.stability_eps)

        return alpha_t, beta_t

    def forward(self, u):
        B, C, H, W = u.shape
        
        # Process each channel separately then apply coupling
        current_time = 0.0
        
        for step in range(self.num_steps):
            u_new = torch.zeros_like(u)
            
            # Process each channel
            for c in range(C):
                alpha_t, beta_t = self.get_alpha_beta_at_time(current_time, channel=c)
                
                # Extract single channel
                u_c = u[:, c, :, :]
                
                # Strang splitting: half step x, full step y, half step x
                u_c = self.diffuse_x_vectorized(u_c, alpha_t, self.dt / 2, self.dx)
                u_c = self.diffuse_y_vectorized(u_c, beta_t, self.dt, self.dx)
                u_c = self.diffuse_x_vectorized(u_c, alpha_t, self.dt / 2, self.dx)
                
                u_new[:, c, :, :] = u_c
            
            # Apply cross-channel coupling
            u = self.apply_channel_coupling(u_new)
            current_time += self.dt

        return u

    def apply_channel_coupling(self, u):
        """Apply learnable cross-channel coupling"""
        B, C, H, W = u.shape
        u_flat = u.view(B, C, -1)  # (B, C, H*W)
        
        # Apply channel coupling: (C, C) @ (B, C, H*W) -> (B, C, H*W)
        u_coupled = torch.einsum('cc,bci->bci', self.channel_coupling, u_flat)
        
        return u_coupled.view(B, C, H, W)

    def diffuse_x_vectorized(self, u, alpha_matrix, dt, dx):
        """Vectorized diffusion in x-direction"""
        B, H, W = u.shape
        device = u.device

        # Reshape for batch processing: (B, H, W) -> (B*H, W)
        u_flat = u.contiguous().view(B * H, W)

        # Expand alpha_matrix for all batches: (H, W) -> (B*H, W)
        alpha_expanded = alpha_matrix.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * H, W)

        # Apply smoothing to coefficients for stability
        alpha_smooth = self.smooth_coefficients(alpha_expanded, dim=1)
        coeff = alpha_smooth * dt / (dx ** 2)

        # Build tridiagonal system coefficients
        a = -coeff  # sub-diagonal
        c = -coeff  # super-diagonal
        b = 1 + 2 * coeff  # main diagonal

        # Apply boundary conditions (Neumann - no flux at boundaries)
        b_modified = b.clone()
        b_modified[:, 0] = 1 + coeff[:, 0]
        b_modified[:, -1] = 1 + coeff[:, -1]

        # Solve all tridiagonal systems in parallel
        result = self.thomas_solver_batch(a, b_modified, c, u_flat)

        return result.view(B, H, W)

    def diffuse_y_vectorized(self, u, beta_matrix, dt, dx):
        """Vectorized diffusion in y-direction"""
        B, H, W = u.shape
        device = u.device

        # Transpose to work on columns: (B, H, W) -> (B, W, H)
        u_t = u.transpose(1, 2).contiguous()
        u_flat = u_t.view(B * W, H)

        # Expand beta_matrix for all batches: (H, W) -> (B*W, H)
        beta_expanded = beta_matrix.t().unsqueeze(0).expand(B, -1, -1).contiguous().view(B * W, H)

        # Apply smoothing to coefficients for stability
        beta_smooth = self.smooth_coefficients(beta_expanded, dim=1)
        coeff = beta_smooth * dt / (dx ** 2)

        # Build tridiagonal system coefficients
        a = -coeff  # sub-diagonal
        c = -coeff  # super-diagonal
        b = 1 + 2 * coeff  # main diagonal

        # Apply boundary conditions
        b_modified = b.clone()
        b_modified[:, 0] = 1 + coeff[:, 0]
        b_modified[:, -1] = 1 + coeff[:, -1]

        # Solve all tridiagonal systems in parallel
        result = self.thomas_solver_batch(a, b_modified, c, u_flat)

        # Transpose back
        return result.view(B, W, H).transpose(1, 2).contiguous()

    def smooth_coefficients(self, coeffs, dim=1, kernel_size=3):
        """Apply smoothing to coefficients for numerical stability"""
        if kernel_size == 1:
            return coeffs

        padding = kernel_size // 2
        if dim == 1:
            coeffs_padded = F.pad(coeffs, (padding, padding), mode='replicate')
            kernel = torch.ones(1, 1, kernel_size, device=coeffs.device) / kernel_size
            smoothed = F.conv1d(coeffs_padded.unsqueeze(1), kernel, padding=0).squeeze(1)
        else:
            raise NotImplementedError("Only dim=1 smoothing implemented")

        return smoothed

    def thomas_solver_batch(self, a, b, c, d):
        """Batch Thomas algorithm for tridiagonal systems"""
        batch_size, N = d.shape
        device = d.device
        eps = self.stability_eps

        # Initialize working arrays
        c_star = torch.zeros_like(d)
        d_star = torch.zeros_like(d)

        # Forward elimination
        denom_0 = b[:, 0] + eps
        c_star = c_star.scatter(1, torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                               (c[:, 0] / denom_0).unsqueeze(1))
        d_star = d_star.scatter(1, torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                               (d[:, 0] / denom_0).unsqueeze(1))

        for i in range(1, N):
            denom = b[:, i] - a[:, i] * c_star[:, i-1] + eps

            if i < N-1:
                c_star = c_star.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                                       (c[:, i] / denom).unsqueeze(1))

            d_val = (d[:, i] - a[:, i] * d_star[:, i-1]) / denom
            d_star = d_star.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                                   d_val.unsqueeze(1))

        # Back substitution
        x = torch.zeros_like(d)
        x = x.scatter(1, torch.full((batch_size, 1), N-1, dtype=torch.long, device=device),
                     d_star[:, -1].unsqueeze(1))

        for i in range(N-2, -1, -1):
            x_val = d_star[:, i] - c_star[:, i] * x[:, i+1]
            x = x.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                         x_val.unsqueeze(1))

        return x


# --- CNN Classifier for TinyImageNet ---
class TinyImageNetPDEClassifier(nn.Module):
    def __init__(self, num_classes=200, dropout_rate=0.3):
        super().__init__()
        
        # Multi-channel PDE diffusion layer
        self.diff = MultiChannelDiffusionLayer(size=64, channels=3)
        
        # CNN backbone for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply PDE diffusion
        x = self.diff(x)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        
        return self.fc3(x)


# --- Data Loading Functions ---
def create_tinyimagenet_data_loaders(data_root='./data', batch_size=64, num_workers=4):
    """Create TinyImageNet data loaders"""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TinyImageNetDataset(data_root, split='train', transform=transform_train)
    val_dataset = TinyImageNetDataset(data_root, split='val', transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


# --- Training Function ---
def train_tinyimagenet_model():
    """Training function for TinyImageNet"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_tinyimagenet_data_loaders(batch_size=32)  # Smaller batch for memory
    
    # Initialize model
    model = TinyImageNetPDEClassifier(num_classes=200).to(device)
    
    # Training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("Starting TinyImageNet PDE training...")
    import time
    
    for epoch in range(15):  # More epochs for complex dataset
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")
        
        # Monitor PDE parameters
        with torch.no_grad():
            for c in range(3):  # RGB channels
                alpha_stats = {
                    'base_mean': model.diff.alpha_base[c].mean().item(),
                    'base_std': model.diff.alpha_base[c].std().item(),
                    'time_mean': model.diff.alpha_time_coeff[c].mean().item(),
                }
                print(f"Channel {['R','G','B'][c]} - α_base: μ={alpha_stats['base_mean']:.3f}±{alpha_stats['base_std']:.3f}, "
                      f"α_time: μ={alpha_stats['time_mean']:.3f}")
        
        print("-" * 80)
    
    return model, val_loader


# --- Evaluation and Visualization ---
def evaluate_and_visualize_tinyimagenet(model, val_loader):
    """Evaluation and visualization for TinyImageNet"""
    device = next(model.parameters()).device
    model.eval()
    
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(imgs)
            pred = output.argmax(dim=1)
            val_correct += pred.eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = 100. * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    # Visualization
    with torch.no_grad():
        images, labels = next(iter(val_loader))
        images = images.to(device)
        
        # Show sample images and their diffused versions
        plt.figure(figsize=(16, 12))
        
        for i in range(6):
            # Original image
            plt.subplot(4, 6, i + 1)
            img = images[i].cpu().permute(1, 2, 0)
            # Denormalize for display
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Original {i+1}")
            
            # After PDE diffusion
            plt.subplot(4, 6, i + 7)
            diffused = model.diff(images[i:i+1]).squeeze(0).cpu().permute(1, 2, 0)
            diffused = diffused * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            diffused = torch.clamp(diffused, 0, 1)
            plt.imshow(diffused)
            plt.axis('off')
            plt.title(f"After PDE {i+1}")
        
        # Visualize learned diffusion coefficients for each channel
        alpha_matrices = model.diff.alpha_base.detach().cpu().numpy()
        channel_names = ['Red', 'Green', 'Blue']
        
        for c in range(3):
            plt.subplot(4, 6, 13 + c)
            plt.imshow(alpha_matrices[c], cmap='RdBu_r')
            plt.colorbar()
            plt.title(f'{channel_names[c]} α-matrix')
            plt.axis('off')
        
        # Cross-channel coupling matrix
        plt.subplot(4, 6, 16)
        coupling = model.diff.channel_coupling.detach().cpu().numpy()
        plt.imshow(coupling, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Channel Coupling')
        plt.axis('off')
        
        plt.suptitle('PDE Diffusion Network on TinyImageNet\nMulti-Channel Time-Dependent Diffusion', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # Print parameter statistics
    print(f"\nPDE Parameter Analysis:")
    print(f"Simulation: dt={model.diff.dt}, steps={model.diff.num_steps}")
    
    with torch.no_grad():
        coupling_matrix = model.diff.channel_coupling
        print(f"Channel coupling eigenvalues: {torch.linalg.eigvals(coupling_matrix).real}")


# --- Main execution ---
if __name__ == "__main__":
    print("Starting PDE diffusion training on TinyImageNet...")
    print("Note: This requires the TinyImageNet dataset to be downloaded.")
    print("Please download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    
    model, val_loader = train_tinyimagenet_model()
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize_tinyimagenet(model, val_loader)
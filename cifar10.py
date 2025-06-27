# PDE Diffusion Neural Network for CIFAR-10/CIFAR-100
# Time-dependent alpha and beta matrices for RGB images (32x32x3)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-100 superclass names (for visualization)
CIFAR100_SUPERCLASSES = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
    'trees', 'vehicles 1', 'vehicles 2'
]

# --- Multi-Channel PDE Diffusion Layer for CIFAR ---
class CIFARDiffusionLayer(nn.Module):
    def __init__(self, size=32, channels=3, dt=0.15, dx=1.0, num_steps=4):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.num_steps = num_steps

        # Per-channel diffusion coefficients as matrices (learnable)
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 1.2)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 1.2)

        # Time-dependent modulation parameters as matrices (learnable)
        self.alpha_time_coeff = nn.Parameter(torch.zeros(channels, size, size))
        self.beta_time_coeff = nn.Parameter(torch.zeros(channels, size, size))

        # Cross-channel coupling for RGB interactions
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
        
        # Process each channel with time-dependent coefficients
        current_time = 0.0
        
        for step in range(self.num_steps):
            u_new = torch.zeros_like(u)
            
            # Process each channel independently
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


# --- CIFAR Classifier with PDE Preprocessing ---
class CIFARPDEClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super().__init__()
        
        # Multi-channel PDE diffusion layer for CIFAR (32x32)
        self.diff = CIFARDiffusionLayer(size=32, channels=3)
        
        # Compact CNN backbone optimized for CIFAR
        self.conv_block1 = self._make_conv_block(3, 64)
        self.conv_block2 = self._make_conv_block(64, 128)
        self.conv_block3 = self._make_conv_block(128, 256)
        self.conv_block4 = self._make_conv_block(256, 512)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with batch norm and activation"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        # Apply PDE diffusion preprocessing
        x = self.diff(x)
        
        # CNN feature extraction
        x = self.conv_block1(x)  # 32x32 -> 16x16
        x = self.conv_block2(x)  # 16x16 -> 8x8
        x = self.conv_block3(x)  # 8x8 -> 4x4
        x = self.conv_block4(x)  # 4x4 -> 2x2
        
        # Global pooling and classification
        x = self.global_avg_pool(x)  # 2x2 -> 1x1
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# --- Data Loading Functions ---
def create_cifar_data_loaders(dataset='cifar10', data_root='./data', batch_size=128, num_workers=4):
    """Create CIFAR-10 or CIFAR-100 data loaders"""
    
    # CIFAR statistics
    if dataset.lower() == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        dataset_class = datasets.CIFAR10
        num_classes = 10
    elif dataset.lower() == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        dataset_class = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError("Dataset must be 'cifar10' or 'cifar100'")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create datasets
    train_dataset = dataset_class(data_root, train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(data_root, train=False, download=True, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


# --- Training Function ---
def train_cifar_model(dataset='cifar10'):
    """Training function for CIFAR-10/100"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on {dataset.upper()}")
    
    # Create data loaders
    train_loader, test_loader, num_classes = create_cifar_data_loaders(
        dataset=dataset, batch_size=128
    )
    
    # Initialize model
    model = CIFARPDEClassifier(num_classes=num_classes).to(device)
    
    # Training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("Starting CIFAR PDE training...")
    import time
    
    for epoch in range(50):  # 10 epochs for CIFAR
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
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")
        
        # Monitor PDE parameters
        if epoch % 2 == 0:  # Every 2 epochs
            with torch.no_grad():
                for c, channel_name in enumerate(['Red', 'Green', 'Blue']):
                    alpha_base = model.diff.alpha_base[c]
                    alpha_time = model.diff.alpha_time_coeff[c]
                    print(f"{channel_name} Channel - α_base: μ={alpha_base.mean().item():.3f}±{alpha_base.std().item():.3f}, "
                          f"α_time: μ={alpha_time.mean().item():.4f}")
                
                # Channel coupling analysis
                coupling = model.diff.channel_coupling
                print(f"Channel coupling matrix diag: {coupling.diag().detach().cpu().numpy()}")
        
        print("-" * 80)
    
    return model, test_loader, num_classes


# --- Evaluation and Visualization ---
def evaluate_and_visualize_cifar(model, test_loader, num_classes, dataset='cifar10'):
    """Evaluation and visualization for CIFAR"""
    device = next(model.parameters()).device
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(imgs)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(labels).sum().item()
            test_total += labels.size(0)
            
            # Track class-wise accuracy
            for i in range(num_classes):
                class_mask = labels == i
                if class_mask.sum() > 0:
                    class_correct[i] += pred[class_mask].eq(labels[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\nOverall Test Accuracy: {test_acc:.2f}%")
    
    # Print class-wise accuracies for CIFAR-10
    if dataset.lower() == 'cifar10':
        print("\nClass-wise Accuracies (CIFAR-10):")
        for i in range(10):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"{CIFAR10_CLASSES[i]}: {acc:.2f}%")
    else:
        print(f"\nCIFAR-100 Average Class Accuracy: {(class_correct / class_total).mean() * 100:.2f}%")
    
    # Visualization
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        
        # Denormalization parameters
        if dataset.lower() == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2470, 0.2435, 0.2616])
            class_names = CIFAR10_CLASSES
        else:
            mean = torch.tensor([0.5071, 0.4867, 0.4408])
            std = torch.tensor([0.2675, 0.2565, 0.2761])
            class_names = [f"Class {i}" for i in range(100)]  # Simplified for CIFAR-100
        
        plt.figure(figsize=(18, 12))
        
        # Sample images, predictions, and diffused versions
        for i in range(8):
            # Original image
            plt.subplot(5, 8, i + 1)
            img = images[i].cpu().permute(1, 2, 0)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            plt.imshow(img)
            plt.axis('off')
            if dataset.lower() == 'cifar10':
                true_label = class_names[labels[i]]
                pred_label = class_names[predicted[i]]
            else:
                true_label = f"True: {labels[i]}"
                pred_label = f"Pred: {predicted[i]}"
            
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"{true_label}", fontsize=8)
            
            # Prediction
            plt.subplot(5, 8, i + 9)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{pred_label}", color=color, fontsize=8)
            
            # After PDE diffusion
            plt.subplot(5, 8, i + 17)
            diffused = model.diff(images[i:i+1]).squeeze(0).cpu().permute(1, 2, 0)
            diffused = diffused * std + mean
            diffused = torch.clamp(diffused, 0, 1)
            plt.imshow(diffused)
            plt.axis('off')
            plt.title("After PDE", fontsize=8)
        
        # Visualize learned diffusion coefficients
        alpha_matrices = model.diff.alpha_base.detach().cpu().numpy()
        channel_names = ['Red α-matrix', 'Green α-matrix', 'Blue α-matrix']
        
        for c in range(3):
            plt.subplot(5, 8, 25 + c)
            plt.imshow(alpha_matrices[c], cmap='RdBu_r', interpolation='nearest')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(channel_names[c], fontsize=9)
            plt.axis('off')
        
        # Cross-channel coupling matrix
        plt.subplot(5, 8, 28)
        coupling = model.diff.channel_coupling.detach().cpu().numpy()
        plt.imshow(coupling, cmap='RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('RGB Coupling', fontsize=9)
        plt.axis('off')
        
        # Time evolution visualization
        plt.subplot(5, 8, 29)
        time_coeffs = model.diff.alpha_time_coeff[0].detach().cpu().numpy()  # Red channel
        plt.imshow(time_coeffs, cmap='RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Time Evolution', fontsize=9)
        plt.axis('off')
        
        plt.suptitle(f'PDE Diffusion Network on {dataset.upper()}\nMulti-Channel Diffusion with Time-Dependent Coefficients', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # Print detailed parameter analysis
    print(f"\nPDE Parameter Analysis:")
    print(f"Simulation: dt={model.diff.dt}, steps={model.diff.num_steps}")
    
    with torch.no_grad():
        # Analyze parameter ranges
        for c, name in enumerate(['Red', 'Green', 'Blue']):
            alpha_base = model.diff.alpha_base[c]
            alpha_time = model.diff.alpha_time_coeff[c]
            print(f"{name} Channel:")
            print(f"  α_base range: [{alpha_base.min().item():.3f}, {alpha_base.max().item():.3f}]")
            print(f"  α_time range: [{alpha_time.min().item():.4f}, {alpha_time.max().item():.4f}]")
        
        # Channel coupling eigenvalues
        coupling_matrix = model.diff.channel_coupling
        eigenvals = torch.linalg.eigvals(coupling_matrix).real
        print(f"Channel coupling eigenvalues: {eigenvals.cpu().numpy()}")


# --- Main execution ---
if __name__ == "__main__":
    # You can choose 'cifar10' or 'cifar100'
    dataset_choice = 'cifar10'  # Change to 'cifar100' for CIFAR-100
    
    print(f"Starting PDE diffusion training on {dataset_choice.upper()}...")
    model, test_loader, num_classes = train_cifar_model(dataset=dataset_choice)
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize_cifar(model, test_loader, num_classes, dataset=dataset_choice)

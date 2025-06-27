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
    def __init__(self, size=32, channels=3, dt=0.05, dx=1.0, num_steps=3, use_implicit=True):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.num_steps = num_steps
        self.use_implicit = use_implicit

        # Simplified coefficients for stability - per channel scalars
        self.alpha_base = nn.Parameter(torch.ones(channels) * 0.3)  # Smaller initial values
        self.beta_base = nn.Parameter(torch.ones(channels) * 0.3)

        # Optional spatial variation (but much smaller)
        self.alpha_spatial = nn.Parameter(torch.zeros(channels, size, size) * 0.01)
        self.beta_spatial = nn.Parameter(torch.zeros(channels, size, size) * 0.01)

        # Cross-channel coupling for RGB interactions
        self.channel_coupling = nn.Parameter(torch.eye(channels) * 0.05)

        # Stability parameters
        self.stability_eps = 1e-6
        self.max_coeff = 0.25 if not use_implicit else 1.0

    def get_alpha_beta_at_time(self, t, channel=None):
        """Get alpha and beta coefficient scalars at time t for specific channel"""
        if channel is None:
            # Return all channels
            alpha_t = self.alpha_base + self.alpha_spatial.mean(dim=(1,2)) * t
            beta_t = self.beta_base + self.beta_spatial.mean(dim=(1,2)) * t
        else:
            # Return specific channel
            alpha_t = self.alpha_base[channel] + self.alpha_spatial[channel].mean() * t
            beta_t = self.beta_base[channel] + self.beta_spatial[channel].mean() * t

        # Ensure positive coefficients for stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps, max=self.max_coeff)
        beta_t = torch.clamp(beta_t, min=self.stability_eps, max=self.max_coeff)

        return alpha_t, beta_t

    def forward(self, u):
        B, C, H, W = u.shape
        
        # Process each diffusion step
        current_time = 0.0
        
        for step in range(self.num_steps):
            u_new = torch.zeros_like(u)
            
            # Process each channel independently
            for c in range(C):
                alpha_t, beta_t = self.get_alpha_beta_at_time(current_time, channel=c)
                
                # Extract single channel
                u_c = u[:, c, :, :]
                
                if self.use_implicit:
                    # Use ADI (Alternating Direction Implicit) method
                    u_c = self.adi_diffusion_step(u_c, alpha_t, beta_t)
                else:
                    # Use explicit finite differences (with stability check)
                    u_c = self.explicit_diffusion_step(u_c, alpha_t, beta_t)
                
                u_new[:, c, :, :] = u_c
            
            # Apply cross-channel coupling
            u = self.apply_channel_coupling(u_new)
            current_time += self.dt

        return u

    def explicit_diffusion_step(self, u, alpha_coeff, beta_coeff):
        """Explicit finite difference diffusion step (no conv2d!)"""
        B, H, W = u.shape
        
        # Ensure stability for explicit scheme
        alpha_coeff = torch.clamp(alpha_coeff, max=0.25)
        beta_coeff = torch.clamp(beta_coeff, max=0.25)
        
        # Apply x-direction diffusion
        u = self.diffuse_x_explicit(u, alpha_coeff)
        
        # Apply y-direction diffusion
        u = self.diffuse_y_explicit(u, beta_coeff)
        
        return u
    
    def diffuse_x_explicit(self, u, coeff):
        """Explicit finite difference in x-direction"""
        B, H, W = u.shape
        u_new = u.clone()
        
        # Interior points: central difference
        if W > 2:
            u_new[:, :, 1:-1] = u[:, :, 1:-1] + coeff * self.dt * (
                u[:, :, 0:-2] - 2 * u[:, :, 1:-1] + u[:, :, 2:]
            ) / (self.dx ** 2)
        
        # Neumann boundary conditions (zero gradient)
        u_new[:, :, 0] = u[:, :, 0] + coeff * self.dt * (
            u[:, :, 1] - u[:, :, 0]
        ) / (self.dx ** 2)
        
        u_new[:, :, -1] = u[:, :, -1] + coeff * self.dt * (
            u[:, :, -2] - u[:, :, -1]
        ) / (self.dx ** 2)
        
        return u_new
    
    def diffuse_y_explicit(self, u, coeff):
        """Explicit finite difference in y-direction"""
        B, H, W = u.shape
        u_new = u.clone()
        
        # Interior points: central difference
        if H > 2:
            u_new[:, 1:-1, :] = u[:, 1:-1, :] + coeff * self.dt * (
                u[:, 0:-2, :] - 2 * u[:, 1:-1, :] + u[:, 2:, :]
            ) / (self.dx ** 2)
        
        # Neumann boundary conditions (zero gradient)
        u_new[:, 0, :] = u[:, 0, :] + coeff * self.dt * (
            u[:, 1, :] - u[:, 0, :]
        ) / (self.dx ** 2)
        
        u_new[:, -1, :] = u[:, -1, :] + coeff * self.dt * (
            u[:, -2, :] - u[:, -1, :]
        ) / (self.dx ** 2)
        
        return u_new
    
    def adi_diffusion_step(self, u, alpha_coeff, beta_coeff):
        """ADI (Alternating Direction Implicit) diffusion step"""
        # Step 1: Implicit in x-direction, explicit in y-direction
        u_half = self.solve_implicit_x(u, alpha_coeff, self.dt/2)
        
        # Step 2: Implicit in y-direction, explicit in x-direction
        u_new = self.solve_implicit_y(u_half, beta_coeff, self.dt/2)
        
        return u_new
    
    def solve_implicit_x(self, u, coeff, dt):
        """Solve implicit diffusion in x-direction using tridiagonal solver"""
        B, H, W = u.shape
        
        # Set up coefficients for implicit scheme
        r = coeff * dt / (self.dx ** 2)
        
        u_new = torch.zeros_like(u)
        
        for h in range(H):
            # Extract row
            row = u[:, h, :]  # Shape: (B, W)
            
            # Tridiagonal system: -r*u[i-1] + (1+2r)*u[i] - r*u[i+1] = u_old[i]
            a = torch.full((B, W), -r, device=u.device)  # sub-diagonal
            b = torch.full((B, W), 1 + 2*r, device=u.device)  # main diagonal
            c = torch.full((B, W), -r, device=u.device)  # super-diagonal
            
            # Boundary conditions (Neumann)
            b[:, 0] = 1 + r
            b[:, -1] = 1 + r
            
            # Solve tridiagonal system
            solution = self.thomas_solver_batch(a, b, c, row)
            u_new[:, h, :] = solution
        
        return u_new
    
    def solve_implicit_y(self, u, coeff, dt):
        """Solve implicit diffusion in y-direction using tridiagonal solver"""
        B, H, W = u.shape
        
        # Set up coefficients for implicit scheme
        r = coeff * dt / (self.dx ** 2)
        
        u_new = torch.zeros_like(u)
        
        for w in range(W):
            # Extract column
            col = u[:, :, w]  # Shape: (B, H)
            
            # Tridiagonal system
            a = torch.full((B, H), -r, device=u.device)  # sub-diagonal
            b = torch.full((B, H), 1 + 2*r, device=u.device)  # main diagonal
            c = torch.full((B, H), -r, device=u.device)  # super-diagonal
            
            # Boundary conditions (Neumann)
            b[:, 0] = 1 + r
            b[:, -1] = 1 + r
            
            # Solve tridiagonal system
            solution = self.thomas_solver_batch(a, b, c, col)
            u_new[:, :, w] = solution
        
        return u_new

    def apply_channel_coupling(self, u):
        """Apply learnable cross-channel coupling"""
        B, C, H, W = u.shape
        u_flat = u.view(B, C, -1)  # (B, C, H*W)
        
        # Apply channel coupling: (C, C) @ (B, C, H*W) -> (B, C, H*W)
        u_coupled = torch.einsum('cc,bci->bci', self.channel_coupling, u_flat)
        
        return u_coupled.view(B, C, H, W)

    def thomas_solver_batch(self, a, b, c, d):
        """
        Batch Thomas algorithm for solving tridiagonal systems
        Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
        """
        batch_size, n = d.shape
        device = d.device
        
        # Forward elimination
        c_prime = torch.zeros_like(c)
        d_prime = torch.zeros_like(d)
        
        # First row
        c_prime[:, 0] = c[:, 0] / (b[:, 0] + self.stability_eps)
        d_prime[:, 0] = d[:, 0] / (b[:, 0] + self.stability_eps)
        
        # Forward sweep
        for i in range(1, n):
            denom = b[:, i] - a[:, i] * c_prime[:, i-1]
            denom = torch.clamp(denom, min=self.stability_eps)
            
            if i < n-1:
                c_prime[:, i] = c[:, i] / denom
            d_prime[:, i] = (d[:, i] - a[:, i] * d_prime[:, i-1]) / denom
        
        # Back substitution
        x = torch.zeros_like(d)
        x[:, -1] = d_prime[:, -1]
        
        for i in range(n-2, -1, -1):
            x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i+1]
            
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
    
    for epoch in range(10):  # 10 epochs for CIFAR
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
                    alpha_spatial = model.diff.alpha_spatial[c]
                    print(f"{channel_name} Channel - α_base: {alpha_base.item():.3f}, "
                          f"α_spatial_std: {alpha_spatial.std().item():.4f}")
                
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
        alpha_coeffs = model.diff.alpha_base.detach().cpu().numpy()
        beta_coeffs = model.diff.beta_base.detach().cpu().numpy()
        channel_names = ['Red α-coeff', 'Green α-coeff', 'Blue α-coeff']
        
        # Display scalar coefficients as bar plot
        plt.subplot(5, 8, 25)
        plt.bar(['R', 'G', 'B'], alpha_coeffs, color=['red', 'green', 'blue'], alpha=0.7)
        plt.title('α Coefficients', fontsize=9)
        plt.ylabel('Value')
        
        plt.subplot(5, 8, 26)
        plt.bar(['R', 'G', 'B'], beta_coeffs, color=['red', 'green', 'blue'], alpha=0.7)
        plt.title('β Coefficients', fontsize=9)
        plt.ylabel('Value')
        
        # Show spatial modulation for one channel
        plt.subplot(5, 8, 27)
        spatial_mod = model.diff.alpha_spatial[0].detach().cpu().numpy()  # Red channel
        plt.imshow(spatial_mod, cmap='RdBu_r', interpolation='nearest')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Red α Spatial Mod', fontsize=9)
        plt.axis('off')
        
        # Cross-channel coupling matrix
        plt.subplot(5, 8, 28)
        coupling = model.diff.channel_coupling.detach().cpu().numpy()
        plt.imshow(coupling, cmap='RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('RGB Coupling', fontsize=9)
        plt.axis('off')
        
        # Additional spatial modulation visualization
        plt.subplot(5, 8, 29)
        spatial_mod_green = model.diff.beta_spatial[1].detach().cpu().numpy()  # Green channel
        plt.imshow(spatial_mod_green, cmap='RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Green β Spatial', fontsize=9)
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
            beta_base = model.diff.beta_base[c]
            alpha_spatial = model.diff.alpha_spatial[c]
            beta_spatial = model.diff.beta_spatial[c]
            print(f"{name} Channel:")
            print(f"  α_base: {alpha_base.item():.3f}")
            print(f"  β_base: {beta_base.item():.3f}")
            print(f"  α_spatial range: [{alpha_spatial.min().item():.4f}, {alpha_spatial.max().item():.4f}]")
            print(f"  β_spatial range: [{beta_spatial.min().item():.4f}, {beta_spatial.max().item():.4f}]")
        
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

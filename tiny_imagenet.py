# Improved PDE Diffusion Neural Network for TinyImageNet
# Addressing poor performance issues with better architecture and training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# --- Fixed and More Stable PDE Diffusion Layer ---
class ImprovedDiffusionLayer(nn.Module):
    def __init__(self, size=64, channels=3, dt=0.05, num_steps=2, use_implicit=True):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.num_steps = num_steps
        self.use_implicit = use_implicit

        # Simplified: single global coefficients per channel instead of matrices
        self.alpha_base = nn.Parameter(torch.ones(channels) * 0.1)  # Smaller initial values
        self.beta_base = nn.Parameter(torch.ones(channels) * 0.1)

        # Optional: learnable spatial modulation (but simpler)
        self.spatial_modulation = nn.Parameter(torch.zeros(channels, size, size) * 0.01)
        
        # Stability parameters
        self.stability_eps = 1e-6
        self.max_coeff = 0.2 if not use_implicit else 1.0

    def forward(self, u):
        B, C, H, W = u.shape
        
        # Apply diffusion to each channel
        for step in range(self.num_steps):
            for c in range(C):
                # Get effective diffusion coefficient (convert to scalar)
                alpha_eff = torch.clamp(
                    self.alpha_base[c] + self.spatial_modulation[c].mean() * 0.1, 
                    min=self.stability_eps, max=self.max_coeff
                ).item()  # Convert to scalar
                
                beta_eff = torch.clamp(
                    self.beta_base[c] + self.spatial_modulation[c].mean() * 0.1,
                    min=self.stability_eps, max=self.max_coeff
                ).item()  # Convert to scalar
                
                u_c = u[:, c, :, :]
                
                if self.use_implicit:
                    # Use implicit scheme for better stability
                    u_c = self.implicit_diffusion_step(u_c, alpha_eff, beta_eff)
                else:
                    # Use explicit scheme (faster but less stable)
                    u_c = self.explicit_diffusion_step(u_c, alpha_eff)
                
                u[:, c, :, :] = u_c
        
        return u

    def explicit_diffusion_step(self, u, coeff):
        """Proper PDE diffusion using finite differences"""
        B, H, W = u.shape
        
        # Ensure stability: CFL condition for explicit scheme
        max_stable_coeff = 0.2
        coeff = min(max(coeff, self.stability_eps), max_stable_coeff)
        
        # Apply finite difference diffusion in x-direction
        u = self.diffuse_x_explicit(u, coeff)
        
        # Apply finite difference diffusion in y-direction  
        u = self.diffuse_y_explicit(u, coeff)
        
        return u
    
    def implicit_diffusion_step(self, u, alpha_coeff, beta_coeff):
        """Implicit diffusion using ADI (Alternating Direction Implicit) method"""
        B, H, W = u.shape
        
        # ADI method: solve in x-direction implicitly, then y-direction implicitly
        
        # Step 1: Implicit solve in x-direction
        u_half = self.solve_implicit_x(u, alpha_coeff, self.dt/2)
        
        # Step 2: Implicit solve in y-direction  
        u_new = self.solve_implicit_y(u_half, beta_coeff, self.dt/2)
        
        return u_new
    
    def solve_implicit_x(self, u, coeff, dt):
        """Solve implicit diffusion in x-direction using tridiagonal solver"""
        B, H, W = u.shape
        
        # Coefficient for implicit scheme (now scalar)
        r = coeff * dt / (1.0 ** 2)  # dx = 1.0
        
        # Process each row independently
        u_new = torch.zeros_like(u)
        
        for h in range(H):
            # Extract row: (B, W)
            row = u[:, h, :]
            
            # Set up tridiagonal system: (1 + 2r)u_i - r*u_{i-1} - r*u_{i+1} = u_old_i
            # Use scalar values for torch.full()
            a = torch.full((B, W), -r, device=u.device, dtype=u.dtype)  # sub-diagonal
            b = torch.full((B, W), 1 + 2*r, device=u.device, dtype=u.dtype)  # main diagonal  
            c = torch.full((B, W), -r, device=u.device, dtype=u.dtype)  # super-diagonal
            
            # Boundary conditions (Neumann): modify first and last equations
            b[:, 0] = 1 + r  # No flux at left boundary
            b[:, -1] = 1 + r  # No flux at right boundary
            c[:, 0] = -r
            a[:, -1] = -r
            
            # Solve tridiagonal system
            solution = self.thomas_algorithm_batch(a, b, c, row)
            u_new[:, h, :] = solution
            
        return u_new
    
    def solve_implicit_y(self, u, coeff, dt):
        """Solve implicit diffusion in y-direction using tridiagonal solver"""
        B, H, W = u.shape
        
        # Coefficient for implicit scheme (now scalar)
        r = coeff * dt / (1.0 ** 2)  # dy = 1.0
        
        # Process each column independently
        u_new = torch.zeros_like(u)
        
        for w in range(W):
            # Extract column: (B, H)
            col = u[:, :, w]
            
            # Set up tridiagonal system using scalar values
            a = torch.full((B, H), -r, device=u.device, dtype=u.dtype)  # sub-diagonal
            b = torch.full((B, H), 1 + 2*r, device=u.device, dtype=u.dtype)  # main diagonal
            c = torch.full((B, H), -r, device=u.device, dtype=u.dtype)  # super-diagonal
            
            # Boundary conditions (Neumann)
            b[:, 0] = 1 + r  # No flux at top boundary
            b[:, -1] = 1 + r  # No flux at bottom boundary
            c[:, 0] = -r
            a[:, -1] = -r
            
            # Solve tridiagonal system
            solution = self.thomas_algorithm_batch(a, b, c, col)
            u_new[:, :, w] = solution
            
        return u_new
    
    def thomas_algorithm_batch(self, a, b, c, d):
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
        c_prime[:, 0] = c[:, 0] / b[:, 0]
        d_prime[:, 0] = d[:, 0] / b[:, 0]
        
        # Forward sweep
        for i in range(1, n):
            denom = b[:, i] - a[:, i] * c_prime[:, i-1]
            denom = torch.clamp(denom, min=self.stability_eps)  # Prevent division by zero
            
            if i < n-1:
                c_prime[:, i] = c[:, i] / denom
            d_prime[:, i] = (d[:, i] - a[:, i] * d_prime[:, i-1]) / denom
        
        # Back substitution
        x = torch.zeros_like(d)
        x[:, -1] = d_prime[:, -1]
        
        for i in range(n-2, -1, -1):
            x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i+1]
            
        return x
    
    def diffuse_x_explicit(self, u, coeff):
        """Explicit finite difference diffusion in x-direction"""
        B, H, W = u.shape
        u_new = u.clone()
        
        # Central differences with Neumann boundary conditions
        # Interior points
        if W > 2:
            u_new[:, :, 1:-1] = u[:, :, 1:-1] + coeff * self.dt * (
                u[:, :, 0:-2] - 2 * u[:, :, 1:-1] + u[:, :, 2:]
            )
        
        # Boundary conditions (Neumann: zero gradient)
        u_new[:, :, 0] = u[:, :, 0] + coeff * self.dt * (u[:, :, 1] - u[:, :, 0])
        u_new[:, :, -1] = u[:, :, -1] + coeff * self.dt * (u[:, :, -2] - u[:, :, -1])
        
        return u_new
    
    def diffuse_y_explicit(self, u, coeff):
        """Explicit finite difference diffusion in y-direction"""
        B, H, W = u.shape
        u_new = u.clone()
        
        # Central differences with Neumann boundary conditions
        # Interior points
        if H > 2:
            u_new[:, 1:-1, :] = u[:, 1:-1, :] + coeff * self.dt * (
                u[:, 0:-2, :] - 2 * u[:, 1:-1, :] + u[:, 2:, :]
            )
        
        # Boundary conditions (Neumann: zero gradient)
        u_new[:, 0, :] = u[:, 0, :] + coeff * self.dt * (u[:, 1, :] - u[:, 0, :])
        u_new[:, -1, :] = u[:, -1, :] + coeff * self.dt * (u[:, -2, :] - u[:, -1, :])
        
        return u_new


# --- Much Improved CNN Architecture ---
class ImprovedTinyImageNetClassifier(nn.Module):
    def __init__(self, num_classes=200, use_pde=True, dropout_rate=0.3):
        super().__init__()
        
        self.use_pde = use_pde
        if use_pde:
            self.diff = ImprovedDiffusionLayer(size=64, channels=3, num_steps=1, use_implicit=False)  # Use explicit for speed
        
        # Much better CNN architecture inspired by ResNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        """Create a layer with residual blocks"""
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Optional PDE preprocessing
        if self.use_pde:
            x = self.diff(x)
        
        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# --- Improved TinyImageNet Dataset with Real Data ---
class ImprovedTinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, download=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if download and not os.path.exists(os.path.join(root_dir, 'tiny-imagenet-200')):
            print("TinyImageNet not found. Creating synthetic dataset for testing...")
            self.create_synthetic_data()
        
        self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200')
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        self.load_data()
    
    def create_synthetic_data(self):
        """Create more realistic synthetic data for testing"""
        dummy_dir = os.path.join(self.root_dir, 'tiny-imagenet-200')
        os.makedirs(dummy_dir, exist_ok=True)
        
        # Create 200 classes with more realistic synthetic images
        for i in range(200):
            class_id = f"n{i:08d}"
            class_dir = os.path.join(dummy_dir, 'train', class_id, 'images')
            os.makedirs(class_dir, exist_ok=True)
            
            # Create more realistic synthetic images with class-specific patterns
            for j in range(50):  # Reduced for faster testing
                # Create images with class-specific color patterns
                base_color = np.array([i % 3, (i // 3) % 3, (i // 9) % 3]) * 85
                noise = np.random.randint(-30, 30, (64, 64, 3))
                img_array = np.clip(base_color.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
                
                # Add some geometric patterns
                center = (32, 32)
                y, x = np.ogrid[:64, :64]
                mask = (x - center[0])**2 + (y - center[1])**2 <= (10 + i % 15)**2
                img_array[mask] = [255 - base_color[0], 255 - base_color[1], 255 - base_color[2]]
                
                dummy_img = Image.fromarray(img_array)
                dummy_img.save(os.path.join(class_dir, f"{class_id}_{j}.JPEG"))
        
        # Create val structure with more samples
        val_dir = os.path.join(dummy_dir, 'val', 'images')
        os.makedirs(val_dir, exist_ok=True)
        
        # Create validation annotations
        with open(os.path.join(dummy_dir, 'val', 'val_annotations.txt'), 'w') as f:
            for i in range(1000):  # 5 per class for faster testing
                class_idx = i % 200
                class_id = f"n{class_idx:08d}"
                
                # Create validation image
                base_color = np.array([class_idx % 3, (class_idx // 3) % 3, (class_idx // 9) % 3]) * 85
                noise = np.random.randint(-30, 30, (64, 64, 3))
                img_array = np.clip(base_color.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
                
                dummy_img = Image.fromarray(img_array)
                dummy_img.save(os.path.join(val_dir, f"val_{i}.JPEG"))
                
                f.write(f"val_{i}.JPEG\t{class_id}\t0\t0\t64\t64\n")
    
    def load_data(self):
        """Load dataset paths and labels"""
        if self.split == 'train':
            train_dir = os.path.join(self.data_dir, 'train')
            if os.path.exists(train_dir):
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
            if os.path.exists(val_dir):
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
        return len(self.image_paths) if self.image_paths else (10000 if self.split == 'train' else 1000)
    
    def __getitem__(self, idx):
        if not self.image_paths:  # fallback
            # Create deterministic synthetic data
            class_idx = idx % 200
            base_color = np.array([class_idx % 3, (class_idx // 3) % 3, (class_idx // 9) % 3]) * 85
            noise = np.random.RandomState(idx).randint(-20, 20, (64, 64, 3))
            img_array = np.clip(base_color.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            label = class_idx
        else:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# --- Improved Training Function ---
def train_improved_tinyimagenet():
    """Improved training with better practices"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Better data augmentation
    transform_train = transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImprovedTinyImageNetDataset('./data', split='train', transform=transform_train)
    val_dataset = ImprovedTinyImageNetDataset('./data', split='val', transform=transform_val)
    
    # Create data loaders with better settings
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model with and without PDE for comparison
    print("Training model WITH PDE diffusion...")
    model_with_pde = ImprovedTinyImageNetClassifier(num_classes=200, use_pde=True).to(device)
    
    # Better optimizer and scheduler
    optimizer = torch.optim.AdamW(model_with_pde.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=10, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    best_acc = 0
    for epoch in range(10):  # Reduced epochs for testing
        model_with_pde.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model_with_pde(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_with_pde.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, LR: {current_lr:.6f}')
        
        # Validation
        model_with_pde.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                output = model_with_pde(imgs)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best validation accuracy: {best_acc:.2f}%")
        
        # Monitor PDE parameters
        if hasattr(model_with_pde, 'diff'):
            with torch.no_grad():
                alpha_base = model_with_pde.diff.alpha_base
                spatial_mod = model_with_pde.diff.spatial_modulation
                print(f"PDE params - α_base: {alpha_base.detach().cpu().numpy()}, "
                      f"spatial_mod_std: {spatial_mod.std().item():.4f}")
        
        print("-" * 80)
    
    print(f"\nBest PDE Model Accuracy: {best_acc:.2f}%")
    return model_with_pde, val_loader


if __name__ == "__main__":
    print("Starting improved TinyImageNet training...")
    model, val_loader = train_improved_tinyimagenet()

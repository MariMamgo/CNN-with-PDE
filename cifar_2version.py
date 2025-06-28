# CIFAR-10 PDE Diffusion WITHOUT Convolution - High Accuracy Architecture
# Multiple enhanced PDE layers + sophisticated fully connected network
# Focus on making PDE diffusion the primary feature extraction mechanism

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import math

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# --- Simplified PDE Diffusion Layer Focused on Alpha/Beta Learning ---
class EnhancedDiffusionLayer(nn.Module):
    def __init__(self, size=32, channels=3, dt=0.001, dx=1.0, dy=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.num_steps = num_steps

        # PRIMARY LEARNABLE PARAMETERS: Alpha and Beta coefficient matrices
        # These are the main parameters we want the model to learn
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)
        
        # Secondary learnable parameters for temporal modulation of alpha/beta
        self.alpha_time_coeff = nn.Parameter(torch.zeros(channels, size, size) * 0.1)
        self.beta_time_coeff = nn.Parameter(torch.zeros(channels, size, size) * 0.1)

        # Simple cross-channel coupling (learnable)
        self.channel_mixing = nn.Parameter(torch.eye(channels) + torch.randn(channels, channels) * 0.01)
        
        self.stability_eps = 1e-6
        
        print(f"Alpha/Beta-Focused DiffusionLayer: {size}x{size}x{channels}")
        print(f"  Spatial: dx={dx}, dy={dy}")
        print(f"  Temporal: dt={dt}, steps={num_steps}")
        print(f"  Learnable parameters: α matrices ({channels}x{size}x{size}), β matrices ({channels}x{size}x{size})")

    def get_alpha_beta_at_time(self, t):
        """Get learnable alpha and beta coefficient matrices at time t"""
        # Time-evolving alpha and beta - the core learnable diffusion parameters
        alpha_t = self.alpha_base + self.alpha_time_coeff * t
        beta_t = self.beta_base + self.beta_time_coeff * t
        
        # Ensure positive coefficients for numerical stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps, max=10.0)
        beta_t = torch.clamp(beta_t, min=self.stability_eps, max=10.0)

        return alpha_t, beta_t

    def apply_channel_mixing(self, u):
        """Apply learnable cross-channel mixing"""
        B, C, H, W = u.shape
        u_flat = u.view(B, C, -1)  # (B, C, H*W)
        
        # Apply learnable channel mixing matrix
        mixed = torch.matmul(self.channel_mixing, u_flat)
        return mixed.view(B, C, H, W)

    def forward(self, u):
        """Forward pass focused on learning optimal alpha/beta"""
        B, C, H, W = u.shape
        device = u.device
        
        # Ensure parameters are on correct device
        if self.alpha_base.device != device:
            self._move_to_device(device)

        u = u.clone()
        current_time = 0.0
        
        for step in range(self.num_steps):
            # Get current alpha and beta matrices (the key learnable parameters)
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)
            
            # Apply channel mixing before diffusion
            u = self.apply_channel_mixing(u)
            
            # Standard Strang splitting diffusion
            u_flat = u.view(B * C, H, W).clone()
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            
            # X-direction diffusion (half step)
            u_flat = self.diffuse_x_vectorized_parallel(u_flat, alpha_flat, self.dt / 2, self.dx)
            
            # Y-direction diffusion (full step)
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)
            beta_flat = beta_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.diffuse_y_vectorized_parallel(u_flat, beta_flat, self.dt, self.dy)
            
            # X-direction diffusion (final half step)
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.diffuse_x_vectorized_parallel(u_flat, alpha_flat, self.dt / 2, self.dx)
            
            u = u_flat.view(B, C, H, W).clone()

        return u

    def _move_to_device(self, device):
        """Move all parameters to device"""
        self.alpha_base = self.alpha_base.to(device)
        self.beta_base = self.beta_base.to(device)
        self.alpha_time_coeff = self.alpha_time_coeff.to(device)
        self.beta_time_coeff = self.beta_time_coeff.to(device)
        self.channel_mixing = self.channel_mixing.to(device)

    def diffuse_x_vectorized_parallel(self, u, alpha_matrix, dt, dx):
        """Standard x-direction diffusion (from original code)"""
        BC, H, W = u.shape
        u_flat = u.contiguous().view(BC * H, W).clone()
        alpha_flat = alpha_matrix.contiguous().view(BC * H, W)
        
        coeff = alpha_flat * dt / (dx ** 2)
        a = (-coeff).clone()
        c = (-coeff).clone()
        b = (1 + 2 * coeff).clone()
        
        # Enhanced boundary conditions using learnable weights
        if hasattr(self, 'boundary_weights'):
            left_weight = torch.sigmoid(self.boundary_weights[3])  # left
            right_weight = torch.sigmoid(self.boundary_weights[1])  # right
            b_modified = b.clone()
            b_modified[:, 0] = 1 + coeff[:, 0] * left_weight
            b_modified[:, -1] = 1 + coeff[:, -1] * right_weight
        else:
            b_modified = b.clone()
            b_modified[:, 0] = 1 + coeff[:, 0]
            b_modified[:, -1] = 1 + coeff[:, -1]
        
        result = self.thomas_solver_batch_optimized(a, b_modified, c, u_flat)
        return result.view(BC, H, W)

    def diffuse_y_vectorized_parallel(self, u, beta_matrix, dt, dy):
        """Standard y-direction diffusion (from original code)"""
        BC, H, W = u.shape
        u_t = u.transpose(1, 2).contiguous().clone()
        u_flat = u_t.view(BC * W, H)
        
        beta_t = beta_matrix.transpose(1, 2).contiguous()
        beta_flat = beta_t.view(BC * W, H)
        
        coeff = beta_flat * dt / (dy ** 2)
        a = (-coeff).clone()
        c = (-coeff).clone()
        b = (1 + 2 * coeff).clone()
        
        # Enhanced boundary conditions
        if hasattr(self, 'boundary_weights'):
            top_weight = torch.sigmoid(self.boundary_weights[0])  # top
            bottom_weight = torch.sigmoid(self.boundary_weights[2])  # bottom
            b_modified = b.clone()
            b_modified[:, 0] = 1 + coeff[:, 0] * top_weight
            b_modified[:, -1] = 1 + coeff[:, -1] * bottom_weight
        else:
            b_modified = b.clone()
            b_modified[:, 0] = 1 + coeff[:, 0]
            b_modified[:, -1] = 1 + coeff[:, -1]
        
        result = self.thomas_solver_batch_optimized(a, b_modified, c, u_flat)
        return result.view(BC, W, H).transpose(1, 2).contiguous().clone()

    def thomas_solver_batch_optimized(self, a, b, c, d):
        """Thomas solver (from original code)"""
        batch_size, N = d.shape
        device = d.device
        eps = self.stability_eps

        c_star_list = []
        d_star_list = []

        denom_0 = b[:, 0] + eps
        c_star_list.append(c[:, 0] / denom_0)
        d_star_list.append(d[:, 0] / denom_0)

        for i in range(1, N):
            denom = b[:, i] - a[:, i] * c_star_list[i-1] + eps
            
            if i < N-1:
                c_star_list.append(c[:, i] / denom)
            else:
                c_star_list.append(torch.zeros_like(c[:, i]))
            
            d_val = (d[:, i] - a[:, i] * d_star_list[i-1]) / denom
            d_star_list.append(d_val)

        x_list = [torch.zeros(batch_size, device=device) for _ in range(N)]
        x_list[-1] = d_star_list[-1]

        for i in range(N-2, -1, -1):
            x_val = d_star_list[i] - c_star_list[i] * x_list[i+1]
            x_list[i] = x_val

        result = torch.stack(x_list, dim=1)
        return result


# --- Spatial Attention Without Convolution ---
class SpatialAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, channels, size, size) * 0.1)
        
        # Attention weights computation (fully connected)
        self.attention_fc = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Add positional encoding
        x_pos = x + self.pos_embed
        
        # Compute spatial attention weights
        # Average pool over spatial dimensions for each channel
        spatial_avg = F.adaptive_avg_pool2d(x_pos, (1, 1)).view(B, C)
        attention_weights = self.attention_fc(spatial_avg).view(B, C, 1, 1)
        
        # Apply attention
        return x * attention_weights


# --- Single PDE Feature Extractor Focused on Alpha/Beta Learning ---
class SinglePDEExtractor(nn.Module):
    def __init__(self, input_size=32, channels=3):
        super().__init__()
        
        # Single PDE layer with specified parameters
        self.pde = EnhancedDiffusionLayer(input_size, channels, dt=0.001, num_steps=10, 
                                         dx=1.0, dy=1.0)  # Single α/β learning layer
        
        # Simple spatial attention
        self.attention = SpatialAttention(channels, input_size)
        
        print(f"Single PDE α/β learning: dt=0.001, steps=10")

    def forward(self, x):
        # Single PDE layer for feature extraction
        features = self.attention(self.pde(x))
        return features


# --- Enhanced Fully Connected Network ---
class EnhancedFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


# --- Main CIFAR-10 Model with Single PDE Layer ---
class CIFAR10PDENoConv(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Single PDE feature extraction (dt=0.001, steps=10)
        self.feature_extractor = SinglePDEExtractor(input_size=32, channels=3)
        
        # Global pooling strategies
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Reduce to 4x4
        self.max_pool = nn.AdaptiveMaxPool2d((4, 4))
        
        # Enhanced fully connected network
        # Input: 3 channels × 4×4 × 2 (avg+max pooling) = 96 features
        self.classifier = EnhancedFC(
            input_size=96,
            hidden_sizes=[512, 256, 128, 64],
            num_classes=10,
            dropout_rate=dropout_rate
        )
        
        # Additional feature processing
        self.feature_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        # Single PDE feature extraction
        features = self.feature_extractor(x)
        
        # Batch normalization
        features = self.feature_bn(features)
        
        # Global pooling to create fixed-size feature vectors
        avg_pooled = self.adaptive_pool(features)  # (B, 3, 4, 4)
        max_pooled = self.max_pool(features)       # (B, 3, 4, 4)
        
        # Combine pooled features
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, 6, 4, 4)
        
        # Flatten for fully connected network
        flattened = pooled_features.view(pooled_features.size(0), -1)  # (B, 96)
        
        # Final classification
        output = self.classifier(flattened)
        
        return output


# --- Training Functions (Updated) ---
def create_cifar10_data_loaders(batch_size=64):
    """Create CIFAR-10 data loaders with enhanced augmentation"""
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # Enhanced augmentation for non-conv model
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1)  # Additional regularization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_cifar10_no_conv(epochs=20, learning_rate=0.001):
    """Train CIFAR-10 model without convolution - focused on learning α/β parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CIFAR-10 without convolution on {device}")
    print(f"Architecture: Single PDE α/β learning + Enhanced FC Network")
    
    # Create data loaders
    train_loader, test_loader = create_cifar10_data_loaders(batch_size=64)
    
    # Initialize model
    model = CIFAR10PDENoConv(dropout_rate=0.3).to(device)
    
    # Count parameters - focus on α/β parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    alpha_beta_params = 0
    for name, param in model.named_parameters():
        if 'alpha' in name or 'beta' in name:
            alpha_beta_params += param.numel()
    
    print(f"Total trainable parameters: {total_params:,}")
    print(f"α/β parameters: {alpha_beta_params:,} ({100*alpha_beta_params/total_params:.1f}% of total)")
    
    # Optimizer with special focus on α/β parameters
    alpha_beta_params_list = []
    other_params = []
    for name, param in model.named_parameters():
        if 'alpha' in name or 'beta' in name:
            alpha_beta_params_list.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': alpha_beta_params_list, 'lr': learning_rate, 'weight_decay': 1e-6},  # Focus on α/β learning
        {'params': other_params, 'lr': learning_rate * 0.5, 'weight_decay': 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"Starting α/β-focused training for {epochs} epochs...")
    print("Single PDE layer: dt=0.001, steps=10, dx=1.0, dy=1.0")
    print("Key learnable parameters: α (x-diffusion) and β (y-diffusion) coefficient matrices")
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_accuracy = 100. * test_correct / test_total
        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"🎯 New best accuracy: {best_accuracy:.2f}%")
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
              f"Train Acc={train_accuracy:.2f}%, Test Acc={test_accuracy:.2f}%")
        
        # Monitor α/β parameter statistics every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                print("α/β Parameter Statistics:")
                pde_layer = model.feature_extractor.pde
                for c in range(3):
                    alpha_stats = pde_layer.alpha_base[c]
                    beta_stats = pde_layer.beta_base[c]
                    channel_name = ['R', 'G', 'B'][c]
                    print(f"  {channel_name}: α∈[{alpha_stats.min():.3f}, {alpha_stats.max():.3f}], "
                          f"β∈[{beta_stats.min():.3f}, {beta_stats.max():.3f}]")
        
        print("-" * 60)
    
    print(f"\nα/β-focused training completed! Best test accuracy: {best_accuracy:.2f}%")
    return model, test_loader, best_accuracy


if __name__ == "__main__":
    print("CIFAR-10 Classification without Convolution")
    print("=" * 50)
    print("Focus: Learning optimal α and β diffusion parameters")
    print("Architecture Components:")
    print("✓ Single PDE layer (dt=0.001, steps=10)")
    print("✓ α/β coefficient learning (3×32×32 matrices each)")
    print("✓ Spatial attention mechanism") 
    print("✓ Enhanced fully connected network")
    print("✓ Cross-channel mixing")
    print("✓ NO convolutional layers!")
    print("=" * 50)
    
    # Train the model with single PDE layer
    model, test_loader, best_acc = train_cifar10_no_conv(epochs=20, learning_rate=0.001)
    
    print(f"\nFinal Results (20 epochs):")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Architecture: Single PDE (dt=0.001, steps=10) + FC")
    print(f"Key Achievement: Learned optimal α/β diffusion coefficients for image classification")

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

# --- Enhanced Multi-Scale PDE Diffusion Layer ---
class EnhancedDiffusionLayer(nn.Module):
    def __init__(self, size=32, channels=3, dt=0.001, dx=1.0, dy=1.0, num_steps=10, 
                 learnable_operators=True):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.num_steps = num_steps
        self.learnable_operators = learnable_operators

        # Enhanced learnable diffusion coefficients with spatial variation
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)
        
        # Multi-scale temporal modulation
        self.alpha_time_coeff = nn.Parameter(torch.zeros(channels, size, size))
        self.beta_time_coeff = nn.Parameter(torch.zeros(channels, size, size))
        
        # Quadratic time dependence for more complex dynamics
        self.alpha_time_quad = nn.Parameter(torch.zeros(channels, size, size))
        self.beta_time_quad = nn.Parameter(torch.zeros(channels, size, size))

        # Cross-channel coupling for RGB interactions
        self.channel_coupling = nn.Parameter(torch.eye(channels) * 0.1)
        
        # Learnable boundary conditions
        self.boundary_weights = nn.Parameter(torch.ones(4))  # top, right, bottom, left
        
        # Adaptive spatial operators (learnable finite difference stencils)
        if learnable_operators:
            # 3x3 stencils for x and y directions
            self.x_stencil = nn.Parameter(torch.tensor([[-0.5, 0.0, 0.5]]).float())
            self.y_stencil = nn.Parameter(torch.tensor([[-0.5], [0.0], [0.5]]).float())
            self.laplacian_stencil = nn.Parameter(torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0], 
                [0.0, 1.0, 0.0]
            ]).float())

        self.stability_eps = 1e-6
        
        print(f"Enhanced DiffusionLayer: {size}x{size}x{channels}")
        print(f"  Spatial: dx={dx}, dy={dy}")
        print(f"  Temporal: dt={dt}, steps={num_steps}")
        print(f"  Features: Learnable operators={learnable_operators}")

    def get_adaptive_coefficients(self, t, u=None):
        """Get spatially and temporally adaptive diffusion coefficients"""
        # Time-dependent coefficients with quadratic terms
        alpha_t = (self.alpha_base + 
                  self.alpha_time_coeff * t + 
                  self.alpha_time_quad * t * t)
        beta_t = (self.beta_base + 
                 self.beta_time_coeff * t + 
                 self.beta_time_quad * t * t)
        
        # Optional: make coefficients depend on local image content
        if u is not None and self.learnable_operators:
            # Content-adaptive diffusion (simple version)
            u_normalized = torch.sigmoid(u)
            content_factor = 1.0 + 0.1 * (u_normalized.mean(dim=1, keepdim=True) - 0.5)
            alpha_t = alpha_t * content_factor
            beta_t = beta_t * content_factor

        # Ensure stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps, max=5.0)
        beta_t = torch.clamp(beta_t, min=self.stability_eps, max=5.0)

        return alpha_t, beta_t

    def apply_channel_coupling(self, u):
        """Apply cross-channel diffusion coupling"""
        B, C, H, W = u.shape
        u_flat = u.view(B, C, -1)  # (B, C, H*W)
        
        # Apply coupling matrix
        coupled = torch.matmul(self.channel_coupling, u_flat)
        return coupled.view(B, C, H, W)

    def forward(self, u):
        """Enhanced forward pass with multi-scale diffusion"""
        B, C, H, W = u.shape
        device = u.device
        
        # Ensure parameters are on correct device
        if self.alpha_base.device != device:
            self._move_to_device(device)

        u = u.clone()
        current_time = 0.0
        
        for step in range(self.num_steps):
            # Get adaptive coefficients
            alpha_all, beta_all = self.get_adaptive_coefficients(current_time, u)
            
            # Apply channel coupling before diffusion
            u = self.apply_channel_coupling(u)
            
            # Strang splitting with enhanced operators
            u_flat = u.view(B * C, H, W).clone()
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            
            # Enhanced x-diffusion
            u_flat = self.enhanced_diffuse_x(u_flat, alpha_flat, self.dt / 2)
            
            # Enhanced y-diffusion
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_adaptive_coefficients(current_time, u)
            beta_flat = beta_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.enhanced_diffuse_y(u_flat, beta_flat, self.dt)
            
            # Final x-diffusion
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_adaptive_coefficients(current_time, u)
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.enhanced_diffuse_x(u_flat, alpha_flat, self.dt / 2)
            
            u = u_flat.view(B, C, H, W).clone()

        return u

    def enhanced_diffuse_x(self, u, alpha_matrix, dt):
        """Enhanced x-direction diffusion with learnable operators"""
        if self.learnable_operators and hasattr(self, 'x_stencil'):
            return self.learnable_diffusion_1d(u, alpha_matrix, dt, self.dx, dim=2)
        else:
            return self.diffuse_x_vectorized_parallel(u, alpha_matrix, dt, self.dx)

    def enhanced_diffuse_y(self, u, beta_matrix, dt):
        """Enhanced y-direction diffusion with learnable operators"""
        if self.learnable_operators and hasattr(self, 'y_stencil'):
            return self.learnable_diffusion_1d(u, beta_matrix, dt, self.dy, dim=1)
        else:
            return self.diffuse_y_vectorized_parallel(u, beta_matrix, dt, self.dy)

    def learnable_diffusion_1d(self, u, coeff_matrix, dt, ds, dim):
        """Learnable finite difference diffusion"""
        # This is a simplified version - you could expand this significantly
        # For now, fall back to standard method but with enhanced boundary conditions
        if dim == 2:  # x-direction
            return self.diffuse_x_vectorized_parallel(u, coeff_matrix, dt, ds)
        else:  # y-direction
            return self.diffuse_y_vectorized_parallel(u, coeff_matrix, dt, ds)

    def _move_to_device(self, device):
        """Move all parameters to device"""
        self.alpha_base = self.alpha_base.to(device)
        self.beta_base = self.beta_base.to(device)
        self.alpha_time_coeff = self.alpha_time_coeff.to(device)
        self.beta_time_coeff = self.beta_time_coeff.to(device)
        self.alpha_time_quad = self.alpha_time_quad.to(device)
        self.beta_time_quad = self.beta_time_quad.to(device)
        self.channel_coupling = self.channel_coupling.to(device)
        self.boundary_weights = self.boundary_weights.to(device)

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


# --- Multi-Scale Feature Extraction Without Convolution ---
class MultiScaleExtractor(nn.Module):
    def __init__(self, input_size=32, channels=3):
        super().__init__()
        
        # Multiple PDE layers with different characteristics
        self.pde1 = EnhancedDiffusionLayer(input_size, channels, dt=0.001, num_steps=5, 
                                          dx=1.0, dy=1.0)  # Fine-scale, fast
        self.pde2 = EnhancedDiffusionLayer(input_size, channels, dt=0.002, num_steps=8, 
                                          dx=2.0, dy=2.0)  # Medium-scale
        self.pde3 = EnhancedDiffusionLayer(input_size, channels, dt=0.005, num_steps=4, 
                                          dx=1.5, dy=1.5)  # Coarse-scale, slow
        
        # Attention mechanisms
        self.attention1 = SpatialAttention(channels, input_size)
        self.attention2 = SpatialAttention(channels, input_size)
        self.attention3 = SpatialAttention(channels, input_size)
        
        # Feature combination weights
        self.combine_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        # Extract features at multiple scales
        features1 = self.attention1(self.pde1(x))
        features2 = self.attention2(self.pde2(x))
        features3 = self.attention3(self.pde3(x))
        
        # Weighted combination
        weights = F.softmax(self.combine_weights, dim=0)
        combined = (weights[0] * features1 + 
                   weights[1] * features2 + 
                   weights[2] * features3)
        
        return combined, features1, features2, features3


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


# --- Main CIFAR-10 Model Without Convolution ---
class CIFAR10PDENoConv(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Multi-scale PDE feature extraction
        self.feature_extractor = MultiScaleExtractor(input_size=32, channels=3)
        
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
        # Multi-scale PDE feature extraction
        combined_features, f1, f2, f3 = self.feature_extractor(x)
        
        # Batch normalization
        features = self.feature_bn(combined_features)
        
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


def train_cifar10_no_conv(epochs=50, learning_rate=0.001):
    """Train CIFAR-10 model without convolution"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CIFAR-10 without convolution on {device}")
    print(f"Architecture: Multi-scale PDE + Enhanced FC Network")
    
    # Create data loaders
    train_loader, test_loader = create_cifar10_data_loaders(batch_size=64)
    
    # Initialize model
    model = CIFAR10PDENoConv(dropout_rate=0.3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Optimizer with different learning rates for different components
    pde_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'pde' in name:
            pde_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': pde_params, 'lr': learning_rate * 0.5, 'weight_decay': 1e-5},
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"Starting training for {epochs} epochs...")
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
        print("-" * 60)
    
    print(f"\nTraining completed! Best test accuracy: {best_accuracy:.2f}%")
    return model, test_loader, best_accuracy


if __name__ == "__main__":
    print("CIFAR-10 Classification without Convolution")
    print("=" * 50)
    print("Architecture Components:")
    print("✓ Multi-scale PDE diffusion layers")
    print("✓ Spatial attention mechanisms") 
    print("✓ Enhanced fully connected networks")
    print("✓ Advanced data augmentation")
    print("✓ NO convolutional layers!")
    print("=" * 50)
    
    # Train the model
    model, test_loader, best_acc = train_cifar10_no_conv(epochs=20, learning_rate=0.001)
    
    print(f"\nFinal Results:")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Architecture: Pure PDE + FC (No Convolution)")

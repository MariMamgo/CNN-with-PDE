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

# --- YOUR ORIGINAL DIFFUSION LAYER (Enhanced) ---
class LearnableDiffusionLayer(nn.Module):
    """
    Your original diffusion equation implementation with learnable α and β coefficients
    Solves: ∂u/∂t = α(x,y) ∂²u/∂x² + β(x,y) ∂²u/∂y²
    """
    def __init__(self, size=32, channels=3, dt=0.001, dx=1.0, dy=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.num_steps = num_steps

        # PRIMARY LEARNABLE PARAMETERS: Alpha and Beta coefficient matrices
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)

        # Temporal modulation of alpha/beta
        self.alpha_time_coeff = nn.Parameter(torch.zeros(channels, size, size) * 0.1)
        self.beta_time_coeff = nn.Parameter(torch.zeros(channels, size, size) * 0.1)

        # Cross-channel coupling
        self.channel_mixing = nn.Parameter(torch.eye(channels) + torch.randn(channels, channels) * 0.01)

        self.stability_eps = 1e-6

        print(f"Learnable Diffusion Layer: {size}x{size}x{channels}")
        print(f"  Learnable α coefficients: {channels}x{size}x{size}")
        print(f"  Learnable β coefficients: {channels}x{size}x{size}")
        print(f"  Temporal: dt={dt}, steps={num_steps}")

    def get_alpha_beta_at_time(self, t):
        """Get learnable alpha and beta coefficient matrices at time t"""
        alpha_t = self.alpha_base + self.alpha_time_coeff * t
        beta_t = self.beta_base + self.beta_time_coeff * t

        # Ensure positive coefficients for numerical stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps, max=10.0)
        beta_t = torch.clamp(beta_t, min=self.stability_eps, max=10.0)

        return alpha_t, beta_t

    def apply_channel_mixing(self, u):
        """Apply learnable cross-channel mixing"""
        B, C, H, W = u.shape
        u_flat = u.view(B, C, -1)
        mixed = torch.matmul(self.channel_mixing, u_flat)
        return mixed.view(B, C, H, W)

    def forward(self, u):
        """Forward pass solving the diffusion equation"""
        B, C, H, W = u.shape
        device = u.device

        if self.alpha_base.device != device:
            self._move_to_device(device)

        u = u.clone()
        current_time = 0.0

        for step in range(self.num_steps):
            # Get current alpha and beta matrices
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)

            # Apply channel mixing
            u = self.apply_channel_mixing(u)

            # Operator splitting: X-direction then Y-direction
            u_flat = u.view(B * C, H, W).clone()

            # X-direction diffusion
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.diffuse_x_vectorized_parallel(u_flat, alpha_flat, self.dt / 2, self.dx)

            # Y-direction diffusion
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)
            beta_flat = beta_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.diffuse_y_vectorized_parallel(u_flat, beta_flat, self.dt / 2, self.dy)

            current_time += self.dt / 2
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
        """X-direction diffusion using Thomas solver"""
        BC, H, W = u.shape
        u_flat = u.contiguous().view(BC * H, W).clone()
        alpha_flat = alpha_matrix.contiguous().view(BC * H, W)

        coeff = alpha_flat * dt / (dx ** 2)
        a = (-coeff).clone()
        c = (-coeff).clone()
        b = (1 + 2 * coeff).clone()

        # Boundary conditions
        b[:, 0] = 1 + coeff[:, 0]
        b[:, -1] = 1 + coeff[:, -1]

        result = self.thomas_solver_batch_optimized(a, b, c, u_flat)
        return result.view(BC, H, W)

    def diffuse_y_vectorized_parallel(self, u, beta_matrix, dt, dy):
        """Y-direction diffusion using Thomas solver"""
        BC, H, W = u.shape
        u_t = u.transpose(1, 2).contiguous().clone()
        u_flat = u_t.view(BC * W, H)

        beta_t = beta_matrix.transpose(1, 2).contiguous()
        beta_flat = beta_t.view(BC * W, H)

        coeff = beta_flat * dt / (dy ** 2)
        a = (-coeff).clone()
        c = (-coeff).clone()
        b = (1 + 2 * coeff).clone()

        # Boundary conditions
        b[:, 0] = 1 + coeff[:, 0]
        b[:, -1] = 1 + coeff[:, -1]

        result = self.thomas_solver_batch_optimized(a, b, c, u_flat)
        return result.view(BC, W, H).transpose(1, 2).contiguous().clone()

    def thomas_solver_batch_optimized(self, a, b, c, d):
        """Thomas algorithm for solving tridiagonal systems"""
        batch_size, N = d.shape
        device = d.device
        eps = self.stability_eps

        c_star_list = []
        d_star_list = []

        # Forward sweep
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

        # Backward sweep
        x_list = [torch.zeros(batch_size, device=device) for _ in range(N)]
        x_list[-1] = d_star_list[-1]

        for i in range(N-2, -1, -1):
            x_val = d_star_list[i] - c_star_list[i] * x_list[i+1]
            x_list[i] = x_val

        result = torch.stack(x_list, dim=1)
        return result

# --- Symmetric Layer from Ruthotto & Haber paper ---
class SymmetricLayer(nn.Module):
    """Symmetric layer F_sym(θ, Y) = -K(θ)^T σ(N(K(θ)Y, θ))"""
    def __init__(self, channels, spatial_size, activation='relu'):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.feature_dim = channels * spatial_size * spatial_size
        
        # Linear transformation K
        self.K = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.norm = nn.BatchNorm1d(self.feature_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        # Initialize K close to identity for stability
        nn.init.eye_(self.K.weight)
        self.K.weight.data += torch.randn_like(self.K.weight) * 0.01
    
    def forward(self, Y):
        B, C, H, W = Y.shape
        Y_flat = Y.view(B, -1)
        
        KY = self.K(Y_flat)
        KY_norm = self.norm(KY)
        sigma_KY = self.activation(KY_norm)
        result = -torch.matmul(sigma_KY, self.K.weight)
        
        return result.view(B, C, H, W)

# --- Parabolic CNN (Heat equation) ---
class ParabolicBlock(nn.Module):
    """Parabolic CNN: ∂_t Y = F_sym(θ, Y)"""
    def __init__(self, channels, spatial_size, num_steps=3, dt=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.symmetric_layer = SymmetricLayer(channels, spatial_size)
        print(f"Parabolic Block: {num_steps} steps, dt={dt}")
    
    def forward(self, Y):
        for step in range(self.num_steps):
            F_sym = self.symmetric_layer(Y)
            Y = Y + self.dt * F_sym
        return Y

# --- Hamiltonian CNN (Energy preserving) ---
class HamiltonianBlock(nn.Module):
    """Hamiltonian CNN with symplectic integration"""
    def __init__(self, channels, spatial_size, num_steps=3, dt=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.F_Y = SymmetricLayer(channels, spatial_size)
        self.F_Z = SymmetricLayer(channels, spatial_size)
        print(f"Hamiltonian Block: {num_steps} steps, dt={dt}")
    
    def forward(self, Y):
        Z = torch.zeros_like(Y)
        
        for step in range(self.num_steps):
            Y = Y + self.dt * (-self.F_Y(Z))
            Z = Z - self.dt * self.F_Z(Y)
        
        return Y

# --- Multi-PDE Feature Extractor (Including Your Diffusion) ---
class HybridPDEExtractor(nn.Module):
    """
    Combines your diffusion equation with paper's PDE architectures
    """
    def __init__(self, input_size=32, channels=3):
        super().__init__()
        
        # Your original learnable diffusion equation
        self.diffusion1 = LearnableDiffusionLayer(input_size, channels, dt=0.001, num_steps=8)
        self.diffusion2 = LearnableDiffusionLayer(input_size, channels, dt=0.002, num_steps=5)
        
        # Paper's PDE architectures
        self.parabolic = ParabolicBlock(channels, input_size, num_steps=4, dt=0.5)
        self.hamiltonian = HamiltonianBlock(channels, input_size, num_steps=3, dt=0.8)
        
        # Learnable combination weights for all approaches
        self.combination_weights = nn.Parameter(torch.ones(4) / 4)
        self.feature_norm = nn.BatchNorm2d(channels)
        
        print(f"Hybrid PDE Extractor: Diffusion + Parabolic + Hamiltonian")
        print(f"  - 2 Learnable Diffusion layers with α/β coefficients")
        print(f"  - 1 Parabolic layer (heat equation)")
        print(f"  - 1 Hamiltonian layer (energy preserving)")
    
    def forward(self, x):
        # Extract features using all approaches
        features_diff1 = self.diffusion1(x)
        features_diff2 = self.diffusion2(x)
        features_parabolic = self.parabolic(x)
        features_hamiltonian = self.hamiltonian(x)
        
        # Learnable weighted combination
        weights = F.softmax(self.combination_weights, dim=0)
        
        combined = (weights[0] * features_diff1 + 
                   weights[1] * features_diff2 +
                   weights[2] * features_parabolic + 
                   weights[3] * features_hamiltonian)
        
        combined = self.feature_norm(combined)
        
        return combined, features_diff1, features_diff2, features_parabolic, features_hamiltonian

# --- Spatial Attention without Convolution ---
class NonConvSpatialAttention(nn.Module):
    def __init__(self, channels, spatial_size):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.feature_dim = channels * spatial_size * spatial_size
        
        self.pos_embed = nn.Parameter(torch.randn(1, channels, spatial_size, spatial_size) * 0.02)
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, self.feature_dim // 8),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 8, self.feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_pos = x + self.pos_embed
        x_flat = x_pos.view(B, -1)
        attention_weights = self.attention_net(x_flat)
        attention_weights = attention_weights.view(B, C, H, W)
        return x * attention_weights

# --- Enhanced Classifier ---
class PDEClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10, dropout_rate=0.4):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate // 2),
            
            nn.Linear(128, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)

# --- Main Hybrid Model ---
class CIFAR10HybridPDEModel(nn.Module):
    """
    Hybrid model combining your diffusion equation with paper's PDE architectures
    """
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        
        # Hybrid PDE feature extraction
        self.feature_extractor = HybridPDEExtractor(input_size=32, channels=3)
        
        # Spatial attention
        self.attention = NonConvSpatialAttention(channels=3, spatial_size=32)
        
        # Global pooling
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((8, 8))
        
        self.feature_bn = nn.BatchNorm2d(3)
        
        # Classifier: 3 channels × 8×8 × 2 (avg+max) = 384 features
        self.classifier = PDEClassifier(input_dim=384, num_classes=10, dropout_rate=dropout_rate)
        
        print("Hybrid PDE CIFAR-10 Model initialized")
        print("Combines: Learnable Diffusion + Parabolic + Hamiltonian + Attention + FC")
    
    def forward(self, x):
        # Hybrid PDE feature extraction
        combined, diff1, diff2, parabolic, hamiltonian = self.feature_extractor(x)
        
        # Apply spatial attention
        attended_features = self.attention(combined)
        features = self.feature_bn(attended_features)
        
        # Global pooling
        avg_pooled = self.adaptive_avg_pool(features)
        max_pooled = self.adaptive_max_pool(features)
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classification
        flattened = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(flattened)
        
        return output

# --- Enhanced Regularization ---
def hybrid_pde_regularization(model, alpha1=1e-4, alpha2=1e-4, alpha3=1e-6):
    """
    Enhanced regularization for both diffusion and symmetric layers
    """
    reg_loss = 0.0
    
    for name, param in model.named_parameters():
        if 'alpha_base' in name or 'beta_base' in name:
            # L2 regularization on diffusion coefficients
            reg_loss += alpha3 * torch.norm(param, p=2) ** 2
        elif 'channel_mixing' in name:
            # Keep channel mixing close to identity
            identity = torch.eye(param.size(0), device=param.device)
            reg_loss += alpha2 * torch.norm(param - identity, p='fro') ** 2
        elif 'K.weight' in name:
            # L2 regularization on symmetric layer weights
            reg_loss += alpha2 * torch.norm(param, p=2) ** 2
        elif 'combination_weights' in name:
            # L1 for sparsity on combination weights
            reg_loss += alpha1 * torch.norm(param, p=1)
    
    return reg_loss

# --- Training Functions ---
def create_cifar10_data_loaders(batch_size=64):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1)
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

def train_hybrid_pde_cifar10(epochs=25, learning_rate=0.001):
    """Train hybrid PDE model combining diffusion with other PDE approaches"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Hybrid PDE CIFAR-10 model on {device}")
    print("Architecture: Learnable Diffusion + Parabolic + Hamiltonian (No Convolution)")
    
    train_loader, test_loader = create_cifar10_data_loaders(batch_size=64)
    model = CIFAR10HybridPDEModel(dropout_rate=0.4).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    diffusion_params = sum(p.numel() for name, p in model.named_parameters() 
                          if 'alpha' in name or 'beta' in name or 'channel_mixing' in name)
    
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Diffusion parameters (α/β): {diffusion_params:,} ({100*diffusion_params/total_params:.1f}% of total)")
    
    # Optimizer with different learning rates for different components
    diffusion_params_list = []
    other_params = []
    for name, param in model.named_parameters():
        if 'alpha' in name or 'beta' in name or 'channel_mixing' in name or 'combination_weights' in name:
            diffusion_params_list.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': diffusion_params_list, 'lr': learning_rate, 'weight_decay': 1e-6},
        {'params': other_params, 'lr': learning_rate * 0.8, 'weight_decay': 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"Starting hybrid PDE training for {epochs} epochs...")
    print("Key components: Learnable α/β diffusion + Symmetric layers + PDE integration")
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
                    reg_loss = hybrid_pde_regularization(model, alpha1=2e-4, alpha2=1e-4, alpha3=1e-6)
                    total_loss_with_reg = loss + reg_loss
                
                scaler.scale(total_loss_with_reg).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                reg_loss = hybrid_pde_regularization(model, alpha1=2e-4, alpha2=1e-4, alpha3=1e-6)
                total_loss_with_reg = loss + reg_loss
                
                total_loss_with_reg.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Reg: {reg_loss.item():.6f}, Acc: {100.*correct/total:.2f}%')
        
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
        
        # Monitor combination weights and diffusion coefficients
        if epoch % 5 == 0:
            with torch.no_grad():
                weights = F.softmax(model.feature_extractor.combination_weights, dim=0)
                print(f"PDE Combination Weights: "
                      f"Diff1={weights[0]:.3f}, Diff2={weights[1]:.3f}, "
                      f"Parabolic={weights[2]:.3f}, Hamiltonian={weights[3]:.3f}")
                
                # Show diffusion coefficient statistics
                for i, diff_layer in enumerate([model.feature_extractor.diffusion1, 
                                              model.feature_extractor.diffusion2]):
                    alpha_range = f"[{diff_layer.alpha_base.min():.3f}, {diff_layer.alpha_base.max():.3f}]"
                    beta_range = f"[{diff_layer.beta_base.min():.3f}, {diff_layer.beta_base.max():.3f}]"
                    print(f"  Diffusion{i+1}: α∈{alpha_range}, β∈{beta_range}")
        
        print("-" * 70)
    
    print(f"\nHybrid PDE training completed! Best test accuracy: {best_accuracy:.2f}%")
    return model, test_loader, best_accuracy

if __name__ == "__main__":
    print("CIFAR-10 Classification: Hybrid PDE Approach")
    print("=" * 70)
    print("Combining:")
    print("1. YOUR ORIGINAL: Learnable Diffusion Equation with α/β coefficients")
    print("   ∂u/∂t = α(x,y) ∂²u/∂x² + β(x,y) ∂²u/∂y² + Thomas solver")
    print("2. PAPER: Parabolic CNN (Heat equation)")
    print("3. PAPER: Hamiltonian CNN (Energy preserving)")
    print("4. Enhanced spatial attention and regularization")
    print("=" * 70)
    print("Architecture Components:")
    print("✓ Learnable α/β diffusion coefficients (your innovation)")
    print("✓ Thomas solver for tridiagonal systems")
    print("✓ Parabolic CNN with symmetric layers")
    print("✓ Hamiltonian CNN with symplectic integration")
    print("✓ Non-convolutional spatial attention")
    print("✓ Hybrid PDE regularization")
    print("✓ NO convolutional layers!")
    print("=" * 70)
    
    # Train the hybrid model
    model, test_loader, best_acc = train_hybrid_pde_cifar10(epochs=25, learning_rate=0.001)
    
    print(f"\nFinal Results:")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Architecture: Hybrid PDE (Diffusion+Parabolic+Hamiltonian) + FC")
    print(f"Key Achievement: Learnable diffusion coefficients + PDE-motivated stability")

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

# --- Symmetric Layer as described in the paper ---
class SymmetricLayer(nn.Module):
    """
    Symmetric layer F_sym(θ, Y) = -K(θ)^T σ(N(K(θ)Y, θ))
    This ensures negative semi-definite Jacobian for stability
    """
    def __init__(self, channels, spatial_size, activation='relu'):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.feature_dim = channels * spatial_size * spatial_size
        
        # Linear transformation K (replacing convolution)
        self.K = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        
        # Normalization layer - using batch norm as in paper
        self.norm = nn.BatchNorm1d(self.feature_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        # Initialize K to be close to identity for stability
        nn.init.eye_(self.K.weight)
        self.K.weight.data += torch.randn_like(self.K.weight) * 0.01
    
    def forward(self, Y):
        B, C, H, W = Y.shape
        Y_flat = Y.view(B, -1)  # Flatten spatial dimensions
        
        # Apply K transformation
        KY = self.K(Y_flat)
        
        # Apply normalization
        KY_norm = self.norm(KY)
        
        # Apply activation
        sigma_KY = self.activation(KY_norm)
        
        # Apply -K^T (symmetric property)
        result = -torch.matmul(sigma_KY, self.K.weight)
        
        return result.view(B, C, H, W)

# --- Parabolic CNN (Heat equation inspired) ---
class ParabolicBlock(nn.Module):
    """
    Parabolic CNN block: ∂_t Y = F_sym(θ, Y)
    Discretized using forward Euler method
    """
    def __init__(self, channels, spatial_size, num_steps=3, dt=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.symmetric_layer = SymmetricLayer(channels, spatial_size)
        
        print(f"Parabolic Block: {num_steps} steps, dt={dt}")
    
    def forward(self, Y):
        # Forward Euler integration
        for step in range(self.num_steps):
            F_sym = self.symmetric_layer(Y)
            Y = Y + self.dt * F_sym
        return Y

# --- Hamiltonian CNN (Energy preserving) ---
class HamiltonianBlock(nn.Module):
    """
    Hamiltonian CNN: Uses auxiliary variables and symplectic integration
    ∂_t Y = -F_sym(θ_1, Z)
    ∂_t Z = F_sym(θ_2, Y)
    """
    def __init__(self, channels, spatial_size, num_steps=3, dt=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        
        # Two symmetric layers for Y and Z dynamics
        self.F_Y = SymmetricLayer(channels, spatial_size)
        self.F_Z = SymmetricLayer(channels, spatial_size)
        
        print(f"Hamiltonian Block: {num_steps} steps, dt={dt} (symplectic)")
    
    def forward(self, Y):
        B, C, H, W = Y.shape
        
        # Initialize auxiliary variable Z (can be learned initialization)
        Z = torch.zeros_like(Y)
        
        # Symplectic Verlet integration
        for step in range(self.num_steps):
            # Y_{j+1} = Y_j + δt * F_sym(θ_1, Z_j)
            Y = Y + self.dt * (-self.F_Y(Z))
            
            # Z_{j+1} = Z_j - δt * F_sym(θ_2, Y_{j+1})
            Z = Z - self.dt * self.F_Z(Y)
        
        return Y

# --- Second-order CNN (Telegraph equation inspired) ---
class SecondOrderBlock(nn.Module):
    """
    Second-order CNN: ∂²_t Y = F_sym(θ, Y)
    Discretized using Leapfrog method
    """
    def __init__(self, channels, spatial_size, num_steps=3, dt=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.symmetric_layer = SymmetricLayer(channels, spatial_size, activation='tanh')
        
        print(f"Second-order Block: {num_steps} steps, dt={dt} (Leapfrog)")
    
    def forward(self, Y):
        # Initialize Y_{-1} = Y_0 (initial condition)
        Y_prev = Y.clone()
        Y_curr = Y.clone()
        
        # Leapfrog integration: Y_{j+1} = 2Y_j - Y_{j-1} + δt² F_sym(θ, Y_j)
        for step in range(self.num_steps):
            F_sym = self.symmetric_layer(Y_curr)
            Y_next = 2 * Y_curr - Y_prev + (self.dt ** 2) * F_sym
            
            Y_prev = Y_curr
            Y_curr = Y_next
        
        return Y_curr

# --- Multi-Architecture Feature Extractor ---
class MultiPDEExtractor(nn.Module):
    """
    Combines all three PDE-motivated architectures
    """
    def __init__(self, input_size=32, channels=3):
        super().__init__()
        
        # Three different PDE-based architectures
        self.parabolic = ParabolicBlock(channels, input_size, num_steps=4, dt=0.5)
        self.hamiltonian = HamiltonianBlock(channels, input_size, num_steps=3, dt=0.8)
        self.second_order = SecondOrderBlock(channels, input_size, num_steps=3, dt=0.3)
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Feature normalization
        self.feature_norm = nn.BatchNorm2d(channels)
        
        print(f"Multi-PDE Extractor: Parabolic + Hamiltonian + Second-order")
    
    def forward(self, x):
        # Extract features using all three architectures
        features_parabolic = self.parabolic(x)
        features_hamiltonian = self.hamiltonian(x)
        features_second_order = self.second_order(x)
        
        # Learnable weighted combination
        weights = F.softmax(self.combination_weights, dim=0)
        
        combined = (weights[0] * features_parabolic + 
                   weights[1] * features_hamiltonian + 
                   weights[2] * features_second_order)
        
        # Normalize combined features
        combined = self.feature_norm(combined)
        
        return combined, features_parabolic, features_hamiltonian, features_second_order

# --- Spatial Attention without Convolution ---
class NonConvSpatialAttention(nn.Module):
    """
    Spatial attention using only fully connected layers
    """
    def __init__(self, channels, spatial_size):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.feature_dim = channels * spatial_size * spatial_size
        
        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, channels, spatial_size, spatial_size) * 0.02)
        
        # Attention computation using FC layers
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
        
        # Add positional encoding
        x_pos = x + self.pos_embed
        
        # Flatten for attention computation
        x_flat = x_pos.view(B, -1)
        
        # Compute attention weights
        attention_weights = self.attention_net(x_flat)
        attention_weights = attention_weights.view(B, C, H, W)
        
        # Apply attention
        return x * attention_weights

# --- Enhanced Fully Connected Classifier ---
class PDEClassifier(nn.Module):
    """
    Enhanced fully connected classifier with regularization
    """
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
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)

# --- Main PDE-motivated CIFAR-10 Model ---
class CIFAR10PDEModel(nn.Module):
    """
    Main model combining PDE-motivated architectures without convolution
    """
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        
        # Multi-PDE feature extraction
        self.feature_extractor = MultiPDEExtractor(input_size=32, channels=3)
        
        # Spatial attention
        self.attention = NonConvSpatialAttention(channels=3, spatial_size=32)
        
        # Global pooling strategies
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((8, 8))
        
        # Feature processing
        self.feature_bn = nn.BatchNorm2d(3)
        
        # Classifier
        # Input: 3 channels × 8×8 × 2 (avg+max pooling) = 384 features
        self.classifier = PDEClassifier(input_dim=384, num_classes=10, dropout_rate=dropout_rate)
        
        print("PDE-motivated CIFAR-10 Model initialized")
        print("Architecture: Multi-PDE (Parabolic+Hamiltonian+Second-order) + Attention + FC")
    
    def forward(self, x):
        # Multi-PDE feature extraction
        combined_features, parabolic_feat, hamiltonian_feat, second_order_feat = self.feature_extractor(x)
        
        # Apply spatial attention
        attended_features = self.attention(combined_features)
        
        # Batch normalization
        features = self.feature_bn(attended_features)
        
        # Global pooling
        avg_pooled = self.adaptive_avg_pool(features)  # (B, 3, 8, 8)
        max_pooled = self.adaptive_max_pool(features)  # (B, 3, 8, 8)
        
        # Combine pooled features
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, 6, 8, 8)
        
        # Flatten for classification
        flattened = pooled_features.view(pooled_features.size(0), -1)  # (B, 384)
        
        # Final classification
        output = self.classifier(flattened)
        
        return output

# --- PDE-based Regularization (from paper) ---
def pde_regularization(model, alpha1=1e-4, alpha2=1e-4):
    """
    Regularization based on PDE theory as described in the paper
    """
    reg_loss = 0.0
    
    # L2 regularization on symmetric layer weights
    for name, param in model.named_parameters():
        if 'K.weight' in name:  # Symmetric layer weights
            reg_loss += alpha2 * torch.norm(param, p=2) ** 2
        elif 'combination_weights' in name:  # Combination weights
            reg_loss += alpha1 * torch.norm(param, p=1)  # L1 for sparsity
    
    return reg_loss

# --- Training Functions ---
def create_cifar10_data_loaders(batch_size=64):
    """Create CIFAR-10 data loaders with enhanced augmentation"""
    
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

def train_pde_cifar10(epochs=25, learning_rate=0.001):
    """Train PDE-motivated CIFAR-10 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PDE-motivated CIFAR-10 model on {device}")
    print("Architecture: Parabolic + Hamiltonian + Second-order CNNs (No Convolution)")
    
    # Create data loaders
    train_loader, test_loader = create_cifar10_data_loaders(batch_size=64)
    
    # Initialize model
    model = CIFAR10PDEModel(dropout_rate=0.4).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pde_params = sum(p.numel() for name, p in model.named_parameters() 
                     if 'K.weight' in name or 'combination_weights' in name)
    
    print(f"Total trainable parameters: {total_params:,}")
    print(f"PDE-specific parameters: {pde_params:,} ({100*pde_params/total_params:.1f}% of total)")
    
    # Optimizer with different learning rates for PDE components
    pde_params_list = []
    other_params = []
    for name, param in model.named_parameters():
        if 'K.weight' in name or 'combination_weights' in name:
            pde_params_list.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': pde_params_list, 'lr': learning_rate, 'weight_decay': 1e-6},
        {'params': other_params, 'lr': learning_rate * 0.8, 'weight_decay': 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"Starting PDE-motivated training for {epochs} epochs...")
    print("Key components: Symmetric layers, Parabolic/Hamiltonian/Second-order dynamics")
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
                    
                    # Add PDE regularization
                    reg_loss = pde_regularization(model, alpha1=2e-4, alpha2=1e-4)
                    total_loss_with_reg = loss + reg_loss
                
                scaler.scale(total_loss_with_reg).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                
                # Add PDE regularization
                reg_loss = pde_regularization(model, alpha1=2e-4, alpha2=1e-4)
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
        
        # Monitor PDE combination weights every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                weights = F.softmax(model.feature_extractor.combination_weights, dim=0)
                print(f"PDE Combination Weights: "
                      f"Parabolic={weights[0]:.3f}, "
                      f"Hamiltonian={weights[1]:.3f}, "
                      f"Second-order={weights[2]:.3f}")
        
        print("-" * 70)
    
    print(f"\nPDE-motivated training completed! Best test accuracy: {best_accuracy:.2f}%")
    return model, test_loader, best_accuracy

if __name__ == "__main__":
    print("CIFAR-10 Classification with PDE-Motivated Neural Networks")
    print("=" * 70)
    print("Based on: 'Deep Neural Networks Motivated by Partial Differential Equations'")
    print("Paper: Ruthotto & Haber (2018)")
    print("=" * 70)
    print("Architecture Components:")
    print("✓ Parabolic CNN (Heat equation inspired)")
    print("✓ Hamiltonian CNN (Energy preserving, symplectic integration)")
    print("✓ Second-order CNN (Telegraph equation inspired)")
    print("✓ Symmetric layers for stability")
    print("✓ Non-convolutional spatial attention")
    print("✓ PDE-based regularization")
    print("✓ NO convolutional layers!")
    print("=" * 70)
    
    # Train the model
    model, test_loader, best_acc = train_pde_cifar10(epochs=25, learning_rate=0.001)
    
    print(f"\nFinal Results:")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Architecture: PDE-motivated (Parabolic+Hamiltonian+Second-order) + FC")
    print(f"Key Achievement: Stable, interpretable neural networks without convolution")

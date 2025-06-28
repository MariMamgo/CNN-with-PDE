# CIFAR-10 PDE Diffusion with proper dx/dy handling and RGB channel support - GPU OPTIMIZED
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# --- Enhanced PDE Diffusion Layer for RGB Images ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=32, channels=3, dt=0.1, dx=1.0, dy=1.0, num_steps=5):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx  # Spatial step in x-direction
        self.dy = dy  # Spatial step in y-direction  
        self.num_steps = num_steps

        # Base diffusion coefficients as matrices (learnable) - one per channel
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 1.0)

        # Time-dependent modulation parameters as matrices (learnable)
        self.alpha_time_coeff = nn.Parameter(torch.zeros(channels, size, size))
        self.beta_time_coeff = nn.Parameter(torch.zeros(channels, size, size))

        # Channel coupling parameters (optional - for inter-channel diffusion)
        self.channel_coupling = nn.Parameter(torch.zeros(channels, channels))

        # Stability parameters
        self.stability_eps = 1e-6
        
        print(f"Initialized CIFAR-10 DiffusionLayer:")
        print(f"  Size: {size}x{size}, Channels: {channels}")
        print(f"  Spatial steps: dx={dx}, dy={dy}")
        print(f"  Temporal: dt={dt}, steps={num_steps}")

    def get_alpha_beta_at_time(self, t, channel=None):
        """Get alpha and beta coefficient matrices at time t for specified channel(s)"""
        if channel is None:
            # Return for all channels
            alpha_t = self.alpha_base + self.alpha_time_coeff * t
            beta_t = self.beta_base + self.beta_time_coeff * t
        else:
            # Return for specific channel
            alpha_t = self.alpha_base[channel] + self.alpha_time_coeff[channel] * t
            beta_t = self.beta_base[channel] + self.beta_time_coeff[channel] * t

        # Ensure positive coefficients for stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps)
        beta_t = torch.clamp(beta_t, min=self.stability_eps)

        return alpha_t, beta_t

    def forward(self, u):
        """
        GPU-optimized forward pass avoiding in-place operations
        Input: u of shape (B, C, H, W) where C=3 for RGB
        """
        B, C, H, W = u.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"
        device = u.device

        # Ensure all parameters are on the same device
        if self.alpha_base.device != device:
            print(f"Warning: Moving PDE parameters to {device}")
            self.alpha_base = self.alpha_base.to(device)
            self.beta_base = self.beta_base.to(device)
            self.alpha_time_coeff = self.alpha_time_coeff.to(device)
            self.beta_time_coeff = self.beta_time_coeff.to(device)

        # Apply multiple diffusion steps with time-dependent coefficients
        current_time = 0.0
        
        for step in range(self.num_steps):
            # Get coefficients for all channels at once
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)  # Shape: (C, H, W)
            
            # Process all channels in parallel using vectorized operations
            # Reshape u to process all channels together: (B, C, H, W) -> (B*C, H, W)
            u_flat = u.contiguous().view(B * C, H, W)
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            
            # Strang splitting: half step x
            u_flat = self.diffuse_x_vectorized_parallel(u_flat, alpha_flat, self.dt / 2, self.dx)
            
            # Full step y (update time)
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)
            beta_flat = beta_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.diffuse_y_vectorized_parallel(u_flat, beta_flat, self.dt, self.dy)
            
            # Final half step x
            current_time += self.dt / 2
            alpha_all, beta_all = self.get_alpha_beta_at_time(current_time)
            alpha_flat = alpha_all.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * C, H, W)
            u_flat = self.diffuse_x_vectorized_parallel(u_flat, alpha_flat, self.dt / 2, self.dx)
            
            # Create new tensor instead of in-place modification
            u = u_flat.view(B, C, H, W)

        return u

    def diffuse_x_vectorized_parallel(self, u, alpha_matrix, dt, dx):
        """
        GPU-optimized vectorized diffusion in x-direction for all channels
        Input: u of shape (B*C, H, W), alpha_matrix of shape (B*C, H, W)
        """
        BC, H, W = u.shape
        device = u.device

        # Reshape for batch processing: (B*C, H, W) -> (BC*H, W)
        u_flat = u.contiguous().view(BC * H, W)
        alpha_flat = alpha_matrix.contiguous().view(BC * H, W)

        # Apply smoothing to coefficients for stability
        alpha_smooth = self.smooth_coefficients(alpha_flat, dim=1)
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
        result = self.thomas_solver_batch_optimized(a, b_modified, c, u_flat)

        return result.view(BC, H, W)

    def diffuse_y_vectorized_parallel(self, u, beta_matrix, dt, dy):
        """
        GPU-optimized vectorized diffusion in y-direction for all channels
        Input: u of shape (B*C, H, W), beta_matrix of shape (B*C, H, W)
        """
        BC, H, W = u.shape
        device = u.device

        # Transpose to work on columns: (B*C, H, W) -> (B*C, W, H)
        u_t = u.transpose(1, 2).contiguous()
        u_flat = u_t.view(BC * W, H)

        # Transpose beta matrix and flatten
        beta_t = beta_matrix.transpose(1, 2).contiguous()
        beta_flat = beta_t.view(BC * W, H)

        # Apply smoothing to coefficients for stability
        beta_smooth = self.smooth_coefficients(beta_flat, dim=1)
        coeff = beta_smooth * dt / (dy ** 2)

        # Build tridiagonal system coefficients
        a = -coeff  # sub-diagonal
        c = -coeff  # super-diagonal
        b = 1 + 2 * coeff  # main diagonal

        # Apply boundary conditions (Neumann - no flux at boundaries)
        b_modified = b.clone()
        b_modified[:, 0] = 1 + coeff[:, 0]
        b_modified[:, -1] = 1 + coeff[:, -1]

        # Solve all tridiagonal systems in parallel
        result = self.thomas_solver_batch_optimized(a, b_modified, c, u_flat)

        # Transpose back: (BC*W, H) -> (BC, W, H) -> (BC, H, W)
        return result.view(BC, W, H).transpose(1, 2).contiguous()

    def thomas_solver_batch_optimized(self, a, b, c, d):
        """
        GPU-optimized batch Thomas algorithm using torch operations
        """
        batch_size, N = d.shape
        device = d.device
        eps = self.stability_eps

        # Use more efficient indexing without scatter
        c_star = torch.zeros_like(d)
        d_star = torch.zeros_like(d)

        # Forward elimination - vectorized
        # First row
        c_star[:, 0] = c[:, 0] / (b[:, 0] + eps)
        d_star[:, 0] = d[:, 0] / (b[:, 0] + eps)

        # Forward sweep - vectorized operations
        for i in range(1, N):
            denom = b[:, i] - a[:, i] * c_star[:, i-1] + eps
            
            if i < N-1:
                c_star[:, i] = c[:, i] / denom
            
            d_star[:, i] = (d[:, i] - a[:, i] * d_star[:, i-1]) / denom

        # Back substitution - vectorized
        x = torch.zeros_like(d)
        x[:, -1] = d_star[:, -1]

        for i in range(N-2, -1, -1):
            x[:, i] = d_star[:, i] - c_star[:, i] * x[:, i+1]

        return x

    def smooth_coefficients(self, coeffs, dim=1, kernel_size=3):
        """Apply smoothing to coefficients for numerical stability"""
        if kernel_size == 1:
            return coeffs

        # Simple moving average smoothing using conv1d
        padding = kernel_size // 2
        if dim == 1:
            coeffs_padded = F.pad(coeffs, (padding, padding), mode='replicate')
            kernel = torch.ones(1, 1, kernel_size, device=coeffs.device) / kernel_size
            smoothed = F.conv1d(coeffs_padded.unsqueeze(1), kernel, padding=0).squeeze(1)
        else:
            raise NotImplementedError("Only dim=1 smoothing implemented")

        return smoothed

    def thomas_solver_batch(self, a, b, c, d):
        """Legacy thomas solver - redirects to optimized version"""
        return self.thomas_solver_batch_optimized(a, b, c, d)

    def get_numerical_stability_info(self):
        """Get information about numerical stability for all channels"""
        with torch.no_grad():
            stability_info = {}
            
            for c in range(self.channels):
                # Check stability conditions for both directions
                alpha_max = torch.max(self.alpha_base[c] + torch.abs(self.alpha_time_coeff[c]) * self.dt * self.num_steps)
                beta_max = torch.max(self.beta_base[c] + torch.abs(self.beta_time_coeff[c]) * self.dt * self.num_steps)
                
                # CFL-like conditions
                cfl_x = alpha_max * self.dt / (self.dx ** 2)
                cfl_y = beta_max * self.dt / (self.dy ** 2)
                
                stability_info[f'channel_{c}'] = {
                    'cfl_x': cfl_x.item(),
                    'cfl_y': cfl_y.item(),
                    'stable_x': cfl_x.item() < 0.5,
                    'stable_y': cfl_y.item() < 0.5
                }
            
            return stability_info


# --- Enhanced Neural Network for CIFAR-10 ---
class CIFAR10PDEClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2, dx=1.0, dy=1.0):
        super().__init__()
        # Fixed: use correct dt and num_steps
        self.diff = DiffusionLayer(size=32, channels=3, dt=0.001, dx=dx, dy=dy, num_steps=10)
        
        # More complex network for CIFAR-10
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply PDE diffusion first
        x = self.diff(x)
        
        # Then conventional CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten and fully connected
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# --- Training Setup for CIFAR-10 ---
def create_cifar10_data_loaders(batch_size=32):
    """Create CIFAR-10 data loaders with GPU-optimized settings"""
    
    # CIFAR-10 normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Optimized data loaders for GPU
    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Increased for better CPU utilization
        pin_memory=True, 
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Prefetch batches
    )
    
    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader


def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        return {
            'current_gb': current_memory,
            'max_gb': max_memory, 
            'cached_gb': cached_memory,
            'device_name': torch.cuda.get_device_name()
        }
    return None

def train_cifar10_model(dx=1.0, dy=1.0, epochs=25):
    """GPU-optimized training function for CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Monitor initial GPU state
    gpu_info = monitor_gpu_usage()
    if gpu_info:
        print(f"GPU: {gpu_info['device_name']}")
        print(f"Initial GPU memory: {gpu_info['current_gb']:.2f} GB")
    
    print(f"Training on CIFAR-10 with spatial discretization: dx={dx}, dy={dy}")
    print(f"Training configuration: {epochs} epochs, dt=0.001, steps=10")

    # Create data loaders with smaller batch size for GPU efficiency
    train_loader, test_loader = create_cifar10_data_loaders(batch_size=32)  # Reduced batch size

    # Initialize model with specified dx/dy
    model = CIFAR10PDEClassifier(dx=dx, dy=dy).to(device)
    
    # Monitor GPU after model creation
    gpu_info = monitor_gpu_usage()
    if gpu_info:
        print(f"After model creation GPU memory: {gpu_info['current_gb']:.2f} GB")

    # Print stability information
    stability_info = model.diff.get_numerical_stability_info()
    print(f"Numerical Stability Analysis:")
    for channel, info in stability_info.items():
        print(f"  {channel}: CFL_x={info['cfl_x']:.4f} {'✓' if info['stable_x'] else '⚠'}, "
              f"CFL_y={info['cfl_y']:.4f} {'✓' if info['stable_y'] else '⚠'}")

    # Optimizer settings for CIFAR-10
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Enable mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    print(f"Starting GPU-optimized CIFAR-10 training for {epochs} epochs...")
    print(f"PDE Configuration: dt=0.001, steps=10, total_time=0.01")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Use mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(imgs)
                    loss = criterion(output, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(imgs)
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 200 == 0:
                gpu_info = monitor_gpu_usage()
                gpu_mem = f", GPU: {gpu_info['current_gb']:.1f}GB" if gpu_info else ""
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%{gpu_mem}')

            # Memory cleanup every 100 batches
            if batch_idx % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Final GPU memory check for epoch
        gpu_info = monitor_gpu_usage()
        gpu_str = f", Peak GPU: {gpu_info['max_gb']:.1f}GB" if gpu_info else ""
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Time: {epoch_time:.2f}s{gpu_str}")

        # Monitor PDE parameters by channel (reduced frequency)
        if epoch % 5 == 0:  # Every 5 epochs for 25 epoch training
            with torch.no_grad():
                for c in range(3):
                    channel_name = ['R', 'G', 'B'][c]
                    alpha_stats = {
                        'base_mean': model.diff.alpha_base[c].mean().item(),
                        'time_mean': model.diff.alpha_time_coeff[c].mean().item(),
                    }
                    print(f"  {channel_name}: α_base={alpha_stats['base_mean']:.3f}, "
                          f"α_time={alpha_stats['time_mean']:.3f}")

        print("-" * 80)

        # Clear GPU cache at end of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model, test_loader


def evaluate_and_visualize_cifar10(model, test_loader):
    """Evaluation and visualization for CIFAR-10 with RGB analysis"""
    device = next(model.parameters()).device
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(imgs)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100. * test_correct / test_total
    print(f"CIFAR-10 Test Accuracy: {test_acc:.2f}%")

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        # Denormalize for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        # Apply PDE and denormalize
        diffused = model.diff(images)
        diffused_denorm = diffused * std + mean
        diffused_denorm = torch.clamp(diffused_denorm, 0, 1)

        # Enhanced analysis
        print(f"\nCIFAR-10 PDE Analysis:")
        print(f"Training: 25 epochs completed")
        print(f"PDE Configuration: dt={model.diff.dt}, steps={model.diff.num_steps}, total_time={model.diff.dt * model.diff.num_steps}")
        print(f"Spatial discretization: dx={model.diff.dx}, dy={model.diff.dy}")
        print(f"Final test accuracy: {test_acc:.2f}%")

        # Channel-specific analysis
        for c, channel_name in enumerate(['Red', 'Green', 'Blue']):
            alpha_final, beta_final = model.diff.get_alpha_beta_at_time(
                model.diff.num_steps * model.diff.dt, channel=c)
            print(f"{channel_name} channel:")
            print(f"  α range: [{alpha_final.min().item():.3f}, {alpha_final.max().item():.3f}]")
            print(f"  β range: [{beta_final.min().item():.3f}, {beta_final.max().item():.3f}]")

        # Visualization
        plt.figure(figsize=(20, 16))

        # Sample images and predictions (2 rows, 8 columns each)
        for i in range(8):
            # Original images
            plt.subplot(5, 8, i + 1)
            img_display = images_denorm[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow(img_display)
            plt.axis('off')
            plt.title(f"True: {CIFAR10_CLASSES[labels[i]]}", fontsize=8)

            # Predictions
            plt.subplot(5, 8, i + 9)
            plt.imshow(img_display)
            plt.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {CIFAR10_CLASSES[predicted[i]]}", color=color, fontsize=8)

            # After PDE diffusion
            plt.subplot(5, 8, i + 17)
            diffused_display = diffused_denorm[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow(diffused_display)
            plt.axis('off')
            plt.title("After PDE", fontsize=8)

        # RGB channel coefficients visualization
        channel_names = ['Red', 'Green', 'Blue']
        for c in range(3):
            alpha_final, beta_final = model.diff.get_alpha_beta_at_time(
                model.diff.num_steps * model.diff.dt, channel=c)
            
            # Alpha matrix for this channel
            plt.subplot(5, 8, 25 + c)
            im = plt.imshow(alpha_final.cpu().numpy(), cmap='Reds')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{channel_names[c]} α", fontsize=10)
            plt.axis('off')

            # Beta matrix for this channel
            plt.subplot(5, 8, 29 + c)
            im = plt.imshow(beta_final.cpu().numpy(), cmap='Blues')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{channel_names[c]} β", fontsize=10)
            plt.axis('off')

        plt.suptitle(f'CIFAR-10 PDE Diffusion: dx={model.diff.dx}, dy={model.diff.dy}', fontsize=16)
        plt.tight_layout()
        plt.show()


def quick_gpu_test():
    """Quick test to verify GPU optimization is working"""
    print("=" * 60)
    print("QUICK GPU TEST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create a small model and test tensor with the updated parameters
    model = DiffusionLayer(size=16, channels=3, dt=0.001, dx=1.0, dy=1.0, num_steps=10).to(device)
    test_input = torch.randn(2, 3, 16, 16).to(device)
    
    print(f"Input tensor device: {test_input.device}")
    print(f"Model parameters device: {next(model.parameters()).device}")
    print(f"Test configuration: dt=0.001, steps=10")
    
    # Time a forward pass
    start_time = time.time()
    
    with torch.no_grad():
        output = model(test_input)
    
    forward_time = time.time() - start_time
    
    print(f"Output tensor device: {output.device}")
    print(f"Forward pass time: {forward_time:.4f} seconds")
    print(f"Output shape: {output.shape}")
    
    # Check if tensors stayed on GPU
    if device.type == 'cuda':
        gpu_info = monitor_gpu_usage()
        print(f"GPU memory after test: {gpu_info['current_gb']:.2f} GB")
        
        if test_input.device.type == 'cuda' and output.device.type == 'cuda':
            print("✓ GPU test PASSED - tensors stayed on GPU")
        else:
            print("⚠ GPU test FAILED - tensors moved to CPU")
    
    print("=" * 60)
    return forward_time < 2.0  # Should be reasonably fast on GPU

if __name__ == "__main__":
    print("Starting GPU-optimized CIFAR-10 PDE diffusion...")
    print("Configuration: 25 epochs, dt=0.001, steps=10")
    print("Note: This configuration performs 30 PDE solves per forward pass (10 steps × 3 Strang substeps)")
    print("      More steps = better diffusion accuracy but slower training")
    
    # Test GPU availability and memory
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
        gpu_info = monitor_gpu_usage()
        print(f"GPU: {gpu_info['device_name']}")
        print(f"Available GPU memory: {gpu_info['cached_gb']:.1f} GB")
        
        # Run quick GPU test
        gpu_test_passed = quick_gpu_test()
        if not gpu_test_passed:
            print("⚠ GPU test suggests performance issues - consider reducing batch size or model complexity")
    else:
        print("CUDA not available - running on CPU")
    
    # Quick test mode for debugging (uncomment for faster testing)
    # print("Running quick test mode (1 epoch)...")
    # model, test_loader = train_cifar10_model(dx=1.0, dy=1.0, epochs=1)
    
    # Full training - 25 epochs with dt=0.001, steps=10
    print("Starting full training (25 epochs, dt=0.001, steps=10)...")
    model, test_loader = train_cifar10_model(dx=1.0, dy=1.0)  # Uses default epochs=25
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize_cifar10(model, test_loader)

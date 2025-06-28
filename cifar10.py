# CIFAR-10 PDE Diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# --- Enhanced PDE Diffusion Layer for RGB Images ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=32, channels=3, dt=0.001, dx=1.0, dy=1.0, num_steps=10):
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
        Forward pass through PDE diffusion layer
        Input: u of shape (B, C, H, W) where C=3 for RGB
        """
        B, C, H, W = u.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        # Apply multiple diffusion steps with time-dependent coefficients
        current_time = 0.0
        
        for step in range(self.num_steps):
            # Process each color channel independently
            for c in range(self.channels):
                # Get alpha and beta matrices for this channel at current time
                alpha_t, beta_t = self.get_alpha_beta_at_time(current_time, channel=c)

                # Extract single channel: (B, H, W)
                u_channel = u[:, c, :, :]

                # Strang splitting: half step x, full step y, half step x
                u_channel = self.diffuse_x_vectorized(u_channel, alpha_t, self.dt / 2, self.dx)
                
                # Update time for middle step
                mid_time = current_time + self.dt / 2
                alpha_t, beta_t = self.get_alpha_beta_at_time(mid_time, channel=c)
                u_channel = self.diffuse_y_vectorized(u_channel, beta_t, self.dt, self.dy)
                
                # Final half step in x
                final_time = current_time + self.dt
                alpha_t, beta_t = self.get_alpha_beta_at_time(final_time, channel=c)
                u_channel = self.diffuse_x_vectorized(u_channel, alpha_t, self.dt / 2, self.dx)

                # Update the channel in the full tensor
                u[:, c, :, :] = u_channel

            current_time += self.dt

        return u

    def diffuse_x_vectorized(self, u, alpha_matrix, dt, dx):
        """
        Vectorized diffusion in x-direction using proper dx spacing
        Input: u of shape (B, H, W), alpha_matrix of shape (H, W)
        """
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

    def diffuse_y_vectorized(self, u, beta_matrix, dt, dy):
        """
        Vectorized diffusion in y-direction using proper dy spacing
        Input: u of shape (B, H, W), beta_matrix of shape (H, W)
        """
        B, H, W = u.shape
        device = u.device

        # Transpose to work on columns: (B, H, W) -> (B, W, H)
        u_t = u.transpose(1, 2).contiguous()
        u_flat = u_t.view(B * W, H)

        # Expand beta_matrix for all batches: (H, W) -> (B*W, H)
        beta_expanded = beta_matrix.t().unsqueeze(0).expand(B, -1, -1).contiguous().view(B * W, H)

        # Apply smoothing to coefficients for stability
        beta_smooth = self.smooth_coefficients(beta_expanded, dim=1)
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
        result = self.thomas_solver_batch(a, b_modified, c, u_flat)

        # Transpose back: (B*W, H) -> (B, W, H) -> (B, H, W)
        return result.view(B, W, H).transpose(1, 2).contiguous()

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

        # Forward sweep
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
        self.diff = DiffusionLayer(size=32, channels=3, dx=dx, dy=dy)
        
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
def create_cifar10_data_loaders(batch_size=64):
    """Create CIFAR-10 data loaders with appropriate augmentation"""
    
    # CIFAR-10 normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_cifar10_model(dx=1.0, dy=1.0, epochs=50):
    """Training function for CIFAR-10 with configurable spatial steps"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on CIFAR-10 with spatial discretization: dx={dx}, dy={dy}")

    # Create data loaders
    train_loader, test_loader = create_cifar10_data_loaders(batch_size=64)

    # Initialize model with specified dx/dy
    model = CIFAR10PDEClassifier(dx=dx, dy=dy).to(device)

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

    # Training loop
    print(f"Starting CIFAR-10 training for {epochs} epochs...")
    import time

    for epoch in range(epochs):
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

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")

        # Monitor PDE parameters by channel
        with torch.no_grad():
            for c in range(3):
                channel_name = ['R', 'G', 'B'][c]
                alpha_stats = {
                    'base_mean': model.diff.alpha_base[c].mean().item(),
                    'base_std': model.diff.alpha_base[c].std().item(),
                    'time_mean': model.diff.alpha_time_coeff[c].mean().item(),
                    'time_std': model.diff.alpha_time_coeff[c].std().item()
                }
                beta_stats = {
                    'base_mean': model.diff.beta_base[c].mean().item(),
                    'base_std': model.diff.beta_base[c].std().item(),
                    'time_mean': model.diff.beta_time_coeff[c].mean().item(),
                    'time_std': model.diff.beta_time_coeff[c].std().item()
                }

                print(f"{channel_name}-channel α: μ={alpha_stats['base_mean']:.3f}±{alpha_stats['base_std']:.3f}, "
                      f"time: μ={alpha_stats['time_mean']:.3f}±{alpha_stats['time_std']:.3f}")

        print("-" * 80)

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
        print(f"Spatial discretization: dx={model.diff.dx}, dy={model.diff.dy}")
        print(f"Temporal discretization: dt={model.diff.dt}, steps={model.diff.num_steps}")

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


# --- Main execution ---
if __name__ == "__main__":
    print("Starting CIFAR-10 PDE diffusion with proper dx/dy handling...")
    
    # Train with equal spacing
    model, test_loader = train_cifar10_model(dx=1.0, dy=1.0, epochs=50)
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize_cifar10(model, test_loader)

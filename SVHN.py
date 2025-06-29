# PDE Diffusion Neural Network adapted for SVHN Dataset
# SVHN: 32x32 RGB images, 10 digit classes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Enhanced PDE Diffusion Layer for RGB Images ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=32, channels=3, dt=0.01, dx=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.channels = channels
        self.dt = dt
        self.dx = dx
        self.num_steps = num_steps

        # Very small diffusion coefficients to preserve image details
        self.alpha_base = nn.Parameter(torch.ones(channels, size, size) * 0.1)
        self.beta_base = nn.Parameter(torch.ones(channels, size, size) * 0.1)

        # Very small time-dependent modulation
        self.alpha_time_coeff = nn.Parameter(torch.randn(channels, size, size) * 0.001)
        self.beta_time_coeff = nn.Parameter(torch.randn(channels, size, size) * 0.001)

        # Minimal channel coupling
        self.channel_coupling = nn.Parameter(torch.eye(channels) * 0.01)

        # Stability parameters
        self.stability_eps = 1e-6

        # Add learnable skip connection weight
        self.skip_weight = nn.Parameter(torch.tensor(0.9))  # Start with mostly original image

    def get_alpha_beta_at_time(self, t):
        """Get alpha and beta coefficient matrices at time t for all channels"""
        alpha_t = self.alpha_base + self.alpha_time_coeff * t
        beta_t = self.beta_base + self.beta_time_coeff * t

        # Ensure positive coefficients for stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps)
        beta_t = torch.clamp(beta_t, min=self.stability_eps)

        return alpha_t, beta_t

    def forward(self, u):
        B, C, H, W = u.shape
        original_u = u.clone()  # Keep original for skip connection

        # Apply very minimal diffusion steps
        current_time = 0.0
        for step in range(self.num_steps):
            # Get alpha and beta matrices at current time
            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)

            # Very light diffusion - mostly preserve original
            u = self.diffuse_x_vectorized(u, alpha_t, self.dt / 2, self.dx)
            current_time += self.dt / 2

            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)
            u = self.diffuse_y_vectorized(u, beta_t, self.dt, self.dx)
            current_time += self.dt / 2

            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)
            u = self.diffuse_x_vectorized(u, alpha_t, self.dt / 2, self.dx)

            # Very minimal channel coupling
            u = self.apply_channel_coupling(u)

        # Learnable skip connection - mostly keep original image
        u = torch.sigmoid(self.skip_weight) * original_u + (1 - torch.sigmoid(self.skip_weight)) * u

        return u

    def apply_channel_coupling(self, u):
        """Apply cross-channel diffusion coupling"""
        B, C, H, W = u.shape
        # Reshape to (B, H*W, C) for matrix multiplication
        u_reshaped = u.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        # Apply channel coupling matrix
        u_coupled = torch.matmul(u_reshaped, self.channel_coupling.t())
        # Reshape back to (B, C, H, W)
        return u_coupled.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def diffuse_x_vectorized(self, u, alpha_matrix, dt, dx):
        """Vectorized diffusion in x-direction for multi-channel data"""
        B, C, H, W = u.shape
        device = u.device

        # Process each channel separately
        results = []
        for c in range(C):
            # Extract channel data: (B, H, W)
            u_channel = u[:, c, :, :]
            alpha_channel = alpha_matrix[c]  # (H, W)

            # Reshape for batch processing: (B, H, W) -> (B*H, W)
            u_flat = u_channel.contiguous().view(B * H, W)

            # Expand alpha_matrix for all batches: (H, W) -> (B*H, W)
            alpha_expanded = alpha_channel.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * H, W)

            # Apply smoothing to coefficients for stability
            alpha_smooth = self.smooth_coefficients(alpha_expanded, dim=1)
            coeff = alpha_smooth * dt / (dx ** 2)

            # Build tridiagonal system coefficients
            a = -coeff  # sub-diagonal
            c_coeff = -coeff  # super-diagonal
            b = 1 + 2 * coeff  # main diagonal

            # Apply boundary conditions (Neumann - no flux at boundaries)
            b_modified = b.clone()
            b_modified[:, 0] = 1 + coeff[:, 0]
            b_modified[:, -1] = 1 + coeff[:, -1]

            # Solve all tridiagonal systems in parallel
            result = self.thomas_solver_batch(a, b_modified, c_coeff, u_flat)
            results.append(result.view(B, H, W))

        return torch.stack(results, dim=1)  # Stack channels back: (B, C, H, W)

    def diffuse_y_vectorized(self, u, beta_matrix, dt, dx):
        """Vectorized diffusion in y-direction for multi-channel data"""
        B, C, H, W = u.shape
        device = u.device

        # Process each channel separately
        results = []
        for c in range(C):
            # Extract channel data: (B, H, W)
            u_channel = u[:, c, :, :]
            beta_channel = beta_matrix[c]  # (H, W)

            # Transpose to work on columns: (B, H, W) -> (B, W, H)
            u_t = u_channel.transpose(1, 2).contiguous()
            u_flat = u_t.view(B * W, H)

            # Expand beta_matrix for all batches: (H, W) -> (B*W, H)
            beta_expanded = beta_channel.t().unsqueeze(0).expand(B, -1, -1).contiguous().view(B * W, H)

            # Apply smoothing to coefficients for stability
            beta_smooth = self.smooth_coefficients(beta_expanded, dim=1)
            coeff = beta_smooth * dt / (dx ** 2)

            # Build tridiagonal system coefficients
            a = -coeff  # sub-diagonal
            c_coeff = -coeff  # super-diagonal
            b = 1 + 2 * coeff  # main diagonal

            # Apply boundary conditions (Neumann - no flux at boundaries)
            b_modified = b.clone()
            b_modified[:, 0] = 1 + coeff[:, 0]
            b_modified[:, -1] = 1 + coeff[:, -1]

            # Solve all tridiagonal systems in parallel
            result = self.thomas_solver_batch(a, b_modified, c_coeff, u_flat)

            # Transpose back: (B*W, H) -> (B, W, H) -> (B, H, W)
            results.append(result.view(B, W, H).transpose(1, 2).contiguous())

        return torch.stack(results, dim=1)  # Stack channels back: (B, C, H, W)

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
        """
        Batch Thomas algorithm - no in-place operations
        All inputs have shape (batch_size, N)
        """
        batch_size, N = d.shape
        device = d.device
        eps = self.stability_eps

        # Initialize working arrays
        c_star = torch.zeros_like(d)
        d_star = torch.zeros_like(d)

        # Forward elimination with numerical stability
        c_star = c_star.clone()
        d_star = d_star.clone()

        # First step
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

        # Backward sweep
        for i in range(N-2, -1, -1):
            x_val = d_star[:, i] - c_star[:, i] * x[:, i+1]
            x = x.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                         x_val.unsqueeze(1))

        return x


# --- Much Improved PDE Neural Network for SVHN ---
class PDEClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # Very conservative PDE Diffusion layer
        self.diff = DiffusionLayer(size=32, channels=3)

        # Much larger network to compensate for limited feature extraction
        self.dropout = nn.Dropout(dropout_rate)

        # Bigger first layer to capture more features
        self.fc1 = nn.Linear(32 * 32 * 3, 2048)  # Much larger
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc5 = nn.Linear(256, 10)  # Output layer

    def forward(self, x):
        # Apply very light PDE diffusion (mostly skip connection)
        x = self.diff(x)

        # Flatten and classify with batch norm and deeper network
        x = x.reshape(x.size(0), -1)

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout(F.relu(self.bn4(self.fc4(x))))

        return self.fc5(x)


# --- Training Setup for SVHN ---
def create_svhn_data_loaders(batch_size=256):
    """Create SVHN data loaders with minimal preprocessing"""
    # Very simple transforms - let the network learn from raw data
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # Just normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    # SVHN dataset
    train_loader = DataLoader(
        datasets.SVHN('./data', split='train', download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        datasets.SVHN('./data', split='test', download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_model():
    """Training function adapted for SVHN"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, test_loader = create_svhn_data_loaders(batch_size=256)

    # Initialize model
    model = PDEClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # More aggressive training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)  # Much higher LR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=15, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting heavily optimized SVHN PDE training...")
    import time

    best_acc = 0

    for epoch in range(15):  # More epochs
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
            scheduler.step()  # Step every batch for OneCycleLR

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:  # More frequent logging
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, LR: {current_lr:.6f}')

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Quick validation check every few epochs
        if epoch % 2 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    output = model(imgs)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100. * val_correct / val_total

            if val_acc > best_acc:
                best_acc = val_acc
                print(f"✓ New best validation accuracy: {best_acc:.2f}%")

            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s")
        else:
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")

        # Monitor PDE parameters
        with torch.no_grad():
            # Alpha statistics across all channels
            alpha_stats = {
                'base_mean': model.diff.alpha_base.mean().item(),
                'base_std': model.diff.alpha_base.std().item(),
                'time_mean': model.diff.alpha_time_coeff.mean().item(),
                'time_std': model.diff.alpha_time_coeff.std().item()
            }
            # Beta statistics across all channels
            beta_stats = {
                'base_mean': model.diff.beta_base.mean().item(),
                'base_std': model.diff.beta_base.std().item(),
                'time_mean': model.diff.beta_time_coeff.mean().item(),
                'time_std': model.diff.beta_time_coeff.std().item()
            }

            print(f"Alpha - Base: μ={alpha_stats['base_mean']:.3f}, σ={alpha_stats['base_std']:.3f} | "
                  f"Time: μ={alpha_stats['time_mean']:.3f}, σ={alpha_stats['time_std']:.3f}")
            print(f"Beta - Base: μ={beta_stats['base_mean']:.3f}, σ={beta_stats['base_std']:.3f} | "
                  f"Time: μ={beta_stats['time_mean']:.3f}, σ={beta_stats['time_std']:.3f}")

            # Channel coupling analysis
            coupling_norm = torch.norm(model.diff.channel_coupling).item()
            print(f"Channel coupling strength: {coupling_norm:.4f}")

        print("-" * 80)

    return model, test_loader


def evaluate_and_visualize(model, test_loader):
    """Evaluation and visualization for SVHN with confusion matrix"""
    device = next(model.parameters()).device
    model.eval()
    test_correct = 0
    test_total = 0

    # For confusion matrix
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(imgs)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(labels).sum().item()
            test_total += labels.size(0)

            # Collect for confusion matrix
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100. * test_correct / test_total
    print(f"SVHN Test Accuracy: {test_acc:.2f}%")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_predictions)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)]))

    # Visualization
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        # Denormalize images for visualization
        mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.1980, 0.2010, 0.1970]).view(1, 3, 1, 1).to(device)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        # Apply PDE diffusion for visualization
        diffused = model.diff(images)
        diffused_denorm = diffused * std + mean
        diffused_denorm = torch.clamp(diffused_denorm, 0, 1)

        # Parameter analysis
        print(f"\nSVHN PDE Parameter Analysis:")
        print(f"Image size: {model.diff.size}x{model.diff.size}, Channels: {model.diff.channels}")
        print(f"Simulation parameters: dt={model.diff.dt}, steps={model.diff.num_steps}")

        # Time evolution analysis
        time_points = torch.linspace(0, model.diff.num_steps * model.diff.dt, 4)
        print(f"\nTime Evolution of Multi-Channel Diffusion Coefficients:")
        for t in time_points:
            alpha_t, beta_t = model.diff.get_alpha_beta_at_time(t.item())
            print(f"t={t.item():.2f}: α(μ±σ)={alpha_t.mean().item():.3f}±{alpha_t.std().item():.3f}, "
                  f"β(μ±σ)={beta_t.mean().item():.3f}±{beta_t.std().item():.3f}")

        # Channel coupling matrix
        print(f"\nChannel Coupling Matrix:")
        coupling_matrix = model.diff.channel_coupling.detach().cpu().numpy()
        print(coupling_matrix)

        # Create comprehensive visualization with confusion matrix
        plt.figure(figsize=(24, 20))

        # Sample images and predictions (3 rows)
        for i in range(8):
            # Original RGB images
            plt.subplot(7, 8, i + 1)
            img_rgb = images_denorm[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f"True: {labels[i]}", fontsize=10)

            # Predictions
            plt.subplot(7, 8, i + 9)
            plt.imshow(img_rgb)
            plt.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {predicted[i]}", color=color, fontsize=10)

            # PDE diffused images
            plt.subplot(7, 8, i + 17)
            diff_rgb = diffused_denorm[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow(diff_rgb)
            plt.axis('off')
            plt.title("After PDE", fontsize=10)

        # Parameter matrices visualization (RGB channels)
        alpha_final, beta_final = model.diff.get_alpha_beta_at_time(model.diff.num_steps * model.diff.dt)

        # Alpha matrices for each channel
        for c in range(3):
            plt.subplot(7, 8, 25 + c)
            im = plt.imshow(alpha_final[c].cpu().numpy(), cmap='RdBu_r')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f'α Matrix Ch{c}', fontsize=10)
            plt.axis('off')

        # Beta matrices for each channel
        for c in range(3):
            plt.subplot(7, 8, 28 + c)
            im = plt.imshow(beta_final[c].cpu().numpy(), cmap='RdBu_r')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f'β Matrix Ch{c}', fontsize=10)
            plt.axis('off')

        # Channel coupling matrix
        plt.subplot(7, 8, 31)
        im = plt.imshow(coupling_matrix, cmap='RdBu_r')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Channel Coupling', fontsize=10)

        # Time coefficients
        plt.subplot(7, 8, 32)
        time_coeff_avg = model.diff.alpha_time_coeff.mean(dim=(1,2)).detach().cpu().numpy()
        plt.bar(['R', 'G', 'B'], time_coeff_avg, color=['red', 'green', 'blue'], alpha=0.7)
        plt.title('Time Coeffs by Channel', fontsize=10)

        # Confusion Matrix (spanning multiple subplots)
        plt.subplot(7, 4, 25)  # Bottom row, larger subplot
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, range(10))
        plt.yticks(tick_marks, range(10))
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

        # Per-class accuracy bar chart
        plt.subplot(7, 4, 26)  # Bottom row, next subplot
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        bars = plt.bar(range(10), per_class_acc, color=plt.cm.viridis(per_class_acc/100))
        plt.xlabel('Digit Class', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14)
        plt.xticks(range(10))
        plt.ylim(0, 100)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.suptitle('PDE Diffusion Neural Network on SVHN Dataset - Original Architecture', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Print additional metrics
        print(f"\nDetailed Metrics:")
        print(f"Overall Accuracy: {test_acc:.2f}%")
        print(f"Best performing class: {per_class_acc.argmax()} ({per_class_acc.max():.2f}%)")
        print(f"Worst performing class: {per_class_acc.argmin()} ({per_class_acc.min():.2f}%)")
        print(f"Standard deviation of class accuracies: {per_class_acc.std():.2f}%")


# --- Main execution ---
if __name__ == "__main__":
    print("Starting PDE diffusion training on SVHN dataset...")
    model, test_loader = train_model()
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize(model, test_loader)

# PDE Diffusion Neural Network for Fashion-MNIST
# Time-dependent alpha and beta matrices for diffusion coefficients
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Fashion-MNIST class names for visualization
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# --- Fixed PDE Diffusion Layer with Matrix Parameters ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=28, dt=0.3, dx=1.0, num_steps=4):  # Adjusted for Fashion-MNIST
        super().__init__()
        self.size = size
        self.dt = dt
        self.dx = dx
        self.num_steps = num_steps

        # Base diffusion coefficients as matrices (learnable)
        self.alpha_base = nn.Parameter(torch.ones(size, size) * 1.8)  # Slightly lower for fashion items
        self.beta_base = nn.Parameter(torch.ones(size, size) * 1.8)

        # Time-dependent modulation parameters as matrices (learnable)
        self.alpha_time_coeff = nn.Parameter(torch.zeros(size, size))
        self.beta_time_coeff = nn.Parameter(torch.zeros(size, size))

        # Stability parameters
        self.stability_eps = 1e-6

    def get_alpha_beta_at_time(self, t):
        """Get alpha and beta coefficient matrices at time t"""
        alpha_t = self.alpha_base + self.alpha_time_coeff * t
        beta_t = self.beta_base + self.beta_time_coeff * t

        # Ensure positive coefficients for stability
        alpha_t = torch.clamp(alpha_t, min=self.stability_eps)
        beta_t = torch.clamp(beta_t, min=self.stability_eps)

        return alpha_t, beta_t

    def forward(self, u):
        B, _, H, W = u.shape
        u = u.squeeze(1)

        # Apply multiple diffusion steps with time-dependent coefficients
        current_time = 0.0
        for step in range(self.num_steps):
            # Get alpha and beta matrices at current time
            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)

            # Strang splitting: half step x, full step y, half step x
            u = self.diffuse_x_vectorized(u, alpha_t, self.dt / 2, self.dx)
            current_time += self.dt / 2

            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)
            u = self.diffuse_y_vectorized(u, beta_t, self.dt, self.dx)
            current_time += self.dt / 2

            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)
            u = self.diffuse_x_vectorized(u, alpha_t, self.dt / 2, self.dx)

        return u.unsqueeze(1)

    def diffuse_x_vectorized(self, u, alpha_matrix, dt, dx):
        """Fixed vectorized diffusion in x-direction - no in-place ops"""
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
        b_modified = b.clone()  # No in-place modification
        b_modified[:, 0] = 1 + coeff[:, 0]
        b_modified[:, -1] = 1 + coeff[:, -1]

        # Solve all tridiagonal systems in parallel
        result = self.thomas_solver_batch(a, b_modified, c, u_flat)

        return result.view(B, H, W)

    def diffuse_y_vectorized(self, u, beta_matrix, dt, dx):
        """Fixed vectorized diffusion in y-direction - no in-place ops"""
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

        # Apply boundary conditions (Neumann - no flux at boundaries)
        b_modified = b.clone()  # No in-place modification
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
        """
        Fixed batch Thomas algorithm - no in-place operations
        All inputs have shape (batch_size, N)
        """
        batch_size, N = d.shape
        device = d.device
        eps = self.stability_eps

        # Initialize working arrays - avoid in-place operations
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

        # Forward sweep - avoiding in-place operations
        for i in range(1, N):
            denom = b[:, i] - a[:, i] * c_star[:, i-1] + eps

            if i < N-1:
                c_star = c_star.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                                       (c[:, i] / denom).unsqueeze(1))

            d_val = (d[:, i] - a[:, i] * d_star[:, i-1]) / denom
            d_star = d_star.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                                   d_val.unsqueeze(1))

        # Back substitution - avoiding in-place operations
        x = torch.zeros_like(d)
        x = x.scatter(1, torch.full((batch_size, 1), N-1, dtype=torch.long, device=device),
                     d_star[:, -1].unsqueeze(1))

        # Backward sweep
        for i in range(N-2, -1, -1):
            x_val = d_star[:, i] - c_star[:, i] * x[:, i+1]
            x = x.scatter(1, torch.full((batch_size, 1), i, dtype=torch.long, device=device),
                         x_val.unsqueeze(1))

        return x


# --- Enhanced Neural Network for Fashion-MNIST ---
class FashionPDEClassifier(nn.Module):
    def __init__(self, dropout_rate=0.15):  # Slightly higher dropout for fashion complexity
        super().__init__()
        self.diff = DiffusionLayer()
        self.dropout = nn.Dropout(dropout_rate)

        # Slightly larger network for fashion complexity
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # Batch normalization for better training
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.diff(x)
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        return self.fc3(x)


# --- Training Setup ---
def create_fashion_data_loaders(batch_size=128):
    """Create Fashion-MNIST data loaders with appropriate augmentation"""
    # More aggressive augmentation for fashion items
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),  # Fashion items can be rotated more
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),  # Fashion items can be flipped
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_loader = DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_fashion_model():
    """Training function for Fashion-MNIST"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, test_loader = create_fashion_data_loaders(batch_size=128)

    # Initialize model
    model = FashionPDEClassifier().to(device)

    # Optimizer settings tuned for Fashion-MNIST
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    print("Starting Fashion-MNIST PDE training...")
    import time

    for epoch in range(25):  # More epochs for fashion complexity
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

        # Monitor parameters
        with torch.no_grad():
            alpha_stats = {
                'base_mean': model.diff.alpha_base.mean().item(),
                'base_std': model.diff.alpha_base.std().item(),
                'time_mean': model.diff.alpha_time_coeff.mean().item(),
                'time_std': model.diff.alpha_time_coeff.std().item()
            }
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

        print("-" * 80)

    return model, test_loader


def evaluate_and_visualize_fashion(model, test_loader):
    """Evaluation and visualization for Fashion-MNIST"""
    device = next(model.parameters()).device
    model.eval()
    test_correct = 0
    test_total = 0

    # Class-wise accuracy tracking
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(imgs)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(labels).sum().item()
            test_total += labels.size(0)

            # Track class-wise accuracy
            for i in range(10):
                class_mask = labels == i
                if class_mask.sum() > 0:
                    class_correct[i] += pred[class_mask].eq(labels[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"Overall Test Accuracy: {test_acc:.2f}%")

    # Print class-wise accuracies
    print("\nClass-wise Accuracies:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"{FASHION_MNIST_CLASSES[i]}: {acc:.2f}%")

    # Visualization
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        # Parameter analysis
        print(f"\nMatrix Parameter Analysis:")
        print(f"Simulation parameters: dt={model.diff.dt}, steps={model.diff.num_steps}")

        # Time evolution analysis
        time_points = torch.linspace(0, model.diff.num_steps * model.diff.dt, 5)
        print(f"\nTime Evolution of Diffusion Coefficients:")
        for t in time_points:
            alpha_t, beta_t = model.diff.get_alpha_beta_at_time(t.item())
            print(f"t={t.item():.2f}: α(μ±σ)={alpha_t.mean().item():.3f}±{alpha_t.std().item():.3f}, "
                  f"β(μ±σ)={beta_t.mean().item():.3f}±{beta_t.std().item():.3f}")

        # Spatial analysis
        alpha_final, beta_final = model.diff.get_alpha_beta_at_time(model.diff.num_steps * model.diff.dt)
        print(f"\nSpatial Variation Analysis:")
        print(f"Alpha matrix - Range: [{alpha_final.min().item():.3f}, {alpha_final.max().item():.3f}]")
        print(f"Beta matrix - Range: [{beta_final.min().item():.3f}, {beta_final.max().item():.3f}]")

        # Visualization
        plt.figure(figsize=(20, 14))

        # Sample images and predictions
        for i in range(8):  # Show more samples for fashion variety
            # Original
            plt.subplot(6, 8, i + 1)
            img = images[i, 0].cpu().numpy()
            # Denormalize for visualization
            img = img * 0.3530 + 0.2860
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f"True: {FASHION_MNIST_CLASSES[labels[i]]}", fontsize=8)

            # Prediction
            plt.subplot(6, 8, i + 9)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {FASHION_MNIST_CLASSES[predicted[i]]}", color=color, fontsize=8)

            # Diffused
            plt.subplot(6, 8, i + 17)
            diffused = model.diff(images[i:i+1]).squeeze().cpu().numpy()
            # Denormalize diffused image
            diffused = diffused * 0.3530 + 0.2860
            plt.imshow(diffused, cmap='gray')
            plt.axis('off')
            plt.title("After PDE", fontsize=8)

        # Parameter matrices visualization
        matrices = [
            (alpha_final.cpu().numpy(), "Final Alpha Matrix", 25),
            (beta_final.cpu().numpy(), "Final Beta Matrix", 26),
            (model.diff.alpha_time_coeff.detach().cpu().numpy(), "Alpha Time Coeff", 33),
            (model.diff.beta_time_coeff.detach().cpu().numpy(), "Beta Time Coeff", 34)
        ]

        for matrix, title, pos in matrices:
            plt.subplot(6, 8, pos)
            im = plt.imshow(matrix, cmap='RdBu_r')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(title, fontsize=10)
            plt.axis('off')

        plt.suptitle('PDE Diffusion Network on Fashion-MNIST\nTime-Dependent Matrix Coefficients', fontsize=16)
        plt.tight_layout()
        plt.show()


# --- Main execution ---
if __name__ == "__main__":
    print("Starting PDE diffusion training on Fashion-MNIST...")
    model, test_loader = train_fashion_model()
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize_fashion(model, test_loader)

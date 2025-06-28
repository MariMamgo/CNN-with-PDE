# total time : 40 minutes, test accuracy : 97.33%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- Enhanced PDE Diffusion Layer with Separate dx/dy Parameters ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=28, dt=0.001, dx=1.0, dy=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.dt = dt
        self.dx = dx  # Spatial step in x-direction
        self.dy = dy  # Spatial step in y-direction
        self.num_steps = num_steps

        # Base diffusion coefficients as matrices (learnable)
        self.alpha_base = nn.Parameter(torch.ones(size, size) * 2.0)
        self.beta_base = nn.Parameter(torch.ones(size, size) * 2.0)

        # Time-dependent modulation parameters as matrices (learnable)
        self.alpha_time_coeff = nn.Parameter(torch.zeros(size, size))
        self.beta_time_coeff = nn.Parameter(torch.zeros(size, size))

        # Stability parameters
        self.stability_eps = 1e-6

        print(f"Initialized DiffusionLayer with dx={dx}, dy={dy}")

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
            u = self.diffuse_y_vectorized(u, beta_t, self.dt, self.dy)
            current_time += self.dt / 2

            alpha_t, beta_t = self.get_alpha_beta_at_time(current_time)
            u = self.diffuse_x_vectorized(u, alpha_t, self.dt / 2, self.dx)

        return u.unsqueeze(1)

    def diffuse_x_vectorized(self, u, alpha_matrix, dt, dx):
        """
        Vectorized diffusion in x-direction using proper dx spacing
        Solves: ∂u/∂t = ∇·(α∇u) in x-direction
        """
        B, H, W = u.shape
        device = u.device

        # Reshape for batch processing: (B, H, W) -> (B*H, W)
        u_flat = u.contiguous().view(B * H, W)

        # Expand alpha_matrix for all batches: (H, W) -> (B*H, W)
        alpha_expanded = alpha_matrix.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * H, W)

        # Apply smoothing to coefficients for stability
        alpha_smooth = self.smooth_coefficients(alpha_expanded, dim=1)
        coeff = alpha_smooth * dt / (dx ** 2)  # Using dx for x-direction

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
        Solves: ∂u/∂t = ∇·(β∇u) in y-direction
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
        coeff = beta_smooth * dt / (dy ** 2)  # Using dy for y-direction

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
        """
        Batch Thomas algorithm for tridiagonal systems
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

    def get_numerical_stability_info(self):
        """Get information about numerical stability"""
        with torch.no_grad():
            # Check stability conditions for both directions
            alpha_max = torch.max(self.alpha_base + torch.abs(self.alpha_time_coeff) * self.dt * self.num_steps)
            beta_max = torch.max(self.beta_base + torch.abs(self.beta_time_coeff) * self.dt * self.num_steps)

            # CFL-like conditions
            cfl_x = alpha_max * self.dt / (self.dx ** 2)
            cfl_y = beta_max * self.dt / (self.dy ** 2)

            return {
                'cfl_x': cfl_x.item(),
                'cfl_y': cfl_y.item(),
                'dx': self.dx,
                'dy': self.dy,
                'dt': self.dt,
                'stable_x': cfl_x.item() < 0.5,
                'stable_y': cfl_y.item() < 0.5
            }


# --- Enhanced Neural Network ---
class PDEClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1, dx=1.0, dy=1.0):
        super().__init__()
        self.diff = DiffusionLayer(dx=dx, dy=dy)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.diff(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# --- Training Setup ---
def create_data_loaders(batch_size=128):
    """Create optimized data loaders with data augmentation"""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05))
    ])

    transform_test = transforms.Compose([transforms.ToTensor()])

    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_model(dx=1.0, dy=1.0):
    """Training function with configurable spatial steps"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Spatial discretization: dx={dx}, dy={dy}")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(batch_size=128)

    # Initialize model with specified dx/dy
    model = PDEClassifier(dx=dx, dy=dy).to(device)

    # Print stability information
    stability_info = model.diff.get_numerical_stability_info()
    print(f"Numerical Stability Analysis:")
    print(f"  CFL condition X: {stability_info['cfl_x']:.4f} {'✓' if stability_info['stable_x'] else '⚠'}")
    print(f"  CFL condition Y: {stability_info['cfl_y']:.4f} {'✓' if stability_info['stable_y'] else '⚠'}")

    # Conservative optimizer settings for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    print("Starting training with proper dx/dy handling...")
    import time

    for epoch in range(1):
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


def evaluate_and_visualize(model, test_loader):
    """Evaluation and visualization with dx/dy analysis"""
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
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Enhanced analysis with dx/dy information
    stability_info = model.diff.get_numerical_stability_info()

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        # Parameter analysis
        print(f"\nEnhanced PDE Analysis:")
        print(f"Spatial discretization: dx={model.diff.dx}, dy={model.diff.dy}")
        print(f"Temporal discretization: dt={model.diff.dt}, steps={model.diff.num_steps}")
        print(f"Stability: CFL_x={stability_info['cfl_x']:.4f}, CFL_y={stability_info['cfl_y']:.4f}")

        # Time evolution analysis
        time_points = torch.linspace(0, model.diff.num_steps * model.diff.dt, 5)
        print(f"\nTime Evolution of Diffusion Coefficients:")
        for t in time_points:
            alpha_t, beta_t = model.diff.get_alpha_beta_at_time(t.item())
            print(f"t={t.item():.2f}: α(μ±σ)={alpha_t.mean().item():.3f}±{alpha_t.std().item():.3f}, "
                  f"β(μ±σ)={beta_t.mean().item():.3f}±{beta_t.std().item():.3f}")

        # Anisotropy analysis
        alpha_final, beta_final = model.diff.get_alpha_beta_at_time(model.diff.num_steps * model.diff.dt)
        effective_diff_x = alpha_final / (model.diff.dx ** 2)
        effective_diff_y = beta_final / (model.diff.dy ** 2)

        print(f"\nAnisotropy Analysis:")
        print(f"Effective diffusion rates:")
        print(f"  X-direction: {effective_diff_x.mean().item():.3f}±{effective_diff_x.std().item():.3f}")
        print(f"  Y-direction: {effective_diff_y.mean().item():.3f}±{effective_diff_y.std().item():.3f}")
        print(f"  Anisotropy ratio: {(effective_diff_x.mean()/effective_diff_y.mean()).item():.3f}")

        # Visualization
        plt.figure(figsize=(20, 15))

        # Sample images and predictions
        for i in range(6):
            # Original
            plt.subplot(6, 6, i + 1)
            plt.imshow(images[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"True: {labels[i]}")

            # Prediction
            plt.subplot(6, 6, i + 7)
            plt.imshow(images[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {predicted[i]}", color=color)

            # Diffused
            plt.subplot(6, 6, i + 13)
            diffused = model.diff(images[i:i+1]).squeeze().cpu().numpy()
            plt.imshow(diffused, cmap='gray')
            plt.axis('off')
            plt.title("After PDE")

        # Parameter matrices visualization
        matrices = [
            (alpha_final.cpu().numpy(), f"Final Alpha Matrix\n(dx={model.diff.dx})", 19),
            (beta_final.cpu().numpy(), f"Final Beta Matrix\n(dy={model.diff.dy})", 20),
            (effective_diff_x.cpu().numpy(), "Effective Diffusion X", 25),
            (effective_diff_y.cpu().numpy(), "Effective Diffusion Y", 26),
            (model.diff.alpha_time_coeff.detach().cpu().numpy(), "Alpha Time Coeff", 31),
            (model.diff.beta_time_coeff.detach().cpu().numpy(), "Beta Time Coeff", 32)
        ]

        for matrix, title, pos in matrices:
            plt.subplot(6, 6, pos)
            im = plt.imshow(matrix, cmap='RdBu_r')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(title, fontsize=9)
            plt.axis('off')

        plt.suptitle(f'Enhanced PDE Diffusion: dx={model.diff.dx}, dy={model.diff.dy}', fontsize=16)
        plt.tight_layout()
        plt.show()


# --- Comparison function for different dx/dy values ---
def compare_spatial_discretizations():
    """Compare performance with different dx/dy ratios"""
    print("=" * 80)
    print("COMPARING DIFFERENT SPATIAL DISCRETIZATIONS")
    print("=" * 80)

    configs = [
        (1.0, 1.0, "Square grid (isotropic)"),
        (1.0, 0.5, "Fine Y resolution"),
        (0.5, 1.0, "Fine X resolution"),
        (2.0, 1.0, "Coarse X resolution")
    ]

    results = []

    for dx, dy, description in configs:
        print(f"\n--- Testing {description}: dx={dx}, dy={dy} ---")
        try:
            model, test_loader = train_model(dx=dx, dy=dy)

            # Quick evaluation
            device = next(model.parameters()).device
            model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    predicted = outputs.argmax(dim=1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            accuracy = 100. * correct / total
            results.append((dx, dy, description, accuracy))
            print(f"Final accuracy: {accuracy:.2f}%")

        except Exception as e:
            print(f"Failed with {description}: {e}")
            results.append((dx, dy, description, 0.0))

    # Summary
    print("\n" + "=" * 80)
    print("SPATIAL DISCRETIZATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'dx':<6} {'dy':<6} {'Description':<25} {'Accuracy':<10}")
    print("-" * 60)
    for dx, dy, desc, acc in results:
        print(f"{dx:<6} {dy:<6} {desc:<25} {acc:<10.2f}%")


# --- Main execution ---
if __name__ == "__main__":
    print("Starting enhanced PDE diffusion with proper dx/dy handling...")

    # Standard run with equal spacing
    model, test_loader = train_model(dx=1.0, dy=1.0)
    print("\nTraining completed! Evaluating...")
    evaluate_and_visualize(model, test_loader)

    # Optional: Compare different discretizations
    # compare_spatial_discretizations()

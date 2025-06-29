# Enhanced PDE Diffusion Neural Network for Face Expression Recognition
# Modified with Time-Dependent Vector Alpha/Beta Coefficients

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Enhanced PDE Diffusion Layer with Time-Dependent Vector Parameters ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=48, dt=0.01, dx=1.0, dy=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.dt = dt
        self.dx = dx  # Spatial step in x-direction
        self.dy = dy  # Spatial step in y-direction
        self.num_steps = num_steps

        # Base diffusion coefficients as 1D vectors (learnable)
        # Alpha vector for x-direction diffusion
        self.alpha_base_x = nn.Parameter(torch.ones(size) * 2.0)
        self.alpha_base_y = nn.Parameter(torch.ones(size) * 2.0)
        
        # Beta vector for y-direction diffusion  
        self.beta_base_x = nn.Parameter(torch.ones(size) * 2.0)
        self.beta_base_y = nn.Parameter(torch.ones(size) * 2.0)

        # Time-dependent modulation parameters as vectors (learnable)
        self.alpha_time_coeff_x = nn.Parameter(torch.zeros(size))
        self.alpha_time_coeff_y = nn.Parameter(torch.zeros(size))
        self.beta_time_coeff_x = nn.Parameter(torch.zeros(size))
        self.beta_time_coeff_y = nn.Parameter(torch.zeros(size))

        # Higher-order time dependence (quadratic terms)
        self.alpha_time_quad_x = nn.Parameter(torch.zeros(size))
        self.alpha_time_quad_y = nn.Parameter(torch.zeros(size))
        self.beta_time_quad_x = nn.Parameter(torch.zeros(size))
        self.beta_time_quad_y = nn.Parameter(torch.zeros(size))

        # Stability parameters
        self.stability_eps = 1e-6

        print(f"Initialized DiffusionLayer with vector coefficients:")
        print(f"  - Size: {size}x{size}, dx={dx}, dy={dy}")
        print(f"  - Vector parameters: 4 base + 4 linear + 4 quadratic = 12 vectors")
        print(f"  - Total learnable parameters: {12 * size}")

    def get_alpha_beta_vectors_at_time(self, t):
        """Get alpha and beta coefficient vectors at time t"""
        # Time-dependent vectors with quadratic terms for richer dynamics
        alpha_x_t = (self.alpha_base_x + 
                     self.alpha_time_coeff_x * t + 
                     self.alpha_time_quad_x * t**2)
        alpha_y_t = (self.alpha_base_y + 
                     self.alpha_time_coeff_y * t + 
                     self.alpha_time_quad_y * t**2)
        
        beta_x_t = (self.beta_base_x + 
                    self.beta_time_coeff_x * t + 
                    self.beta_time_quad_x * t**2)
        beta_y_t = (self.beta_base_y + 
                    self.beta_time_coeff_y * t + 
                    self.beta_time_quad_y * t**2)

        # Ensure positive coefficients for stability
        alpha_x_t = torch.clamp(alpha_x_t, min=self.stability_eps)
        alpha_y_t = torch.clamp(alpha_y_t, min=self.stability_eps)
        beta_x_t = torch.clamp(beta_x_t, min=self.stability_eps)
        beta_y_t = torch.clamp(beta_y_t, min=self.stability_eps)

        return alpha_x_t, alpha_y_t, beta_x_t, beta_y_t

    def expand_vector_to_matrix(self, vec_x, vec_y, direction='x'):
        """
        Convert vectors to coefficient matrices for diffusion
        vec_x: coefficients varying along x-direction (size,)
        vec_y: coefficients varying along y-direction (size,)
        direction: 'x' or 'y' to determine how to combine the vectors
        """
        if direction == 'x':
            # For x-direction diffusion: broadcast vec_x along y-axis
            # Each row has the same x-varying coefficients
            matrix = vec_x.unsqueeze(0).expand(self.size, -1)  # (size, size)
        else:  # direction == 'y'
            # For y-direction diffusion: broadcast vec_y along x-axis  
            # Each column has the same y-varying coefficients
            matrix = vec_y.unsqueeze(1).expand(-1, self.size)  # (size, size)
            
        return matrix

    def forward(self, u):
        B, _, H, W = u.shape
        u = u.squeeze(1)

        # Apply multiple diffusion steps with time-dependent vector coefficients
        current_time = 0.0
        for step in range(self.num_steps):
            # Get vector coefficients at current time
            alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(current_time)
            
            # Strang splitting: half step x, full step y, half step x
            # Convert vectors to matrices for x-direction diffusion
            alpha_matrix_x = self.expand_vector_to_matrix(alpha_x_t, alpha_y_t, direction='x')
            u = self.diffuse_x_vectorized(u, alpha_matrix_x, self.dt / 2, self.dx)
            current_time += self.dt / 2

            # Update coefficients and do y-direction diffusion
            alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(current_time)
            beta_matrix_y = self.expand_vector_to_matrix(beta_x_t, beta_y_t, direction='y')
            u = self.diffuse_y_vectorized(u, beta_matrix_y, self.dt, self.dy)
            current_time += self.dt / 2

            # Final half step in x-direction
            alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(current_time)
            alpha_matrix_x = self.expand_vector_to_matrix(alpha_x_t, alpha_y_t, direction='x')
            u = self.diffuse_x_vectorized(u, alpha_matrix_x, self.dt / 2, self.dx)

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
            # Check stability conditions for both directions across all time steps
            max_time = self.dt * self.num_steps
            
            # Get maximum coefficients across time evolution
            alpha_x_max = torch.max(self.alpha_base_x + 
                                   torch.abs(self.alpha_time_coeff_x) * max_time +
                                   torch.abs(self.alpha_time_quad_x) * max_time**2)
            alpha_y_max = torch.max(self.alpha_base_y + 
                                   torch.abs(self.alpha_time_coeff_y) * max_time +
                                   torch.abs(self.alpha_time_quad_y) * max_time**2)
            beta_x_max = torch.max(self.beta_base_x + 
                                  torch.abs(self.beta_time_coeff_x) * max_time +
                                  torch.abs(self.beta_time_quad_x) * max_time**2)
            beta_y_max = torch.max(self.beta_base_y + 
                                  torch.abs(self.beta_time_coeff_y) * max_time +
                                  torch.abs(self.beta_time_quad_y) * max_time**2)

            # CFL-like conditions
            cfl_x = torch.max(alpha_x_max, beta_x_max) * self.dt / (self.dx ** 2)
            cfl_y = torch.max(alpha_y_max, beta_y_max) * self.dt / (self.dy ** 2)

            return {
                'cfl_x': cfl_x.item(),
                'cfl_y': cfl_y.item(),
                'dx': self.dx,
                'dy': self.dy,
                'dt': self.dt,
                'stable_x': cfl_x.item() < 0.5,
                'stable_y': cfl_y.item() < 0.5,
                'max_alpha_x': alpha_x_max.item(),
                'max_alpha_y': alpha_y_max.item(),
                'max_beta_x': beta_x_max.item(),
                'max_beta_y': beta_y_max.item(),
                'total_vector_params': 12 * self.size
            }

    def get_coefficient_evolution(self):
        """Get the evolution of coefficients over time for visualization"""
        time_points = torch.linspace(0, self.dt * self.num_steps, 50)
        evolutions = {
            'time': time_points.numpy(),
            'alpha_x': [],
            'alpha_y': [],
            'beta_x': [],
            'beta_y': []
        }
        
        with torch.no_grad():
            for t in time_points:
                alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(t)
                evolutions['alpha_x'].append(alpha_x_t.mean().item())
                evolutions['alpha_y'].append(alpha_y_t.mean().item())
                evolutions['beta_x'].append(beta_x_t.mean().item())
                evolutions['beta_y'].append(beta_y_t.mean().item())
        
        return evolutions


# --- Face Expression Dataset Class (unchanged) ---
class FaceExpressionDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []

        # Define emotion mapping
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}

        # Load the dataset
        if os.path.isfile(data_path):
            # If it's a single CSV file (like FER2013)
            self.data = pd.read_csv(data_path)
            self._load_from_csv()
        else:
            # If it's a directory with image folders
            self._load_from_directory(data_path)

        print(f"Loaded {'train' if train else 'test'} dataset with {len(self.image_paths) if not hasattr(self, 'data') else len(self.data)} samples")

        # Print class distribution
        self._print_class_distribution()

    def _load_from_csv(self):
        """Load dataset from CSV file"""
        if 'Usage' in self.data.columns:
            # FER2013 format with Usage column
            if self.train:
                self.data = self.data[self.data['Usage'] == 'Training']
            else:
                self.data = self.data[self.data['Usage'].isin(['PublicTest', 'PrivateTest'])]
        else:
            # Split data manually if no Usage column
            total_len = len(self.data)
            if self.train:
                self.data = self.data[:int(0.8 * total_len)]
            else:
                self.data = self.data[int(0.8 * total_len):]

    def _load_from_directory(self, data_path):
        """Load dataset from directory structure"""
        print(f"Loading from directory: {data_path}")

        # Check if emotion folders exist directly in data_path
        emotion_folders = []
        for emotion in self.emotion_to_idx.keys():
            emotion_path = os.path.join(data_path, emotion)
            if os.path.exists(emotion_path):
                emotion_folders.append((emotion, emotion_path))

        if not emotion_folders:
            # Try alternative naming conventions
            alternative_names = {
                'angry': ['angry', 'anger', '0'],
                'disgust': ['disgust', '1'],
                'fear': ['fear', '2'],
                'happy': ['happy', 'happiness', '3'],
                'sad': ['sad', 'sadness', '4'],
                'surprise': ['surprise', 'surprised', '5'],
                'neutral': ['neutral', '6']
            }

            for emotion, alternatives in alternative_names.items():
                for alt_name in alternatives:
                    emotion_path = os.path.join(data_path, alt_name)
                    if os.path.exists(emotion_path):
                        emotion_folders.append((emotion, emotion_path))
                        break

        if not emotion_folders:
            raise ValueError(f"No emotion folders found in {data_path}. Expected folders: {list(self.emotion_to_idx.keys())}")

        print(f"Found emotion folders: {[folder[0] for folder in emotion_folders]}")

        # Load images from each emotion folder
        for emotion, emotion_path in emotion_folders:
            emotion_idx = self.emotion_to_idx[emotion]

            # Get all image files in the emotion folder
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_files.extend([f for f in os.listdir(emotion_path) if f.lower().endswith(ext)])

            print(f"Found {len(image_files)} images for emotion '{emotion}'")

            for image_file in image_files:
                image_path = os.path.join(emotion_path, image_file)
                self.image_paths.append(image_path)
                self.labels.append(emotion_idx)

        # Convert to numpy arrays for easier manipulation
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        # Shuffle the data
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = self.image_paths[indices]
        self.labels = self.labels[indices]

        # Split into train/test if needed (80/20 split)
        total_len = len(self.image_paths)
        if self.train:
            split_idx = int(0.8 * total_len)
            self.image_paths = self.image_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            split_idx = int(0.8 * total_len)
            self.image_paths = self.image_paths[split_idx:]
            self.labels = self.labels[split_idx:]

    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset"""
        if hasattr(self, 'data'):
            # CSV format
            if 'emotion' in self.data.columns:
                emotion_counts = self.data['emotion'].value_counts().sort_index()
                print("Emotion distribution:", dict(emotion_counts))
        else:
            # Directory format
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            distribution = {self.idx_to_emotion[label]: count for label, count in zip(unique_labels, counts)}
            print("Emotion distribution:", distribution)

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            # CSV format
            row = self.data.iloc[idx]

            # Handle different data formats
            if 'pixels' in row:
                # FER2013 format: pixels are space-separated string
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                image = pixels.reshape(48, 48)
            elif 'pixel_values' in row:
                # Alternative format
                pixels = np.array([int(p) for p in row['pixel_values'].split()], dtype=np.uint8)
                image = pixels.reshape(48, 48)
            else:
                # Assume image path
                image_path = row['image_path'] if 'image_path' in row else row[0]
                image = Image.open(image_path).convert('L')
                image = np.array(image)

            # Convert to PIL Image for transforms
            image = Image.fromarray(image, mode='L')

            # Get label
            label = row['emotion'] if 'emotion' in row else row['label']
        else:
            # Directory format
            image_path = self.image_paths[idx]
            label = self.labels[idx]

            # Load and process image
            try:
                image = Image.open(image_path).convert('L')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a black image as fallback
                image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode='L')

        if self.transform:
            image = self.transform(image)

        return image, label


# --- Enhanced Neural Network for Face Expression Recognition ---
class FaceExpressionPDEClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1, dx=1.0, dy=1.0):
        super().__init__()
        # Updated for 48x48 images and 7 emotion classes with vector coefficients
        self.diff = DiffusionLayer(size=48, dx=dx, dy=dy)
        self.dropout = nn.Dropout(dropout_rate)

        # Adjusted network for 48x48 images
        self.fc1 = nn.Linear(48 * 48, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)  # 7 emotion classes

        # Batch normalization for better convergence
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.diff(x)
        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        return self.fc3(x)


# --- Dataset Download and Setup Functions (unchanged) ---
def download_and_setup_dataset():
    """Download and organize the emotion recognition dataset"""
    print("📥 Downloading dataset...")
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    return path

def find_data_directories(dataset_path):
    """Find training and validation directories in the dataset"""
    possible_train_names = ['train', 'training', 'Train', 'Training']
    possible_val_names = ['test', 'validation', 'val', 'Test', 'Validation', 'Val']

    train_dir = None
    val_dir = None

    print(f"Searching for data directories in: {dataset_path}")

    # Check for direct subdirectories
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            print(f"Found directory: {item}")
            if item in possible_train_names:
                train_dir = item_path
                print(f"  -> Identified as training directory")
            elif item in possible_val_names:
                val_dir = item_path
                print(f"  -> Identified as validation directory")
            elif item == 'images':
                # Check inside 'images' folder
                images_path = item_path
                for sub_item in os.listdir(images_path):
                    sub_item_path = os.path.join(images_path, sub_item)
                    if os.path.isdir(sub_item_path):
                        if sub_item in possible_train_names:
                            train_dir = sub_item_path
                            print(f"  -> Found training directory in images: {sub_item_path}")
                        elif sub_item in possible_val_names:
                            val_dir = sub_item_path
                            print(f"  -> Found validation directory in images: {sub_item_path}")

    # If not found, look deeper
    if train_dir is None or val_dir is None:
        print("Searching deeper in directory structure...")
        for root, dirs, files in os.walk(dataset_path):
            for dir_name in dirs:
                if dir_name in possible_train_names and train_dir is None:
                    train_dir = os.path.join(root, dir_name)
                    print(f"Found training directory: {train_dir}")
                elif dir_name in possible_val_names and val_dir is None:
                    val_dir = os.path.join(root, dir_name)
                    print(f"Found validation directory: {val_dir}")

    # If still no validation directory found, we'll use the training directory
    # and split it internally
    if train_dir and not val_dir:
        print("No separate validation directory found, will split training data")
        val_dir = train_dir

    return train_dir, val_dir


# --- Data Loading Functions (unchanged) ---
def create_face_expression_loaders(data_path, batch_size=64):
    """Create data loaders for face expression dataset"""

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    transform_test = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        print(f"Creating datasets from path: {data_path}")
        train_dataset = FaceExpressionDataset(data_path, train=True, transform=transform_train)
        test_dataset = FaceExpressionDataset(data_path, train=False, transform=transform_test)

        # Ensure we have data
        if len(train_dataset) == 0:
            print("Warning: Training dataset is empty!")
            return None, None
        if len(test_dataset) == 0:
            print("Warning: Test dataset is empty!")
            return None, None

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False  # Reduced workers for compatibility
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False  # Reduced workers for compatibility
        )

        # Test loading a batch to ensure everything works
        try:
            test_batch = next(iter(train_loader))
            print(f"Successfully loaded test batch: {test_batch[0].shape}, {test_batch[1].shape}")
        except Exception as e:
            print(f"Error loading test batch: {e}")
            return None, None

        return train_loader, test_loader

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# --- Training Function ---
def train_face_expression_model(data_path, dx=1.0, dy=1.0, epochs=25):
    """Training function for face expression recognition with vector coefficients"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Spatial discretization: dx={dx}, dy={dy}")

    # Create data loaders
    train_loader, test_loader = create_face_expression_loaders(data_path, batch_size=64)

    if train_loader is None:
        print("Failed to create data loaders")
        return None, None

    # Initialize model
    model = FaceExpressionPDEClassifier(dx=dx, dy=dy).to(device)

    # Print stability information
    stability_info = model.diff.get_numerical_stability_info()
    print(f"Numerical Stability Analysis (Vector Coefficients):")
    print(f"  CFL condition X: {stability_info['cfl_x']:.4f} {'✓' if stability_info['stable_x'] else '⚠'}")
    print(f"  CFL condition Y: {stability_info['cfl_y']:.4f} {'✓' if stability_info['stable_y'] else '⚠'}")
    print(f"  Total vector parameters: {stability_info['total_vector_params']}")
    print(f"  Max coefficients - αx: {stability_info['max_alpha_x']:.3f}, αy: {stability_info['max_alpha_y']:.3f}")
    print(f"                   - βx: {stability_info['max_beta_x']:.3f}, βy: {stability_info['max_beta_y']:.3f}")

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use class weights for imbalanced dataset (common in emotion recognition)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    print("Starting training for face expression recognition with vector coefficients...")
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

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")

        # Monitor vector parameters
        with torch.no_grad():
            alpha_x_stats = {
                'base_mean': model.diff.alpha_base_x.mean().item(),
                'base_std': model.diff.alpha_base_x.std().item(),
                'time_mean': model.diff.alpha_time_coeff_x.mean().item(),
                'quad_mean': model.diff.alpha_time_quad_x.mean().item()
            }
            
            beta_y_stats = {
                'base_mean': model.diff.beta_base_y.mean().item(),
                'base_std': model.diff.beta_base_y.std().item(),
                'time_mean': model.diff.beta_time_coeff_y.mean().item(),
                'quad_mean': model.diff.beta_time_quad_y.mean().item()
            }

            print(f"Vector Coefficients - αx: base μ={alpha_x_stats['base_mean']:.3f}±{alpha_x_stats['base_std']:.3f}, "
                  f"linear={alpha_x_stats['time_mean']:.3f}, quad={alpha_x_stats['quad_mean']:.3f}")
            print(f"                    - βy: base μ={beta_y_stats['base_mean']:.3f}±{beta_y_stats['base_std']:.3f}, "
                  f"linear={beta_y_stats['time_mean']:.3f}, quad={beta_y_stats['quad_mean']:.3f}")

        print("-" * 80)

    return model, test_loader


# --- Enhanced Evaluation and Visualization ---
def evaluate_and_visualize_emotions(model, test_loader):
    """Evaluation and visualization for emotion recognition with vector coefficients"""
    device = next(model.parameters()).device
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(imgs)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))

    # Enhanced visualization with vector coefficient analysis
    plt.figure(figsize=(24, 18))

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        # Sample images and predictions
        for i in range(min(8, len(images))):
            # Original
            plt.subplot(5, 8, i + 1)
            img = images[i, 0].cpu().numpy()
            # Denormalize for display
            img = (img + 1) / 2  # Convert from [-1,1] to [0,1]
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f"True: {emotion_labels[labels[i]]}", fontsize=8)

            # Prediction
            plt.subplot(5, 8, i + 9)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {emotion_labels[predicted[i]]}", color=color, fontsize=8)

            # Diffused
            plt.subplot(5, 8, i + 17)
            diffused = model.diff(images[i:i+1]).squeeze().cpu().numpy()
            # Denormalize for display
            diffused = (diffused + 1) / 2
            plt.imshow(diffused, cmap='gray')
            plt.axis('off')
            plt.title("After PDE", fontsize=8)

        # Vector coefficient visualization at final time
        final_time = model.diff.num_steps * model.diff.dt
        alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = model.diff.get_alpha_beta_vectors_at_time(final_time)

        # Alpha vectors
        plt.subplot(5, 8, 25)
        plt.plot(alpha_x_t.cpu().numpy(), 'b-', label='αx(t_final)', linewidth=2)
        plt.plot(alpha_y_t.cpu().numpy(), 'r--', label='αy(t_final)', linewidth=2)
        plt.title("Alpha Vectors", fontsize=10)
        plt.xlabel("Spatial Index")
        plt.ylabel("Coefficient Value")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

        # Beta vectors
        plt.subplot(5, 8, 26)
        plt.plot(beta_x_t.cpu().numpy(), 'g-', label='βx(t_final)', linewidth=2)
        plt.plot(beta_y_t.cpu().numpy(), 'm--', label='βy(t_final)', linewidth=2)
        plt.title("Beta Vectors", fontsize=10)
        plt.xlabel("Spatial Index")
        plt.ylabel("Coefficient Value")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

        # Time evolution of mean coefficients
        evolution = model.diff.get_coefficient_evolution()
        plt.subplot(5, 8, 27)
        plt.plot(evolution['time'], evolution['alpha_x'], 'b-', label='mean(αx)', linewidth=2)
        plt.plot(evolution['time'], evolution['alpha_y'], 'r--', label='mean(αy)', linewidth=2)
        plt.plot(evolution['time'], evolution['beta_x'], 'g:', label='mean(βx)', linewidth=2)
        plt.plot(evolution['time'], evolution['beta_y'], 'm-.', label='mean(βy)', linewidth=2)
        plt.title("Coefficient Evolution", fontsize=10)
        plt.xlabel("Time")
        plt.ylabel("Mean Coefficient Value")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

        # Confusion Matrix
        plt.subplot(5, 8, 28)
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotion_labels, yticklabels=emotion_labels,
                   cbar=False)
        plt.title("Confusion Matrix", fontsize=10)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)

        # Coefficient matrices visualization (convert vectors to matrices for display)
        alpha_matrix_x = model.diff.expand_vector_to_matrix(alpha_x_t, alpha_y_t, direction='x')
        beta_matrix_y = model.diff.expand_vector_to_matrix(beta_x_t, beta_y_t, direction='y')

        plt.subplot(5, 8, 33)
        im = plt.imshow(alpha_matrix_x.cpu().numpy(), cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Alpha Matrix (X-dir)", fontsize=8)
        plt.axis('off')

        plt.subplot(5, 8, 34)
        im = plt.imshow(beta_matrix_y.cpu().numpy(), cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Beta Matrix (Y-dir)", fontsize=8)
        plt.axis('off')

        # Vector parameter statistics
        stability_info = model.diff.get_numerical_stability_info()
        plt.subplot(5, 8, 35)
        plt.axis('off')
        info_text = f"""Vector Coefficient Model
        
Total Parameters: {stability_info['total_vector_params']}
CFL X: {stability_info['cfl_x']:.3f}
CFL Y: {stability_info['cfl_y']:.3f}
Stable: {stability_info['stable_x'] and stability_info['stable_y']}

Max Coefficients:
αx: {stability_info['max_alpha_x']:.3f}
αy: {stability_info['max_alpha_y']:.3f}
βx: {stability_info['max_beta_x']:.3f}
βy: {stability_info['max_beta_y']:.3f}"""
        plt.text(0.1, 0.9, info_text, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.suptitle(f'PDE-based Face Expression Recognition with Vector Coefficients (Test Acc: {test_acc:.2f}%)', fontsize=16)
        plt.tight_layout()
        plt.show()


# --- Main execution function ---
def run_pde_emotion_recognition():
    """Complete pipeline for PDE-based emotion recognition with vector coefficients"""
    print("🚀 Starting PDE-based Emotion Recognition System (Vector Coefficients)")
    print("=" * 70)

    # Download dataset
    dataset_path = download_and_setup_dataset()

    # First, try to find CSV files
    data_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))

    if data_files:
        # Use CSV format
        data_path = data_files[0]
        print(f"Using CSV file: {data_path}")
    else:
        # Use directory format
        print("No CSV files found. Looking for image directories...")
        train_dir, val_dir = find_data_directories(dataset_path)

        if train_dir:
            data_path = train_dir
            print(f"Using training directory: {data_path}")

            # Check if this directory contains emotion subdirectories
            emotion_dirs = []
            emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

            for emotion in emotion_names:
                emotion_path = os.path.join(data_path, emotion)
                if os.path.exists(emotion_path):
                    emotion_dirs.append(emotion)
                    file_count = len([f for f in os.listdir(emotion_path)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    print(f"  - {emotion}: {file_count} images")

            if not emotion_dirs:
                print(f"No emotion directories found in {data_path}")
                # Try looking for numbered directories or other formats
                subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
                print(f"Available subdirectories: {subdirs}")
                return None, None

        else:
            print(f"No suitable data directories found. Using base directory: {dataset_path}")
            data_path = dataset_path

    print(f"\n🧠 Training PDE-Enhanced Neural Network with Vector Coefficients...")
    print("=" * 70)

    # Train model with vector-based diffusion coefficients
    model, test_loader = train_face_expression_model(data_path, dx=1.0, dy=1.0, epochs=25)

    if model is not None:
        print("\n📊 Evaluating Model Performance...")
        print("=" * 70)
        evaluate_and_visualize_emotions(model, test_loader)

        print("\n✅ Training and evaluation completed successfully!")
        print("🔬 Key improvements with vector coefficients:")
        print("   - Reduced parameter count from matrix to vector form")
        print("   - Time-dependent quadratic evolution for richer dynamics")
        print("   - Separate x/y direction control for anisotropic diffusion")
        print("   - Enhanced numerical stability monitoring")
        return model, test_loader
    else:
        print("❌ Training failed. Please check the dataset format and path.")
        return None, None


# --- Run the complete system ---
if __name__ == "__main__":
    # Run the complete PDE emotion recognition system with vector coefficients
    model, test_loader = run_pde_emotion_recognition()

# Advanced Feature Extraction for Emotion Recognition without CNNs
# Focus on sophisticated feature engineering and better architectures

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
import math

# --- Advanced PDE Layer with Multiple Operators ---
class AdvancedPDELayer(nn.Module):
    def __init__(self, size=48, num_steps=2):
        super().__init__()
        self.size = size
        self.num_steps = num_steps
        
        # Learnable coefficients for different operators
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
        self.edge_enhance_coeff = nn.Parameter(torch.tensor(0.05))
        self.smoothing_coeff = nn.Parameter(torch.tensor(0.02))
        
        print(f"Initialized AdvancedPDELayer with {num_steps} steps")

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.squeeze(1)  # (B, H, W)
        
        # Store original for residual connection
        x_orig = x.clone()
        
        for step in range(self.num_steps):
            x = self.pde_step(x)
        
        # Learnable residual mixing
        alpha = torch.sigmoid(self.smoothing_coeff)
        x = alpha * x + (1 - alpha) * x_orig
        
        return x.unsqueeze(1)  # (B, 1, H, W)

    def pde_step(self, u):
        """Combined PDE operations: diffusion + edge enhancement"""
        # Clamp coefficients for stability
        diff_coeff = torch.clamp(self.diffusion_coeff, 0.001, 0.5)
        edge_coeff = torch.clamp(self.edge_enhance_coeff, 0.001, 0.3)
        
        # Compute Laplacian (diffusion)
        laplacian = self.compute_laplacian(u)
        
        # Compute gradient magnitude (edge enhancement)
        grad_mag = self.compute_gradient_magnitude(u)
        
        # Combined update
        dt = 0.01
        u_new = u + dt * (diff_coeff * laplacian - edge_coeff * grad_mag)
        
        return u_new

    def compute_laplacian(self, u):
        """Compute discrete Laplacian"""
        B, H, W = u.shape
        laplacian = torch.zeros_like(u)
        
        if H > 2 and W > 2:
            laplacian[:, 1:-1, 1:-1] = (
                u[:, 2:, 1:-1] + u[:, :-2, 1:-1] + 
                u[:, 1:-1, 2:] + u[:, 1:-1, :-2] - 
                4 * u[:, 1:-1, 1:-1]
            )
        return laplacian

    def compute_gradient_magnitude(self, u):
        """Compute gradient magnitude for edge enhancement"""
        B, H, W = u.shape
        grad_x = torch.zeros_like(u)
        grad_y = torch.zeros_like(u)
        
        # Compute gradients with central differences
        if W > 2:
            grad_x[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / 2
        if H > 2:
            grad_y[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / 2
        
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_mag

# --- Multi-Scale Patch Feature Extractor ---
class MultiScalePatchExtractor(nn.Module):
    def __init__(self, image_size=48):
        super().__init__()
        self.image_size = image_size
        
        # Different patch sizes for multi-scale analysis
        self.patch_sizes = [3, 5, 7, 9, 11]
        
        # Feature extractors for each scale
        self.patch_extractors = nn.ModuleList()
        for patch_size in self.patch_sizes:
            extractor = nn.Sequential(
                nn.Linear(patch_size * patch_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            self.patch_extractors.append(extractor)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, H, W)  # Remove channel dimension
        
        all_features = []
        
        for i, patch_size in enumerate(self.patch_sizes):
            # Extract patches
            stride = max(1, patch_size // 2)
            patches = self.extract_patches(x, patch_size, stride)
            
            if patches.size(1) > 0:  # If we have patches
                # Process patches through extractor
                patch_features = self.patch_extractors[i](patches)
                
                # Global pooling over patches
                global_features = patch_features.mean(dim=1)  # (B, 8)
                all_features.append(global_features)
        
        if all_features:
            return torch.cat(all_features, dim=1)
        else:
            return torch.zeros(B, len(self.patch_sizes) * 8, device=x.device)
    
    def extract_patches(self, x, patch_size, stride):
        """Extract patches from image"""
        B, H, W = x.shape
        patches = []
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = x[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch.reshape(B, -1))
        
        if patches:
            return torch.stack(patches, dim=1)  # (B, num_patches, patch_size^2)
        else:
            return torch.zeros(B, 0, patch_size * patch_size, device=x.device)

# --- Spatial Statistics Extractor ---
class SpatialStatsExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, H, W)
        
        features = []
        
        # Global statistics
        features.extend([
            x.mean(dim=[1, 2]),
            x.std(dim=[1, 2]),
            x.min(dim=2)[0].min(dim=1)[0],
            x.max(dim=2)[0].max(dim=1)[0]
        ])
        
        # Quantiles
        x_flat = x.view(B, -1)
        for q in [0.25, 0.5, 0.75]:
            features.append(torch.quantile(x_flat, q, dim=1))
        
        # Regional statistics (divide into 3x3 grid)
        h_step, w_step = H // 3, W // 3
        for i in range(3):
            for j in range(3):
                h_start, h_end = i * h_step, (i + 1) * h_step if i < 2 else H
                w_start, w_end = j * w_step, (j + 1) * w_step if j < 2 else W
                region = x[:, h_start:h_end, w_start:w_end]
                features.extend([
                    region.mean(dim=[1, 2]),
                    region.std(dim=[1, 2])
                ])
        
        return torch.stack(features, dim=1)

# --- Frequency Domain Features ---
class FrequencyFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, H, W)
        
        # Apply 2D FFT
        fft = torch.fft.fft2(x)
        fft_mag = torch.abs(fft)
        
        # Extract frequency domain features
        features = []
        
        # Low frequency energy (center region)
        center_h, center_w = H // 2, W // 2
        low_freq_region = fft_mag[:, center_h-2:center_h+3, center_w-2:center_w+3]
        features.append(low_freq_region.mean(dim=[1, 2]))
        
        # High frequency energy (corners)
        corner_size = min(H, W) // 8
        corners = [
            fft_mag[:, :corner_size, :corner_size],  # Top-left
            fft_mag[:, :corner_size, -corner_size:],  # Top-right
            fft_mag[:, -corner_size:, :corner_size],  # Bottom-left
            fft_mag[:, -corner_size:, -corner_size:]  # Bottom-right
        ]
        
        for corner in corners:
            features.append(corner.mean(dim=[1, 2]))
        
        # Frequency magnitude statistics
        features.extend([
            fft_mag.mean(dim=[1, 2]),
            fft_mag.std(dim=[1, 2])
        ])
        
        return torch.stack(features, dim=1)

# --- Advanced Classifier Architecture ---
class AdvancedEmotionClassifier(nn.Module):
    def __init__(self, use_pde=True):
        super().__init__()
        
        # PDE processing (optional)
        self.use_pde = use_pde
        if use_pde:
            self.pde_layer = AdvancedPDELayer(size=48, num_steps=2)
        
        # Feature extractors
        self.patch_extractor = MultiScalePatchExtractor(image_size=48)
        self.stats_extractor = SpatialStatsExtractor()
        self.freq_extractor = FrequencyFeatureExtractor()
        
        # Calculate feature dimensions
        patch_features = len([3, 5, 7, 9, 11]) * 8  # 5 scales * 8 features = 40
        stats_features = 4 + 3 + 18  # global + quantiles + regional = 25
        freq_features = 7  # 1 low + 4 corners + 2 stats = 7
        total_features = patch_features + stats_features + freq_features  # 72
        
        # Advanced classifier with attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(total_features, total_features),
            nn.Tanh(),
            nn.Linear(total_features, total_features),
            nn.Sigmoid()
        )
        
        # Main classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            ResidualBlock(256, 256, 0.3),
            ResidualBlock(256, 128, 0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 7)
        )
        
        print(f"Initialized AdvancedEmotionClassifier with {total_features} features, PDE: {use_pde}")
    
    def forward(self, x):
        # Optional PDE processing
        if self.use_pde:
            x = self.pde_layer(x)
        
        # Extract multiple types of features
        patch_features = self.patch_extractor(x)
        stats_features = self.stats_extractor(x)
        freq_features = self.freq_extractor(x)
        
        # Combine all features
        all_features = torch.cat([patch_features, stats_features, freq_features], dim=1)
        
        # Apply attention to features
        attention_weights = self.feature_attention(all_features)
        attended_features = all_features * attention_weights
        
        # Classify
        output = self.classifier(attended_features)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        
        out += identity
        out = F.relu(out)
        return out

# --- Enhanced Dataset with Better Augmentation ---
class EnhancedFaceDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []

        # Emotion mapping
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}

        # Load dataset
        self._load_data(data_path)
        print(f"Loaded {'train' if train else 'test'} dataset: {len(self.labels)} samples")
        self._print_distribution()

    def _load_data(self, data_path):
        """Load data from directory structure"""
        if os.path.isfile(data_path) and data_path.endswith('.csv'):
            self._load_csv(data_path)
        else:
            self._load_directory(data_path)

    def _load_csv(self, csv_path):
        """Load from CSV file"""
        df = pd.read_csv(csv_path)
        
        if 'Usage' in df.columns:
            if self.train:
                df = df[df['Usage'] == 'Training']
            else:
                df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]
        
        # Use more data for better training
        if len(df) > 10000 and self.train:
            df = df.sample(n=10000, random_state=42)
            print(f"Sampled {len(df)} training examples")
        elif len(df) > 3000 and not self.train:
            df = df.sample(n=3000, random_state=42)
            print(f"Sampled {len(df)} test examples")
        
        self.data = df
        
    def _load_directory(self, data_path):
        """Load from directory structure"""
        all_images = []
        all_labels = []
        
        # Load more samples per class for better training
        samples_per_class = 800 if self.train else 200
        
        for emotion, idx in self.emotion_to_idx.items():
            emotion_path = os.path.join(data_path, emotion)
            if os.path.exists(emotion_path):
                images = [f for f in os.listdir(emotion_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(images) > samples_per_class:
                    images = images[:samples_per_class]
                    
                for img in images:
                    all_images.append(os.path.join(emotion_path, img))
                    all_labels.append(idx)
                    
                print(f"Found {len(images)} images for {emotion}")
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        if len(all_images) > 0:
            train_imgs, test_imgs, train_labels, test_labels = train_test_split(
                all_images, all_labels, test_size=0.2, random_state=42, 
                stratify=all_labels
            )
            
            if self.train:
                self.image_paths = train_imgs
                self.labels = train_labels
            else:
                self.image_paths = test_imgs
                self.labels = test_labels

    def _print_distribution(self):
        """Print class distribution"""
        if hasattr(self, 'data'):
            dist = self.data['emotion'].value_counts().sort_index()
            print("Class distribution:", dict(dist))
        else:
            unique, counts = np.unique(self.labels, return_counts=True)
            dist = {self.idx_to_emotion[u]: c for u, c in zip(unique, counts)}
            print("Class distribution:", dist)

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        return len(self.labels)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            row = self.data.iloc[idx]
            if 'pixels' in row:
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                image = Image.fromarray(pixels.reshape(48, 48), mode='L')
            else:
                image = Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8), mode='L')
            label = row['emotion']
        else:
            try:
                image = Image.open(self.image_paths[idx]).convert('L')
                image = image.resize((48, 48))
            except:
                image = Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8), mode='L')
            label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Dataset Download Functions ---
def download_and_setup_dataset():
    """Download dataset"""
    print("📥 Downloading dataset...")
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    return path

def find_data_path(dataset_path):
    """Find the actual data path"""
    # Look for CSV files first
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and 'fer' in file.lower():
                return os.path.join(root, file)
    
    # Look for image directories
    for root, dirs, files in os.walk(dataset_path):
        if any(d in ['angry', 'happy', 'sad', 'neutral'] for d in dirs):
            return root
    
    return dataset_path

# --- Advanced Training Function ---
def train_advanced_model(data_path, use_pde=True, epochs=15):
    """Advanced training with better optimization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enhanced data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Create datasets
    print("Creating enhanced datasets...")
    train_dataset = EnhancedFaceDataset(data_path, train=True, transform=transform_train)
    test_dataset = EnhancedFaceDataset(data_path, train=False, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Test data loading
    sample_batch = next(iter(train_loader))
    print(f"Batch shape: {sample_batch[0].shape}, Labels shape: {sample_batch[1].shape}")

    # Initialize model
    model = AdvancedEmotionClassifier(use_pde=use_pde).to(device)

    # Advanced optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.patch_extractor.parameters(), 'lr': 0.002},
        {'params': model.stats_extractor.parameters(), 'lr': 0.001},
        {'params': model.freq_extractor.parameters(), 'lr': 0.001},
        {'params': model.feature_attention.parameters(), 'lr': 0.001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ] + ([{'params': model.pde_layer.parameters(), 'lr': 0.0005}] if use_pde else []), 
    weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Starting advanced training for {epochs} epochs...")

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 30 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_advanced_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping. Best validation accuracy: {best_val_acc:.2f}%")
            break

        print("-" * 60)

    # Load best model
    model.load_state_dict(torch.load('best_advanced_model.pth'))
    return model, test_loader

# --- Evaluation Function ---
def evaluate_advanced_model(model, test_loader):
    """Advanced evaluation with detailed analysis"""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100. * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # Calculate confidence statistics
    all_probs = np.array(all_probs)
    confidences = np.max(all_probs, axis=1)
    avg_confidence = np.mean(confidences)
    
    print(f"\n🎯 Advanced Model Results:")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Improvement over random: {accuracy - 100/7:.2f}%")
    
    # Detailed analysis
    print("\nPrediction distribution:")
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    for pred, count in zip(unique_preds, pred_counts):
        print(f"  {emotion_labels[pred]}: {count} ({100*count/len(all_preds):.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels, zero_division=0))
    
    return accuracy

# --- Main execution ---
def run_advanced_emotion_recognition():
    """Run advanced emotion recognition comparison"""
    print("🚀 Advanced PDE-based Emotion Recognition")
    print("=" * 60)

    # Download dataset
    dataset_path = download_and_setup_dataset()
    data_path = find_data_path(dataset_path)
    print(f"Using data path: {data_path}")

    # Test advanced model without PDE first
    print("\n=== Testing Advanced Model (No PDE) ===")
    model_no_pde, test_loader = train_advanced_model(data_path, use_pde=False, epochs=12)
    acc_no_pde = evaluate_advanced_model(model_no_pde, test_loader)

    # Test advanced model with PDE
    print("\n=== Testing Advanced Model (With PDE) ===")
    model_with_pde, _ = train_advanced_model(data_path, use_pde=True, epochs=12)
    acc_with_pde = evaluate_advanced_model(model_with_pde, test_loader)

    print(f"\n📊 Final Comparison:")
    print(f"Advanced Model (No PDE): {acc_no_pde:.2f}%")
    print(f"Advanced Model (With PDE): {acc_with_pde:.2f}%")
    print(f"Random Baseline: {100/7:.2f}%")
    print(f"PDE Improvement: {acc_with_pde - acc_no_pde:.2f}%")

    return model_with_pde, test_loader

if __name__ == "__main__":
    run_advanced_emotion_recognition()

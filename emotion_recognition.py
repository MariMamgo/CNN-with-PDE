# Enhanced PDE Diffusion Neural Network for Face Expression Recognition
# Improved version with better feature extraction and architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from collections import Counter

# --- Improved PDE Diffusion Layer ---
class ImprovedDiffusionLayer(nn.Module):
    def __init__(self, size=48, dt=0.01, dx=1.0, dy=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.num_steps = num_steps

        # Learnable diffusion coefficients - made smaller and more focused
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        
        # Learnable anisotropic factors
        self.aniso_x = nn.Parameter(torch.ones(1))
        self.aniso_y = nn.Parameter(torch.ones(1))
        
        # Edge enhancement parameters
        self.edge_strength = nn.Parameter(torch.ones(1) * 0.1)
        
        print(f"Initialized ImprovedDiffusionLayer with size={size}, steps={num_steps}")

    def forward(self, u):
        B, C, H, W = u.shape
        u = u.squeeze(1)  # Remove channel dimension
        
        # Store original for residual connection
        u_orig = u.clone()
        
        for step in range(self.num_steps):
            u = self.diffusion_step(u)
        
        # Residual connection with learnable mixing
        alpha_mix = torch.sigmoid(self.edge_strength)
        u = alpha_mix * u + (1 - alpha_mix) * u_orig
        
        return u.unsqueeze(1)

    def diffusion_step(self, u):
        # Compute gradients
        grad_x = torch.zeros_like(u)
        grad_y = torch.zeros_like(u)
        
        # Central differences for interior points
        grad_x[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * self.dx)
        grad_y[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * self.dy)
        
        # Boundary conditions (zero gradient)
        grad_x[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / self.dx
        grad_x[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / self.dx
        grad_y[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / self.dy
        grad_y[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / self.dy
        
        # Compute divergence of diffusion flux
        flux_x = self.alpha * self.aniso_x * grad_x
        flux_y = self.beta * self.aniso_y * grad_y
        
        div_flux = torch.zeros_like(u)
        div_flux[:, :, 1:-1] += (flux_x[:, :, 2:] - flux_x[:, :, :-2]) / (2 * self.dx)
        div_flux[:, 1:-1, :] += (flux_y[:, 2:, :] - flux_y[:, :-2, :]) / (2 * self.dy)
        
        # Boundary handling
        div_flux[:, :, 0] += (flux_x[:, :, 1] - flux_x[:, :, 0]) / self.dx
        div_flux[:, :, -1] += (flux_x[:, :, -1] - flux_x[:, :, -2]) / self.dx
        div_flux[:, 0, :] += (flux_y[:, 1, :] - flux_y[:, 0, :]) / self.dy
        div_flux[:, -1, :] += (flux_y[:, -1, :] - flux_y[:, -2, :]) / self.dy
        
        # Update
        u_new = u + self.dt * div_flux
        
        return u_new

# --- Multi-Scale Feature Extractor ---
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_size=48):
        super().__init__()
        self.input_size = input_size
        
        # Patch-based features at different scales
        self.patch_sizes = [3, 5, 7, 9]
        self.patch_features = nn.ModuleList([
            self._create_patch_extractor(ps) for ps in self.patch_sizes
        ])
        
        # Statistical feature extractors
        self.spatial_stats = SpatialStatisticsExtractor(input_size)
        self.gradient_features = GradientFeatureExtractor(input_size)
        
        # Learnable feature weights
        self.feature_weights = nn.Parameter(torch.ones(len(self.patch_sizes) + 2))
        
    def _create_patch_extractor(self, patch_size):
        # Create a simple patch-based feature extractor
        stride = max(1, patch_size // 2)
        return nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=stride, padding=patch_size//2),
            nn.Linear(patch_size * patch_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, H, W)  # Remove channel dimension
        
        features = []
        
        # Extract patch features at different scales
        for i, extractor in enumerate(self.patch_features):
            x_input = x.unsqueeze(1)  # Add channel back for unfold
            patches = extractor[0](x_input)  # Unfold
            patches = patches.transpose(1, 2)  # (B, num_patches, patch_dim)
            patch_features = extractor[1:](patches)  # Apply linear layers
            # Global average pooling over patches
            patch_features = patch_features.mean(dim=1)  # (B, 16)
            features.append(patch_features * self.feature_weights[i])
        
        # Extract statistical features
        stats = self.spatial_stats(x)
        features.append(stats * self.feature_weights[-2])
        
        # Extract gradient features
        grad_feats = self.gradient_features(x)
        features.append(grad_feats * self.feature_weights[-1])
        
        # Concatenate all features
        return torch.cat(features, dim=1)

class SpatialStatisticsExtractor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
    def forward(self, x):
        B, H, W = x.shape
        
        # Global statistics
        mean_val = x.mean(dim=[1, 2])
        std_val = x.std(dim=[1, 2])
        min_val = x.min(dim=2)[0].min(dim=1)[0]
        max_val = x.max(dim=2)[0].max(dim=1)[0]
        
        # Regional statistics (divide image into quadrants)
        h_mid, w_mid = H // 2, W // 2
        regions = [
            x[:, :h_mid, :w_mid],      # Top-left
            x[:, :h_mid, w_mid:],      # Top-right
            x[:, h_mid:, :w_mid],      # Bottom-left
            x[:, h_mid:, w_mid:]       # Bottom-right
        ]
        
        regional_means = torch.stack([r.mean(dim=[1, 2]) for r in regions], dim=1)
        regional_stds = torch.stack([r.std(dim=[1, 2]) for r in regions], dim=1)
        
        # Combine features
        features = torch.cat([
            mean_val.unsqueeze(1), std_val.unsqueeze(1),
            min_val.unsqueeze(1), max_val.unsqueeze(1),
            regional_means, regional_stds
        ], dim=1)
        
        return features

class GradientFeatureExtractor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
    def forward(self, x):
        B, H, W = x.shape
        
        # Compute gradients
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        
        grad_x[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        grad_y[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Edge statistics
        edge_density = (grad_mag > grad_mag.mean(dim=[1, 2], keepdim=True)).float().mean(dim=[1, 2])
        edge_strength = grad_mag.mean(dim=[1, 2])
        edge_max = grad_mag.max(dim=2)[0].max(dim=1)[0]
        
        # Directional features
        grad_angle = torch.atan2(grad_y, grad_x + 1e-8)
        # Histogram of gradient directions (simplified)
        angle_features = []
        for i in range(4):  # 4 bins for angle histogram
            angle_min = -np.pi + i * np.pi/2
            angle_max = -np.pi + (i+1) * np.pi/2
            mask = (grad_angle >= angle_min) & (grad_angle < angle_max)
            angle_features.append((mask * grad_mag).sum(dim=[1, 2]) / (mask.sum(dim=[1, 2]) + 1e-8))
        
        angle_features = torch.stack(angle_features, dim=1)
        
        features = torch.cat([
            edge_density.unsqueeze(1), edge_strength.unsqueeze(1),
            edge_max.unsqueeze(1), angle_features
        ], dim=1)
        
        return features

# --- Improved Classifier with Residual Connections ---
class ImprovedFaceExpressionClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3, dx=1.0, dy=1.0):
        super().__init__()
        
        # PDE diffusion layer
        self.diffusion = ImprovedDiffusionLayer(size=48, dx=dx, dy=dy, num_steps=3)
        
        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(input_size=48)
        
        # Calculate feature dimensions
        # Patch features: 4 scales * 16 features = 64
        # Spatial stats: 4 global + 8 regional = 12
        # Gradient features: 3 edge + 4 directional = 7
        total_features = 64 + 12 + 7
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(512, 512, dropout_rate),
            ResidualBlock(512, 256, dropout_rate),
            ResidualBlock(256, 256, dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 7)  # 7 emotion classes
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Apply PDE diffusion
        x_diffused = self.diffusion(x)
        
        # Extract multi-scale features
        features = self.feature_extractor(x_diffused)
        
        # Classify
        output = self.classifier(features)
        
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

# --- Improved Dataset with Better Preprocessing ---
class ImprovedFaceExpressionDataset(Dataset):
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
            self.data = pd.read_csv(data_path)
            self._load_from_csv()
        else:
            self._load_from_directory(data_path)

        print(f"Loaded {'train' if train else 'test'} dataset with {len(self.image_paths)} samples")
        self._print_class_distribution()

    def _load_from_csv(self):
        """Load dataset from CSV file with better filtering"""
        if 'Usage' in self.data.columns:
            if self.train:
                self.data = self.data[self.data['Usage'] == 'Training']
            else:
                self.data = self.data[self.data['Usage'].isin(['PublicTest', 'PrivateTest'])]
        else:
            # Stratified split to maintain class balance
            from sklearn.model_selection import train_test_split
            if 'emotion' in self.data.columns:
                train_data, test_data = train_test_split(
                    self.data, test_size=0.2, stratify=self.data['emotion'], 
                    random_state=42
                )
                self.data = train_data if self.train else test_data

    def _load_from_directory(self, data_path):
        """Improved directory loading with better error handling"""
        print(f"Loading from directory: {data_path}")
        
        emotion_folders = []
        for emotion in self.emotion_to_idx.keys():
            emotion_path = os.path.join(data_path, emotion)
            if os.path.exists(emotion_path):
                emotion_folders.append((emotion, emotion_path))

        if not emotion_folders:
            # Try numbered folders
            for i, emotion in enumerate(self.emotion_to_idx.keys()):
                emotion_path = os.path.join(data_path, str(i))
                if os.path.exists(emotion_path):
                    emotion_folders.append((emotion, emotion_path))

        if not emotion_folders:
            raise ValueError(f"No emotion folders found in {data_path}")

        print(f"Found emotion folders: {[folder[0] for folder in emotion_folders]}")

        # Load images with validation
        for emotion, emotion_path in emotion_folders:
            emotion_idx = self.emotion_to_idx[emotion]
            image_files = []
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_files.extend([f for f in os.listdir(emotion_path) 
                                  if f.lower().endswith(ext)])

            # Validate images
            valid_images = []
            for image_file in image_files:
                image_path = os.path.join(emotion_path, image_file)
                try:
                    # Quick validation
                    with Image.open(image_path) as img:
                        img.verify()
                    valid_images.append(image_file)
                except:
                    continue

            print(f"Found {len(valid_images)} valid images for emotion '{emotion}'")

            for image_file in valid_images:
                image_path = os.path.join(emotion_path, image_file)
                self.image_paths.append(image_path)
                self.labels.append(emotion_idx)

        # Convert to numpy arrays and stratified split
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        from sklearn.model_selection import train_test_split
        if len(np.unique(self.labels)) > 1:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                self.image_paths, self.labels, test_size=0.2, 
                stratify=self.labels, random_state=42
            )
            
            if self.train:
                self.image_paths = train_paths
                self.labels = train_labels
            else:
                self.image_paths = test_paths
                self.labels = test_labels

    def _print_class_distribution(self):
        """Print class distribution and return class weights"""
        if hasattr(self, 'data'):
            if 'emotion' in self.data.columns:
                emotion_counts = self.data['emotion'].value_counts().sort_index()
                print("Emotion distribution:", dict(emotion_counts))
                return compute_class_weight('balanced', classes=np.unique(self.data['emotion']), 
                                          y=self.data['emotion'])
        else:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            distribution = {self.idx_to_emotion[label]: count 
                          for label, count in zip(unique_labels, counts)}
            print("Emotion distribution:", distribution)
            return compute_class_weight('balanced', classes=unique_labels, y=self.labels)

    def get_class_weights(self):
        """Get class weights for handling imbalanced dataset"""
        if hasattr(self, 'data'):
            if 'emotion' in self.data.columns:
                return compute_class_weight('balanced', classes=np.unique(self.data['emotion']), 
                                          y=self.data['emotion'])
        else:
            unique_labels = np.unique(self.labels)
            return compute_class_weight('balanced', classes=unique_labels, y=self.labels)

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            row = self.data.iloc[idx]
            if 'pixels' in row:
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                image = pixels.reshape(48, 48)
                image = Image.fromarray(image, mode='L')
            else:
                image_path = row['image_path'] if 'image_path' in row else row[0]
                image = Image.open(image_path).convert('L')
            
            label = row['emotion'] if 'emotion' in row else row['label']
        else:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            try:
                image = Image.open(image_path).convert('L')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode='L')

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Improved Training Function ---
def train_improved_model(data_path, dx=1.0, dy=1.0, epochs=30):
    """Improved training function with better optimization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Improved data transforms
    transform_train = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Add some noise for robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Better normalization
    ])

    transform_test = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Create datasets
    train_dataset = ImprovedFaceExpressionDataset(data_path, train=True, transform=transform_train)
    test_dataset = ImprovedFaceExpressionDataset(data_path, train=False, transform=transform_test)

    # Handle class imbalance with weighted sampling
    class_weights = train_dataset.get_class_weights()
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                           num_workers=0, pin_memory=True)

    # Initialize model
    model = ImprovedFaceExpressionClassifier(dx=dx, dy=dy, dropout_rate=0.4).to(device)
    
    # Use weighted loss for class imbalance
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    # Improved optimizer with different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': model.diffusion.parameters(), 'lr': 0.001},
        {'params': model.feature_extractor.parameters(), 'lr': 0.002},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=1e-4)

    # Better learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training loop with early stopping
    best_acc = 0
    patience = 8
    patience_counter = 0

    print("Starting improved training...")
    
    for epoch in range(epochs):
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
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                output = model(imgs)
                val_loss += criterion(output, labels).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"           Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_acc:.2f}%")
            break
            
        print("-" * 80)

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, test_loader

# --- Updated evaluation function ---
def evaluate_improved_model(model, test_loader):
    """Evaluation function for improved model"""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            
            # Get predictions and confidences
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            confidence = probs.max(dim=1)[0]
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    test_acc = 100. * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    avg_confidence = np.mean(all_confidences)
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))

    # Visualization
    plt.figure(figsize=(20, 12))
    
    # Sample predictions
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        confidences = F.softmax(outputs, dim=1).max(dim=1)[0]

        for i in range(min(8, len(images))):
            # Original image
            plt.subplot(3, 8, i + 1)
            img = images[i, 0].cpu().numpy()
            # Denormalize
            img = img * 0.229 + 0.485
            img = np.clip(img, 0, 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f"True: {emotion_labels[labels[i]]}", fontsize=8)

            # Prediction with confidence
            plt.subplot(3, 8, i + 9)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {emotion_labels[predicted[i]]}\nConf: {confidences[i]:.2f}", 
                     color=color, fontsize=8)

            # Processed by diffusion
            plt.subplot(3, 8, i + 17)
            diffused = model.diffusion(images[i:i+1]).squeeze().cpu().numpy()
            diffused = diffused * 0.229 + 0.485
            diffused = np.clip(diffused, 0, 1)
            plt.imshow(diffused, cmap='gray')
            plt.axis('off')
            plt.title("After PDE", fontsize=8)

    # Confusion Matrix
    plt.subplot(3, 8, 24)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=emotion_labels, yticklabels=emotion_labels,
               cbar=False)
    plt.title("Confusion Matrix", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.suptitle(f'Improved PDE-based Emotion Recognition (Test Acc: {test_acc:.2f}%)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return test_acc

# --- Dataset Download and Setup Functions ---
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

# --- Main execution ---
def run_improved_emotion_recognition():
    """Run the improved emotion recognition system"""
    print("🚀 Starting Improved PDE-based Emotion Recognition System")
    print("=" * 60)

    # Download and setup dataset
    dataset_path = download_and_setup_dataset()
    
    # Find data path
    data_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))

    if data_files:
        data_path = data_files[0]
        print(f"Using CSV file: {data_path}")
    else:
        # Use directory format
        print("Looking for image directories...")
        train_dir, val_dir = find_data_directories(dataset_path)
        data_path = train_dir if train_dir else dataset_path
        print(f"Using directory: {data_path}")

    print(f"\n🧠 Training Improved Model...")
    print("=" * 60)

    # Train improved model
    model, test_loader = train_improved_model(data_path, dx=1.0, dy=1.0, epochs=30)

    print("\n📊 Evaluating Improved Model...")
    print("=" * 60)
    test_acc = evaluate_improved_model(model, test_loader)

    print(f"\n✅ Training completed! Final test accuracy: {test_acc:.2f}%")
    return model, test_loader

# Run the improved system
if __name__ == "__main__":
    model, test_loader = run_improved_emotion_recognition()

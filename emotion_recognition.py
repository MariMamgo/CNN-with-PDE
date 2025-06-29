# Simplified PDE Emotion Recognition - Focus on Core Functionality
# Debugging version to identify and fix issues

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

# --- Simple PDE Diffusion Layer ---
class SimplePDELayer(nn.Module):
    def __init__(self, size=48, num_steps=3):
        super().__init__()
        self.size = size
        self.num_steps = num_steps
        
        # Simple learnable diffusion coefficient
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
        
        print(f"Initialized SimplePDELayer with {num_steps} steps")

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.squeeze(1)  # Remove channel dimension: (B, H, W)
        
        for step in range(self.num_steps):
            x = self.diffusion_step(x)
        
        return x.unsqueeze(1)  # Add channel back: (B, 1, H, W)

    def diffusion_step(self, u):
        """Simple diffusion step using finite differences"""
        B, H, W = u.shape
        u_new = u.clone()
        
        # Simple diffusion: u_new = u + dt * alpha * laplacian(u)
        dt = 0.01
        alpha = torch.clamp(self.diffusion_coeff, 0.01, 0.5)  # Keep stable
        
        # Compute Laplacian using simple finite differences
        # Interior points
        if H > 2 and W > 2:
            laplacian = (u[:, 2:, 1:-1] + u[:, :-2, 1:-1] + 
                        u[:, 1:-1, 2:] + u[:, 1:-1, :-2] - 
                        4 * u[:, 1:-1, 1:-1])
            
            u_new[:, 1:-1, 1:-1] = u[:, 1:-1, 1:-1] + dt * alpha * laplacian
        
        return u_new

# --- Simple Baseline Classifier ---
class BaselineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple fully connected network
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(48 * 48, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)

# --- PDE-Enhanced Classifier ---
class SimplePDEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pde_layer = SimplePDELayer(size=48, num_steps=3)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(48 * 48, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )
        
    def forward(self, x):
        x = self.pde_layer(x)
        x = self.flatten(x)
        return self.classifier(x)

# --- Simple Dataset Class ---
class SimpleFaceDataset(Dataset):
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
        print(f"CSV columns: {df.columns.tolist()}")
        
        if 'Usage' in df.columns:
            if self.train:
                df = df[df['Usage'] == 'Training']
            else:
                df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]
        
        # Sample subset for debugging
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
            print(f"Sampled {len(df)} examples for faster debugging")
        
        self.data = df
        
    def _load_directory(self, data_path):
        """Load from directory structure"""
        all_images = []
        all_labels = []
        
        # Look for emotion folders
        for emotion, idx in self.emotion_to_idx.items():
            emotion_path = os.path.join(data_path, emotion)
            if os.path.exists(emotion_path):
                images = [f for f in os.listdir(emotion_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Limit samples per class for debugging
                if len(images) > 500:
                    images = images[:500]
                    
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
                # Handle other CSV formats
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

# --- Training Function ---
def train_simple_model(data_path, model_type='baseline', epochs=10):
    """Simple training function with debugging"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simple transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleFaceDataset(data_path, train=True, transform=transform)
    test_dataset = SimpleFaceDataset(data_path, train=False, transform=transform)

    # Check if we have data
    if len(train_dataset) == 0:
        print("ERROR: No training data found!")
        return None, None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Test data loading
    print("Testing data loading...")
    try:
        sample_batch = next(iter(train_loader))
        print(f"Batch shape: {sample_batch[0].shape}, Labels shape: {sample_batch[1].shape}")
        print(f"Sample labels: {sample_batch[1][:5]}")
        print(f"Pixel value range: [{sample_batch[0].min():.3f}, {sample_batch[0].max():.3f}]")
    except Exception as e:
        print(f"ERROR in data loading: {e}")
        return None, None

    # Initialize model
    if model_type == 'pde':
        model = SimplePDEClassifier().to(device)
        print("Using PDE-enhanced model")
    else:
        model = BaselineClassifier().to(device)
        print("Using baseline model")

    # Simple optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training for {epochs} epochs...")

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
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

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
        print("-" * 60)

        # Early stopping if no improvement
        if epoch > 3 and val_acc < 20:
            print("Poor performance detected - stopping early for debugging")
            break

    return model, test_loader

# --- Evaluation Function ---
def evaluate_model(model, test_loader):
    """Simple evaluation"""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_labels = []
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Random chance: {100/7:.2f}%")
    
    # Print detailed results
    print("\nPrediction distribution:")
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    for pred, count in zip(unique_preds, pred_counts):
        print(f"  Class {pred} ({emotion_labels[pred]}): {count} predictions ({100*count/len(all_preds):.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels, zero_division=0))
    
    return accuracy

# --- Debug Function ---
def debug_data_and_model():
    """Debug function to test data loading and model basics"""
    print("🔍 Debugging Data and Model...")
    
    # Download dataset
    dataset_path = download_and_setup_dataset()
    data_path = find_data_path(dataset_path)
    print(f"Using data path: {data_path}")
    
    # Test baseline model first
    print("\n=== Testing Baseline Model ===")
    baseline_model, test_loader = train_simple_model(data_path, model_type='baseline', epochs=5)
    
    if baseline_model is not None:
        baseline_acc = evaluate_model(baseline_model, test_loader)
        
        if baseline_acc > 20:  # If baseline works reasonably
            print("\n=== Testing PDE Model ===")
            pde_model, _ = train_simple_model(data_path, model_type='pde', epochs=5)
            if pde_model is not None:
                pde_acc = evaluate_model(pde_model, test_loader)
                
                print(f"\n📊 Results Summary:")
                print(f"Baseline Accuracy: {baseline_acc:.2f}%")
                print(f"PDE Accuracy: {pde_acc:.2f}%")
                print(f"Random Chance: {100/7:.2f}%")
        else:
            print("⚠️ Baseline model performance is too low - check data loading")
    else:
        print("❌ Failed to create baseline model - check data path and format")

# --- Main execution ---
if __name__ == "__main__":
    debug_data_and_model()

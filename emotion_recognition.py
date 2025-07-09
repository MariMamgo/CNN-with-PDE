import kagglehub

# Download latest version
path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")

print("Path to dataset files:", path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# Import transforms explicitly
try:
    import torchvision.transforms as transforms
except ImportError:
    print("torchvision not available, defining basic ToTensor transform")
    class BasicTransforms:
        @staticmethod
        def ToTensor():
            def to_tensor(img):
                if isinstance(img, Image.Image):
                    return torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
                return img
            return to_tensor

        @staticmethod
        def Compose(transform_list):
            def compose(img):
                for transform in transform_list:
                    img = transform(img)
                return img
            return compose

    transforms = BasicTransforms()

#PDE Layer with Extended Fourier Series (updated alpha and beta functions)
class PDELayer(nn.Module):
    def __init__(self, Nx=48, Ny=48, Lx=1.0, Ly=1.0, T=0.01, dt=0.001):
        super().__init__()
        self.Nx, self.Ny, self.Lx, self.Ly = Nx, Ny, Lx, Ly
        self.T, self.dt = T, dt
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.Nt = int(T / dt)

        # Initialize with positive values (abs() will keep them positive)
        self.alpha_w1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_w2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_w3 = nn.Parameter(torch.tensor(0.1))


        self.beta_w1 = nn.Parameter(torch.tensor(0.3))
        self.beta_w2 = nn.Parameter(torch.tensor(0.2))
        self.beta_w3 = nn.Parameter(torch.tensor(0.1))

        self.register_buffer('x', torch.linspace(0, Lx, Nx))
        self.register_buffer('y', torch.linspace(0, Ly, Ny))

    def alpha(self, y_val):
        # Ensure ALL coefficients are positive using abs()
        fourier_terms = (
            torch.abs(self.alpha_w1) + 
            torch.abs(self.alpha_w2) * torch.sin(2 * torch.pi * y_val) +
            torch.abs(self.alpha_w3) * torch.cos(2 * torch.pi * y_val) 
        )
        return 0.5 * self.dt * fourier_terms / self.dx**2

    def beta(self, x_val):
        # Ensure ALL coefficients are positive using abs()
        fourier_terms = (
            torch.abs(self.beta_w1) + 
            torch.abs(self.beta_w2) * torch.cos(2 * torch.pi * x_val) +
            torch.abs(self.beta_w3) * torch.sin(2 * torch.pi * x_val) 
        )
        return self.dt * fourier_terms / self.dy**2

    def forward(self, u0):
        u = u0.squeeze(1)  # (B, 48, 48)
        B, Nx, Ny = u.shape
        u = F.pad(u, (1, 1, 1, 1), mode='reflect')  # (B, 50, 50)

        yy, xx = torch.meshgrid(self.y, self.x, indexing='ij')  # (48, 48)
        alpha_grid = self.alpha(yy).to(u.device)
        beta_grid = self.beta(xx).to(u.device)

        alpha_grid = alpha_grid.unsqueeze(0)
        beta_grid = beta_grid.unsqueeze(0)

        for _ in range(self.Nt):
            u_inner = u[:, 1:-1, 1:-1]
            u_xx = u[:, 2:, 1:-1] - 2 * u_inner + u[:, :-2, 1:-1]
            u_yy = u[:, 1:-1, 2:] - 2 * u_inner + u[:, 1:-1, :-2]
            u[:, 1:-1, 1:-1] = u_inner + alpha_grid * u_xx + beta_grid * u_yy

        return u[:, 1:-1, 1:-1].unsqueeze(1)  # (B, 1, 48, 48)

#Emotion Dataset Loader for Image Folders
class EmotionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Define emotion mapping
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }

        # Look for images in the specified split directory
        split_dir = os.path.join(root_dir, 'images', split)
        if not os.path.exists(split_dir):
            # Try alternative split names
            possible_splits = ['train', 'test', 'validation', 'val']
            split_dir = None
            for alt_split in possible_splits:
                alt_path = os.path.join(root_dir, 'images', alt_split)
                if os.path.exists(alt_path):
                    split_dir = alt_path
                    print(f"Using {alt_split} directory")
                    break

            if split_dir is None:
                # Check if images are directly organized by emotion
                images_dir = os.path.join(root_dir, 'images')
                subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
                print(f"Found subdirectories in images: {subdirs}")

                # Use the images directory directly and look for emotion folders
                for emotion_folder in subdirs:
                    if emotion_folder.lower() in self.emotion_to_idx:
                        emotion_path = os.path.join(images_dir, emotion_folder)
                        emotion_idx = self.emotion_to_idx[emotion_folder.lower()]

                        for img_file in os.listdir(emotion_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.images.append(os.path.join(emotion_path, img_file))
                                self.labels.append(emotion_idx)
                return

        # Load images from split directory organized by emotion
        for emotion_folder in os.listdir(split_dir):
            emotion_path = os.path.join(split_dir, emotion_folder)
            if os.path.isdir(emotion_path) and emotion_folder.lower() in self.emotion_to_idx:
                emotion_idx = self.emotion_to_idx[emotion_folder.lower()]

                for img_file in os.listdir(emotion_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(emotion_path, img_file))
                        self.labels.append(emotion_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale

        # Resize to 48x48 if needed
        if img.size != (48, 48):
            img = img.resize((48, 48))

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

#Updated Model
class DiffusionClassifier(nn.Module):
    def __init__(self, img_size=48, num_classes=7):
        super().__init__()
        self.pde = PDELayer(Nx=img_size, Ny=img_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pde(x)
        return self.classifier(x)


# Simplified training function - no regularization needed
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss.item()}")
            continue
            
        loss.backward()
        
        # Light gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    print(f"  alpha_w1={model.pde.alpha_w1.item():.4f}, alpha_w2={model.pde.alpha_w2.item():.4f}, alpha_w3={model.pde.alpha_w3.item():.4f}")
    print(f"  alpha_w4={model.pde.alpha_w4.item():.4f}, alpha_w5={model.pde.alpha_w5.item():.4f}")
    print(f"  beta_w1= {model.pde.beta_w1.item():.4f}, beta_w2= {model.pde.beta_w2.item():.4f}, beta_w3= {model.pde.beta_w3.item():.4f}")
    print(f"  beta_w4= {model.pde.beta_w4.item():.4f}, beta_w5= {model.pde.beta_w5.item():.4f}")
    
    return avg_loss
    
def evaluate(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100*correct/total:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the actual downloaded path instead of hardcoded Kaggle path
    dataset_path = path  # Use the path variable from kagglehub.dataset_download()
    print(f"Using dataset path: {dataset_path}")
    
    print(f"Checking files in {dataset_path}:")
    try:
        files = os.listdir(dataset_path)
        print("Available files:", files)

        # Check images directory structure
        images_path = os.path.join(dataset_path, 'images')
        if os.path.exists(images_path):
            print(f"Contents of images directory:")
            subdirs = os.listdir(images_path)
            print("Subdirectories:", subdirs)

            # Check deeper structure
            for subdir in subdirs[:3]:  # Check first 3 subdirectories
                subdir_path = os.path.join(images_path, subdir)
                if os.path.isdir(subdir_path):
                    contents = os.listdir(subdir_path)
                    print(f"Contents of {subdir}: {contents[:5]}...")  # Show first 5 items

    except FileNotFoundError:
        print(f"Directory {dataset_path} not found!")
        return

    transform = transforms.Compose([transforms.ToTensor()])

    try:
        # Try to create datasets with image folder structure
        train_dataset = EmotionDataset(dataset_path, split='train', transform=transform)
        print(f"Train dataset loaded: {len(train_dataset)} images")

        # Try different splits for test data
        test_splits = ['test', 'validation', 'val']
        test_dataset = None
        for test_split in test_splits:
            try:
                test_dataset = EmotionDataset(dataset_path, split=test_split, transform=transform)
                print(f"Test dataset loaded from '{test_split}': {len(test_dataset)} images")
                break
            except:
                continue

        if test_dataset is None or len(test_dataset) == 0:
            print("No test dataset found, using 20% of train data for testing")
            train_size = int(0.8 * len(train_dataset))
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, test_size]
            )

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if len(train_dataset) == 0:
        print("No training data found!")
        return

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = DiffusionClassifier(img_size=48, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Back to normal LR

    print(f"Starting training with {len(train_dataset)} training samples and {len(test_dataset)} test samples")

    for epoch in range(70):
        train(model, device, train_loader, optimizer, criterion, epoch)

    evaluate(model, device, test_loader)

    # Visualize 5 random predictions
    model.eval()
    indices = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))

    # Handle both regular dataset and subset (from random_split)
    if hasattr(test_dataset, 'dataset'):  # It's a subset
        images = torch.stack([test_dataset.dataset[test_dataset.indices[i]][0] for i in range(len(indices))]).to(device)
        labels = torch.tensor([test_dataset.dataset[test_dataset.indices[i]][1] for i in range(len(indices))])
    else:  # It's a regular dataset
        images = torch.stack([test_dataset[i][0] for i in indices]).to(device)
        labels = torch.tensor([test_dataset[i][1] for i in indices])

    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()

    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    fig, axes = plt.subplots(1, len(indices), figsize=(3*len(indices), 3))
    if len(indices) == 1:
        axes = [axes]

    for i in range(len(indices)):
        axes[i].imshow(images[i].squeeze().cpu(), cmap='gray')
        axes[i].set_title(f"Pred: {emotion_names[preds[i]]}\nTrue: {emotion_names[labels[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

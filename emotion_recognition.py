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
import kagglehub
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

        @staticmethod
        def RandomHorizontalFlip(p=0.5):
            def flip(img):
                if random.random() < p:
                    return transforms.functional.hflip(img)
                return img
            return flip

        @staticmethod
        def RandomRotation(degrees):
            def rotate(img):
                angle = random.uniform(-degrees, degrees)
                return transforms.functional.rotate(img, angle)
            return rotate

    transforms = BasicTransforms()

#PDE Layer (UNCHANGED - same mathematical model)
class PDELayer(nn.Module):
    def __init__(self, Nx=48, Ny=48, Lx=1.0, Ly=1.0, T=0.01, dt=0.001):
        super().__init__()
        self.Nx, self.Ny, self.Lx, self.Ly = Nx, Ny, Lx, Ly
        self.T, self.dt = T, dt
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.Nt = int(T / dt)

        self.alpha_w1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_w2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_w3 = nn.Parameter(torch.tensor(0.1))
        self.beta_w1 = nn.Parameter(torch.tensor(0.3))
        self.beta_w2 = nn.Parameter(torch.tensor(0.2))
        self.beta_w3 = nn.Parameter(torch.tensor(0.2))

        # Register as buffers so they move with the model to GPU
        self.register_buffer('x', torch.linspace(0, Lx, Nx))
        self.register_buffer('y', torch.linspace(0, Ly, Ny))

    def alpha(self, y_val):
        return 0.5 * self.dt * (self.alpha_w1 + self.alpha_w2 * torch.sin(2 * torch.pi * y_val) + self.alpha_w3 * torch.sin(4 * torch.pi * y_val)) / self.dx**2

    def beta(self, x_val):
        return self.dt * (self.beta_w1 + self.beta_w2 * torch.cos(2 * torch.pi * x_val)+ self.beta_w3 * torch.cos(4 * torch.pi * x_val)) / self.dy**2

    def forward(self, u0):
        u = u0.squeeze(1)  # (B, 48, 48)
        B, Nx, Ny = u.shape
        u = F.pad(u, (1, 1, 1, 1), mode='reflect')  # (B, 50, 50)

        yy, xx = torch.meshgrid(self.y, self.x, indexing='ij')  # (48, 48)
        alpha_grid = self.alpha(yy).unsqueeze(0)
        beta_grid = self.beta(xx).unsqueeze(0)

        for _ in range(self.Nt):
            u_inner = u[:, 1:-1, 1:-1]
            u_xx = u[:, 2:, 1:-1] - 2 * u_inner + u[:, :-2, 1:-1]
            u_yy = u[:, 1:-1, 2:] - 2 * u_inner + u[:, 1:-1, :-2]
            u[:, 1:-1, 1:-1] = u_inner + alpha_grid * u_xx + beta_grid * u_yy

        return u[:, 1:-1, 1:-1].unsqueeze(1)  # (B, 1, 48, 48)

#Improved Emotion Dataset with Data Augmentation
class EmotionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, balance_classes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.balance_classes = balance_classes

        # Define emotion mapping
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }

        # Look for images in the specified split directory
        split_dir = os.path.join(root_dir, 'images', split)
        if not os.path.exists(split_dir):
            # Try alternative split names
            possible_splits = ['test', 'val', 'train', 'validation']
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
                if os.path.exists(images_dir):
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

        # Balance classes if requested
        if self.balance_classes:
            self._balance_dataset()

    def _balance_dataset(self):
        # Count samples per class
        from collections import Counter
        label_counts = Counter(self.labels)
        min_count = min(label_counts.values())
        
        # Keep only min_count samples per class
        balanced_images = []
        balanced_labels = []
        class_counts = {i: 0 for i in range(7)}
        
        for img, label in zip(self.images, self.labels):
            if class_counts[label] < min_count:
                balanced_images.append(img)
                balanced_labels.append(label)
                class_counts[label] += 1
        
        self.images = balanced_images
        self.labels = balanced_labels
        print(f"Balanced dataset: {len(self.images)} images total")

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

#Improved Model with Better Classifier
class DiffusionClassifier(nn.Module):
    def __init__(self, img_size=48, num_classes=7, dropout_rate=0.3):
        super().__init__()
        self.pde = PDELayer(Nx=img_size, Ny=img_size)  # Keep PDE layer unchanged
        
        # Improved classifier with batch normalization and dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pde(x)  # PDE layer unchanged
        return self.classifier(x)

# Improved Training with Better Monitoring
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    print(f"  alpha_w1={model.pde.alpha_w1.item():.4f}, alpha_w2={model.pde.alpha_w2.item():.4f}, alpha_w3={model.pde.alpha_w3.item():.4f}")
    print(f"  beta_w1={model.pde.beta_w1.item():.4f}, beta_w2={model.pde.beta_w2.item():.4f}, beta_w3={model.pde.beta_w3.item():.4f}")
    
    return avg_loss, accuracy

def evaluate(model, device, test_loader, emotion_names):
    model.eval()
    correct, total = 0, 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=emotion_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy

def main(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_path = path
    print(f"Checking files in {dataset_path}:")
    
    # Improved data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    try:
        files = os.listdir(dataset_path)
        print("Available files:", files)

        # Check images directory structure
        images_path = os.path.join(dataset_path, 'images')
        if os.path.exists(images_path):
            print(f"Contents of images directory:")
            subdirs = os.listdir(images_path)
            print("Subdirectories:", subdirs)

        # Create datasets with improvements
        train_dataset = EmotionDataset(dataset_path, split='train', 
                                     transform=train_transform, balance_classes=True)
        print(f"Train dataset loaded: {len(train_dataset)} images")

        # Try different splits for test data
        test_splits = ['test', 'validation', 'val']
        test_dataset = None
        for test_split in test_splits:
            try:
                test_dataset = EmotionDataset(dataset_path, split=test_split, 
                                            transform=test_transform)
                if len(test_dataset) > 0:
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
        print("No training data found! Please check your dataset structure.")
        return

    # Improved data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # Improved model with dropout
    model = DiffusionClassifier(img_size=48, num_classes=7, dropout_rate=0.3).to(device)
    
    # Better optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)

    print(f"Starting training with {len(train_dataset)} training samples and {len(test_dataset)} test samples")

    # Training with early stopping
    best_accuracy = 0
    patience = 10
    patience_counter = 0
    
    train_losses = []
    train_accuracies = []

    for epoch in range(70):
        loss, acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(loss)
        train_accuracies.append(acc)
        
        scheduler.step()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            test_acc = evaluate(model, device, test_loader, emotion_names)
            
            # Early stopping
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    final_accuracy = evaluate(model, device, test_loader, emotion_names)

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.show()

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
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    print("Path to dataset files:", path)
    main(path)

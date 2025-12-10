import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import models


class PaperPreprocessing(object):
    """
    Applies the specific preprocessing steps defined in the paper:
    1. Histogram Equalization (Contrast)
    2. Bilateral Filter (Noise Reduction)
    """
    def __call__(self, img):
        # Convert PIL Image to Numpy Array (Grayscale)
        img_np = np.array(img)
        
        # Ensure it is grayscale (if not already)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # 1. Histogram Equalization
        # "histogram equalization is performed to increase the contrast" 
        img_eq = cv2.equalizeHist(img_np)
        
        # 2. Bilateral Filter
        img_blur = cv2.bilateralFilter(img_eq, 9, 75, 75)
        
        # Return as PIL Image (for PyTorch compatibility)
        return transforms.ToPILImage()(img_blur)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)



class GoatNet(nn.Module):
    def __init__(self, num_classes):
        super(GoatNet, self).__init__()
        
        # Feature Extractor (5 Blocks as per Paper)
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32x32 -> 16x16 
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 5: 4x4 -> 2x2 
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input size is now 128 channels * 2 * 2 pixels = 512
            nn.Linear(128 * 4 * 4, 256), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Dropout(0.0), 
            
            nn.Linear(256, num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PretrainedGoatNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PretrainedGoatNet, self).__init__()
        
        # 1. Load Pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # 2. Modify Input Layer for Grayscale
        # Original ResNet expects 3 channels (RGB). We change it to 1 channel.
        # We keep the same kernel size and stride as the original.
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. Modify Output Layer for Goat Classes
        # The original fc layer outputs 1000 classes (ImageNet). We change it to num_classes (10).
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(NUM_CLASSES=10):
    model = GoatNet(NUM_CLASSES).to(device)
    
    # Paper uses SGD 
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # 

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\nStarting Training...")
    
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        
        # Store History
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | Val Loss: {epoch_val_loss:.4f}")

   
if __name__ == "__main__":
    train_model()
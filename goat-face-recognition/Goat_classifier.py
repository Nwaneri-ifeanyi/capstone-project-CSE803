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

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

DATA_ROOT = '../../processed_dataset_highres' # Path to your high-res folders
SAVE_PATH = 'goat_recognition_model.pth'
IMAGE_SIZE = 64     # Paper specifies 64x64 input [cite: 118, 123]
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 0.01

# Check Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ====================================================================
# 2. CUSTOM PREPROCESSING (From Paper Methodology)
# ====================================================================

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
        # "bilateral filter used for noise reduction" 
        # Parameters (9, 75, 75) are standard for this operation
        img_blur = cv2.bilateralFilter(img_eq, 9, 75, 75)
        
        # Return as PIL Image (for PyTorch compatibility)
        return transforms.ToPILImage()(img_blur)

# ====================================================================
# 3. DATA TRANSFORMS & LOADING
# ====================================================================

# Training Transform: Includes Augmentation to fix overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to 64x64 [cite: 123]
    PaperPreprocessing(),                        # Apply HistEq + Bilateral 
    transforms.RandomHorizontalFlip(p=0.5),      # Augmentation: Flip
    transforms.RandomRotation(10),               # Augmentation: Rotate
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Validation Transform: No random augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    PaperPreprocessing(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

print("Loading dataset...")
# Load raw data first
full_dataset = ImageFolder(root=DATA_ROOT) 

# Split indices
train_size = int(0.8 * len(full_dataset)) # 80% Training [cite: 179]
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], 
                                        generator=torch.Generator().manual_seed(42))

# Apply the specific transforms to the subsets
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_data = TransformedSubset(train_subset, train_transform)
val_data = TransformedSubset(val_subset, val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(full_dataset.classes)
print(f"Classes: {NUM_CLASSES} | Train Images: {len(train_data)} | Val Images: {len(val_data)}")

# ====================================================================
# 4. MODEL ARCHITECTURE
# ====================================================================

class GoatNet(nn.Module):
    def __init__(self, num_classes):
        super(GoatNet, self).__init__()
        
        # Feature Extractor (4 Blocks) [cite: 124]
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization to prevent overfitting
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(256, num_classes) # [cite: 159]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ====================================================================
# 5. TRAINING LOOP
# ====================================================================

def train_model():
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

    # ====================================================================
    # 6. SAVING & PLOTTING
    # ====================================================================
    
    # Save Model Artifacts
    print(f"\nSaving model to {SAVE_PATH}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': {v: k for k, v in full_dataset.class_to_idx.items()},
        'image_size': IMAGE_SIZE,
        'num_classes': NUM_CLASSES
    }, SAVE_PATH)
    
    # Plot Results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.savefig('final_training_results.png')
    print("Training Complete. Results saved.")

if __name__ == "__main__":
    train_model()
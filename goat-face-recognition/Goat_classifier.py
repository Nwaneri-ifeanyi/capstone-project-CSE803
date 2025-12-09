
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
from scipy.ndimage import zoom # Used for resizing heatmaps

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision.datasets import ImageFolder 

# ====================================================================
# 1. DATA CONFIGURATION AND LOADING (Goat Identity Recognition)
# ====================================================================

DATA_ROOT = '/cse_project/1selected'
NUM_CLASSES = 10 
IMAGE_SIZE = 64 

# --- Define the Transformation Pipeline ---
goat_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

print("Loading Goat Identity Recognition Datasets...")

# Load Datasets using ImageFolder
try:
    full_dataset = ImageFolder(root=DATA_ROOT, transform=goat_transform)
except RuntimeError as e:
    print(f"\n[ERROR] Failed to load ImageFolder. Please ensure the DATA_ROOT is correct: {DATA_ROOT}")
    print(f"Original Error: {e}")
    # Fallback to a dummy dataset 
    class DummyDataset(Dataset):
        def __init__(self): 
            self.data = torch.randn(100, 1, 64, 64); 
            self.targets = torch.randint(0, NUM_CLASSES, (100,)); 
            self.classes = [f"Goat ID {i}" for i in range(NUM_CLASSES)]
            self.samples = [(self.data[i], self.targets[i]) for i in range(100)]
        def __len__(self): return 100
        def __getitem__(self, idx): return self.data[idx], self.targets[idx]
        @property
        def class_to_idx(self): return {cls: i for i, cls in enumerate(self.classes)}
    full_dataset = DummyDataset()
    print("Using dummy data for demonstration.")


# Split Data (80% Training, 20% Test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

if len(full_dataset) >= NUM_CLASSES:
    trainset, testset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    print(f"Data Loaded. Total Images: {len(full_dataset)}. Train: {len(trainset)}, Test: {len(testset)}")
else:
    print("Dataset too small for 80/20 split. Using full set for both train/test.")
    trainset = full_dataset
    testset = full_dataset


# ====================================================================
# 2. MODEL DEFINITION (Custom CNN from the Paper)
# ====================================================================

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # --- CNN Feature Extractor (4 MaxPool blocks, 128 channels output) ---
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 4x4 (128 channels)
        )
        
        # --- Classifier (Fully Connected Layers) ---
        self.fc1 = nn.Linear(128 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 256)         
        self.fc_final = nn.Linear(256, NUM_CLASSES)  
        
        # --- Visualization Layer (FIXED) ---
        self.conv_visualization = nn.Conv2d(128, NUM_CLASSES, 1) 


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_final(x)
        return x

    def transfer(self):
        # Skipped for simplicity of visualization, as discussed.
        pass

    def visualize(self, x):
        feature_map = self.features(x)
        x = self.conv_visualization(feature_map)
        return x

# ====================================================================
# 3. TRAINING AND EVALUATION FUNCTIONS (with Plotting Data Collection)
# ====================================================================

def calculate_accuracy(model, loader):
    """Calculates model accuracy and loss on a given DataLoader."""
    model.eval() 
    correct = 0
    total_loss = 0
    with torch.no_grad(): 
        for batch, label in loader:
            if isinstance(label, list): label = torch.tensor(label, dtype=torch.long)
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            loss = criterion(pred, label)
            total_loss += loss.item() * batch.size(0)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return acc, avg_loss

def train(model, train_loader, val_loader, num_epoch = 50): 
    """Trains the model and tracks history for plotting."""
    print("Start training...")
    
    # --- History tracking lists ---
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # -----------------------------
    
    for i in range(num_epoch):
        model.train() # Set model to training mode
        running_loss = []
        for batch, label in tqdm(train_loader, desc=f"Epoch {i+1}/{num_epoch}"):
            if isinstance(label, list): label = torch.tensor(label, dtype=torch.long)
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() 
            pred = model(batch) 
            loss = criterion(pred, label) 
            running_loss.append(loss.item())
            loss.backward() 
            optimizer.step() 
        
        # Calculate training metrics after each epoch
        train_loss = np.mean(running_loss)
        train_acc, _ = calculate_accuracy(model, train_loader)
        
        # Calculate validation metrics after each epoch
        val_acc, val_loss = calculate_accuracy(model, val_loader)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {i+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    print("Done!")
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Generates and saves the loss and accuracy plots."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # --- Plot Loss (Similar to Fig. 3 (C)) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # --- Plot Accuracy (Similar to Fig. 3 (D)) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy ')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('goat_training_history.png')
    print("Training history plots saved as 'goat_training_history.png'")
    # 

# --- DEVICE CONFIGURATION ---
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Configuring device: {device}")

# --- Execute Training, Collect History, and Plot ---
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 
num_epoch = 50 

# Run the training and collect history
train_loss_history, val_loss_history, train_acc_history, val_acc_history = \
    train(model, trainloader, testloader, num_epoch)

# Plot the collected history
plot_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

# Final evaluation on the test set
final_acc, _ = calculate_accuracy(model, testloader)
print(f"Final Test Accuracy: {final_acc:.4f}")

# ====================================================================
# 4. VISUALIZATION SECTION (for Class Activation Maps)
# ====================================================================

print("Preparing Visualization...")
model.transfer() 

model.eval()
target_idx = -1
target_img = None
target_label = None

try:
    goat_classes = full_dataset.classes
except AttributeError:
    goat_classes = [f"Goat ID {i}" for i in range(NUM_CLASSES)]


with torch.no_grad():
    for i in range(len(testset)):
        img, label = testset[i]
        
        if isinstance(img, tuple) and len(img) == 2:
            img, label = img

        img_tensor = img.unsqueeze(0).to(device)
        pred_scores = model(img_tensor)
        pred_label = torch.argmax(pred_scores, dim=1).item()
        
        if pred_label == label:
            target_idx = i
            target_img = img
            target_label = label
            print(f"Found correctly classified image at index {i}. Label: {goat_classes[label]}")
            break

if target_img is not None:
    img_tensor = target_img.unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model.visualize(img_tensor) 
    
    heatmaps = heatmaps.cpu().numpy()[0] 
    original_img = target_img.squeeze().cpu().numpy() 
    
    resized_heatmaps = []
    for i in range(NUM_CLASSES):
        resized_heatmaps.append(zoom(heatmaps[i], 16, order=1)) 
    resized_heatmaps = np.array(resized_heatmaps)
    
    # Plotting setup
    fig, axes = plt.subplots(4, 3, figsize=(15, 20)) 
    fig.suptitle(f"Activation Maps for Test Image #{target_idx}\nTrue Label: {goat_classes[target_label]}", fontsize=16)
    
    # Plot Original Image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title("Original Input (64x64)")
    axes[0, 0].axis('off')
    
    # Blank out extra spaces in the first row
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

    # Plot Heatmaps for all 10 classes
    for i in range(NUM_CLASSES):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        ax = axes[row, col]
        
        ax.imshow(original_img, cmap='gray', alpha=0.5) 
        im = ax.imshow(resized_heatmaps[i], cmap='jet', alpha=0.5) 
        
        class_name = goat_classes[i]
        title_text = f"Class: {class_name}"
        if i == target_label:
            ax.set_title(title_text + " (Target)", color='green', fontweight='bold')
        else:
            ax.set_title(title_text)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('goat_activation_visualization.png')
    print("Visualization saved as 'goat_activation_visualization.png'")
else:
    print("Could not find a correctly classified image.")
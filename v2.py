import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths relative to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMILE_DIR = os.path.join(BASE_DIR, 'images/smile')
NON_SMILE_DIR = os.path.join(BASE_DIR, 'images/non_smile')
TEST_DIR = os.path.join(BASE_DIR, 'images/test')
TEST_IMAGE_PATH = os.path.join(BASE_DIR, 'images/a.jpg')  # Non-smile test image
TEST_IMAGE_PATH2 = os.path.join(BASE_DIR, 'images/ad.jpg')  # Smile test image
TEST_IMAGE_PATH3 = os.path.join(BASE_DIR, 'images/a1.jpg') 
TEST_IMAGE_PATH4 = os.path.join(BASE_DIR, 'images/a2.jpg') 

# Define the dataset class
class SmileDataset(Dataset):
    def __init__(self, smile_dir, non_smile_dir, transform=None):
        self.smile_files = [os.path.join(smile_dir, f) for f in os.listdir(smile_dir) if f.lower().endswith('.jpg')]
        self.non_smile_files = [os.path.join(non_smile_dir, f) for f in os.listdir(non_smile_dir) if f.lower().endswith('.jpg')]
        self.all_files = self.smile_files + self.non_smile_files
        self.labels = [1] * len(self.smile_files) + [0] * len(self.non_smile_files)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Improved neural network architecture
class ImprovedSmileNet(nn.Module):
    def __init__(self):
        super(ImprovedSmileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Enhanced transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),     # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization
])

# Check if directories exist
if not os.path.exists(SMILE_DIR):
    raise FileNotFoundError(f"Smile directory not found at {SMILE_DIR}")
if not os.path.exists(NON_SMILE_DIR):
    raise FileNotFoundError(f"Non-smile directory not found at {NON_SMILE_DIR}")

# Create dataset
dataset = SmileDataset(SMILE_DIR, NON_SMILE_DIR, transform=transform)

# Print dataset balance
print(f"\nDataset balance: {len(dataset.smile_files)} smile vs {len(dataset.non_smile_files)} non-smile images")

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the improved model
model = ImprovedSmileNet().to(device)

# Add class weighting for imbalanced datasets
weight = torch.tensor([len(dataset.non_smile_files)/len(dataset.smile_files)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

# Training loop
num_epochs = 20  # More epochs
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train
    train_losses.append(epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

# Enhanced prediction function with debugging
def predict_image(image_path, model, transform, device, show_image=False):
    try:
        image = Image.open(image_path).convert('L')
        
        if show_image:
            plt.imshow(image, cmap='gray')
            plt.title("Input Image")
            plt.show()
            
        image = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(image)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.7 else 0
        
        return prediction, probability
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

# Test on the test images with visualization
print("\nTesting non-smile image:")
if os.path.exists(TEST_IMAGE_PATH):
    prediction, probability = predict_image(TEST_IMAGE_PATH, model, transform, device, show_image=False)
    if prediction is not None:
        print(f'Prediction: {"Smile" if prediction == 1 else "Non-smile"} with probability: {probability:.4f}')
else:
    print(f"Test image not found at {TEST_IMAGE_PATH}")

print("\nTesting smile image:")
if os.path.exists(TEST_IMAGE_PATH2):
    prediction, probability = predict_image(TEST_IMAGE_PATH2, model, transform, device, show_image=False)
    if prediction is not None:
        print(f'Prediction: {"Smile" if prediction == 1 else "Non-smile"} with probability: {probability:.4f}')
else:
    print(f"Test image not found at {TEST_IMAGE_PATH2}")

print("\nTesting smile image:")
if os.path.exists(TEST_IMAGE_PATH3):
    prediction, probability = predict_image(TEST_IMAGE_PATH3, model, transform, device, show_image=False)
    if prediction is not None:
        print(f'Prediction: {"Smile" if prediction == 1 else "Non-smile"} with probability: {probability:.4f}')
else:
    print(f"Test image not found at {TEST_IMAGE_PATH3}")

print("\nTesting smile image:")
if os.path.exists(TEST_IMAGE_PATH4):
    prediction, probability = predict_image(TEST_IMAGE_PATH4, model, transform, device, show_image=False)
    if prediction is not None:
        print(f'Prediction: {"Smile" if prediction == 1 else "Non-smile"} with probability: {probability:.4f}')
else:
    print(f"Test image not found at {TEST_IMAGE_PATH4}")

# Optional: Test on the test folder
if os.path.exists(TEST_DIR):
    test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith('.jpg')]
    
    if test_files:
        correct = 0
        total = 0
        for file in test_files:
            true_label = 1 if 'smile' in os.path.basename(file).lower() else 0
            prediction, _ = predict_image(file, model, transform, device)
            if prediction is not None:
                total += 1
                correct += 1 if prediction == true_label else 0

        if total > 0:
            print(f'\nTest Folder Accuracy: {correct / total:.4f} ({correct}/{total})')
    else:
        print("\nNo JPG images found in test folder")
else:
    print(f"\nTest directory not found at {TEST_DIR}")
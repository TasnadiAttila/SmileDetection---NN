import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

# Enhanced face detection with better error handling
def detect_and_align_face(image_path, output_size=(64, 64)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Improved face detection parameters
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            
            # Dynamic margin based on face size
            margin_w = int(w * 0.25)
            margin_h = int(h * 0.25)
            x = max(0, x - margin_w)
            y = max(0, y - margin_h)
            w = min(gray.shape[1] - x, w + 2*margin_w)
            h = min(gray.shape[0] - y, h + 2*margin_h)
            
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, output_size)
            
            # Enhanced preprocessing
            face = cv2.equalizeHist(face)
            face = cv2.GaussianBlur(face, (3, 3), 0)
            
            return Image.fromarray(face)
    except Exception as e:
        print(f"Face detection warning for {image_path}: {str(e)}")
    
    # Fallback with enhanced preprocessing
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize(output_size)
        return img
    except:
        return Image.new('L', output_size)

# Using pretrained ResNet with modifications (updated weights parameter)
class PretrainedSmileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify first conv layer for grayscale
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final layers
        self.base.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1))
        
    def forward(self, x):
        return self.base(x)

# Enhanced dataset class
class AlignedSmileDataset(Dataset):
    def __init__(self, smile_dir, non_smile_dir, transform=None):
        self.smile_files = self._get_image_paths(smile_dir)
        self.non_smile_files = self._get_image_paths(non_smile_dir)
        self.all_files = self.smile_files + self.non_smile_files
        self.labels = [1] * len(self.smile_files) + [0] * len(self.non_smile_files)
        self.transform = transform

    def _get_image_paths(self, dir_path):
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        return [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                if f.lower().endswith(valid_ext)]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        try:
            image = detect_and_align_face(img_path)
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return zero image with non-smile label as fallback
            dummy_img = torch.zeros(1, 64, 64) if self.transform is None else self.transform(Image.new('L', (64, 64)))
            return dummy_img, 0

# Stronger augmentations
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Enhanced prediction with confidence intervals
def predict_image(image_path, model, transform, device, threshold=0.5):
    try:
        aligned_face = detect_and_align_face(image_path)
        if transform:
            aligned_face = transform(aligned_face).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(aligned_face)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > threshold else 0
            
            # Calculate confidence level
            confidence = prob if pred == 1 else (1 - prob)
                
        return pred, prob, confidence
    except Exception as e:
        print(f"Prediction failed for {image_path}: {str(e)}")
        return None, None, None

# Main execution
if __name__ == '__main__':
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SMILE_DIR = os.path.join(BASE_DIR, 'images/smile')
    NON_SMILE_DIR = os.path.join(BASE_DIR, 'images/non_smile')
    TEST_IMAGES = [
        os.path.join(BASE_DIR, 'images/a.jpg'),
        os.path.join(BASE_DIR, 'images/a1.jpg'),
        os.path.join(BASE_DIR, 'images/a2.jpg'),
        os.path.join(BASE_DIR, 'images/a3.jpg'),
        os.path.join(BASE_DIR, 'images/a4.jpg'),
        os.path.join(BASE_DIR, 'images/a5.jpg'),
        os.path.join(BASE_DIR, 'images/a6.jpg'),
        # Add new test images here
        os.path.join(BASE_DIR, 'images/test/new_image1.jpg'),
        os.path.join(BASE_DIR, 'images/test/new_image2.jpg')
    ]
    
    # Dataset with different transforms for train/val
    dataset = AlignedSmileDataset(SMILE_DIR, NON_SMILE_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transform()
    val_dataset.dataset.transform = get_val_transform()
    
    # Data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model with pretrained weights (updated to new API)
    model = PretrainedSmileNet().to(device)
    
    # Class-balanced loss
    class_counts = [len(dataset.non_smile_files), len(dataset.smile_files)]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0]).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Learning rate scheduler (removed verbose parameter)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2, factor=0.5)
    
    # Training loop with early stopping
    num_epochs = 15
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        val_loss /= len(val_loader)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_smile_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_smile_model.pth'))
    
    # Find optimal threshold
    def find_optimal_threshold(model, loader):
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images.to(device))
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(torch.sigmoid(outputs).cpu().numpy())
        
        thresholds = np.linspace(0.2, 0.8, 100)
        best_acc = 0
        best_thresh = 0.5
        for thresh in thresholds:
            acc = np.mean((np.array(y_prob) > thresh) == np.array(y_true))
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        return best_thresh
    
    optimal_thresh = find_optimal_threshold(model, val_loader)
    print(f"\nOptimal prediction threshold: {optimal_thresh:.4f}")
    
    # Test on all images
    print("\nTesting images:")
    for img_path in TEST_IMAGES:
        if os.path.exists(img_path):
            pred, prob, confidence = predict_image(img_path, model, get_val_transform(), device, optimal_thresh)
            if pred is not None:
                print(f"{os.path.basename(img_path)}: {'Smile' if pred == 1 else 'Non-smile'} | "
                      f"Score: {prob:.4f} | "
                      f"Confidence: {confidence*100:.1f}%")
        else:
            print(f"Image not found: {img_path}")
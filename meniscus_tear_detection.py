import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import random
from tqdm import tqdm

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
DATA_DIR = 'DeepLearning'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TRAIN_CSV = os.path.join(DATA_DIR, 'train-meniscus.csv')
VALID_CSV = os.path.join(DATA_DIR, 'valid-meniscus.csv')

# MRI Dataset class
class MRIDataset(Dataset):
    def __init__(self, csv_file, data_dir, plane='sagittal', transform=None, max_slices=26, max_samples=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the MRI volumes.
            plane (string): MRI plane ('sagittal', 'coronal', or 'axial').
            transform (callable, optional): Optional transform to be applied on a sample.
            max_slices (int): Maximum number of slices to use from each volume.
            max_samples (int, optional): Maximum number of samples to use from the dataset.
        """
        self.labels_df = pd.read_csv(csv_file, header=None, names=['case_id', 'label'])
        self.data_dir = os.path.join(data_dir, plane)
        self.transform = transform
        self.max_slices = max_slices
        
        # Filter to only include files that exist
        self.labels_df['case_id'] = self.labels_df['case_id'].astype(str).str.zfill(4)
        self.available_files = [f.split('.')[0] for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        self.labels_df = self.labels_df[self.labels_df['case_id'].isin(self.available_files)]
        
        # Limit the dataset to max_samples if provided
        if max_samples is not None:
            self.labels_df = self.labels_df.head(max_samples)
        
        print(f"Loaded {len(self.labels_df)} cases for {plane} plane")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        case_id = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        
        # Load MRI volume
        mri_path = os.path.join(self.data_dir, f"{case_id}.npy")
        volume = np.load(mri_path)
        
        # Ensure we have a consistent number of slices
        if volume.shape[0] > self.max_slices:
            # Take center slices
            start = (volume.shape[0] - self.max_slices) // 2
            volume = volume[start:start+self.max_slices]
        elif volume.shape[0] < self.max_slices:
            # Pad with zeros
            pad_width = ((0, self.max_slices - volume.shape[0]), (0, 0), (0, 0))
            volume = np.pad(volume, pad_width, mode='constant')
        
        # Convert to tensor and normalize
        volume = torch.from_numpy(volume).float() / 255.0
        
        # Add channel dimension
        volume = volume.unsqueeze(1)  # Shape: [slices, 1, height, width]
        
        # Apply transforms if any
        if self.transform:
            transformed_slices = []
            for i in range(volume.shape[0]):
                slice_tensor = volume[i]  # Shape: [1, height, width]
                transformed_slice = self.transform(slice_tensor)
                transformed_slices.append(transformed_slice)
            volume = torch.stack(transformed_slices)
        
        return volume, torch.tensor(label, dtype=torch.float32)

# Define the MRNet model
class MRNetModel(nn.Module):
    def __init__(self, backbone='resnet18', num_slices=26):
        super(MRNetModel, self).__init__()
        self.num_slices = num_slices
        
        # Load pre-trained model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer to accept 1 channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the number of features in the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Add a custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, num_slices, 1, height, width]
        batch_size = x.size(0)
        
        # Process each slice through the backbone
        slice_features = []
        for i in range(self.num_slices):
            slice_i = x[:, i, :, :, :]  # Shape: [batch_size, 1, height, width]
            features_i = self.backbone(slice_i)  # Shape: [batch_size, num_features]
            slice_features.append(features_i)
        
        # Stack features from all slices
        slice_features = torch.stack(slice_features, dim=1)  # Shape: [batch_size, num_slices, num_features]
        
        # Apply attention
        attention_weights = self.attention(slice_features)  # Shape: [batch_size, num_slices, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights to features
        weighted_features = slice_features * attention_weights
        
        # Sum across slices
        aggregated_features = torch.sum(weighted_features, dim=1)  # Shape: [batch_size, num_features]
        
        # Final classification
        output = self.classifier(aggregated_features)
        
        return output

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_valid_auc = -1  # or float('nan') if you prefer
    train_losses = []
    valid_losses = []
    valid_aucs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for volumes, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(volumes).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * volumes.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for volumes, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                volumes = volumes.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(volumes).squeeze()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * volumes.size(0)
                
                # Store predictions and labels for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(epoch_valid_loss)
        
        # Calculate AUC
        try:
            valid_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            valid_auc = float('nan')  # Set to nan if AUC cannot be calculated
        
        valid_aucs.append(valid_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Valid Loss: {epoch_valid_loss:.4f}, "
              f"Valid AUC: {valid_auc:.4f}")
        
        # Save the best model
        if valid_auc > best_valid_auc or epoch == 0:
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), 'best_meniscus_model.pth')
            print(f"Saved new best model with AUC: {valid_auc:.4f}")
            
         # Update best_valid_auc if valid_auc is not nan
        if not np.isnan(valid_auc) and valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            print(f"Updated best_valid_auc to: {best_valid_auc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(valid_aucs, label='Valid AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, valid_losses, valid_aucs

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for volumes, labels in tqdm(data_loader, desc="Evaluating"):
            volumes = volumes.to(device)
            
            # Forward pass
            outputs = model(volumes).squeeze()
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    return auc, cm, report

# Main function
def main():
    # Initialize best_valid_auc to a representative value
    best_valid_auc = -1  # or float('nan') if you prefer

    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets with a limit of 10 samples
    train_dataset = MRIDataset(
        csv_file=TRAIN_CSV,
        data_dir=TRAIN_DIR,
        plane='sagittal',
        transform=transform,
        max_samples=500  # Limit to 10 samples
    )
    
    valid_dataset = MRIDataset(
        csv_file=VALID_CSV,
        data_dir=VALID_DIR,
        plane='sagittal',
        transform=transform,
        max_samples=500  # Limit to 10 samples
    )
    
    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Create data loaders with adjustments for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Adjust based on your memory capacity
        shuffle=True,
        num_workers=num_cores-1,  # Use half of the available cores
        pin_memory=False  # Not needed for CPU
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=8,  # Adjust based on your memory capacity
        shuffle=False,
        num_workers=num_cores-1,  # Use half of the available cores
        pin_memory=False  # Not needed for CPU
    )
    
    # Initialize model
    model = MRNetModel(backbone='resnet18').to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training...")
    train_losses, valid_losses, valid_aucs = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5
    )
    
    # Load the best model
    try:
        model.load_state_dict(torch.load('best_meniscus_model.pth'))
    except FileNotFoundError:
        print("Model file not found. Please ensure the model is trained and saved correctly.")
        return

    # Check if best_valid_auc was updated
    if best_valid_auc == -1:  # or float('nan') if you used that
        best_valid_auc = float('nan')
        print("Warning: Loaded model had a validation AUC of nan during training.")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    evaluate_model(model, valid_loader)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_aucs': valid_aucs
    }, 'meniscus_model_final.pth')
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main() 
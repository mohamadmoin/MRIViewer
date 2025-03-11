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

# MRI Dataset class for multiple planes
class MultiPlaneMRIDataset(Dataset):
    def __init__(self, csv_file, data_dir, planes=None, transform=None, max_slices=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the MRI volumes.
            planes (list): List of MRI planes to use ['sagittal', 'coronal', 'axial'].
            transform (callable, optional): Optional transform to be applied on a sample.
            max_slices (dict): Maximum number of slices to use from each plane.
        """
        self.labels_df = pd.read_csv(csv_file, header=None, names=['case_id', 'label'])
        self.data_dir = data_dir
        self.transform = transform
        
        if planes is None:
            self.planes = ['sagittal', 'coronal', 'axial']
        else:
            self.planes = planes
            
        if max_slices is None:
            self.max_slices = {'sagittal': 26, 'coronal': 26, 'axial': 26}
        else:
            self.max_slices = max_slices
        
        # Filter to only include files that exist in all planes
        self.labels_df['case_id'] = self.labels_df['case_id'].astype(str).str.zfill(4)
        
        # Check which cases exist in all planes
        valid_cases = set(self.labels_df['case_id'])
        for plane in self.planes:
            plane_dir = os.path.join(self.data_dir, plane)
            available_files = {f.split('.')[0] for f in os.listdir(plane_dir) if f.endswith('.npy')}
            valid_cases = valid_cases.intersection(available_files)
        
        # Filter the dataframe to only include valid cases
        self.labels_df = self.labels_df[self.labels_df['case_id'].isin(valid_cases)]
        
        print(f"Loaded {len(self.labels_df)} cases for {', '.join(self.planes)} planes")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        case_id = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        
        # Load MRI volumes for each plane
        volumes = {}
        for plane in self.planes:
            mri_path = os.path.join(self.data_dir, plane, f"{case_id}.npy")
            volume = np.load(mri_path)
            
            # Ensure we have a consistent number of slices
            max_slices = self.max_slices[plane]
            if volume.shape[0] > max_slices:
                # Take center slices
                start = (volume.shape[0] - max_slices) // 2
                volume = volume[start:start+max_slices]
            elif volume.shape[0] < max_slices:
                # Pad with zeros
                pad_width = ((0, max_slices - volume.shape[0]), (0, 0), (0, 0))
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
            
            volumes[plane] = volume
        
        return volumes, torch.tensor(label, dtype=torch.float32)

# Define the single-plane model
class SinglePlaneModel(nn.Module):
    def __init__(self, backbone='resnet18', num_slices=26):
        super(SinglePlaneModel, self).__init__()
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
        self.num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.num_features, 128),
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
        
        return aggregated_features

# Define the multi-plane model
class MultiPlaneMRNetModel(nn.Module):
    def __init__(self, backbone='resnet18', planes=None, max_slices=None):
        super(MultiPlaneMRNetModel, self).__init__()
        
        if planes is None:
            self.planes = ['sagittal', 'coronal', 'axial']
        else:
            self.planes = planes
            
        if max_slices is None:
            self.max_slices = {'sagittal': 26, 'coronal': 26, 'axial': 26}
        else:
            self.max_slices = max_slices
        
        # Create a model for each plane
        self.plane_models = nn.ModuleDict({
            plane: SinglePlaneModel(backbone=backbone, num_slices=self.max_slices[plane])
            for plane in self.planes
        })
        
        # Get the feature dimension from the single plane model
        dummy_model = SinglePlaneModel(backbone=backbone)
        feature_dim = dummy_model.num_features
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * len(self.planes), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict):
        # Process each plane
        plane_features = []
        for plane in self.planes:
            if plane in x_dict:
                features = self.plane_models[plane](x_dict[plane])
                plane_features.append(features)
        
        # Concatenate features from all planes
        combined_features = torch.cat(plane_features, dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        
        return output

# Custom collate function for batching dictionaries
def dict_collate_fn(batch):
    volumes_dict = {plane: [] for plane in batch[0][0].keys()}
    labels = []
    
    for volumes, label in batch:
        for plane in volumes:
            volumes_dict[plane].append(volumes[plane])
        labels.append(label)
    
    # Stack tensors in each plane
    for plane in volumes_dict:
        volumes_dict[plane] = torch.stack(volumes_dict[plane])
    
    return volumes_dict, torch.stack(labels)

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_valid_auc = 0.0
    train_losses = []
    valid_losses = []
    valid_aucs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for volumes_dict, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            for plane in volumes_dict:
                volumes_dict[plane] = volumes_dict[plane].to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(volumes_dict).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for volumes_dict, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                for plane in volumes_dict:
                    volumes_dict[plane] = volumes_dict[plane].to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(volumes_dict).squeeze()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                
                # Store predictions and labels for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(epoch_valid_loss)
        
        # Calculate AUC
        valid_auc = roc_auc_score(all_labels, all_preds)
        valid_aucs.append(valid_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Valid Loss: {epoch_valid_loss:.4f}, "
              f"Valid AUC: {valid_auc:.4f}")
        
        # Save the best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), 'best_multiplane_meniscus_model.pth')
            print(f"Saved new best model with AUC: {valid_auc:.4f}")
    
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
    plt.savefig('multiplane_training_curves.png')
    plt.close()
    
    return train_losses, valid_losses, valid_aucs

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for volumes_dict, labels in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            for plane in volumes_dict:
                volumes_dict[plane] = volumes_dict[plane].to(device)
            
            # Forward pass
            outputs = model(volumes_dict).squeeze()
            
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
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = MultiPlaneMRIDataset(
        csv_file=TRAIN_CSV,
        data_dir=TRAIN_DIR,
        planes=['sagittal', 'coronal', 'axial'],
        transform=transform
    )
    
    valid_dataset = MultiPlaneMRIDataset(
        csv_file=VALID_CSV,
        data_dir=VALID_DIR,
        planes=['sagittal', 'coronal', 'axial'],
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Smaller batch size due to increased memory requirements
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dict_collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dict_collate_fn
    )
    
    # Initialize model
    model = MultiPlaneMRNetModel(backbone='resnet18').to(device)
    
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
        num_epochs=20
    )
    
    # Load the best model
    model.load_state_dict(torch.load('best_multiplane_meniscus_model.pth'))
    
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
    }, 'multiplane_meniscus_model_final.pth')
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main() 
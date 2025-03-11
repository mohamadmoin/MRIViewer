import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from tqdm import tqdm
import argparse
from torchvision import transforms
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
DATA_DIR = 'DeepLearning'
VALID_DIR = os.path.join(DATA_DIR, 'valid')
VALID_CSV = os.path.join(DATA_DIR, 'valid-meniscus.csv')

# Import models from our scripts
from meniscus_tear_detection import MRNetModel
from multiplane_meniscus_detection import MultiPlaneMRNetModel

# GradCAM implementation for visualization
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x):
        # Forward pass
        model_output = self.model(x)
        
        # Backward pass
        self.model.zero_grad()
        model_output.backward(retain_graph=True)
        
        # Get weights
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Create heatmap
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
        
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        cam = F.relu(cam)
        cam = cam / (torch.max(cam) + 1e-10)
        
        return cam.cpu().numpy()

# Function to load a single-plane model
def load_single_plane_model(model_path, backbone='resnet18', num_slices=26):
    model = MRNetModel(backbone=backbone, num_slices=num_slices)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to load a multi-plane model
def load_multi_plane_model(model_path, backbone='resnet18'):
    model = MultiPlaneMRNetModel(backbone=backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to load and preprocess an MRI volume
def load_mri_volume(case_id, plane, data_dir=VALID_DIR, max_slices=26):
    mri_path = os.path.join(data_dir, plane, f"{case_id}.npy")
    volume = np.load(mri_path)
    
    # Ensure we have a consistent number of slices
    if volume.shape[0] > max_slices:
        # Take center slices
        start = (volume.shape[0] - max_slices) // 2
        volume = volume[start:start+max_slices]
    elif volume.shape[0] < max_slices:
        # Pad with zeros
        pad_width = ((0, max_slices - volume.shape[0]), (0, 0), (0, 0))
        volume = np.pad(volume, pad_width, mode='constant')
    
    # Convert to tensor and normalize
    volume_tensor = torch.from_numpy(volume).float() / 255.0
    
    # Add channel dimension
    volume_tensor = volume_tensor.unsqueeze(1)  # Shape: [slices, 1, height, width]
    
    # Apply normalization
    transform = transforms.Normalize(mean=[0.5], std=[0.5])
    normalized_slices = []
    for i in range(volume_tensor.shape[0]):
        slice_tensor = volume_tensor[i]
        normalized_slice = transform(slice_tensor)
        normalized_slices.append(normalized_slice)
    
    volume_tensor = torch.stack(normalized_slices)
    
    # Add batch dimension
    volume_tensor = volume_tensor.unsqueeze(0)  # Shape: [1, slices, 1, height, width]
    
    return volume, volume_tensor

# Function to visualize predictions with GradCAM
def visualize_predictions(model, case_id, plane, data_dir=VALID_DIR, save_dir='visualizations'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load MRI volume
    original_volume, volume_tensor = load_mri_volume(case_id, plane, data_dir)
    
    # Move to device
    volume_tensor = volume_tensor.to(device)
    
    # Get model prediction
    with torch.no_grad():
        if isinstance(model, MultiPlaneMRNetModel):
            # For multi-plane model, we need to create a dictionary
            volumes_dict = {plane: volume_tensor}
            prediction = model(volumes_dict).item()
        else:
            prediction = model(volume_tensor).item()
    
    # Get GradCAM for each slice
    if isinstance(model, MultiPlaneMRNetModel):
        # For multi-plane model, we need to get the target layer from the specific plane model
        target_layer = model.plane_models[plane].backbone.layer4[-1]
    else:
        target_layer = model.backbone.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)
    
    # Create a figure to display the results
    num_slices = original_volume.shape[0]
    fig_rows = (num_slices + 4) // 5  # Ceiling division
    fig = plt.figure(figsize=(20, 4 * fig_rows))
    
    # Create a custom colormap for the heatmap
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # Transparent to red with alpha
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    for i in range(num_slices):
        # Get the original slice
        original_slice = original_volume[i]
        
        # Get the input for GradCAM
        if isinstance(model, MultiPlaneMRNetModel):
            # For multi-plane model, we need to create a dictionary with a single slice
            single_slice = volume_tensor[:, i:i+1, :, :, :]
            volumes_dict = {plane: single_slice}
            
            # We need to modify the model temporarily to work with a single slice
            original_num_slices = model.plane_models[plane].num_slices
            model.plane_models[plane].num_slices = 1
            
            # Get prediction for this slice
            model.zero_grad()
            output = model(volumes_dict)
            output.backward()
            
            # Get GradCAM
            cam = grad_cam(volumes_dict)
            
            # Restore the original number of slices
            model.plane_models[plane].num_slices = original_num_slices
        else:
            # For single-plane model, we can just use the slice directly
            single_slice = volume_tensor[:, i:i+1, :, :, :]
            
            # Temporarily change the model's num_slices
            original_num_slices = model.num_slices
            model.num_slices = 1
            
            # Get prediction for this slice
            model.zero_grad()
            output = model(single_slice)
            output.backward()
            
            # Get GradCAM
            cam = grad_cam(single_slice)
            
            # Restore the original number of slices
            model.num_slices = original_num_slices
        
        # Resize CAM to match the original image size
        cam = cv2.resize(cam, (original_slice.shape[1], original_slice.shape[0]))
        
        # Plot the original slice
        ax = fig.add_subplot(fig_rows, 5, i + 1)
        ax.imshow(original_slice, cmap='gray')
        ax.imshow(cam, cmap=cmap)
        ax.set_title(f"Slice {i+1}/{num_slices}")
        ax.axis('off')
    
    # Add overall prediction
    plt.suptitle(f"Case ID: {case_id}, Plane: {plane}, Prediction: {prediction:.4f} " + 
                 f"({'Positive' if prediction > 0.5 else 'Negative'})", 
                 fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{case_id}_{plane}_prediction.png"))
    plt.close()
    
    return prediction

# Function to analyze a set of cases
def analyze_cases(model, case_ids, planes, data_dir=VALID_DIR, save_dir='visualizations'):
    results = []
    
    for case_id in tqdm(case_ids, desc="Analyzing cases"):
        case_results = {'case_id': case_id}
        
        for plane in planes:
            try:
                prediction = visualize_predictions(model, case_id, plane, data_dir, save_dir)
                case_results[f"{plane}_prediction"] = prediction
            except Exception as e:
                print(f"Error processing case {case_id}, plane {plane}: {e}")
                case_results[f"{plane}_prediction"] = None
        
        results.append(case_results)
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "analysis_results.csv"), index=False)
    
    return results_df

# Main function
def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions on MRI images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='single',
                        help='Type of model: single-plane or multi-plane')
    parser.add_argument('--case_id', type=str, help='Specific case ID to analyze')
    parser.add_argument('--plane', type=str, choices=['sagittal', 'coronal', 'axial'], default='sagittal',
                        help='MRI plane to visualize (for single-plane model)')
    parser.add_argument('--data_dir', type=str, default=VALID_DIR, help='Directory with MRI data')
    parser.add_argument('--save_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--num_cases', type=int, default=5, help='Number of random cases to analyze')
    
    args = parser.parse_args()
    
    # Load the model
    if args.model_type == 'single':
        model = load_single_plane_model(args.model_path)
        planes = [args.plane]
    else:
        model = load_multi_plane_model(args.model_path)
        planes = ['sagittal', 'coronal', 'axial']
    
    # Load validation labels
    labels_df = pd.read_csv(VALID_CSV, header=None, names=['case_id', 'label'])
    labels_df['case_id'] = labels_df['case_id'].astype(str).str.zfill(4)
    
    # If a specific case ID is provided, analyze only that case
    if args.case_id:
        case_ids = [args.case_id]
    else:
        # Otherwise, select random cases
        positive_cases = labels_df[labels_df['label'] == 1]['case_id'].tolist()
        negative_cases = labels_df[labels_df['label'] == 0]['case_id'].tolist()
        
        # Select an equal number of positive and negative cases
        num_each = min(args.num_cases // 2, len(positive_cases), len(negative_cases))
        selected_positive = np.random.choice(positive_cases, num_each, replace=False)
        selected_negative = np.random.choice(negative_cases, num_each, replace=False)
        
        case_ids = np.concatenate([selected_positive, selected_negative])
    
    # Analyze the selected cases
    results_df = analyze_cases(model, case_ids, planes, args.data_dir, args.save_dir)
    
    print(f"Analysis completed. Results saved to {os.path.join(args.save_dir, 'analysis_results.csv')}")

if __name__ == "__main__":
    main() 
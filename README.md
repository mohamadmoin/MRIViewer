# MRI Visualization Tools

This repository contains tools to visualize MRI data stored as .npy files in three orientations: axial, coronal, and sagittal.

## Dataset Organization

The MRI dataset should be organized as follows:

```
DeepLearning/
└── train/
    ├── axial/
    │   ├── 0114.npy
    │   ├── 0575.npy
    │   └── ...
    ├── coronal/
    │   ├── 0114.npy
    │   ├── 0575.npy
    │   └── ...
    └── sagittal/
    │   ├── 0114.npy
    │   ├── 0575.npy
    │   └── ...
```

Each .npy file contains MRI slices in its respective orientation, and files with the same ID (e.g., 0114.npy) correspond to the same subject.

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Applications

There are two different visualization applications available:

### 1. MRI 3D Viewer (Advanced)

This application provides an interactive 3D volume visualization using PyVista and 2D slice views using matplotlib.

To run:

```bash
python mri_3d_viewer.py
```

Features:
- Interactive 3D volume rendering that can be rotated
- 2D slice view with slider for browsing through slices
- Toggle between axial, coronal, and sagittal orientations
- List of available MRI IDs to select from

### 2. MRI Simple Viewer

This is a simplified version that doesn't require PyVista, using only matplotlib for visualization.

To run:

```bash
python mri_simple_viewer.py
```

Features:
- Side-by-side view of axial, coronal, and sagittal slices
- Sliders for each orientation to browse through slices
- Simple 3D visualization showing orthogonal slices
- List of available MRI IDs to select from

## Usage for Deep Learning

For using this MRI data in deep learning tasks:

1. Load the .npy files using NumPy
2. Preprocess the data (normalize, resize, etc.)
3. Use libraries like TensorFlow or PyTorch for model building

Example preprocessing code:

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
axial_data = np.load('DeepLearning/train/axial/0114.npy')

# Normalize to [0,1] range
axial_data_normalized = axial_data.astype('float32') / 255.0

# Split into training and validation sets
slices = [axial_data_normalized[i, :, :] for i in range(axial_data_normalized.shape[0])]
train_slices, val_slices = train_test_split(slices, test_size=0.2, random_state=42)
```

## Troubleshooting

If you encounter issues with PyVista, try using the simplified viewer which only requires matplotlib.

For other issues or questions, please open an issue on the GitHub repository.

# DeepLearning codes

## How to Use

### Train the single-plane model:

```bash
python meniscus_tear_detection.py
```

### Train the multi-plane model:

```bash
python multiplane_meniscus_detection.py
```

### Visualize predictions:

```bash
python visualize_predictions.py --model_path best_meniscus_model.pth --model_type single --plane sagittal
```

### For the multi-plane model:

```bash
python visualize_predictions.py --model_path best_multiplane_meniscus_model.pth --model_type multi
```

## Technical Details:

The MRI data is stored as 3D volumes in .npy format, with dimensions [slices, height, width].
The models are implemented in PyTorch and use pretrained ResNet models as the backbone.
The attention mechanism helps the model focus on the most relevant slices in the MRI volume.
The multi-plane model combines features from all three planes for a more comprehensive analysis.
This solution provides a state-of-the-art approach to meniscus tear detection based on recent research in the field, and the visualization tools can help in understanding and interpreting the model's predictions.

# MRI Viewer

A medical image viewer for MRI scans with advanced visualization features.

## Features

- Load and view MRI datasets in different orientations (Axial, Coronal, Sagittal)
- Interactive slice navigation
- Adjustable window level/width (contrast/brightness)
- Multiple image enhancement methods:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Adaptive Histogram Equalization
  - Gaussian Filtering
  - Unsharp Mask
  - Non-local Means Denoising
  - Adaptive Edge Enhancement
  - Super Resolution
- Zoom functionality with different interpolation methods
- Tissue-specific presets (Brain, Bone, Soft Tissue)

## Using the Application

1. Launch the MRI_Viewer executable
2. Click "Select Dataset Folder" to choose a folder containing your MRI data
   - The folder should have three subdirectories: "axial", "coronal", and "sagittal"
   - Each subdirectory should contain .npy files with matching IDs
3. Select an MRI ID from the dropdown menu
4. Click "Load MRI" to load the selected dataset
5. Use the controls to:
   - Navigate through slices
   - Switch between views (Axial/Coronal/Sagittal)
   - Adjust window level/width
   - Apply image enhancements
   - Zoom and pan the image

## Data Directory Structure

```
your_dataset_folder/
├── axial/
│   ├── 1.npy
│   ├── 2.npy
│   └── ...
├── coronal/
│   ├── 1.npy
│   ├── 2.npy
│   └── ...
└── sagittal/
    ├── 1.npy
    ├── 2.npy
    └── ...
```

## Tips

- Use the "Show Advanced Controls" button to access additional features
- Try different enhancement combinations for optimal visualization
- Use tissue presets as starting points for window level/width adjustment
- The zoom function supports different interpolation methods for quality control
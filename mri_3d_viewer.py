#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI Viewer - Medical image viewer with zooming capability and window level/width adjustment
Supports loading different datasets and provides various image enhancement options
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.widgets import Slider, Button
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QComboBox, QLabel, 
                           QFileDialog, QSlider, QGroupBox, QGridLayout,
                           QCheckBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt
import cv2
from skimage import exposure, restoration

class MRIViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data containers
        self.axial_data = None
        self.coronal_data = None
        self.sagittal_data = None
        self.current_id = None
        
        # Store original data for reset functionality
        self.original_axial_data = None
        self.original_coronal_data = None
        self.original_sagittal_data = None
        
        # Initialize window level/width settings
        self.window_level = 0.5
        self.window_width = 1.0
        
        # Initialize enhancement settings
        self.enhancement_methods = {
            "clahe": False,
            "adaptive_histogram": False,
            "gaussian": False,
            "unsharp_mask": False,
            "denoise": False,
            "edge_enhancement": False,
            "super_resolution": False
        }
        self.interpolation_method = "bicubic"
        
        # Data directory
        self.data_dir = os.getcwd()
        
        # Setup UI
        self.init_ui()
    
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        self.control_frame = QFrame()
        self.control_frame.setFrameShape(QFrame.StyledPanel)
        self.control_frame.setFrameShadow(QFrame.Raised)
        control_layout = QVBoxLayout(self.control_frame)
        
        # Create primary controls row
        primary_controls = QWidget()
        primary_layout = QHBoxLayout(primary_controls)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        
        # Dataset selection
        select_folder_btn = QPushButton("Select Dataset Folder")
        select_folder_btn.clicked.connect(self.select_dataset_folder)
        primary_layout.addWidget(select_folder_btn)
        
        # ID selection
        self.id_combo = QComboBox()
        primary_layout.addWidget(QLabel("MRI ID:"))
        primary_layout.addWidget(self.id_combo)
        
        # Load button
        load_btn = QPushButton("Load MRI")
        load_btn.clicked.connect(self.load_mri_data)
        primary_layout.addWidget(load_btn)
        
        # Add spacer
        primary_layout.addStretch(1)
        
        # View selection
        view_label = QLabel("Cross-section View:")
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.view_combo.currentIndexChanged.connect(self.update_cross_section)
        primary_layout.addWidget(view_label)
        primary_layout.addWidget(self.view_combo)
        
        # Slice slider
        self.slice_slider_label = QLabel("Slice: 0")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        primary_layout.addWidget(self.slice_slider_label)
        primary_layout.addWidget(self.slice_slider)
        
        # Toggle button for expanding additional controls
        self.expand_btn = QPushButton("▼ Show Advanced Controls")
        self.expand_btn.setCheckable(True)
        self.expand_btn.clicked.connect(self.toggle_advanced_controls)
        
        # Create advanced controls section (initially hidden)
        self.advanced_controls = QScrollArea()
        self.advanced_controls.setWidgetResizable(True)
        self.advanced_controls.setVisible(False)
        
        # Container for advanced controls
        advanced_container = QWidget()
        advanced_layout = QVBoxLayout(advanced_container)
        self.advanced_controls.setWidget(advanced_container)
        
        # Enhancement controls group
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QGridLayout()
        
        # Add checkboxes for each enhancement method
        enhancement_title = QLabel("<b>Enhancement Methods:</b>")
        enhancement_layout.addWidget(enhancement_title, 0, 0, 1, 2)
        
        # Basic enhancement methods
        self.clahe_check = QCheckBox("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
        self.clahe_check.setToolTip("Enhances local contrast while preserving overall appearance")
        self.clahe_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("clahe", state))
        enhancement_layout.addWidget(self.clahe_check, 1, 0, 1, 2)
        
        self.hist_check = QCheckBox("Adaptive Histogram Equalization")
        self.hist_check.setToolTip("Improves contrast throughout the image")
        self.hist_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("adaptive_histogram", state))
        enhancement_layout.addWidget(self.hist_check, 2, 0, 1, 2)
        
        self.gaussian_check = QCheckBox("Gaussian Filtering (Noise Reduction)")
        self.gaussian_check.setToolTip("Smooths noise while preserving edges")
        self.gaussian_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("gaussian", state))
        enhancement_layout.addWidget(self.gaussian_check, 3, 0, 1, 2)
        
        self.unsharp_check = QCheckBox("Unsharp Mask (Edge Enhancement)")
        self.unsharp_check.setToolTip("Enhances edges to make details more visible")
        self.unsharp_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("unsharp_mask", state))
        enhancement_layout.addWidget(self.unsharp_check, 4, 0, 1, 2)
        
        # Advanced enhancement methods
        advanced_title = QLabel("<b>Advanced Methods:</b>")
        enhancement_layout.addWidget(advanced_title, 5, 0, 1, 2)
        
        self.denoise_check = QCheckBox("Non-local Means Denoising")
        self.denoise_check.setToolTip("Advanced noise reduction that preserves fine details")
        self.denoise_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("denoise", state))
        enhancement_layout.addWidget(self.denoise_check, 6, 0, 1, 2)
        
        self.edge_check = QCheckBox("Adaptive Edge Enhancement")
        self.edge_check.setToolTip("Highlights edges while respecting tissue boundaries")
        self.edge_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("edge_enhancement", state))
        enhancement_layout.addWidget(self.edge_check, 7, 0, 1, 2)
        
        self.super_res_check = QCheckBox("Super Resolution (Experimental)")
        self.super_res_check.setToolTip("Increases apparent resolution for better detail when zoomed")
        self.super_res_check.stateChanged.connect(lambda state: self.toggle_enhancement_method("super_resolution", state))
        enhancement_layout.addWidget(self.super_res_check, 8, 0, 1, 2)
        
        # Interpolation method
        interp_label = QLabel("<b>Zoom Interpolation:</b>")
        enhancement_layout.addWidget(interp_label, 9, 0, 1, 2)
        
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"])
        self.interp_combo.setCurrentText("Bicubic")
        self.interp_combo.currentTextChanged.connect(self.set_interpolation_method)
        enhancement_layout.addWidget(self.interp_combo, 10, 0, 1, 2)
        
        # Reset enhancement button
        reset_enhancement_btn = QPushButton("Reset All Enhancements")
        reset_enhancement_btn.clicked.connect(self.reset_enhancements)
        enhancement_layout.addWidget(reset_enhancement_btn, 11, 0, 1, 2)
        
        enhancement_group.setLayout(enhancement_layout)
        advanced_layout.addWidget(enhancement_group)
        
        # Window level/width controls
        window_group = QGroupBox("Window Level/Width Adjustment")
        window_layout = QGridLayout()
        
        # Window level slider
        level_label = QLabel("Window Level:")
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(0, 100)
        self.level_slider.setValue(50)
        self.level_slider.valueChanged.connect(self.update_window_settings)
        window_layout.addWidget(level_label, 0, 0)
        window_layout.addWidget(self.level_slider, 0, 1)
        
        # Window width slider
        width_label = QLabel("Window Width:")
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 100)
        self.width_slider.setValue(100)
        self.width_slider.valueChanged.connect(self.update_window_settings)
        window_layout.addWidget(width_label, 1, 0)
        window_layout.addWidget(self.width_slider, 1, 1)
        
        # Preset buttons
        presets_layout = QHBoxLayout()
        brain_btn = QPushButton("Brain")
        bone_btn = QPushButton("Bone")
        soft_tissue_btn = QPushButton("Soft Tissue")
        
        brain_btn.clicked.connect(lambda: self.apply_preset("brain"))
        bone_btn.clicked.connect(lambda: self.apply_preset("bone"))
        soft_tissue_btn.clicked.connect(lambda: self.apply_preset("soft_tissue"))
        
        presets_layout.addWidget(brain_btn)
        presets_layout.addWidget(bone_btn)
        presets_layout.addWidget(soft_tissue_btn)
        
        # Reset window settings button
        reset_window_btn = QPushButton("Reset Window Settings")
        reset_window_btn.clicked.connect(self.reset_window_settings)
        
        window_layout.addLayout(presets_layout, 2, 0, 1, 2)
        window_layout.addWidget(reset_window_btn, 3, 0, 1, 2)
        window_group.setLayout(window_layout)
        
        advanced_layout.addWidget(window_group)
        
        # Global reset button
        reset_all_btn = QPushButton("Reset All Adjustments")
        reset_all_btn.clicked.connect(self.reset_all)
        advanced_layout.addWidget(reset_all_btn)
        
        # Add all controls to the control layout
        control_layout.addWidget(primary_controls)
        control_layout.addWidget(self.expand_btn)
        control_layout.addWidget(self.advanced_controls)
        
        # Add control panel to main layout
        main_layout.addWidget(self.control_frame)
        
        # 2D slice visualization panel with controls
        self.slice_widget = QWidget()
        slice_layout = QVBoxLayout(self.slice_widget)
        
        # Create matplotlib figure for the slice view
        self.slice_fig = plt.figure(figsize=(8, 8))
        self.slice_canvas = self.slice_fig.canvas
        
        # Add matplotlib toolbar for zoom functionality
        self.mpl_toolbar = NavigationToolbar2QT(self.slice_canvas, self.slice_widget)
        slice_layout.addWidget(self.mpl_toolbar)
        slice_layout.addWidget(self.slice_canvas)
        
        # Add slice widget to main layout
        main_layout.addWidget(self.slice_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready - Please select a dataset folder")

    def select_dataset_folder(self):
        """Open folder dialog to select dataset directory"""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.data_dir = folder
            self.populate_id_list()
            self.statusBar().showMessage(f"Selected dataset folder: {folder}")

    def toggle_advanced_controls(self):
        """Toggle visibility of advanced controls"""
        is_visible = self.advanced_controls.isVisible()
        self.advanced_controls.setVisible(not is_visible)
        
        # Update button text
        if not is_visible:
            self.expand_btn.setText("▲ Hide Advanced Controls")
        else:
            self.expand_btn.setText("▼ Show Advanced Controls")
    
    def toggle_enhancement_method(self, method_name, state):
        """Toggle specific enhancement method on/off"""
        self.enhancement_methods[method_name] = (state == Qt.Checked)
        if self.axial_data is not None:
            self.update_slice()
    
    def set_interpolation_method(self, method):
        """Set the interpolation method for zooming"""
        self.interpolation_method = method.lower()
        if self.axial_data is not None:
            self.update_slice()
    
    def reset_enhancements(self):
        """Reset all enhancement settings"""
        # Uncheck all enhancement checkboxes
        self.clahe_check.setChecked(False)
        self.hist_check.setChecked(False)
        self.gaussian_check.setChecked(False)
        self.unsharp_check.setChecked(False)
        self.denoise_check.setChecked(False)
        self.edge_check.setChecked(False)
        self.super_res_check.setChecked(False)
        
        # Reset all enhancement methods
        for method in self.enhancement_methods:
            self.enhancement_methods[method] = False
        
        # Reset interpolation
        self.interp_combo.setCurrentText("Bicubic")
        
        # Update the display
        if self.axial_data is not None:
            self.update_slice()
        
        self.statusBar().showMessage("Enhancement settings reset")
    
    def reset_window_settings(self):
        """Reset window level/width settings"""
        self.level_slider.setValue(50)
        self.width_slider.setValue(100)
        self.update_window_settings()
        self.statusBar().showMessage("Window settings reset")
    
    def reset_all(self):
        """Reset all adjustments and return to original state"""
        # Reset window settings
        self.reset_window_settings()
        
        # Reset enhancement settings
        self.reset_enhancements()
        
        # If data is loaded, restore from original
        if self.original_axial_data is not None:
            self.axial_data = self.original_axial_data.copy()
            self.coronal_data = self.original_coronal_data.copy()
            self.sagittal_data = self.original_sagittal_data.copy()
            self.update_slice()
        
        self.statusBar().showMessage("All settings reset to default")
    
    def apply_preset(self, preset_type):
        """Apply predefined window level/width presets"""
        if preset_type == "brain":
            self.level_slider.setValue(40)
            self.width_slider.setValue(80)
        elif preset_type == "bone":
            self.level_slider.setValue(70)
            self.width_slider.setValue(30)
        elif preset_type == "soft_tissue":
            self.level_slider.setValue(50)
            self.width_slider.setValue(60)
        
        self.update_window_settings()
        self.statusBar().showMessage(f"Applied {preset_type.title()} preset")
    
    def update_window_settings(self):
        """Update window level and width based on slider values"""
        self.window_level = self.level_slider.value() / 100.0
        self.window_width = self.width_slider.value() / 100.0
        
        # Update the current slice with new window settings
        self.update_slice()
    
    def populate_id_list(self):
        """Populate the ID combo box with available MRI IDs"""
        try:
            axial_dir = os.path.join(self.data_dir, "axial")
            coronal_dir = os.path.join(self.data_dir, "coronal")
            sagittal_dir = os.path.join(self.data_dir, "sagittal")
            
            # Get all IDs from axial directory
            axial_files = [f.split('.')[0] for f in os.listdir(axial_dir) if f.endswith('.npy')]
            
            # Filter to IDs that exist in all three directories
            valid_ids = []
            for id_str in axial_files:
                if (os.path.exists(os.path.join(coronal_dir, f"{id_str}.npy")) and 
                    os.path.exists(os.path.join(sagittal_dir, f"{id_str}.npy"))):
                    valid_ids.append(id_str)
            
            # Sort IDs numerically
            valid_ids.sort(key=lambda x: int(x))
            
            # Add to combo box
            self.id_combo.clear()
            self.id_combo.addItems(valid_ids)
            
            self.statusBar().showMessage(f"Found {len(valid_ids)} MRI datasets")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading IDs: {str(e)}")
    
    def load_mri_data(self):
        """Load MRI data for the selected ID"""
        try:
            id_str = self.id_combo.currentText()
            if not id_str:
                self.statusBar().showMessage("Please select an ID")
                return
            
            self.statusBar().showMessage(f"Loading MRI data for ID {id_str}...")
            
            # Load data from all three orientations
            axial_path = os.path.join(self.data_dir, "axial", f"{id_str}.npy")
            coronal_path = os.path.join(self.data_dir, "coronal", f"{id_str}.npy")
            sagittal_path = os.path.join(self.data_dir, "sagittal", f"{id_str}.npy")
            
            self.axial_data = np.load(axial_path)
            self.coronal_data = np.load(coronal_path)
            self.sagittal_data = np.load(sagittal_path)
            
            # Store original data for reset functionality
            self.original_axial_data = self.axial_data.copy()
            self.original_coronal_data = self.coronal_data.copy()
            self.original_sagittal_data = self.sagittal_data.copy()
            
            # Update UI
            self.current_id = id_str
            self.slice_slider.setMaximum(self.axial_data.shape[0] - 1)
            self.slice_slider.setValue(self.axial_data.shape[0] // 2)
            
            # Update visualizations
            self.update_slice()
            
            self.statusBar().showMessage(f"Loaded MRI data for ID {id_str}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading MRI data: {str(e)}")
    
    def update_cross_section(self):
        """Update the 2D cross-section view"""
        if self.axial_data is None:
            return
            
        view_type = self.view_combo.currentText()
        current_slice = self.slice_slider.value()
        
        # Update slider maximum based on view
        if view_type == "Axial":
            self.slice_slider.setMaximum(self.axial_data.shape[0] - 1)
            if current_slice >= self.axial_data.shape[0]:
                current_slice = self.axial_data.shape[0] - 1
                self.slice_slider.setValue(current_slice)
        elif view_type == "Coronal":
            self.slice_slider.setMaximum(self.coronal_data.shape[0] - 1)
            if current_slice >= self.coronal_data.shape[0]:
                current_slice = self.coronal_data.shape[0] - 1
                self.slice_slider.setValue(current_slice)
        elif view_type == "Sagittal":
            self.slice_slider.setMaximum(self.sagittal_data.shape[0] - 1)
            if current_slice >= self.sagittal_data.shape[0]:
                current_slice = self.sagittal_data.shape[0] - 1
                self.slice_slider.setValue(current_slice)
        
        self.update_slice()
    
    def update_slice(self):
        """Update the displayed slice based on slider value and window settings"""
        if self.axial_data is None:
            return
        
        try:    
            view_type = self.view_combo.currentText()
            current_slice = self.slice_slider.value()
            
            # Update slice label
            self.slice_slider_label.setText(f"Slice: {current_slice}")
            
            # Clear the figure
            self.slice_fig.clear()
            ax = self.slice_fig.add_subplot(111)
            
            # Get data for selected view
            if view_type == "Axial":
                # Use a copy to avoid modifying original data
                slice_data = self.axial_data[current_slice].copy()
                title = f"Axial Slice {current_slice+1}/{self.axial_data.shape[0]}"
            elif view_type == "Coronal":
                slice_data = self.coronal_data[current_slice].copy()
                title = f"Coronal Slice {current_slice+1}/{self.coronal_data.shape[0]}"
            elif view_type == "Sagittal":
                slice_data = self.sagittal_data[current_slice].copy()
                title = f"Sagittal Slice {current_slice+1}/{self.sagittal_data.shape[0]}"
            
            # Apply all enabled enhancements in sequence
            enhanced_slice = self.apply_all_enhancements(slice_data)
            
            # Apply window level/width adjustments
            vmin, vmax = self.calculate_window_range(enhanced_slice)
            
            # Set interpolation method for better zoom quality
            interpolation = self.get_matplotlib_interpolation()
            
            # Display slice with window level/width applied and enhanced interpolation
            im = ax.imshow(enhanced_slice, cmap='gray', vmin=vmin, vmax=vmax, interpolation=interpolation)
            ax.set_title(title)
            self.slice_fig.colorbar(im)
            
            # Redraw canvas
            self.slice_canvas.draw()
        except Exception as e:
            self.statusBar().showMessage(f"Error updating slice: {str(e)}")
    
    def apply_all_enhancements(self, image):
        """Apply all enabled enhancement methods in sequence"""
        enhanced_image = image.copy()
        
        # Normalize to 0-255 uint8 for OpenCV processing if needed
        if np.max(enhanced_image) <= 1.0:
            # Assume image is already normalized to [0, 1]
            image_8bit = (enhanced_image * 255).astype(np.uint8)
        else:
            # Normalize to [0, 255]
            img_min, img_max = enhanced_image.min(), enhanced_image.max()
            image_8bit = (((enhanced_image - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
        
        try:
            # Apply each enabled enhancement method in sequence
            if self.enhancement_methods["clahe"]:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image_8bit = clahe.apply(image_8bit)
            
            if self.enhancement_methods["adaptive_histogram"]:
                image_8bit = cv2.equalizeHist(image_8bit)
            
            if self.enhancement_methods["gaussian"]:
                image_8bit = cv2.GaussianBlur(image_8bit, (5, 5), 0)
            
            if self.enhancement_methods["unsharp_mask"]:
                blurred = cv2.GaussianBlur(image_8bit, (5, 5), 0)
                image_8bit = cv2.addWeighted(image_8bit, 1.5, blurred, -0.5, 0)
            
            # Advanced methods
            if self.enhancement_methods["denoise"]:
                # Non-local means denoising
                image_8bit = cv2.fastNlMeansDenoising(image_8bit, None, 10, 7, 21)
            
            if self.enhancement_methods["edge_enhancement"]:
                # Edge enhancement using gradient
                sobelx = cv2.Sobel(image_8bit, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(image_8bit, cv2.CV_64F, 0, 1, ksize=3)
                gradient = np.sqrt(sobelx**2 + sobely**2)
                gradient = (gradient / gradient.max() * 255).astype(np.uint8)
                image_8bit = cv2.addWeighted(image_8bit, 0.7, gradient, 0.3, 0)
            
            if self.enhancement_methods["super_resolution"]:
                # Simple upscaling followed by downscaling for pseudo-super resolution
                upscaled = cv2.resize(image_8bit, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                image_8bit = cv2.resize(upscaled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            
            # Convert back to original data type and range
            if np.max(enhanced_image) <= 1.0:
                # Return to [0, 1] range
                return image_8bit.astype(np.float32) / 255.0
            else:
                # Return to original range
                img_min, img_max = enhanced_image.min(), enhanced_image.max()
                return (image_8bit.astype(np.float32) / 255.0) * (img_max - img_min) + img_min
                
        except Exception as e:
            self.statusBar().showMessage(f"Enhancement error: {str(e)}")
            # Return original image if enhancement fails
            return enhanced_image
    
    def get_matplotlib_interpolation(self):
        """Convert interpolation method name to matplotlib parameter"""
        if self.interpolation_method == "nearest":
            return "nearest"
        elif self.interpolation_method == "bilinear":
            return "bilinear"
        elif self.interpolation_method == "bicubic":
            return "bicubic"
        elif self.interpolation_method == "lanczos":
            return "lanczos"
        else:
            return "bicubic"  # Default
    
    def calculate_window_range(self, data):
        """Calculate the min/max values based on window level and width settings"""
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        
        # Calculate the center point based on window_level (0-1)
        center = data_min + (self.window_level * data_range)
        
        # Calculate the width of the window based on window_width (0-1)
        width = self.window_width * data_range
        
        # Calculate min/max values for display
        vmin = center - (width / 2)
        vmax = center + (width / 2)
        
        return vmin, vmax


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MRIViewer()
    viewer.show()
    sys.exit(app.exec_()) 
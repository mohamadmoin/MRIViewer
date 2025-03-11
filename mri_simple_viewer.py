#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI Simple Viewer - Views axial, coronal, and sagittal MRI slices using matplotlib
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QComboBox, QLabel, 
                           QSlider, QSplitter, QGridLayout)
from PyQt5.QtCore import Qt

class MRISliceCanvas(FigureCanvas):
    """Canvas for displaying MRI slices"""
    def __init__(self, parent=None, width=5, height=5):
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
    
    def display_slice(self, slice_data, title):
        """Display a single MRI slice"""
        self.ax.clear()
        im = self.ax.imshow(slice_data, cmap='gray')
        self.ax.set_title(title)
        self.fig.colorbar(im, ax=self.ax)
        self.draw()

class MRISimpleViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Simple Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data containers
        self.axial_data = None
        self.coronal_data = None
        self.sagittal_data = None
        self.current_id = None
        
        # Data directories
        self.data_dir = os.path.join(os.getcwd(), "DeepLearning", "train")
        
        # Setup UI
        self.init_ui()
    
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # ID selection
        self.id_combo = QComboBox()
        self.populate_id_list()
        control_layout.addWidget(QLabel("MRI ID:"))
        control_layout.addWidget(self.id_combo)
        
        # Load button
        load_btn = QPushButton("Load MRI")
        load_btn.clicked.connect(self.load_mri_data)
        control_layout.addWidget(load_btn)
        
        # Add spacer
        control_layout.addStretch(1)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Create a grid layout for the three views
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        main_layout.addWidget(self.grid_widget)
        
        # Create canvas for each view
        self.axial_canvas = MRISliceCanvas(self.grid_widget)
        self.coronal_canvas = MRISliceCanvas(self.grid_widget)
        self.sagittal_canvas = MRISliceCanvas(self.grid_widget)
        
        # Create sliders for each view
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(0)
        self.axial_slider.valueChanged.connect(lambda: self.update_slice("axial"))
        
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.setMaximum(0)
        self.coronal_slider.valueChanged.connect(lambda: self.update_slice("coronal"))
        
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.setMaximum(0)
        self.sagittal_slider.valueChanged.connect(lambda: self.update_slice("sagittal"))
        
        # Add canvases and sliders to grid
        self.grid_layout.addWidget(QLabel("Axial View"), 0, 0, 1, 1, Qt.AlignCenter)
        self.grid_layout.addWidget(self.axial_canvas, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.axial_slider, 2, 0, 1, 1)
        
        self.grid_layout.addWidget(QLabel("Coronal View"), 0, 1, 1, 1, Qt.AlignCenter)
        self.grid_layout.addWidget(self.coronal_canvas, 1, 1, 1, 1)
        self.grid_layout.addWidget(self.coronal_slider, 2, 1, 1, 1)
        
        self.grid_layout.addWidget(QLabel("Sagittal View"), 0, 2, 1, 1, Qt.AlignCenter)
        self.grid_layout.addWidget(self.sagittal_canvas, 1, 2, 1, 1)
        self.grid_layout.addWidget(self.sagittal_slider, 2, 2, 1, 1)
        
        # Create a separate panel for 3D view (using MPL's 3D capabilities)
        self.fig_3d = Figure(figsize=(5, 5))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
        main_layout.addWidget(self.canvas_3d)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
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
            
            # Update UI
            self.current_id = id_str
            
            # Set slider ranges
            self.axial_slider.setMaximum(self.axial_data.shape[0] - 1)
            self.axial_slider.setValue(self.axial_data.shape[0] // 2)
            
            self.coronal_slider.setMaximum(self.coronal_data.shape[0] - 1)
            self.coronal_slider.setValue(self.coronal_data.shape[0] // 2)
            
            self.sagittal_slider.setMaximum(self.sagittal_data.shape[0] - 1)
            self.sagittal_slider.setValue(self.sagittal_data.shape[0] // 2)
            
            # Update all views
            self.update_slice("axial")
            self.update_slice("coronal")
            self.update_slice("sagittal")
            self.update_3d_view()
            
            self.statusBar().showMessage(f"Loaded MRI data for ID {id_str}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading MRI data: {str(e)}")
    
    def update_slice(self, view_type):
        """Update the specified slice view"""
        if self.axial_data is None:
            return
        
        if view_type == "axial":
            slice_idx = self.axial_slider.value()
            slice_data = self.axial_data[slice_idx]
            title = f"Axial Slice {slice_idx+1}/{self.axial_data.shape[0]}"
            self.axial_canvas.display_slice(slice_data, title)
        
        elif view_type == "coronal":
            slice_idx = self.coronal_slider.value()
            slice_data = self.coronal_data[slice_idx]
            title = f"Coronal Slice {slice_idx+1}/{self.coronal_data.shape[0]}"
            self.coronal_canvas.display_slice(slice_data, title)
        
        elif view_type == "sagittal":
            slice_idx = self.sagittal_slider.value()
            slice_data = self.sagittal_data[slice_idx]
            title = f"Sagittal Slice {slice_idx+1}/{self.sagittal_data.shape[0]}"
            self.sagittal_canvas.display_slice(slice_data, title)
    
    def update_3d_view(self):
        """Update the 3D visualization with current data"""
        if self.axial_data is None:
            return
        
        # Clear existing plot
        self.ax_3d.clear()
        
        # Get slices from middle of each dimension
        axial_slice = self.axial_data[self.axial_data.shape[0]//2]
        coronal_slice = self.coronal_data[self.coronal_data.shape[0]//2]
        sagittal_slice = self.sagittal_data[self.sagittal_data.shape[0]//2]
        
        # Get dimensions
        z_dim, y_dim, x_dim = self.axial_data.shape[0], axial_slice.shape[0], axial_slice.shape[1]
        
        # Create the coordinate meshgrids
        x, y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
        
        # Create a simplified 3D representation
        # We'll show three orthogonal slices from the volume
        
        # Axial slice (xy plane at mid-z)
        self.ax_3d.contourf(x, y, axial_slice, zdir='z', offset=z_dim//2, cmap='gray', alpha=0.8)
        
        # Coronal slice (xz plane at mid-y)
        z, x = np.meshgrid(np.arange(z_dim), np.arange(x_dim))
        coronal_slice_rotated = np.rot90(coronal_slice) if coronal_slice.shape[1] == x_dim else coronal_slice
        self.ax_3d.contourf(x, coronal_slice_rotated, z, zdir='y', offset=y_dim//2, cmap='gray', alpha=0.8)
        
        # Sagittal slice (yz plane at mid-x)
        z, y = np.meshgrid(np.arange(z_dim), np.arange(y_dim))
        sagittal_slice_rotated = np.rot90(sagittal_slice) if sagittal_slice.shape[1] == y_dim else sagittal_slice
        self.ax_3d.contourf(sagittal_slice_rotated, y, z, zdir='x', offset=x_dim//2, cmap='gray', alpha=0.8)
        
        # Set axis labels and limits
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim(0, x_dim)
        self.ax_3d.set_ylim(0, y_dim)
        self.ax_3d.set_zlim(0, z_dim)
        
        # Add a title
        self.ax_3d.set_title(f"3D View of MRI ID: {self.current_id}")
        
        # Update the canvas
        self.canvas_3d.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MRISimpleViewer()
    viewer.show()
    sys.exit(app.exec_()) 
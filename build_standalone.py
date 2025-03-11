import PyInstaller.__main__
import sys
import os

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define icon path (you can add an .ico file later)
# icon_path = os.path.join(script_dir, "icon.ico")

# PyInstaller command line arguments
args = [
    'mri_3d_viewer.py',  # Your main script
    '--name=MRI_Viewer',  # Name of the executable
    '--onefile',  # Create a single executable file
    '--windowed',  # Don't show console window when running the app
    # '--icon=' + icon_path,  # Uncomment if you add an icon
    '--add-data=README.md;.',  # Add README file
    '--hidden-import=numpy',
    '--hidden-import=matplotlib',
    '--hidden-import=cv2',
    '--hidden-import=skimage',
    '--hidden-import=PyQt5',
    '--clean',  # Clean PyInstaller cache
    '--noconfirm',  # Replace output directory without asking
]

# Run PyInstaller
PyInstaller.__main__.run(args) 
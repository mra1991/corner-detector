"""
Graphical user interface for the corner detection application.

This module provides a simple Tkinter-based interface that allows the user
to:
    1. Select an input image.
    2. Adjust the parameters for Harris corner detection algorithm.
    3. Run the corner detection pipeline.
    4. Display the original image and all intermediate results.
"""

import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

from image_utils import load_image, add_figure, show_figures, init_figures, mark_corners
from corner_detector import detect_corners


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "sample_images"
DEFAULT_IMAGE = IMAGE_DIR / "building1.jpg"


class CornerDetectorGUI:
    """
    Tkinter GUI for the corner detection application.

    The GUI manages image selection, user input for filter parameters,
    execution of the Harris corner detection pipeline, and visualization
    of the results.
    """
    def __init__(self, root):
        """
        Initialize the graphical user interface.

        Parameters
        ----------
        root : tkinter.Tk
            The main application window.
        """
        self.root = root
        self.root.title("Harris Corner Detector")

        self.img = load_image(DEFAULT_IMAGE)

        self.window_slider = tk.Scale(
            root,
            from_=3,
            to=15,
            resolution=2,
            orient="horizontal",
            label="Window size"
        )
        self.window_slider.set(9)
        self.window_slider.grid(row=0, column=0)
        
        self.k_slider = tk.Scale(
            root,
            from_=0.03,
            to=0.08,
            resolution=0.005,
            orient="horizontal",
            label="Harris constant k"
        )
        self.k_slider.set(0.06)
        self.k_slider.grid(row=0, column=1)
        
        self.percentile_slider = tk.Scale(
            root,
            from_=90,
            to=99,
            resolution=1,
            orient="horizontal",
            label="Threshold percentile"
        )
        self.percentile_slider.set(97)
        self.percentile_slider.grid(row=1, column=0)
        
        self.min_slider = tk.Scale(
            root,
            from_=1,
            to=10,
            resolution=1,
            orient="horizontal",
            label="Min distance"
        )
        self.min_slider.set(5)
        self.min_slider.grid(row=1, column=1)

        self.browse_button = tk.Button(
            root,
            text="Browse Image",
            command=self.browse_image
        )
        self.browse_button.grid(row=2, column=0)

        self.detect_button = tk.Button(
            root,
            text="Detect Corners",
            command=self.detect_corners_clicked
        )
        self.detect_button.grid(row=2, column=1, pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.grid(row=3, pady=10)

        self.update_image_preview()
    
    def update_image_preview(self):
        """
        Update the image preview displayed in the GUI.

        The current grayscale image is converted to an 8-bit PIL image,
        resized to fit the preview area while preserving its aspect ratio,
        and displayed in the application window.
        """
        preview = self.img

        if preview.max() <= 1.0:
            preview = preview * 255

        preview = preview.astype(np.uint8)

        pil_image = Image.fromarray(preview)
        pil_image.thumbnail((500, 400))

        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.tk_image)

    def browse_image(self):
        """
        Open a file dialog and load a new input image.

        If the user selects a valid image file, it replaces the currently
        loaded image.
        """
        filepath = filedialog.askopenfilename(
            initialdir=IMAGE_DIR,
            initialfile=DEFAULT_IMAGE.name,
            title="Select an image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            self.img = load_image(filepath)
            self.update_image_preview()

    def detect_corners_clicked(self):
        """
        Run the corner detection pipeline using the current GUI settings.

        The window size, k, threshold percentile and minimum distance
        are read from the sliders, the Harris corner detector
        is executed, and the resulting images are displayed.
        """
        window_size = self.window_slider.get()
        k = self.k_slider.get()
        percent = self.percentile_slider.get()
        min_distance = self.min_slider.get()

        response_map, thresh_response, coords_max = detect_corners(self.img, window_size, k, percent, min_distance)

        self.display_results(response_map, thresh_response, coords_max)

    def display_results(self, response_map, thresh_response, coords_max):
        """
        Display the original image and all intermediate processing results.        
        """
        images = [response_map, thresh_response, self.img]
        titles = [
            'Harris response map (raw)',
            'Harris response map (after threshold)',
            'Detected corners after NMS'
        ]
        
        init_figures()
        for i, (image, title) in enumerate(zip(images, titles), start=1):
            add_figure(image, i, title)

        mark_corners(coords_max)
        show_figures()


def start_gui():
    """
    Create and start the graphical user interface.

    This function creates the main Tkinter window, initializes the GUI,
    and starts the application's event loop.
    """
    root = tk.Tk()
    app = CornerDetectorGUI(root)
    root.mainloop()
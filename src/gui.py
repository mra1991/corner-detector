"""
Graphical user interface for the corner detection application.

This module provides a Tkinter-based interface that allows the user to:
    1. Select an input image.
    2. Adjust the Harris corner detector parameters.
    3. Run the corner detection pipeline.
    4. View the source image and processing results.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from corner_detector import detect_corners
from image_utils import (
    load_image,
    add_figure,
    show_figures,
    init_figures,
    mark_corners,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "sample_images"


class CornerDetectorGUI:
    """
    Tkinter GUI for the Harris corner detection application.

    The GUI manages image selection, parameter controls, execution of the
    Harris corner detection pipeline, and visualization of the results.
    """

    def __init__(self, root):
        """
        Initialize the corner detector graphical interface.

        Parameters
        ----------
        root : tkinter.Tk
            Main application window.
        """
        self.root = root
        self.root.title("Harris Corner Detector")
        self.root.resizable(False, False)

        self.img = None
        self.image_path = None
        self.tk_image = None

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        self._build_controls()
        self._build_actions()
        self._build_status()
        self._build_image_preview()
        self._update_button_states()

    def _build_controls(self):
        """Create the Harris detector parameter controls."""
        controls = ttk.LabelFrame(
            self.root,
            text="Harris Corner Detection Settings"
        )
        controls.grid(
            row=0,
            column=0,
            padx=12,
            pady=(10, 6),
            sticky="ew"
        )

        self.window_slider = tk.Scale(
            controls,
            from_=3,
            to=15,
            resolution=2,
            orient="horizontal",
            label="Window size",
            length=180
        )
        self.window_slider.set(9)
        self.window_slider.grid(row=0, column=0, padx=8, pady=6)

        self.k_slider = tk.Scale(
            controls,
            from_=0.03,
            to=0.08,
            resolution=0.005,
            orient="horizontal",
            label="Harris constant k",
            length=180
        )
        self.k_slider.set(0.06)
        self.k_slider.grid(row=0, column=1, padx=8, pady=6)

        self.percentile_slider = tk.Scale(
            controls,
            from_=90,
            to=99,
            resolution=1,
            orient="horizontal",
            label="Threshold percentile",
            length=180
        )
        self.percentile_slider.set(97)
        self.percentile_slider.grid(row=1, column=0, padx=8, pady=6)

        self.min_slider = tk.Scale(
            controls,
            from_=1,
            to=10,
            resolution=1,
            orient="horizontal",
            label="Minimum corner distance",
            length=180
        )
        self.min_slider.set(5)
        self.min_slider.grid(row=1, column=1, padx=8, pady=6)

    def _build_actions(self):
        """Create the image-selection and corner-detection buttons."""
        actions = ttk.LabelFrame(
            self.root,
            text="Actions"
        )
        actions.grid(
            row=1,
            column=0,
            padx=12,
            pady=6,
            sticky="ew"
        )

        self.browse_button = ttk.Button(
            actions,
            text="Browse Image",
            command=self.browse_image
        )
        self.browse_button.grid(row=0, column=0, padx=10, pady=10)

        self.detect_button = ttk.Button(
            actions,
            text="Detect Corners",
            command=self.detect_corners_clicked
        )
        self.detect_button.grid(row=0, column=1, padx=10, pady=10)

        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

    def _build_status(self):
        """Create the application status message."""
        self.status_var = tk.StringVar(
            value="Select an image to begin."
        )

        status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="center"
        )
        status_label.grid(
            row=2,
            column=0,
            padx=12,
            pady=(2, 6),
            sticky="ew"
        )

    def _build_image_preview(self):
        """Create the source-image preview area."""
        image_frame = ttk.LabelFrame(
            self.root,
            text="Source Image"
        )
        image_frame.grid(
            row=3,
            column=0,
            padx=12,
            pady=(6, 12),
            sticky="nsew"
        )

        self.image_label = ttk.Label(
            image_frame,
            text="No image selected",
            anchor="center",
            width=70
        )
        self.image_label.pack(
            padx=12,
            pady=12
        )

    def _update_button_states(self):
        """Enable or disable buttons according to the current state."""
        if self.img is None:
            self.detect_button.config(state="disabled")
        else:
            self.detect_button.config(state="normal")

    def update_image_preview(self):
        """
        Update the image preview displayed in the GUI.

        The current grayscale image is converted to an unsigned 8-bit PIL
        image, resized while preserving its aspect ratio, and displayed in
        the source-image frame.
        """
        if self.img is None:
            self.tk_image = None
            self.image_label.config(
                image="",
                text="No image selected"
            )
            return

        preview = np.asarray(self.img)

        if preview.max() <= 1.0:
            preview = preview * 255.0

        preview = np.clip(preview, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(preview)
        pil_image.thumbnail(
            (600, 420),
            Image.Resampling.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(pil_image)

        self.image_label.config(
            image=self.tk_image,
            text=""
        )

    def browse_image(self):
        """
        Open a file dialog and load a grayscale input image.

        If the selected image cannot be loaded, an error message is displayed
        and the previously loaded image remains unchanged.
        """
        filepath = filedialog.askopenfilename(
            initialdir=IMAGE_DIR,
            title="Select an image",
            filetypes=[
                (
                    "Image Files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                ),
                ("All Files", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            image = load_image(filepath)
        except (FileNotFoundError, OSError, ValueError) as error:
            messagebox.showerror(
                "Unable to Load Image",
                str(error)
            )
            return

        self.img = image
        self.image_path = Path(filepath)

        self.update_image_preview()
        self._update_button_states()

        self.status_var.set(
            f"Loaded {self.image_path.name} "
            f"({self.img.shape[1]} × {self.img.shape[0]} pixels)."
        )

    def detect_corners_clicked(self):
        """
        Run Harris corner detection with the selected parameter values.

        The response map, thresholded response, and detected corner
        coordinates are produced and displayed in a Matplotlib window.
        """
        if self.img is None:
            messagebox.showwarning(
                "No Image Selected",
                "Select an image before detecting corners."
            )
            return

        window_size = self.window_slider.get()
        k = self.k_slider.get()
        percentile = self.percentile_slider.get()
        min_distance = self.min_slider.get()

        try:
            response_map, thresh_response, coords_max = detect_corners(
                self.img,
                window_size,
                k,
                percentile,
                min_distance
            )
        except (ValueError, TypeError) as error:
            messagebox.showerror(
                "Corner Detection Failed",
                str(error)
            )
            return
        except Exception as error:
            messagebox.showerror(
                "Unexpected Error",
                f"An unexpected error occurred:\n{error}"
            )
            return

        self.status_var.set(
            f"Detected {len(coords_max)} corner points."
        )

        self.display_results(
            response_map,
            thresh_response,
            coords_max
        )

    def display_results(
        self,
        response_map,
        thresh_response,
        coords_max
    ):
        """
        Display the intermediate and final corner detection results.

        Parameters
        ----------
        response_map : numpy.ndarray
            Raw Harris corner response map.

        thresh_response : numpy.ndarray
            Harris response map after percentile thresholding.

        coords_max : numpy.ndarray
            Detected corner coordinates after non-maximum suppression.
        """
        images = [
            response_map,
            thresh_response,
            self.img
        ]

        titles = [
            "Harris response map (raw)",
            "Harris response map (after threshold)",
            "Detected corners after NMS"
        ]

        init_figures()

        for index, (image, title) in enumerate(
            zip(images, titles),
            start=1
        ):
            add_figure(
                image,
                index,
                title
            )

        if len(coords_max) > 0:
            mark_corners(coords_max)

        show_figures()


def start_gui():
    """
    Create and start the Harris Corner Detector interface.

    This function creates the Tkinter root window, initializes the
    application, and starts the Tk event loop.
    """
    root = tk.Tk()
    CornerDetectorGUI(root)
    root.mainloop()
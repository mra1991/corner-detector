"""
Entry point for the corner detection application.

This module launches the graphical user interface, allowing the user
to interactively select an image, adjust corner detection parameters,
and visualize the processing results.

The application follows these main steps:

1. Load the input image as a grayscale image.
2. Compute horizontal and vertical image gradients.
3. Compute products of image gradients.
4. Build the second moment matrix for each pixel.
5. Compute the Harris response value.
6. Threshold weak corner responses.
7. Apply non-maximum suppression.
8. Display the detected corners on the original image.

"""
from gui import start_gui

if __name__ == "__main__":
    start_gui()
    
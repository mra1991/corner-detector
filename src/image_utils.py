"""
Image loading and visualization utilities.

This module provides helper functions for loading grayscale images,
creating Matplotlib figures, and displaying one or more images during
the edge detection process.
"""
import matplotlib.pylab as plt
from skimage import io

def load_image(filepath):
    """
    Load an image from disk as a grayscale floating-point image.

    Args:
        filepath: Path to the image file.

    Returns:
        A 2D NumPy array representing the grayscale image.
    """
    return io.imread(filepath, as_gray=True)

def add_figure(image, plot_counter, title="Figure"):
    """
    Create a new Matplotlib figure and display an image.

    Parameters
    ----------
    image : numpy.ndarray
        Image to display.
        
    plot_counter : int
        index of sub-plot starting from one.

    title : str, optional
        Title displayed above the image.

    Returns
    -------
    None
    """
    plt.subplot(2, 2, plot_counter)
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

def show_figures():
    """
    Display all previously created Matplotlib figures.
    """
    plt.tight_layout()
    plt.show()
    
def mark_corners(coords_max):
    # coords_max is an array of [row, col] pairs
    # plot an 'x' at each detected corner location
    plt.plot(coords_max[:, 1], coords_max[:, 0], 'x')
    
def init_figures():
    """
    Set the size and caption 
    """
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title("Corner Detection Results")


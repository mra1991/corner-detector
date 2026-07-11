"""
Image filtering and convolution operations.

This module provides functions for zero padding, 2D filtering
and computation of horizontal and vertical image
gradients using the Sobel operator.
"""
import numpy as np
from scipy.ndimage import correlate

def zero_pad(image, pad_height, pad_width):
    """
    Pad an image with zeros around its border.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    pad_height : int
        Number of rows added to the top and bottom.

    pad_width : int
        Number of columns added to the left and right.

    Returns
    -------
    numpy.ndarray
        Zero-padded image.
    """
    image_height, image_width = image.shape
    out = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width))
    out[pad_height: image_height + pad_height, pad_width: image_width + pad_width] = image
    return out

def filter2d(image, kernel):
    """
    Apply a 2D cross-correlation filter to a grayscale image using zero padding.

    Args:
        image: 2D NumPy array representing a grayscale image.
        kernel: 2D NumPy array representing the filter kernel.

    Returns:
        A 2D NumPy array with the filtered image.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    out = np.zeros((image_height, image_width)) 
    image = zero_pad(image, kernel_height//2, kernel_width//2 )
    for h in range(image_height):
        for w in range(image_width):
            window = image[h: h + kernel_height, w: w + kernel_width]
            out[h, w] = np.sum(window * kernel)
    return out

def fast_filter2d(image, kernel):
    """
    Apply fast 2D cross-correlation to a grayscale image.

    This function performs the same operation as the educational
    ``filter2d`` implementation, but uses SciPy's optimized compiled
    routines for significantly faster processing. The image is padded
    with zeros at its boundaries, and the kernel is applied without
    being flipped.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional grayscale input image.

    kernel : numpy.ndarray
        Two-dimensional filter kernel. Odd kernel dimensions are
        recommended so that the kernel has a well-defined center.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same dimensions as the input image.

    Raises
    ------
    ValueError
        If the image or kernel is not two-dimensional.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    if image.ndim != 2:
        raise ValueError("The input image must be a 2D grayscale array.")

    if kernel.ndim != 2:
        raise ValueError("The kernel must be a 2D array.")

    return correlate(
        image,
        kernel,
        mode="constant",
        cval=0.0
    )

def partial_x(image):
    """
    Compute the horizontal image gradient using the Sobel operator.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    numpy.ndarray
        Gradient in the x-direction.
    """
    sobel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])
    return fast_filter2d(image, sobel_x)

def partial_y(image):
    """
    Compute the vertical image gradient using the Sobel operator.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    numpy.ndarray
        Gradient in the y-direction.
    """
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    return fast_filter2d(image, sobel_y)



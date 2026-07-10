"""
Image filtering and convolution operations.

This module provides functions for zero padding, 2D filtering
and computation of horizontal and vertical image
gradients using the Sobel operator.
"""
import numpy as np

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
    return filter2d(image, sobel_x)

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
    return filter2d(image, sobel_y)



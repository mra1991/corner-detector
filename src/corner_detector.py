"""
Core Harris corner detection algorithms.

This module implements the Harris corner detection pipeline by computing
image gradients, constructing the second moment matrix, evaluating the
Harris response, thresholding weak responses, and applying non-maximum
suppression to identify corner locations.
"""
import numpy as np
from filters import filter2d, fast_filter2d, partial_x, partial_y
from skimage.feature import peak_local_max

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute the Harris corner response map for a grayscale image.

    The algorithm computes image gradients using the Sobel operator,
    constructs the second moment matrix over a local window, and evaluates
    the Harris response

    R = det(M) - k(trace(M))²

    at every pixel.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image.

    window_size : int, optional
        Size of the window used to compute the second moment matrix.

    k : float, optional
        Harris detector sensitivity parameter.

    Returns
    -------
    numpy.ndarray
        Harris corner response map of the same size as the input image.
    """
    response = None
    
    # step 1: compute partial derivatives in x and y directions
    Ix = partial_x(img)
    Iy = partial_y(img)

    # step 2: compute products of derivatives at each pixel
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # step 3: compute second moment matrix in a window around each pixel 
    window = np.ones((window_size, window_size))
    Sxx = fast_filter2d(Ixx, window)
    Syy = fast_filter2d(Iyy, window)
    Sxy = fast_filter2d(Ixy, window)

    # step 4: compute Harris response:
    # R = det(M) - k * (trace(M))^2
    # where M = [[Sxx, Sxy],
    #            [Sxy, Syy]]
    # det(M) = Sxx*Syy - Sxy^2
    # trace(M) = Sxx + Syy
    det_M = (Sxx * Syy) - (Sxy * Sxy)
    trace_M = Sxx + Syy
    response = det_M - k * (trace_M * trace_M)

    return response

def detect_corners(img, window_size, k, thresh_percentile, min_dis):
    """
    Detect Harris corners in a grayscale image.

    This function computes the Harris response map, thresholds weak
    responses, and applies non-maximum suppression to identify the final
    corner locations.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image.

    window_size : int
        Size of the window used to compute the second moment matrix.

    k : float
        Harris detector sensitivity parameter.

    thresh_percentile : float
        Percentile used to threshold the Harris response map.

    min_dis : int
        Minimum distance between detected corner points during
        non-maximum suppression.

    Returns
    -------
    response_map : numpy.ndarray
        Raw Harris corner response map.

    thresh_response : numpy.ndarray
        Thresholded Harris response map.

    coords_max : numpy.ndarray
        Coordinates of the detected corner points after non-maximum
        suppression.
    """
 
    # Compute Harris corner response
    response_map = harris_corners(img, window_size, k)   # window_size and k to be played with for best results-----------

    # Threshold on response
    threshold = np.percentile(response_map, thresh_percentile)

    # creating a threshold response image
    thresh_response = response_map.copy()
    thresh_response[thresh_response < threshold] = 0    # keep only strong responses

    # Perform non-max suppression by finding peak local maximum
    # finding the coordinates of local maximum 
    coords_max = peak_local_max(
        response_map,
        min_dis,       # how far away the corners should be. play with this for best results ------------------------------
        threshold_abs=threshold     # to ensures we only keep strong corners
    )
    
    return response_map, thresh_response, coords_max

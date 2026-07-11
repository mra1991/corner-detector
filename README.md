# Harris Corner Detector

A Python application for detecting corners in grayscale images using the **Harris Corner Detection** algorithm.

Created by **Mohammadreza Abolhassani**.

## Overview

This project detects **corners**, which are important feature points in computer vision. Corners are useful because they are visually distinctive and can be used in tasks such as image matching, object recognition, motion tracking, and 3D reconstruction.

The application includes a simple **Tkinter GUI** that allows the user to load an image, adjust Harris detector parameters, preview the selected image, and display the results.

## Features

- Load a grayscale image from disk
- Preview the selected image in the GUI
- Adjust Harris corner detection parameters
- Compute image gradients using Sobel filters
- Compute the Harris response map
- Apply thresholding
- Apply non-maximum suppression
- Display detected corners on the original image

## Project Structure

```text
corner-detector/
│
├── sample_images/
│   └── building.jpg
│
└── src/
    ├── main.py
    ├── gui.py
    ├── corner_detector.py
    ├── filters.py
    └── image_utils.py
```

## Requirements

Install the required libraries:

```bash
pip install numpy matplotlib scikit-image pillow scipy
```

## How to Run

From the `src` folder, run:

```bash
python main.py
```

## Harris Corner Detection Pipeline

The application follows these main steps:

1. Load the input image as a grayscale image.
2. Compute horizontal and vertical image gradients.
3. Compute products of image gradients.
4. Build the second moment matrix for each pixel.
5. Compute the Harris response value.
6. Threshold weak corner responses.
7. Apply non-maximum suppression.
8. Display the detected corners on the original image.

## Algorithm Explanation

The Harris detector begins by computing image gradients:

```text
Ix = horizontal gradient
Iy = vertical gradient
```

These gradients measure how pixel intensity changes in the x and y directions.

Then the algorithm computes:

```text
Ixx = Ix²
Iyy = Iy²
Ixy = IxIy
```

These values describe the local intensity structure around each pixel.

For each pixel, the gradient products are summed over a local window to create the second moment matrix:

```text
M = [ Sxx  Sxy ]
    [ Sxy  Syy ]
```

The Harris response is then computed as:

```text
R = det(M) - k(trace(M))²
```

where:

```text
det(M) = SxxSyy - Sxy²
trace(M) = Sxx + Syy
```

The response value helps classify each region:

- Large positive R means the region is likely a corner.
- Large negative R means the region is likely an edge.
- Near-zero R means the region is likely flat.

## GUI Parameters

| Parameter | Meaning |
|---|---|
| Window size | Neighborhood size used to compute the second moment matrix |
| k | Harris sensitivity parameter |
| Threshold percentile | Controls how strong a response must be to count as a corner |
| Minimum distance | Minimum spacing between detected corners |

## Output

The output window shows:

1. Raw Harris response map
2. Thresholded Harris response map
3. Original image with detected corners marked

## Notes

This project demonstrates the Harris corner detection pipeline step by step instead of relying on a fully built-in corner detector. The goal is to understand how gradients, local structure, thresholding, and non-maximum suppression work together to detect corners.
